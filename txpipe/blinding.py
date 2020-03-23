###
#
# TO BE DEPRECATED VERY SOON
# Leaving here to be able to see what is going on
#
####




from .base_stage import PipelineStage
from .data_types import SACCFile
import numpy as np
import warnings

class TXBlinding(PipelineStage):
    """
    Blinding the data vectors.

    """
    name='TXBlinding'
    inputs = [
        ('twopoint_data', SACCFile),
    ]
    outputs = [
        ('twopoint_data', SACCFile),
    ]
    config_options = {
        'seed': 1972,  ## seed uniquely specifies the shift in parameters
        'Omega_b': [0.0485, 0.001], ## fiducial_model_value, shift_sigma
        'Omega_c': [0.2545, 0.01],
        'w0': [-1.0, 0.1],
        'h': [0.682, 0.02],
        'sigma8': [0.801, 0.01],
        'n_s': [0.971, 0.03],
        'b0': 0.95,  ### we assume bias to be of the form b0/growth
        'delete_unblinded': True 
    }


    def run(self):
        """
        Run the analysis for this stage.
        
         - Load two point SACC file
         - Blinding it 
         - Output blinded data
         - Deletete unblinded data
        """
        import sacc
        sys.stdout.flush()

        if self.rank==0:
            unblinded_fname = self.get_input('twopoint_data')+'.UNBLINDED.RABID'
            sacc = sacc.load_fits(unblinded_fname)
            blinded_sacc = self.blind_muir(sacc, meta)
            blinded_sacc.save_fits(self.get_output('twopoint_data'), overwrite=True)
            if config['delete_unblinded']:
                print ("Deleting unblinded file...")
                os.remove (unblinded_fname)


    def blind_muir(self, sacc, meta):
        import pyccl as ccl
        import firecrown
        import io
        ## here we actually do blinding
        if self.rank==0:
            print(f"Blinding... ")
        np.random.seed(self.config["blind_seed"])
        # blind signature -- this ensures seed is consistent across
        # numpy versions
        blind_sig = ''.join(format(x, '02x') for x in np.random.bytes(4))
        if self.rank==0:
            print ("Blinding signature: %s"%(blind_sig))


        fid_params = {
            'Omega_b':  self.config['blind_Omega_b'][0],
            'Omega_c':  self.config['blind_Omega_c'][0],
            'h': self.config['blind_h'][0],
            'w0': self.config['blind_w0'][0],
            'sigma8': self.config['blind_sigma8'][0],
            'n_s': self.config['blind_n_s'][0],
        }
        ## now get biases
        bz={}
        fidCosmo=ccl.Cosmology(**fid_params)
        for key,tracer in sacc.tracers.items():
            if 'lens' in key:
                zeff = (tracer.z*tracer.nz).sum()/tracer.nz.sum()
                bz[key] = self.config['blind_b0']/ccl.growth_factor(fidCosmo,1/(1+zeff)) 
        
        offset_params = copy.copy(fid_params)
        for par in fid_params.keys():
            offset_params [par] += self.config['blind_'+par][1]*np.random.normal(0.,1.)
        fc_config = """
           parameters:
             ## missing params set in fid_params and offset_params
             Omega_k: 0.0
             wa: 0.0

             one: 1
"""
        for k,v in bz.items():
            fc_config+="             bias_{lens}: {bias}\n".format(lens=k,bias=v)
        fc_config += """
           two_point:
             module: firecrown.ccl.two_point
           
             systematics:
               dummy:
                  kind: PhotoZShiftBias
                  delta_z: one

        """

        ### sources
        fc_config+="     sources: \n"
        srclist=[]
        lenslist=[]
        for key,tracer in sacc.tracers.items():
            ## This is a hack, need to think how to do this better
            if 'source' in key:
                fc_config+="""
                                  {src}:
                                     kind: WLSource
                                     sacc_tracer: {src}\n""".format(src=key)
                srclist.append(key)
            if 'lens' in key:
                fc_config+="""
                                  {lens}:
                                     kind: NumberCountsSource
                                     bias: bias_{lens}
                                     sacc_tracer: {lens}\n""".format(lens=key)
                lenslist.append(key)
        fc_config+="             statistics: \n"
        ##
        ## list(dict.fromkeys()) gives unique elements while preserving order
        types=list(dict.fromkeys([(p.data_type, p.tracers,
                                   "{dtype}_{tracer1}_{tracer2}".format(dtype=p.data_type,
                                            tracer1=p.tracers[0],tracer2=p.tracers[1]))
                                   for p in sacc.data]))
        for dtype,(tracer1,tracer2),fcname in types:
            fc_config+="""
                                 {fcname}:
                                   sources: ['{tracer1}', '{tracer2}']
                                   sacc_data_type: {dtype}
                   """.format(fcname=fcname, dtype=dtype,tracer1=tracer1,tracer2=tracer2)
        ## now try to get predictions
        pred={}
        for name,pars in [('fid',fid_params),('ofs',offset_params)]:
            print ("Calling firecrown : %s"%(name))

            config, data = firecrown.parse(io.StringIO(fc_config),
               settings={'two_point':{'sacc_data':sacc},'parameters':pars})
            cosmo = firecrown.get_ccl_cosmology(config['parameters'])
            firecrown.compute_loglike(cosmo=cosmo, data=data)
            pred[name] = np.hstack([data['two_point']['data']['statistics'][n].predicted_statistic_ for
                                  _,_,n in types])
            ## if there is some sanity this should work
            assert(len(pred[name])==len(sacc))

        diffvec = pred['ofs']-pred['fid']
        ## now add offsets
        for p, delta in zip(sacc.data,diffvec):
            p.value+=delta
        print ("Blinding done.")
            
        return  sacc

            
