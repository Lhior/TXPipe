from .base_stage import PipelineStage
from .data_types import ShearCatalog, HDFFile, YamlFile, SACCFile
import numpy as np
import warnings
import os
import pickle

# require TJPCov to be in PYTHONPATH
d2r=np.pi/180

# Needed changes: 1) ell and theta spacing could be further optimized 2) coupling matrix

class TXFourierGaussianCovariance(PipelineStage):
    name='TXFourierGaussianCovariance'
    do_xi=False
    
    inputs = [
        ('fiducial_cosmology', YamlFile),    # For the cosmological parameters
        ('twopoint_data_fourier', SACCFile), # For the binning information
        ('tracer_metadata', HDFFile),        # For metadata
    ]

    outputs = [
        ('summary_statistics_fourier', SACCFile),
    ]

    config_options = {
        'pickled_wigner_transform': '',
        'use_true_shear': False,
        
    }


    def run(self):
        import pyccl as ccl
        import sacc
        import tjpcov
        import threadpoolctl

        # read the fiducial cosmology
        cosmo = self.read_cosmology()

        # read binning
        two_point_data = self.read_sacc()

        # read the n(z) and f_sky from the source summary stats        
        meta = self.read_number_statistics()
        
        # Binning choices. The ell binning is a linear piece with all the
        # integer values up to 500 -- these are from firecrown, might need 
        # to change later
        meta['ell'] = np.concatenate(
            (np.linspace(2, 500-1, 500-2), 
             np.logspace(np.log10(500), np.log10(6e4), 500))
            )

        # Theta binning - log spaced between 1 .. 300 arcmin.
        meta['theta'] = np.logspace(np.log10(1/60), np.log10(300./60), 3000) 

        #C_ell covariance
        cov = self.compute_covariance(cosmo, meta, two_point_data=two_point_data)
        
        self.save_outputs(two_point_data, cov)

    def save_outputs(self, two_point_data, cov):
        filename = self.get_output('summary_statistics_fourier')
        two_point_data.add_covariance(cov)
        two_point_data.save_fits(filename, overwrite=True)


    def read_cosmology(self):
        import pyccl as ccl
        filename = self.get_input('fiducial_cosmology')
        cosmo = ccl.Cosmology.read_yaml(filename)

        print("COSMOLOGY OBJECT:")
        print(cosmo)
        return cosmo

    def read_sacc(self):
        import sacc
        f = self.get_input('twopoint_data_fourier')
        two_point_data = sacc.Sacc.load_fits(f)

        # Remove the data types that we won't use for inference
        mask = [
            two_point_data.indices(sacc.standard_types.galaxy_shear_cl_ee),
            two_point_data.indices(sacc.standard_types.galaxy_shearDensity_cl_e),
            two_point_data.indices(sacc.standard_types.galaxy_density_cl),
            # not doing b-modes, do we want to?
        ]
        print("Length before cuts = ", len(two_point_data))
        mask = np.concatenate(mask)
        two_point_data.keep_indices(mask)
        print("Length after cuts = ", len(two_point_data))
        two_point_data.to_canonical_order()

        return two_point_data


    def read_number_statistics(self):
        input_data = self.open_input('tracer_metadata')

        # per-bin quantities
        N_eff = input_data['tracers/N_eff'][:]
        N_lens = input_data['tracers/lens_counts'][:]
        if self.config['use_true_shear']:
            nbins = len(input_data['tracers/sigma_e'][:])
            sigma_e = np.array([0. for i in range(nbins)])
        else:
            sigma_e = input_data['tracers/sigma_e'][:]

        # area in sq deg
        area_deg2 = input_data['tracers'].attrs['area']
        area_unit = input_data['tracers'].attrs['area_unit']
        if area_unit != 'deg^2':
            raise ValueError("Units of area have changed")

        input_data.close()

        # area in steradians and sky fraction
        area = area_deg2 * np.radians(1)**2
        area_arcmin2 = area_deg2 * 60**2
        full_sky = 4*np.pi
        f_sky = area / full_sky

        # Density information from counts
        n_eff = N_eff / area
        n_lens = N_lens / area

        # for printing out only
        n_eff_arcmin = N_eff / area_arcmin2
        n_lens_arcmin = N_lens / area_arcmin2

        # Feedback
        print(f"area =  {area_deg2:.1f} deg^2")
        print(f"f_sky:  {f_sky}")
        print(f"N_eff:  {N_eff} (totals)")
        print(f"N_lens: {N_lens} (totals)")
        print(f"n_eff:  {n_eff} / steradian")
        print(f"     =  {np.around(n_eff_arcmin,2)} / sq arcmin")
        print(f"lens density: {n_lens} / steradian")
        print(f"            = {np.around(n_lens_arcmin,2)} / arcmin")

        # Pass all this back as a dictionary
        meta = {
            'f_sky': f_sky,
            'sigma_e': sigma_e,
            'n_eff': n_eff,
            'n_lens': n_lens,
        }

        return meta

    def get_tracer_info(self, cosmo, meta, two_point_data):
        # Generates CCL tracers from n(z) information in the data file
        import pyccl as ccl
        ccl_tracers={}
        tracer_noise={}

        for tracer in two_point_data.tracers:

            # Pull out the integer corresponding to the tracer index
            tracer_dat = two_point_data.get_tracer(tracer)
            nbin = int(two_point_data.tracers[tracer].name.split("_")[1])

            z = tracer_dat.z.copy().flatten()
            nz = tracer_dat.nz.copy().flatten()

            # Identify source tracers and gnerate WeakLensingTracer objects
            # based on them
            if 'source' in tracer or 'src' in tracer:
                sigma_e = meta['sigma_e'][nbin]
                n_eff = meta['n_eff'][nbin]
                ccl_tracers[tracer] = ccl.WeakLensingTracer(cosmo, dndz=(z, nz)) #CCL automatically normalizes dNdz
                tracer_noise[tracer] = sigma_e**2 / n_eff

            # or if it is a lens bin then generaete the corresponding
            # CCL tracer class
            elif 'lens' in tracer:
                b = 1.0*np.ones(len(z))  # place holder
                n_gal = meta['n_lens'][nbin]
                tracer_noise[tracer] = 1 / n_gal
                ccl_tracers[tracer] = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z,nz), bias=(z,b))
        
        return ccl_tracers, tracer_noise

    def get_spins(self, tracer_comb):
        # Get the Wigner Transform factors
        WT_factors={}
        WT_factors['lens','source'] = (0, 2)
        WT_factors['source','lens'] = (2, 0) #same as (0,2)
        WT_factors['source','source'] = {'plus':(2,2), 'minus':(2, -2)}
        WT_factors['lens','lens'] = (0, 0)

        tracers=[]
        for i in tracer_comb:
            if 'lens' in i:
                tracers+=['lens']
            if 'source' in i:
                tracers+=['source']
        return WT_factors[tuple(tracers)]

    # compute a single covariance matrix for a given pair of C_ell or xi.  
    def compute_covariance_block(self, cosmo, meta, ell_bins,
        tracer_comb1=None, tracer_comb2=None, ccl_tracers=None, tracer_Noise=None,
        two_point_data=None,
        xi_plus_minus1='plus', xi_plus_minus2='plus',
        cache=None, WT=None,
        ):
        import pyccl as ccl
        from tjpcov import bin_cov

        cl = {}

        # tracers 1,2,3,4 = tracer_comb1[0], tracer_comb1[1], tracer_comb2[0], tracer_comb2[1]
        # In the dicts below we use '13' to indicate C_ell_(1,3), etc.
        # This index maps to this usae
        reindex = {
            (0, 0): 13,
            (1, 1): 24,
            (0, 1): 14,
            (1, 0): 23,
        }

        ell = meta['ell']

        # Getting all the C_ell that we need, saving the results in a cache
        # for later re-use
        for i in (0,1):
            for j in (0,1):
                local_key = reindex[(i,j)]
                # For symmetric pairs we may have saved the C_ell the other
                # way around, so try both keys
                cache_key1 = (tracer_comb1[i], tracer_comb2[j])
                cache_key2 = (tracer_comb2[j], tracer_comb1[i])
                if cache_key1 in cache:
                    cl[local_key] = cache[cache_key1]
                elif cache_key2 in cache:
                    cl[local_key] = cache[cache_key2]
                else:
                    # If not cached then we must compute
                    t1 = tracer_comb1[i]
                    t2 = tracer_comb2[j]
                    c = ccl.angular_cl(cosmo, ccl_tracers[t1], ccl_tracers[t2], ell)
                    print("Computed C_ell for ", cache_key1)
                    cache[cache_key1] = c
                    cl[local_key] = c

        # The shape noise C_ell values.
        # These are zero for cross bins and as computed earlier for auto bins
        SN={}
        SN[13] = tracer_Noise[tracer_comb1[0]] if tracer_comb1[0] == tracer_comb2[0] else 0
        SN[24] = tracer_Noise[tracer_comb1[1]] if tracer_comb1[1] == tracer_comb2[1] else 0
        SN[14] = tracer_Noise[tracer_comb1[0]] if tracer_comb1[0] == tracer_comb2[1] else 0
        SN[23] = tracer_Noise[tracer_comb1[1]] if tracer_comb1[1] == tracer_comb2[0] else 0


        # The overall normalization factor at the front of the matrix
        if self.do_xi:
            norm = np.pi * 4 * meta['f_sky']
        else: 
            norm = (2*ell + 1) * np.gradient(ell) * meta['f_sky']

        # The coupling is an identity matrix at least when we neglect
        # the mask
        coupling_mat = {}
        coupling_mat[1324] = np.eye(len(ell))
        coupling_mat[1423] = np.eye(len(ell))

        # Initial covariance of C_ell components
        cov = {}
        cov[1324] = np.outer(cl[13] + SN[13], cl[24] + SN[24]) * coupling_mat[1324]
        cov[1423] = np.outer(cl[14] + SN[14], cl[23] + SN[23]) * coupling_mat[1423]

        # for shear-shear components we also add a B-mode contribution
        first_is_shear_shear = ('source' in tracer_comb1[0]) and ('source' in tracer_comb1[1])
        second_is_shear_shear = ('source' in tracer_comb2[0]) and ('source' in tracer_comb2[1])

        if self.do_xi and (first_is_shear_shear or second_is_shear_shear):
            # this adds the B-mode shape noise contribution.
            # We assume B-mode power (C_ell) is 0
            Bmode_F = 1
            if xi_plus_minus1 != xi_plus_minus2:
                # in the cross term, this contribution is subtracted.
                # eq. 29-31 of https://arxiv.org/pdf/0708.0387.pdf
                Bmode_F=-1 
            # below the we multiply zero to maintain the shape of the Cl array, these are effectively 
            # B-modes
            cov[1324] += np.outer(cl[13]*0 + SN[13], cl[24]*0 + SN[24]) * coupling_mat[1324] * Bmode_F
            cov[1423] += np.outer(cl[14]*0 + SN[14], cl[23]*0 + SN[23]) * coupling_mat[1423] * Bmode_F

        cov['final']=cov[1423]+cov[1324]

        if self.do_xi:
            s1_s2_1 = self.get_spins(tracer_comb1)
            s1_s2_2 = self.get_spins(tracer_comb2)

            # For the shear-shear we have two sets of spins, plus and minus,
            # which are returned as a dict, so we need to pull out the one we need
            # Otherwise it's just specified as a tuple, e.g. (2,0)
            if isinstance(s1_s2_1, dict):
                s1_s2_1 = s1_s2_1[xi_plus_minus1]
            if isinstance(s1_s2_2, dict):
                s1_s2_2 = s1_s2_2[xi_plus_minus2]

            # Use these terms to project the covariance from C_ell to xi(theta)
            th, cov['final']=WT.projected_covariance2(
                l_cl=ell, s1_s2=s1_s2_1, s1_s2_cross=s1_s2_2, cl_cov=cov['final'])

        # Normalize
        cov['final'] /= norm

        # Put the covariance into bins. 
        # This is optional in the case of a C_ell covariance (only if bins in ell are
        # supplied, otherwise the matrix is for each ell value individually).  It is
        # required for real-space covariances since these are always binned.
        if self.do_xi:
            thb, cov['final_b'] = bin_cov(r=th/d2r, r_bins=ell_bins, cov=cov['final'])
        else:
            if ell_bins is not None:
                lb, cov['final_b'] = bin_cov(r=ell, r_bins=ell_bins, cov=cov['final'])
        return cov
    
    def get_angular_bins(self, two_point_data):
        # Assume that the ell binning is the same for each of the bins.
        # This is true in the current pipeline.
        X = two_point_data.get_data_points('galaxy_shear_cl_ee',i=0,j=0)
        # Further assume that the ell ranges are contiguous, so that
        # the max value of one window is the min value of the next.
        # So we just need the lower edges of each bin and then the
        # final maximum value of the last bin
        ell_edges = [x['window'].min for x in X]
        ell_edges.append(X[-1]['window'].max)

        return np.array(ell_edges)

    def make_wigner_transform(self, meta):
        import threadpoolctl
        from tjpcov import wigner_transform

        path = self.config['pickled_wigner_transform']
        if path:
            if os.path.exists(path):
                print(f"Loading precomputed wigner transform from {path}")
                WT = pickle.load(open(path, 'rb'))
                return WT
            else:
                print(f"Precomputed wigner transform {path} not found.")
                print("Will compute it and then save it.")

        # We don't want to use n processes with n threads each by accident,
        # where n is the number of CPUs we have
        # so for this bit of the code, which uses python's multiprocessing,
        # we limit the number of threads that numpy etc can use.
        # After this is finished this will switch back to allowing all the CPUs
        # to be used for threading instead.
        num_processes = int(os.environ.get("OMP_NUM_THREADS", 1))
        print("Generating Wigner Transform.")
        with threadpoolctl.threadpool_limits(1):
            WT = wigner_transform(
                l = meta['ell'],
                theta = meta['theta'] * d2r,
                s1_s2 = [(2,2), (2,-2), (0,2), (2,0), (0,0)],
                ncpu = num_processes,
                )
            print("Computed Wigner Transform.")

        if path:
            try:
                pickle.dump(WT, open(path, 'wb'))
            except OSError:
                sys.stderr.write(f"Could not save wigner transform to {path}")
        return WT


    #compute all the covariances and then combine them into one single giant matrix
    def compute_covariance(self, cosmo, meta, two_point_data):
        from tjpcov import bin_cov

        ccl_tracers,tracer_Noise = self.get_tracer_info(cosmo, meta, two_point_data=two_point_data)
        # we will loop over all these
        tracer_combs = two_point_data.get_tracer_combinations() 
        N2pt = len(tracer_combs)


        WT = self.make_wigner_transform(meta)
        
        # the bit below is just counting the number of 2pt functions, and accounting 
        # for the fact that xi needs to be double counted
        N2pt0 = 0
        if self.do_xi:
            N2pt0 = N2pt
            tracer_combs_temp = tracer_combs.copy()
            for combo in tracer_combs:
                if ('source' in combo[0]) and ('source' in combo[1]):
                    N2pt += 1
                    tracer_combs_temp += [combo]
            tracer_combs = tracer_combs_temp.copy()

        ell_bins = self.get_angular_bins(two_point_data)
        Nell_bins = len(ell_bins) - 1


        cov_full=np.zeros((Nell_bins*N2pt, Nell_bins*N2pt))
        count_xi_pm1 = 0
        count_xi_pm2 = 0
        cl_cache = {}
        xi_pm = [[('plus','plus'), ('plus', 'minus')], [('minus','plus'), ('minus', 'minus')]]

        # Look through the chunk of matrix, tracer pair by tracer pair
        for i in range(N2pt):
            tracer_comb1 = tracer_combs[i]

            if i == N2pt0:
                count_xi_pm1 = 1

            for j in range(i, N2pt):
                tracer_comb2 = tracer_combs[j]
                print(f"Computing {tracer_comb1} x {tracer_comb2}: chunk ({i},{j}) of ({N2pt},{N2pt})")

                if j == N2pt0:
                    count_xi_pm2 = 1

                if self.do_xi and ('source' in tracer_comb1) and ('source' in tracer_comb2):
                    cov_ij = self.compute_covariance_block(
                        cosmo,
                        meta,
                        ell_bins, 
                        tracer_comb1=tracer_comb1,
                        tracer_comb2=tracer_comb2,
                        ccl_tracers=ccl_tracers,
                        tracer_Noise=tracer_Noise, 
                        two_point_data=two_point_data,
                        xi_plus_minus1=xi_pm[count_xi_pm1, count_xi_pm2][0],
                        xi_plus_minus2=xi_pm[count_xi_pm1, count_xi_pm2][1],
                        cache=cl_cache,
                        WT=WT,
                    )

                else:
                    cov_ij = self.compute_covariance_block(
                        cosmo,
                        meta,
                        ell_bins,
                        tracer_comb1=tracer_comb1,
                        tracer_comb2=tracer_comb2,
                        ccl_tracers=ccl_tracers,
                        tracer_Noise=tracer_Noise,
                        two_point_data=two_point_data,
                        cache=cl_cache,
                        WT=WT,
                    )

                # Fill in this chunk of the matrix
                cov_ij = cov_ij['final_b']
                # Find the right location in the matrix
                start_i = i * Nell_bins
                start_j = j * Nell_bins
                end_i = start_i + Nell_bins
                end_j = start_j + Nell_bins
                # and fill it in, and the transpose component
                cov_full[start_i:end_i, start_j:end_j] = cov_ij
                cov_full[start_j:end_j, start_i:end_i] = cov_ij.T

        try:
            np.linalg.cholesky(cov_full)
        except:        
            print("liAnalg.LinAlgError: Covariance not positive definite! "
                "Most likely this is a problem in xim. "
                "We will continue for now but this needs to be fixed.")

        return cov_full


class TXRealGaussianCovariance(TXFourierGaussianCovariance):
    name='TXRealGaussianCovariance'
    do_xi = True

    inputs = [
        ('fiducial_cosmology', YamlFile),     # For the cosmological parameters
        ('twopoint_data_real', SACCFile),     # For the binning information
        ('tracer_metadata', HDFFile),         # For metadata

    ]

    outputs = [
        ('summary_statistics_real', SACCFile),
    ]

    config_options = {
        'min_sep':2.5,  # arcmin
        'max_sep':250,
        'nbins':20,
        'pickled_wigner_transform': '',
    }

    def run(self):
        super().run()

    def get_angular_bins(self, two_point_data):
        # this should be changed to read from sacc file
        th_arcmin = np.logspace(np.log10(self.config['min_sep']), np.log10(self.config['max_sep']), self.config['nbins']+1)
        return th_arcmin/60.0


    def read_sacc(self):
        import sacc
        f = self.get_input('twopoint_data_real')
        two_point_data = sacc.Sacc.load_fits(f)

        mask = [
            two_point_data.indices(sacc.standard_types.galaxy_density_xi),
            two_point_data.indices(sacc.standard_types.galaxy_shearDensity_xi_t),
            two_point_data.indices(sacc.standard_types.galaxy_shear_xi_plus),
            two_point_data.indices(sacc.standard_types.galaxy_shear_xi_minus),
        ]
        mask = np.concatenate(mask)
        two_point_data.keep_indices(mask)

        two_point_data.to_canonical_order()

        return two_point_data


    def save_outputs(self, two_point_data, cov):
        filename = self.get_output('summary_statistics_real')
        two_point_data.add_covariance(cov)
        two_point_data.save_fits(filename, overwrite=True)
