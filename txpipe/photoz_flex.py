from ceci import PipelineStage
from txpipe.data_types import HDFFile
import os
import sys
import numpy as np
import pandas as pd
import pickle
import time
import flexcode
from flexcode.helpers import make_grid #need to make proper z_grid

# This class runs the python3 version of BPZ from the command line                         
class PZPDFlex(PipelineStage):
    """Pipeline stage to run a trained model of FlexZBoost in python
       The code will be set up to read and write hdf5 files in the
       same formats as BPZPipe as a TXPipe pipeline stage
                                                                                           
    """
    name = "PZPDFlex"

    inputs = [
        ('photometry_catalog', HDFFile),]

    outputs = [
#        ('photoz_pdfs', PhotozPDFFile),                                                   
        ('photoz_pdfs', HDFFile),]

    config_options = {
        "chunk_rows": 1000,
        "bands": "ugrizy",
        "sigma_intrins": 0.05, #"intrinsic" assumed scatter, used in ODDS                  
        "odds_int": 0.99445, #number of sigma_intrins to integrate +/- around peak         
        # note that 1.95993 is the number of sigma you get for old "ODDS" =0.95            
        #in old BPZ, 0.68 is 0.99445                                                       
        #"point_estimate": "mode",  # mean, mode, or median          
        "has_redshift": False, #does the test file have redshift?
        #if so, read in and append to output file.
        "nz": 300, #Number of grid points that FZboost will calculate
        "model_picklefile": str,
        "do_metacal": False, #switch for whether or not to run metacal suffices
    }

    def run(self):
        # Columns we will need from the data                                               


        if self.config['do_metacal']:
            self.suffices = ["", "_1p", "_1m", "_2p", "_2m"]
        else:
            self.suffices = [""]

        bands = self.config['bands']


        cols =  [f'{band}_mag{suffix}' for band in bands for suffix in self.suffices]
        cols += [f'{band}_mag_err' for band in bands]
        cols += ["objectId"]

        has_sz = self.config['has_redshift']
        if has_sz:
            cols += ["redshift"]

        #read in the pre-trained FlexZBoost model
        model_file = self.config['model_picklefile']
        fz_model = pickle.load(open(model_file,'rb'))


        #set up redshift grid
        nz = self.config['nz']
        tmp_grid = make_grid(nz,fz_model.z_min,fz_model.z_max)
        z_grid = tmp_grid.T.flatten()
        # Prepare the output HDF5 file                                                     
        output_file = self.prepare_output(z_grid)

        # Amount of data to load at once                                                   
        chunk_rows = self.config['chunk_rows']
        # Loop through chunks of the data.                                                 
        # Parallelism is handled in the iterate_input function -                           
        # each processor will only be given the sub-set of data it is                      
        # responsible for.  The HDF5 parallel output mode means they can                   
        # all write to the file at once too.                                               
        for start, end, data in self.iterate_hdf('photometry_catalog', "photometry", cols,
                                                 chunk_rows):
            print(f"Process {self.rank} running photo-z for rows {start}-{end}")

            # Calculate the pseudo-fluxes that we need                                     
            new_data = self.preprocess_data(data)
            # Actually run BPZ                                                             
            point_estimates, pdfs = self.estimate_pdfs(fz_model, new_data, nz)

            # Save this chunk of data                                                      
            self.write_output(output_file, start, end, pdfs, point_estimates)

        # Synchronize processors                                                           
        if self.is_mpi():
            self.comm.Barrier()
        output_file.close()

    def prepare_output(self, z_grid):
        """                                                                                
        Prepare the output HDF5 file for writing.                                          
        Note that this is done by all the processes if running in parallel;                
        that is part of the design of HDF5.                                                
                                                                                           
        Parameters                                                                         
        ----------                                                                         
        nobj: int                                                                          
            Number of objects in the catalog                                               
                                                                                           
        z_grid: np 1d array                                                                          
            Redshift grid points that p(z) will be calculated on.
        Returns                                                                            
        -------                                                                            
        f: h5py.File object                                                                
            The output file, opened for writing.                                           
                                                                                           
        """
        has_sz = self.config['has_redshift']
        print(f'has_sz: {has_sz}')
        # Work out how much space we will need.                                            
        cat = self.open_input("photometry_catalog")
        ids = cat['photometry/objectId'][:]
        nobj = ids.size
        nz = len(z_grid)

        if has_sz:
            szs = cat['photometry/redshift'][:]
        cat.close()


        
        # Open the output file.                                                            
        # This will automatically open using the HDF5 mpi-io driver                        
        # if we are running under MPI and the output type is parallel                      
        f = self.open_output('photoz_pdfs', parallel=True)

        # Create the space for output data                                                 
        groupid = f.create_group('id')
        groupid.create_dataset('objectId', (nobj,), dtype = 'i8')

        if has_sz:
            groupsz =f.create_group('true_redshift')
            groupsz.create_dataset('specz', (nobj,), dtype='f4')

        grouppt = f.create_group('point_estimates')
        grouppt.create_dataset('mode', (nobj,), dtype='f4')
        grouppt.create_dataset('mean', (nobj,), dtype='f4')
        grouppt.create_dataset('median', (nobj,), dtype='f4')
        grouppt.create_dataset('odds', (nobj,), dtype = 'f4')

        group = f.create_group('pdf')
        group.create_dataset("z", (nz,), dtype='f4')
        group.create_dataset("pdf", (nobj,nz), dtype='f4')

        # One processor writes the redshift axis to output.                                
        if self.rank==0:
            groupid['objectId'][:] = ids
            group['zgrid'][:] = z_grid
            if has_sz == True:
                groupsz['specz'][:] = szs
        return f


    def preprocess_data(self, data):
        """
        This function makes a new set of data with the i-magnitude and
        the colors and color errors (just mag errors in quadrature)
        input:
          data: iterate_hdf data 
        returns:
          df: pandas dataframe of data (incl multiple suffices if 
          specified)
          
        """
        bands = self.config['bands']
        n_filter = len(bands)
        #read in the i-band magnitude, calculate colors and color
        #errors for the other bands, stick in a dataframe for simplicity
        #add all suffices to single pandas dataframe, will pull out
        #specific columns when estimating pdfs
        
        for ii,suffix in enumerate(self.suffices):
            i_mag = data[f'i_mag{suffix}']
            if ii==0:
                d = {f'i_mag{suffix}': i_mag}
                df = pd.DataFrame(d)

            for i in range(n_filter-1):
                b1 = bands[i]
                b2 = bands[i+1]
                df[f'color_{b1}{b2}{suffix}'] = (
                    np.array(data[f'{b1}_mag{suffix}']) -
                    np.array(data[f'{b2}_mag{suffix}'])
                    )

                df[f'color_err_{b1}{b2}{suffix}'] = np.sqrt(
                    np.array(data[f'{b1}_mag_err{suffix}'])**2.0 +
                    np.array(data[f'{b2}_mag_err{suffix}'])**2.0
                    )

        return df
        
    def estimate_pdfs(self, fz_model, new_data, nz):
        """
        function to actually compute the PDFs and point estimates
        inputs:
        fz_model: flexzboost trained model object
        new_data: pandas dataframe 
          pandas df of data columns
        nz: integer
          number of redshift grid points to evaluate the model on
        Returns:
        point_estimates: numpy nd-array
          point estimates for each of the suffices
        pdfs:
          p(z) evaluated on nz grid points for the suffix == '' data
       """
        bands = self.config['bands']
        n_filter = len(bands)
        n_suffices = len(self.suffices)
        n_gal = len(new_data['i_mag'])

        point_estimates = np.zeros((6*n_suffices, n_gal))
        columns = []
        for s,suffix in enumerate(self.suffices):
            #pull out just the part of the data for the correct suffix
            columns.append(f'i_mag{suffix}')
            for i in range(n_filter-1):
                b1 = bands[i]
                b2 = bands[i+1]
                columns.append(f'color_{b1}{b2}{suffix}')
                columns.append(f'color_err_{b1}{b2}{suffix}')

            suffix_df = new_data[columns]
            suffix_data = suffix_df.to_numpy()
            
            suffix_pdfs,zgridx = fz_model.predict(suffix_data, n_grid=nz)
            zgrid = np.array(zgridx).flatten() #need to flatten zgrid 
            
            if suffix =="":
                pdfs = suffix_pdfs

            for i in range(n_gal):
                pdf = np.array(pdfs[i,:])
                # calculate mean redshift
                point_estimates[6*s+0, i] = (pdf * zgrid).sum()/pdf.sum()
                # calculate z_mode                                                  
                point_estimates[6*s+1, i] = zgrid[np.argmax(pdf)]
                #"median" redshift
                point_estimates[6*s+2, i] = self.pdf_median(zgrid, pdf)
                #calcualte ODDS param
                point_estimates[6*s+3, i] = self.calculate_odds(zgrid,
                                                           point_estimates[1,i],
                                                           pdf)
        return point_estimates, pdfs


    def calculate_odds(self, z, zb, pdf):
        """                                                                                
        calculates the integrated pdf between -N*sigma_intrins and +N*sigma_intrins        
        around the mode of the PDF, zb                                                     
        parameters:                                                                        
          -sigma_intrins: intrinsic scatter of distn, read in from config                  
          -odds_int: number of sigma_intrins to multiply by to define interval             
          (read in from config)                                                            
          -z : redshift grid of pdf                                                        
          -pdf: posterior redshift estimate                                                
          -zb: mode of posterior                                                           
        """
        cumpdf = np.cumsum(pdf)
        if np.isclose(cumpdf[-1],0.0):
            return 0.0
        else:
            zo1 = zb - self.config['sigma_intrins']*self.config['odds_int']*(1.+zb)
            zo2 = zb + self.config['sigma_intrins']*self.config['odds_int']*(1.+zb)
            i1 = np.searchsorted(z,zo1)-1
            i2 = np.searchsorted(z,zo2)
            if i1<0:
                return cumpdf[i2]/cumpdf[-1]
            if i2>len(z)-1:
                return 1. - cumpdf[i1]/cumpdf[-1]
            return(cumpdf[i2]-cumpdf[i1])/cumpdf[-1]

    def pdf_median(self, z, p):
        psum = p.sum()
        cdf = np.cumsum(p)
        if np.isclose(psum,0.0):
            #print("problem with p(z), forcing to 0.0")                                        
            return 0.0
        else:
            cdf = np.concatenate([[0.0], cdf])
            i = np.where(cdf<psum/2.0)[0].max()
            return z[i]


    def write_output(self, output_file, start, end, pdfs, point_estimates):
        """                                                                                
        Write out a chunk of the computed PZ data.                                       
        Parameters                                                                         
        ----------                                                                         
        output_file: h5py.File                                                             
            The object we are writing out to                                               
        start: int                                                                         
            The index into the full range of data that this chunk starts at                
        end: int                                                                           
            The index into the full range of data that this chunk ends at                  
        pdfs: array of shape (n_chunk, n_z)                                                
            The output PDF values                                                          
        point_estimates: array of shape (3, n_chunk)                                       
            Point-estimated photo-zs for each of the 5 metacalibrated variants             
        """

        # First write the 
        group = output_file['pdf']
        group['pdf'][start:end] = pdfs

        # Now write the point-estimate info.
        # This may hav
        group = output_file['point_estimates']
        for s, suffix in enumerate(self.suffices):
            group[f'z_mean{suffix}'][start:end] = point_estimates[6*s+0]
            group[f'z_mode{suffix}'][start:end] = point_estimates[6*s+1]
            group[f'z_median{suffix}'][start:end] = point_estimates[6*s+2]
            group[f'odds{suffix}'][start:end] = point_estimates[6*s+3]
