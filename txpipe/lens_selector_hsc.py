from .base_stage import PipelineStage
from .lens_selector import TXBaseLensSelector
from .data_types import YamlFile, TomographyCatalog, HDFFile, TextFile
from .utils import LensNumberDensityStats
import numpy as np
import warnings




class TXHSCLensSelector(TXBaseLensSelector):
    """
    This pipeline stage selects objects to be used
    as the lens sample for the galaxy clustering and
    shear-position calibrations.
    """

    name='TXBaseLensSelector'

    outputs = [
        ('lens_tomography_catalog', TomographyCatalog)
    ]

    config_options = {
        'verbose': False,
        'chunk_rows':10000,
        'lens_zbin_edges':[float],
        # Mag cuts
        # Default photometry cuts based on the BOSS Galaxy Target Selection:                                                     
        # http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php                                           
        'cperp_cut':0.2,
        'r_cpar_cut':13.5,
        'r_lo_cut':16.0,
        'r_hi_cut':19.6,
        'i_lo_cut':17.5,
        'i_hi_cut':19.9,
        'r_i_cut':2.0,
        'random_seed': 42,
    }

    def run(self):
        """
        Run the analysis for this stage.
        
         - Collect the list of columns to read
         - Create iterators to read chunks of those columns
         - Loop through chunks:
            - select objects for each bin
            - write them out
            - accumulate selection bias values
         - Average the selection biases
         - Write out biases and close the output
        """
        import astropy.table
        import sklearn.ensemble

        if self.name == "TXBaseLensSelector":
            raise ValueError("Do not run TXBaseLensSelector - run a sub-class")

        # Suppress some warnings from numpy that are not relevant
        original_warning_settings = np.seterr(all='ignore')  

        # The output file we will put the tomographic
        # information into
        output_file = self.setup_output()

        iterator = self.data_iterator()

        # We will collect the selection biases for each bin
        # as a matrix.  We will collect together the different
        # matrices for each chunk and do a weighted average at the end.
        nbin_lens = len(self.config['lens_zbin_edges']) - 1

        number_density_stats = LensNumberDensityStats(nbin_lens, self.comm)

        # Loop through the input data, processing it chunk by chunk
        for (start, end, phot_data) in iterator:
            print(f"Process {self.rank} running selection for rows {start:,}-{end:,}")

            pz_data = self.apply_redshift_cut(phot_data)

            # Select lens bin objects
            lens_gals = self.select_lens(phot_data)

            # Combine this selection with size and snr cuts to produce a source selection
            # and calculate the shear bias it would generate
            tomo_bin, counts = self.calculate_tomography(pz_data, phot_data, lens_gals)

            # Save the tomography for this chunk
            self.write_tomography(output_file, start, end, tomo_bin)

            # Accumulate information on the number counts and the selection biases.
            # These will be brought together at the end.
            number_density_stats.add_data(tomo_bin)

        # Do the selection bias averaging and output that too.
        self.write_global_values(output_file, number_density_stats)

        # Save and complete
        output_file.close()

        # Restore the original warning settings in case we are being called from a library
        np.seterr(**original_warning_settings)


    def apply_redshift_cut(self, phot_data):

        pz_data = {}
        nbin = len(self.config['lens_zbin_edges']) - 1

        z = phot_data[f'z']

        zbin = np.repeat(-1, len(z))
        for zi in range(nbin):
            mask_zbin = (
                  (z >= self.config['lens_zbin_edges'][zi]) 
                & (z<self.config['lens_zbin_edges'][zi+1])
            )
            zbin[mask_zbin] = zi
            
        pz_data[f'zbin'] = zbin

        return pz_data


    def setup_output(self):
        """
        Set up the output data file.

        Creates the data sets and groups to put module output
        in the tomography_catalog output file.
        """
        n = self.open_input('photometry_catalog')['photometry/ra'].size
        nbin_lens = len(self.config['lens_zbin_edges'])-1

        outfile = self.open_output('lens_tomography_catalog', parallel=True)
        group = outfile.create_group('tomography')
        group.create_dataset('lens_bin', (n,), dtype='i')
        group.create_dataset('lens_counts', (nbin_lens,), dtype='i')

        group.attrs['nbin_lens'] = nbin_lens
        group.attrs[f'lens_zbin_edges'] = self.config['lens_zbin_edges']

        return outfile

    def write_tomography(self, outfile, start, end, lens_bin):
        """
        Write out a chunk of tomography and response.

        Parameters
        ----------
        outfile: h5py.File

        start: int
            The index into the output this chunk starts at

        end: int
            The index into the output this chunk ends at

        tomo_bin: array of shape (nrow,)
            The bin index for each output object

        R: array of shape (nrow,2,2)
            Multiplicative bias calibration factor for each object
        """

        group = outfile['tomography']
        group['lens_bin'][start:end] = lens_bin

    def write_global_values(self, outfile, number_density_stats):
        """
        Write out overall selection biases

        Parameters
        ----------
        outfile: h5py.File
        """
        lens_counts = number_density_stats.collect()


        if self.rank==0:
            group = outfile['tomography']
            group['lens_counts'][:] = lens_counts


    def select_lens(self, phot_data):
        """
        Apply magnitude cut to the HSC data as described in Tab. 4 of Mandelbaum et al., 2018.
        Parameters
        ----------
        phot_data

        Returns
        -------

        """

        # Magnitude cut
        mag_i_cut = self.config['mag_i_cut']

        mag_i = phot_data['mag_i']
        a_i = phot_data['a_i']

        n = len(mag_i)
        # HDF does not support bools, so we will prepare a binary array
        # where 1 is a lens and 0 is not
        lens_gals = np.repeat(0, n)

        # If a galaxy is a lens give it a one
        lens_mask = ((mag_i - a_i) <= mag_i_cut)
        lens_gals[lens_mask] = 1

        return lens_gals

    def calculate_tomography(self, pz_data, phot_data, lens_gals):
    
        nbin = len(self.config['lens_zbin_edges']) - 1
        n = len(phot_data['mag_i'])

        # The main output data - the tomographic
        # bin index for each object, or -1 for no bin.
        tomo_bin = np.repeat(-1, n)

        # We also keep count of total count of objects in each bin
        counts = np.zeros(nbin, dtype=int)

        for i in range(nbin):
            sel_00 = (pz_data['zbin'] == i) & (lens_gals == 1)
            tomo_bin[sel_00] = i
            counts[i] = sel_00.sum()

        return tomo_bin, counts


class TXTruthLensSelector(TXBaseLensSelector):
    name = "TXTruthLensSelector"

    inputs = [
        ('photometry_catalog', HDFFile),
    ]

    def data_iterator(self):
        print(f"We are cheating and using the true redshift.")
        chunk_rows = self.config['chunk_rows']
        if self.config['true_z']:
            phot_cols = ['mag_i','mag_r','mag_g', 'redshift_true']
        else:
            phot_cols = ['mag_i','mag_r','mag_g', 'mean_z']

        # Input data.  These are iterators - they lazily load chunks
        # of the data one by one later when we do the for loop.
        # This code can be run in parallel, and different processes will
        # each get different chunks of the data 
        for s, e, data in self.iterate_hdf('photometry_catalog', 'photometry', phot_cols, chunk_rows):
            if self.config['true_z']:
                data['z'] = data['redshift_true']
            elif self.config['input_pz']:
                data['z'] = data['mean_z']
            yield s, e, data



class TXMeanLensSelector(TXBaseLensSelector):
    name = "TXMeanLensSelector"
    inputs = [
        ('photometry_catalog', HDFFile),
        ('photoz_pdfs', HDFFile),
    ]


    def data_iterator(self):
        chunk_rows = self.config['chunk_rows']
        phot_cols = ['mag_i','mag_r','mag_g']
        z_cols = ['z_mean']
        iter_phot = self.iterate_hdf('photometry_catalog', 'photometry', phot_cols, chunk_rows)
        iter_pz = self.iterate_hdf('photoz_pdfs', 'point_estimates', z_cols, chunk_rows)
        for (s, e, data), (_, _, z_data) in zip(iter_phot, iter_pz):
            data['z'] = z_data['z_mean']
            yield s, e, data



def flatten_list(lst):
    return [item for sublist in lst for item in sublist]



if __name__ == '__main__':
    PipelineStage.main()

