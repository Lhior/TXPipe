from .base_stage import PipelineStage
from .data_types import YamlFile, TomographyCatalog, HDFFile, TextFile
from .utils import LensNumberDensityStats
import numpy as np
import warnings




class TXDaskLensSelector(PipelineStage):
    """
    This pipeline stage selects objects to be used
    as the lens sample for the galaxy clustering and
    shear-position calibrations.
    """

    name='TXDaskLensSelector'

    inputs = [
        ('photometry_catalog', HDFFile),
        ('photoz_pdfs', HDFFile),
    ]

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
        import dask.array as da
        import dask_mpi
        import zarr
        self.setup_dask()

        # Suppress some warnings from numpy that are not relevant
        data = self.load_data()
        edges = self.config['lens_zbin_edges']
        nedge = len(edges)

        sel = self.select_lens(data['mag_g'], data['mag_r'], data['mag_i'])
        bins = sel * (da.digitize(data['z'], edges) % nedge)

        counts = da.bincount(bins, minlength=nedge)[1:]
        counts2d = counts.sum()
        bins -= 1

        weight = da.ones(sel.size)

        output = self.get_output('lens_tomography_catalog')

        store = zarr.NestedDirectoryStore(output)

        da.to_zarr(bins, store, "tomography/lens_bin", overwrite=True)
        da.to_zarr(bins, store, "tomography/lens_weight", overwrite=True)
        da.to_zarr(bins, store, "tomography/lens_counts", overwrite=True)
        da.to_zarr(bins, store, "tomography/lens_counts_2d", overwrite=True)
        # da.to_zarr(output, {
        #     'tomography/lens_bin': bins,
        #     'tomography/lens_weight': weight,
        #     'tomography/lens_counts': counts,
        #     'tomography/lens_counts_2d': counts2d,
        # }) 

        # TODO: metadata, provenance


    def setup_dask(self):
        # import dask_mpi
        import multiprocessing.popen_spawn_posix
        import dask.distributed
        self.dask_client = dask.distributed.Client()  # Connect this local process to remote workers
        print("Initializing DASK:")
        print(self.dask_client)
        # if self.comm is not None:
        #     print("Initializing DASK")
        #     # dask_mpi.initialize()
        #     self.dask_client = dask.distributed.Client()  # Connect this local process to remote workers
        #     print(self.dask_client)
        #     print(dir(self.dask_client))

    def load_data(self):
        import dask
        f = self.open_input("photometry_catalog")
        g = self.open_input("photoz_pdfs")
        data = {
            'mag_g': dask.array.from_array(f['photometry/mag_g']),
            'mag_r': dask.array.from_array(f['photometry/mag_r']),
            'mag_i': dask.array.from_array(f['photometry/mag_i']),
            'z': dask.array.from_array(g['point_estimates/z_mode']),
        }
        return data


    def select_lens(self, mag_g, mag_r, mag_i):
        """Photometry cuts based on the BOSS Galaxy Target Selection:
        http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php
        """

        # Mag cuts 
        cperp_cut_val = self.config['cperp_cut']
        r_cpar_cut_val = self.config['r_cpar_cut']
        r_lo_cut_val = self.config['r_lo_cut']
        r_hi_cut_val = self.config['r_hi_cut']
        i_lo_cut_val = self.config['i_lo_cut']
        i_hi_cut_val = self.config['i_hi_cut']
        r_i_cut_val = self.config['r_i_cut']

        # HDF does not support bools, so we will prepare a binary array
        # where 0 is a lens and 1 is not

        cpar = 0.7 * (mag_g - mag_r) + 1.2 * ((mag_r - mag_i) - 0.18)
        cperp = (mag_r - mag_i) - ((mag_g - mag_r) / 4.0) - 0.18
        dperp = (mag_r - mag_i) - ((mag_g - mag_r) / 8.0)

        # LOWZ
        cperp_cut = np.abs(cperp) < cperp_cut_val #0.2
        r_cpar_cut = mag_r < r_cpar_cut_val + cpar / 0.3
        r_lo_cut = mag_r > r_lo_cut_val #16.0
        r_hi_cut = mag_r < r_hi_cut_val #19.6

        lowz_cut = (cperp_cut) & (r_cpar_cut) & (r_lo_cut) & (r_hi_cut)

        # CMASS
        i_lo_cut = mag_i > i_lo_cut_val #17.5
        i_hi_cut = mag_i < i_hi_cut_val #19.9
        r_i_cut = (mag_r - mag_i) < r_i_cut_val #2.0
        #dperp_cut = dperp > 0.55 # this cut did not return any sources...

        cmass_cut = (i_lo_cut) & (i_hi_cut) & (r_i_cut)

        # If a galaxy is a lens under either LOWZ or CMASS give it a zero
        lens_mask =  lowz_cut | cmass_cut

        return lens_mask


if __name__ == '__main__':
    PipelineStage.main()


