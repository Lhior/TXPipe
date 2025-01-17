from .base_stage import PipelineStage
from .data_types import TomographyCatalog, MapsFile, HDFFile, ShearCatalog
import numpy as np
from .utils import unique_list, choose_pixelization
from .utils.calibration_tools import read_shear_catalog_type, apply_metacal_response
from .utils.calibrators import LensfitCalibrator
from .mapping import Mapper, FlagMapper


SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2

# These generic mapping options are used by multiple different
# map types.
# TODO: consider dropping support for gnomonic maps.
# Also consider adding support for pixell
map_config_options = {
    "chunk_rows": 100000,  # The number of rows to read in each chunk of data at a time
    "pixelization": "healpix",  # The pixelization scheme to use, currently just healpix
    "nside": 0,  # The Healpix resolution parameter for the generated maps. Only req'd if using healpix
    "sparse": True,  # Whether to generate sparse maps - faster and less memory for small sky areas,
    "ra_cent": np.nan,  # These parameters are only required if pixelization==tan
    "dec_cent": np.nan,
    "npix_x": -1,
    "npix_y": -1,
    "pixel_size": np.nan,  # Pixel size of pixelization scheme
}


class TXBaseMaps(PipelineStage):
    """
    This is an abstract base mapping class, which other subclasses
    inherit from to use the same basic structure, which is:
    - select pixelization
    - prepare some mapper objects
    - iterate through selected columns
        - update each mapper with each chunk
    - finalize the mappers
    - save the maps
    """

    name = "TXBaseMaps"
    inputs = []
    outputs = []
    config_options = {}

    def run(self):
        # Read input configuration information
        # and select pixel scheme. Also save the scheme
        # metadata so we can save it later
        pixel_scheme = self.choose_pixel_scheme()
        self.config.update(pixel_scheme.metadata)

        # Initialize our maps
        mappers = self.prepare_mappers(pixel_scheme)

        # Loop through the data
        for s, e, data in self.data_iterator():
            # Give an idea of progress
            print(f"Process {self.rank} read data chunk {s:,} - {e:,}")
            # Build up any maps we are using
            self.accumulate_maps(pixel_scheme, data, mappers)

        # Finalize and output the maps
        maps = self.finalize_mappers(pixel_scheme, mappers)
        if self.rank == 0:
            self.save_maps(pixel_scheme, maps)

    def choose_pixel_scheme(self):
        """
        Subclasses can override to instead load pixelization
        from an existing map
        """
        return choose_pixelization(**self.config)

    def prepare_mappers(self, pixel_scheme):
        """
        Subclasses must override to init any mapper objects
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def data_iterator(self):
        """
        Subclasses must override to create an iterator looping over
        input data
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def accumulate_maps(self, pixel_scheme, data, mappers):
        """
        Subclasses must override to supply the next chunk "data" to
        their mappers
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def finalize_mappers(self, pixel_scheme, mappers):
        """
        Subclasses must override to finalize their maps and return
        a dictionary of (output_tag, map_name) -> (pixels, values)
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def save_maps(self, pixel_scheme, maps):
        """
        Subclasses can use this directly, by generating maps as described
        in finalize_mappers
        """
        # Find and open all output files
        tags = unique_list([outfile for outfile, _ in maps.keys()])
        output_files = {
            tag: self.open_output(tag, wrapper=True)
            for tag in tags
            if tag.endswith("maps")
        }

        # add a maps section to each
        for output_file in output_files.values():
            output_file.file.create_group("maps")
            output_file.file["maps"].attrs.update(self.config)

        # same the relevant maps in each
        for (tag, map_name), (pix, map_data) in maps.items():
            output_files[tag].write_map(map_name, pix, map_data, self.config)


class TXSourceMaps(TXBaseMaps):
    """
    Make g1, g2, var(g1), var(g2), and lensing weight maps
    from shear catalogs and tomography
    """

    name = "TXSourceMaps"

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
    ]

    outputs = [
        ("source_maps", MapsFile),
    ]

    # Generic mapping options + one option
    # to use the truth shear columns
    config_options = {"true_shear": False, **map_config_options}

    def prepare_mappers(self, pixel_scheme):
        # read shear cols and calibration info
        nbin_source, cal = self.get_calibrators()

        # store in config so it is saved later
        self.config["nbin_source"] = nbin_source
        # create basic mapper object
        source_bins = list(range(nbin_source))
        lens_bins = []
        mapper = Mapper(
            pixel_scheme,
            lens_bins,
            source_bins,
            do_lens=False,
            sparse=self.config["sparse"],
        )
        return [mapper, cal]

    def get_calibrators(self):
        shear_catalog_type = read_shear_catalog_type(self)
        with self.open_input("shear_tomography_catalog") as f:
            nbin_source = f["tomography"].attrs["nbin_source"]

            if shear_catalog_type == "metacal":
                R = f['/metacal_response/R_total'][:] # nbin x 2 x 2
                cal = {i:R[i] for i in range(nbin_source)}
                cal['2D'] = f['/metacal_response/R_total_2d'][:]
            elif shear_catalog_type == "lensfit":
                K = f['/response/K'][:]
                c = f['/response/C'][:]
                cal = {i: (K[i], c[i]) for i in range(nbin_source)}
                cal['2D'] = (
                    f['/response/K_2d'][0],
                    f['/response/C_2d'][:],
                )
            elif shear_catalog_type == "hsc":
                R = f['/response/R'][:]
                K = f['/response/K'][:]
                cal = {i: (R[i], K[i]) for i in range(nbin_source)}
                cal['2D'] = (
                    f['/response/R_2d'][0],
                    f['/response/K_2d'][:],
                )
            else:
                raise ValueError("Unknown calibration")
        return nbin_source, cal

    def data_iterator(self):

        # can optionally read truth values
        if self.config["true_shear"]:
            shear_cols = ["true_g1", "true_g2", "ra", "dec", "weight"]
        elif self.config["shear_catalog_type"] == "metacal":
            shear_cols = ["mcal_g1", "mcal_g2", "ra", "dec", "weight"]
        else:
            shear_cols = ["g1", "g2", "ra", "dec", "weight"]

        # use utility function that combines data chunks
        # from different files. Reading from n file sections
        # takes 3n+1 arguments
        return self.combined_iterators(
            self.config["chunk_rows"],  # number of rows to iterate at once
            # first file info
            "shear_catalog",  # tag of input file to iterate through
            "shear",  # data group within file to look at
            shear_cols,  # column(s) to read
            # next file
            "shear_tomography_catalog",  # tag of input file to iterate through
            "tomography",  # data group within file to look at
            ["source_bin"],  # column(s) to read
        )

    def accumulate_maps(self, pixel_scheme, data, mappers):
        # rename columns, if needed
        if self.config["true_shear"]:
            data["g1"] = data["true_g1"]
            data["g2"] = data["true_g2"]
        elif self.config["shear_catalog_type"] == "metacal":
            data["g1"] = data["mcal_g1"]
            data["g2"] = data["mcal_g2"]
        # for other catalogs they're just called g1, g2 aready

        # send data to map
        mapper = mappers[0]
        mapper.add_data(data)

    def calibrate_map_metacal(self, g1, g2, var_g1, var_g2, R):
        g1, g2 = apply_metacal_response(R, 0, g1, g2)

        std_g1 = np.sqrt(var_g1)
        std_g2 = np.sqrt(var_g2)
        std_g1, std_g2 = apply_metacal_response(R, 0, std_g1, std_g2)

        var_g1 = std_g1 ** 2
        var_g2 = std_g2 ** 2

        return g1, g2, var_g1, var_g2


    def calibrate_map_lensfit(self, g1, g2, var_g1, var_g2, K, c):
        calib = LensfitCalibrator(K,c)

        g1,g2 = calib.apply(g1,g2,subtract_mean=True)

        var_g1,var_g2 = calib.apply(g1,g2,subtract_mean=False)

        return g1, g2, var_g1, var_g2

    def calibrate_map_hsc(self, g1, g2, var_g1, var_g2, K, c):
        return "need to write calibrate map hsc function"

    def calibrate_maps(self, g1, g2, var_g1, var_g2, cal):
        import healpy

        # We will return lists of calibrated maps
        g1_out = {}
        g2_out = {}
        var_g1_out = {}
        var_g2_out = {}

        # We calibrate the 2D case separately
        for i in g1.keys():
            # we want to avoid accidentally calibrating any pixels
            # that should be masked.
            mask = (
                  (g1[i] == healpy.UNSEEN)
                | (g2[i] == healpy.UNSEEN)
                | (var_g1[i] == healpy.UNSEEN)
                | (var_g2[i] == healpy.UNSEEN)
            )
            if self.config['shear_catalog_type'] == 'metacal':
                out = self.calibrate_map_metacal(g1[i], g2[i], var_g1[i], var_g2[i], cal[i])
            elif self.config['shear_catalog_type'] == 'lensfit':
                K, c = cal[i]
                out = self.calibrate_map_lensfit(g1[i], g2[i], var_g1[i], var_g2[i], K, c)
            elif self.config['shear_catalog_type'] == 'hsc':
                R, K, c = cal[i]
                out = self.calibrate_map_hsc(g1[i], g2[i], var_g1[i], var_g2[i], R, K, c)
            else:
                raise ValueError("Unknown calibration")

            # re-apply the masking, just to make sure
            for x in out:
                x[mask] = healpy.UNSEEN

            # append our results for this tomographic bin
            g1_out[i] = out[0]
            g2_out[i] = out[1]
            var_g1_out[i] = out[2]
            var_g2_out[i] = out[3]

        return g1_out, g2_out, var_g1_out, var_g2_out


    def finalize_mappers(self, pixel_scheme, mappers):
        # only one mapper here - we call its finalize method
        # to collect everything
        mapper, cal = mappers
        pix, _, _, g1, g2, var_g1, var_g2, weights_g = mapper.finalize(self.comm)

        # build up output
        maps = {}

        # only master gets full stuff
        if self.rank != 0:
            return maps

        # Calibrate the maps
        g1, g2, var_g1, var_g2 = self.calibrate_maps(g1, g2, var_g1, var_g2, cal)

        for b in mapper.source_bins:
            # keys are the output tag and the map name
            maps["source_maps", f"g1_{b}"] = (pix, g1[b])
            maps["source_maps", f"g2_{b}"] = (pix, g2[b])
            maps["source_maps", f"var_g1_{b}"] = (pix, var_g1[b])
            maps["source_maps", f"var_g2_{b}"] = (pix, var_g2[b])
            maps["source_maps", f"lensing_weight_{b}"] = (pix, weights_g[b])

        return maps


class TXLensMaps(TXBaseMaps):
    """
    Make galaxy number count maps from photometry
    and lens tomography.

    Density maps are made later once masks are generated.
    """

    name = "TXLensMaps"

    inputs = [
        ("photometry_catalog", HDFFile),
        ("lens_tomography_catalog", TomographyCatalog),
    ]

    outputs = [
        ("lens_maps", MapsFile),
    ]

    config_options = {**map_config_options}

    def prepare_mappers(self, pixel_scheme):
        # read nbin_lens and save
        with self.open_input("lens_tomography_catalog") as f:
            nbin_lens = f["tomography"].attrs["nbin_lens"]
        self.config["nbin_lens"] = nbin_lens

        # create lone mapper
        lens_bins = list(range(nbin_lens))
        source_bins = []
        mapper = Mapper(
            pixel_scheme,
            lens_bins,
            source_bins,
            do_g=False,
            sparse=self.config["sparse"],
        )
        return [mapper]

    def data_iterator(self):
        print("TODO: add use of lens weights here")
        # see TXSourceMaps abov for info on this
        return self.combined_iterators(
            self.config["chunk_rows"],
            # first file
            "photometry_catalog",
            "photometry",
            ["ra", "dec"],
            # next file
            "lens_tomography_catalog",
            "tomography",
            ["lens_bin"],
        )

    def accumulate_maps(self, pixel_scheme, data, mappers):
        # no need to rename cols here since just ra, dec, lens_bin
        mapper = mappers[0]
        mapper.add_data(data)

    def finalize_mappers(self, pixel_scheme, mappers):
        # Again just the one mapper
        mapper = mappers[0]
        # Ignored return values are empty dicts for shear
        pix, ngal, weighted_ngal, _, _, _, _, _ = mapper.finalize(self.comm)
        maps = {}

        if self.rank != 0:
            return maps

        for b in mapper.lens_bins:
            # keys are the output tag and the map name
            maps["lens_maps", f"ngal_{b}"] = (pix, ngal[b])
            maps["lens_maps", f"weighted_ngal_{b}"] = (pix, weighted_ngal[b])

        return maps


class TXExternalLensMaps(TXLensMaps):
    """
    Same as TXLensMaps except it reads from an external
    lens catalog.
    """

    name = "TXExternalLensMaps"

    inputs = [
        ("lens_catalog", HDFFile),
        ("lens_tomography_catalog", TomographyCatalog),
    ]

    config_options = {**map_config_options}

    def data_iterator(self):
        # See TXSourceMaps for an explanation of thsis
        return self.combined_iterators(
            self.config["chunk_rows"],
            # first file
            "lens_catalog",
            "lens",
            ["ra", "dec"],
            # next file
            "lens_tomography_catalog",
            "tomography",
            ["lens_bin", "lens_weight"],
            # another section in the same file
        )


class TXMainMaps(TXSourceMaps, TXLensMaps):
    """
    Combined source and photometric lens maps, from the
    same photometry catalog. This might be slightly faster than
    running two maps separately, but it only works if the source 
    and lens catalogs are the same set of objects. Otherwise use
    TXSourceMaps and TXLensMaps.
    """

    name = "TXMainMaps"

    inputs = [
        ("photometry_catalog", HDFFile),
        ("lens_tomography_catalog", TomographyCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        ("shear_catalog", ShearCatalog),
    ]
    outputs = [
        ("lens_maps", MapsFile),
        ("source_maps", MapsFile),
    ]

    config_options = {"true_shear": False, **map_config_options}

    def data_iterator(self):
        # This is just the combination of
        # the source and lens map columns
        print("TODO: no lens weights here")

        with self.open_input("photometry_catalog") as f:
            sz1 = f['photometry/ra'].size
        with self.open_input("shear_catalog") as f:
            sz2 = f['shear/ra'].size

        if sz1 != sz2:
            raise ValueError("Shear and photometry catalogs in TXMainMaps are "
                             "different sizes. To use separate source and lens "
                             "samples use TXSourceMaps and TXLensMaps separately."
                             )

        # metacal, lensfit, etc.
        shear_catalog_type = read_shear_catalog_type(self)

        # can optionally read truth values, or otherwise will look for
        # lensfit or metacal col names
        if self.config["true_shear"]:
            shear_cols = ["true_g1", "true_g1", "ra", "dec", "weight"]
        elif shear_catalog_type == "metacal":
            shear_cols = ["mcal_g1", "mcal_g2", "ra", "dec", "weight"]
        else:
            shear_cols = ["g1", "g2", "ra", "dec", "weight"]

        return self.combined_iterators(
            self.config["chunk_rows"],
            # first file
            "photometry_catalog",
            "photometry",
            ["ra", "dec"],
            # next file
            "shear_catalog",
            "shear",
            shear_cols,
            # next file
            "lens_tomography_catalog",
            "tomography",
            ["lens_bin", "lens_weight"],
            # next file
            "shear_tomography_catalog",
            "tomography",
            ["source_bin"],
        )

    def prepare_mappers(self, pixel_scheme):
        nbin_source, cal = self.get_calibrators()

        with self.open_input("lens_tomography_catalog") as f:
            nbin_lens = f["tomography"].attrs["nbin_lens"]

        self.config["nbin_source"] = nbin_source
        self.config["nbin_lens"] = nbin_lens

        source_bins = list(range(nbin_source))
        lens_bins = list(range(nbin_lens))

        # still a single mapper doing source and lens
        mapper = Mapper(
            pixel_scheme, lens_bins, source_bins, sparse=self.config["sparse"]
        )
        return [mapper, cal]

    # accumulate_maps is inherited from TXSourceMaps because
    # that appears first in the parent classes

    def finalize_mappers(self, pixel_scheme, mappers):
        # Still one mapper, but now we read both source and
        # lens maps from it.
        mapper, cal = mappers
        pix, ngal, weighted_ngal, g1, g2, var_g1, var_g2, weights_g = mapper.finalize(self.comm)
        maps = {}

        if self.rank != 0:
            return maps

        g1, g2, var_g1, var_g2 = self.calibrate_maps(g1, g2, var_g1, var_g2, cal)

        # Now both loops, source and lens
        for b in mapper.lens_bins:
            maps["lens_maps", f"ngal_{b}"] = (pix, ngal[b])
            maps["lens_maps", f"weighted_ngal_{b}"] = (pix, weighted_ngal[b])

        for b in mapper.source_bins:
            maps["source_maps", f"g1_{b}"] = (pix, g1[b])
            maps["source_maps", f"g2_{b}"] = (pix, g2[b])
            maps["source_maps", f"var_g1_{b}"] = (pix, var_g1[b])
            maps["source_maps", f"var_g2_{b}"] = (pix, var_g2[b])
            maps["source_maps", f"lensing_weight_{b}"] = (pix, weights_g[b])

        return maps


class TXDensityMaps(PipelineStage):
    """
    Convert n_gal maps to overdensity delta maps
    delta = (ngal - <ngal>) / <ngal>

    This has to be separate from the lens mappers above
    because it requires the mask, which is created elsewhere
    (right now in masks.py)
    """

    name = "TXDensityMaps"
    inputs = [
        ("lens_maps", MapsFile),
        ("mask", MapsFile),
    ]
    outputs = [
        ("density_maps", MapsFile),
    ]

    def run(self):
        import healpy

        # Read the mask
        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_map("mask")

        # set unseen pixels to weight zero
        mask[mask == healpy.UNSEEN] = 0
        mask[np.isnan(mask)] = 0
        mask = mask.flatten()
        pix = np.where(mask > 0)[0]

        # Read the count maps
        with self.open_input("lens_maps", wrapper=True) as f:
            meta = dict(f.file["maps"].attrs)
            nbin_lens = meta["nbin_lens"]
            ngal_maps = [f.read_map(f"weighted_ngal_{b}").flatten() for b in range(nbin_lens)]

        # Convert count maps into density maps
        density_maps = []
        for i, ng in enumerate(ngal_maps):
            mask_copy = mask.copy()
            mask_copy[ng == healpy.UNSEEN] = 0
            ng[np.isnan(ng)] = 0.0
            ng[ng == healpy.UNSEEN] = 0
            # Convert the number count maps to overdensity maps.
            # First compute the overall mean object count per bin.
            # mean clustering galaxies per pixel in this map
            mu = np.average(ng, weights=mask_copy)
            print(f"Mean number density in bin {i} = {mu}")
            # and then use that to convert to overdensity
            d = (ng - mu) / mu
            # remove nans
            d[mask == 0] = 0
            density_maps.append(d)

        # write output
        with self.open_output("density_maps", wrapper=True) as f:
            # create group and save metadata there too.
            f.file.create_group("maps")
            f.file["maps"].attrs.update(meta)
            # save each density map
            for i, rho in enumerate(density_maps):
                f.write_map(f"delta_{i}", pix, rho[pix], meta)
