import xarray as xr
from dep_tools.processors import Processor
from dep_tools.searchers import search_across_180
from odc.algo import mask_cleanup
from odc.geo import Geometry
from odc.stac import load
from pystac_client import Client
from xarray import DataArray, Dataset

OUTPUT_NODATA = 255


class MangrovesProcessor(Processor):
    def __init__(self, areas: Geometry):
        super().__init__()
        self.areas = areas

    def process(self, data: DataArray, debug: bool = False) -> DataArray:
        # Remove time dimension and load into memory now...
        data = data.squeeze().compute()

        # Scale data, clip to valid range
        data = (data.where(data != 0) * 0.0001).clip(0.0001, 1)

        # AMMI
        nir = data["nir"]
        swir = data["swir16"]
        red = data["red"]
        green = data["green"]

        data["ammi"] = ((nir - red) / (red + swir)) * (
            (nir - swir) / (swir - 0.65 * red)
        )

        # AMMI_THRESHOLD = 4.0 - 20
        AMMI_THRESHOLD = range(4, 20)
        mangrove_mask = data["ammi"] >= list(AMMI_THRESHOLD)[0]

        num_vals = len(AMMI_THRESHOLD)
        for i, val in enumerate(AMMI_THRESHOLD, 1):
            density_percentage = 10 + (i - 1) * (90 / (num_vals - 1))
            mangrove_mask = xr.where(
                data["ammi"] >= val, density_percentage, mangrove_mask
            )

        # Store this thing
        mangrove_mask = mangrove_mask.astype("uint8")
        data["mangroves"] = mangrove_mask.where(mangrove_mask != 0, drop=True)
        mangroves_pre_mask = data["mangroves"]

        # Morphological Filters and Elevation Masking
        data["ndwi"] = (green - nir) / (green + nir)
        data["mndwi"] = (green - swir) / (green + swir)

        # water mask
        water = (data.mndwi + data.ndwi).squeeze() < 0
        water_mask = mask_cleanup(water, [["dilation", 5], ["erosion", 5]])
        data["mangroves"] = apply_mask(data["mangroves"], water_mask)

        # elevation mask (30-50m)
        data["mangroves"], elevation_mask = mask_elevation(
            data["mangroves"], threshold=30, return_mask=True
        )

        if not debug:
            # Drop everything except mangroves
            data = data[["mangroves"]]
        else:
            data = data.drop_vars(["red", "green", "nir", "swir16", "ndwi", "mndwi"])

            data["ammi"] = data["ammi"]

            data["mangroves_pre_mask"] = mangroves_pre_mask
            data["elevation_mask"] = elevation_mask

            data["water"] = water
            data["water_mask"] = water_mask

        # Change to uint8 and set nodata to 255
        data = data.fillna(OUTPUT_NODATA).astype("uint8")
        data["mangroves"].odc.nodata = OUTPUT_NODATA

        return data


def apply_mask(
    ds: Dataset,
    mask: DataArray,
    ds_to_mask: Dataset | None = None,
    return_mask: bool = False,
) -> Dataset:
    """Applies a mask to a dataset"""
    to_mask = ds if ds_to_mask is None else ds_to_mask
    masked = to_mask.where(mask)

    if return_mask:
        return masked, mask
    else:
        return masked


def mask_elevation(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    threshold: float = 10,
    return_mask: bool = False,
) -> Dataset:
    """
    Mask elevation. Returns 1 for high areas, 0 for low
    """
    e84_catalog = "https://earth-search.aws.element84.com/v1/"
    e84_client = Client.open(e84_catalog)
    collection = "cop-dem-glo-30"

    items = search_across_180(
        region=ds.odc.geobox, client=e84_client, collections=[collection]
    )

    # Using geobox means it will load the elevation data the same shape as the other data
    elevation = load(items, measurements=["data"], like=ds.odc.geobox).squeeze()

    # True where data is above elevation
    mask = elevation.data < (threshold * 1.0)

    return apply_mask(ds, mask, ds_to_mask, return_mask)
