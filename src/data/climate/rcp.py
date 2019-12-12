import os
import cdsapi
import xarray as xr
from typing import Type, Union


from zipfile import ZipFile, BadZipFile
from src.data.climate.era5 import get_era5_data
from src.data.climate.ncep import get_ncep_data

DATA_DIR = "/home/emmanuel/projects/2020_rbig_rs/data/climate/raw/rcp/"

c = cdsapi.Client()

xr_types = Union[xr.Dataset, xr.DataArray]


def get_base_model(base_model: str, variable: str) -> xr_types:

    if base_model == "era5":
        ds = get_era5_data()[variable]
    elif base_model == "ncep":
        ds = get_ncep_data()[variable]
    else:
        raise ValueError("Unrecognized base model:", base_model)
    ds.attrs["model_id"] = base_model
    return ds


class DataDownloader:
    def __init__(self):
        pass

    def download_all(self) -> None:

        # Russian Model
        download_inmcm4()
        self.zip_2_netcdf("inmcm4")

        # Austrailian Model
        download_access1_0()
        self.zip_2_netcdf("access1_0")

        # Austrailian Model
        download_access1_3()
        self.zip_2_netcdf("access1_3")

        # French Model II
        download_ipsl_cm5a_lr()
        self.zip_2_netcdf("ipsl_cm5a_lr")

        # French Model III
        download_ipsl_cm5a_mr()
        self.zip_2_netcdf("ipsl_cm5a_mr")

        # German Model I
        download_mpi_esm_lr()
        self.zip_2_netcdf("mpi_esm_lr")

        # German Model II
        download_mpi_esm_mr()
        self.zip_2_netcdf("mpi_esm_mr")

        # Norweigian Model I
        download_noresm1_m()
        self.zip_2_netcdf("noresm1_m")

    def zip_2_netcdf(self, name=str) -> None:

        # Unzip files
        print(f"{DATA_DIR}{name}/{name}.zip")
        try:
            with ZipFile(f"{DATA_DIR}{name}/{name}.zip", "r") as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(f"{DATA_DIR}{name}/")
            print("Removing file:", f"{DATA_DIR}{name}/")
            os.remove(f"{DATA_DIR}{name}/{name}.zip")
        except BadZipFile:

            # rename to netcdf
            print("Already nc...")
            print("Changing name: \n", f"{DATA_DIR}{name}/{name}.nc")
            os.rename(f"{DATA_DIR}{name}/{name}.zip", f"{DATA_DIR}{name}/{name}.nc")


class DataLoader:
    def __init__(self):
        pass

    def load_rcp_data(self, data: str = "inmcm4") -> Union[xr.DataArray, xr.Dataset]:
        """Loads AMIP models previously downloaded.
        
        Available Models
        ----------------
        * inmcm4
        * access1_0
        * bcc_csm1_1
        * bcc_csm1_1_m
        * bnu_esm
        * cnrm_cm5
        * giss_e2_r
        * ipsl_cm5a_lr
        * ipsl_cm5a_mr
        * ipsl_cm5b_lr
        * mpi_esm_lr
        * mpi_esm_mr
        * noresm1_m
        """
        if data in [
            "inmcm4",
            "access1_0",
            "access1_3",
            "bcc_csm1_1_m",
            "ipsl_cm5a_lr",
            "ipsl_cm5a_mr",
            "mpi_esm_lr",
            "mpi_esm_mr",
            "noresm1_m",
        ]:
            return xr.open_mfdataset(f"{DATA_DIR}{data}/*.nc", combine="by_coords")
        else:
            raise ValueError(f"Unrecognized dataset:", data)


# inmcm4 (INM, Russia)
def download_inmcm4():
    name = "inmcm4"
    make_directory(DATA_DIR, name)

    c.retrieve(
        "projections-cmip5-monthly-single-levels",
        {
            "ensemble_member": "r1i1p1",
            "format": "zip",
            "experiment": "rcp_8_5",
            "variable": "mean_sea_level_pressure",
            "period": ["200601-210012"],
            "model": "inmcm4",
        },
        f"{DATA_DIR}{name}/inmcm4.zip",
    )


# ACCESS1-0 (BoM-CSIRO, Australia)
def download_access1_0():

    name = "access1_0"
    make_directory(DATA_DIR, name)

    c.retrieve(
        "projections-cmip5-monthly-single-levels",
        {
            "ensemble_member": "r1i1p1",
            "format": "zip",
            "experiment": "rcp_8_5",
            "variable": "mean_sea_level_pressure",
            "period": "200601-210012",
            "model": "access1_0",
        },
        f"{DATA_DIR}{name}/access1_0.zip",
    )


# ACCESS1-3 (BoM-CSIRO, Australia)
def download_access1_3():

    name = "access1_3"
    make_directory(DATA_DIR, name)

    c.retrieve(
        "projections-cmip5-monthly-single-levels",
        {
            "ensemble_member": "r1i1p1",
            "format": "zip",
            "experiment": "rcp_8_5",
            "variable": "mean_sea_level_pressure",
            "period": "200601-210012",
            "model": "access1_3",
        },
        f"{DATA_DIR}{name}/access1_3.zip",
    )


# bcc-csm1-1 (BCC, China)
def download_bcc_csm1_1_m():

    name = "bcc_csm1_1_m"
    make_directory(DATA_DIR, name)

    c.retrieve(
        "projections-cmip5-monthly-single-levels",
        {
            "ensemble_member": "r1i1p1",
            "format": "zip",
            "experiment": "rcp_8_5",
            "variable": "mean_sea_level_pressure",
            "period": "200601-210012",
            "model": "bnu_esm",
        },
        f"{DATA_DIR}{name}/bcc_csm1_1_m.zip",
    )


# IPSL-CM5A-MR (IPSL, France)
def download_ipsl_cm5a_mr():

    name = "ipsl_cm5a_mr"
    make_directory(DATA_DIR, name)

    c.retrieve(
        "projections-cmip5-monthly-single-levels",
        {
            "ensemble_member": "r1i1p1",
            "format": "zip",
            "experiment": "rcp_8_5",
            "variable": "mean_sea_level_pressure",
            "period": "200601-210012",
            "model": "ipsl_cm5a_mr",
        },
        f"{DATA_DIR}{name}/ipsl_cm5a_mr.zip",
    )


# IPSL-CM5A-LR (IPSL, France)
def download_ipsl_cm5a_lr():

    name = "ipsl_cm5a_lr"
    make_directory(DATA_DIR, name)

    c.retrieve(
        "projections-cmip5-monthly-single-levels",
        {
            "ensemble_member": "r1i1p1",
            "format": "zip",
            "experiment": "rcp_8_5",
            "variable": "mean_sea_level_pressure",
            "period": "200601-210012",
            "model": "ipsl_cm5b_lr",
        },
        f"{DATA_DIR}{name}/ipsl_cm5a_lr.zip",
    )


# MPI-ESM-LR (MPI, Germany)
def download_mpi_esm_lr():

    name = "mpi_esm_lr"
    make_directory(DATA_DIR, name)

    c.retrieve(
        "projections-cmip5-monthly-single-levels",
        {
            "ensemble_member": "r1i1p1",
            "format": "zip",
            "experiment": "rcp_8_5",
            "variable": "mean_sea_level_pressure",
            "period": "200601-210012",
            "model": "mpi_esm_lr",
        },
        f"{DATA_DIR}{name}/mpi_esm_lr.zip",
    )


# MPI-ESM-MR (MPI, Germany)
def download_mpi_esm_mr():

    name = "mpi_esm_mr"
    make_directory(DATA_DIR, name)

    c.retrieve(
        "projections-cmip5-monthly-single-levels",
        {
            "ensemble_member": "r1i1p1",
            "format": "zip",
            "experiment": "rcp_8_5",
            "variable": "mean_sea_level_pressure",
            "period": "200601-210012",
            "model": "mpi_esm_mr",
        },
        f"{DATA_DIR}{name}/mpi_esm_mr.zip",
    )


# NorESM1-M (NCC, Norway)
def download_noresm1_m():

    name = "noresm1_m"
    make_directory(DATA_DIR, name)

    c.retrieve(
        "projections-cmip5-monthly-single-levels",
        {
            "ensemble_member": "r1i1p1",
            "format": "zip",
            "experiment": "rcp_8_5",
            "variable": "mean_sea_level_pressure",
            "period": "200601-210012",
            "model": "noresm1_m",
        },
        f"{DATA_DIR}{name}/noresm1_m.zip",
    )


def make_directory(directory: str, name: str):
    if not os.path.exists(directory + name):
        os.makedirs(directory + name)
