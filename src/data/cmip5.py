import cdsapi

DATA_DIR = "/home/emmanuel/projects/2020_rbig_rs/data/climate/raw/"


def get_data():

    c = cdsapi.Client()

    c.retrieve(
        "projections-cmip5-monthly-single-levels",
        {
            "experiment": "rcp_8_5",
            "variable": "mean_sea_level_pressure",
            "model": "giss_e2_r",
            "ensemble_member": "r1i1p1",
            "period": "200601-202512",
            "format": "netcdf",
        },
        f"{DATA_DIR}CMIP5.nc",
    )


def main():

    get_data()

    return None


if __name__ == "__main__":
    main()
