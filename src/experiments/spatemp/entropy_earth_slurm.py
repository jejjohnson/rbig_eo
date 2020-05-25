import sys, os
from pyprojroot import here

root = here(project_files=[".here"])
sys.path.append(str(here()))


from src.experiments.spatemp import entropy_earth
from src.experiments.utils import dict_product
from typing import Dict
import argparse


def get_parameters(resample: bool = False, region: str = "world") -> Dict:

    parameters = {}
    # ======================
    # Variable
    # ======================
    parameters["variable"] = [
        "gpp",
        "sm",
        "rm",
        "lst",
        "lai",
        "precip",
    ]
    parameters["region"] = [region]
    return list(dict_product(parameters))


def main(args):
    # get parameters
    parameters = get_parameters(resample=args.resample, region=args.region)
    # print(len(parameters))
    # break
    if args.nparams:
        print(f"Total # of params: {len(parameters)}")
    else:
        iparams = parameters[args.job]

        args.variable = iparams["variable"]
        args.region = iparams["region"]

        entropy_earth.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for GP experiment.")

    parser.add_argument(
        "--res", default="low", type=str, help="Resolution for datacube"
    )

    parser.add_argument("-j", "--job", default=0, type=int, help="job array")

    parser.add_argument(
        "-s", "--save", default="v0", type=str, help="Save name for experiment."
    )
    parser.add_argument(
        "--njobs", type=int, default=-1, help="number of processes in parallel",
    )
    parser.add_argument(
        "--subsample", type=int, default=None, help="subset points to take"
    )
    parser.add_argument(
        "--region", type=str, default="world", help="Region for the Gaussianization"
    )
    parser.add_argument(
        "--period", type=str, default="2010", help="Period to do the Gaussianization"
    )
    parser.add_argument(
        "-rs", "--resample", type=str, default=None, help="Resample Frequency"
    )
    parser.add_argument("-m", "--method", type=str, default="old", help="RBIG Method")
    parser.add_argument("-sm", "--smoke-test", action="store_true")
    parser.add_argument("-tm", "--temporal-mean", action="store_true")
    parser.add_argument("-rc", "--remove-climatology", action="store_true")
    parser.add_argument("-np", "--nparams", action="store_true")

    main(parser.parse_args())
