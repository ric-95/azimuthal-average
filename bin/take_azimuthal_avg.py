import os
import sys

sys.path.append("../")

import numpy as np
from azimuthal_average import take_azimuthal_average
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--case-dir",
                     help="Path to OpenFOAM case directory.",
                     type=str, default=".")
args = parser.parse_args()
CASE_DIR = args.case_dir
RELATIVE_CSV_DIR = "slice_csv_files"
CSV_DIR = os.path.join(CASE_DIR, RELATIVE_CSV_DIR)


def main():
    azim_avg_df = take_azimuthal_average(CSV_DIR, write_intermediate=False, copy=False)
    output_file = os.path.join(CSV_DIR, "average_output", "azim_average.csv")
    if not os.path.isdir(os.path.dirname(output_file)):
        os.mkdir(os.path.dirname(output_file))
    azim_avg_df.to_csv(output_file, sep=",")
    return


if __name__ == "__main__":
    main()
