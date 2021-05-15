import pandas as pd
import numpy as np
import os
import glob


def get_csv_files_list(csv_dir):
    """Return a list containing all found csv files within a given directory."""
    return glob.glob(f"{csv_dir}/*.csv")


def load_single_csv_file_as_df(csv_file):
    """Returns the contents of a csv file as a DataFrame."""
    return pd.read_csv(csv_file, sep=",")

