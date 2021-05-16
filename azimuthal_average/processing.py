import os
import pandas as pd
import numpy as np
import azimuthal_average.common as common
import azimuthal_average.data_loading as dl
from azimuthal_average.data_types import (CartesianVector, CylindricalVector,
                                          SymmetricCartesianTensor,
                                          SymmetricCylindricalTensor)


def get_cartesian_to_cylindrical_rotation_matrix(radial_vector):
    sin_t, cos_t = radial_vector.get_sin_t_and_cos_t()
    return np.array(
        [[cos_t, sin_t, 0],
         [-sin_t, cos_t, 0],
         [0, 0, 1]]
    )


def get_rotation_matrix(theta):
    return np.array(
        [[np.cos(theta), np.sin(theta), 0],
         [-np.sin(theta), np.cos(theta), 0],
         [0, 0, 1]]
    )


def write_intermediate_file(dir_, slice_number, slice):
    common.check_create_dir(dir_)
    intermediate_file = f"cyl_clice_{slice_number}.csv"
    intermediate_file = os.path.join(dir_, intermediate_file)
    slice.to_csv(intermediate_file, sep=",")


def update_running_average(running_average, n, new_obs):
    """Updates a running average while avoiding large values and provoke overflow.

    Parameters
    ----------
    cumulative_avg: value of the cumulative average so far
    n: Number of samples including new observation
    new_obs: New observation"""
    a = 1/n
    b = 1 - a
    return a*new_obs + b*running_average


def process_slice(slice_df, slice_number, theta, running_average, copy=True,
                   write_intermediate=False):


    cylindrical_slice_df = transform_df_to_cylindric_coordinates(slice_df,
                                                              theta,
                                                              copy)
    cylindrical_slice = cylindrical_slice_df.to_numpy()
    if running_average is None:
        running_average = cylindrical_slice
    running_average = update_running_average(running_average, slice_number+1,
                                             cylindrical_slice)
    if write_intermediate:
        intermediate_dir = os.path.join(csv_dir, "intermediate")
        write_intermediate_file(intermediate_dir,
                                 slice_number, cylindrical_slice_df)
    return running_average, cylindrical_slice_df


def calculate_slice_theta(slice_number, total_slices):
    return (slice_number/total_slices)*np.pi*2

def take_azimuthal_average(csv_dir, write_intermediate=False, copy=True):
    """Take the azimuthal average of all the csv files within a given directory.

    Args:
        csv_dir (str): Path to directory containing """
    csv_list = dl.get_csv_files_list(csv_dir)
    total_slices = len(csv_list)
    running_average = None
    for slice_number, csv_file in enumerate(csv_list):
        print("Transforming", csv_file)
        slice_df = dl.load_single_csv_file_as_df(csv_file)
        theta = calculate_slice_theta(slice_number, total_slices)
        running_average, cylindrical_slice_df = process_slice(
            slice_df, slice_number, theta, running_average, copy,
            write_intermediate)
    cols = cylindrical_slice_df.columns
    idx = cylindrical_slice_df.index
    return pd.DataFrame(data=running_average, columns=cols, index=idx)


def transform_df_to_cylindric_coordinates(dataframe, theta, copy=True):
    """Transforms dataframe to cylindrical coordinates

    Args:
        dataframe (DataFrame): Dataframe containing a cartesian slice
        theta (float): Angle of rotation in radians."""
    """Initialize output df"""
    if copy:
        transformed_df = dataframe.copy()
    else:
        transformed_df = dataframe
    "Transform coordinate positions to cylindrical"
    print("Transforming coordinate positions...")
    coords_df = transform_coordinates_to_cylindrical(dataframe)
    "Transform velocity to cylindrical coordinates"
    print("Transforming velocity vectors...")
    u_cyl_df = transform_velocity_to_cylindrical(dataframe, theta)
    "Transform Reynolds tensor to cylindrical coordinates"
    print("Transforming stress tensors..")
    r_cyl_df = transform_stress_tensor_to_cylindrical(dataframe, theta)
    print("Transformation completed.")
    dfs_to_concat = [coords_df, u_cyl_df, r_cyl_df]
    transformed_df = pd.concat([transformed_df, *dfs_to_concat], axis=1)
    wall_shear_stress_labels = [f"wallShearStressMean_{i}" for i in range(3)]
    return transformed_df.drop(
        columns=[
            "Points_Magnitude",
            "UMean_Magnitude",
            "UPrime2Mean_Magnitude",
            "wallShearStressMean_Magnitude",
            "yPlusMean",
            "Point ID",
            *common.paraview_cart_coords(),
            *wall_shear_stress_labels,
            *common.r_stress_paraview_cart_coords(),
            *common.umean_paraview_cart_coords()
        ]
    )


def transform_coordinates_to_cylindrical(dataframe):
    cart_pos_array = dataframe[common.paraview_cart_coords()].to_numpy()
    cart_pos_vectors = CartesianVector(cart_pos_array)
    radial_pos = np.sqrt(cart_pos_vectors.x**2 + cart_pos_vectors.y**2)
    axial_pos = cart_pos_vectors.z
    return pd.DataFrame(data={"r": radial_pos, "z": axial_pos})


def transform_velocity_to_cylindrical(dataframe, theta):
    u_keys = common.umean_paraview_cart_coords()
    u_array = dataframe[u_keys].to_numpy()
    u_cart_vectors = CartesianVector(vector_array=u_array)
    return (u_cart_vectors
            .convert_to_cylindrical(theta=theta)
            .as_dataframe(prefix="u_mean"))


def transform_stress_tensor_to_cylindrical(dataframe, theta):
    cart_stress_tensors = SymmetricCartesianTensor(
        dataframe[common.r_stress_paraview_cart_coords()].to_numpy()
    )
    cyl_stress_tensors = cart_stress_tensors.convert_to_cylindrical(theta)
    return cyl_stress_tensors.as_dataframe(prefix="R")
