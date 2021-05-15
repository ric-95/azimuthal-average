import os
import pandas as pd
import numpy as np
import azimuthal_average.common as common
import azimuthal_average.data.data_types as dt
import azimuthal_average.data.data_loading as dl


def take_azimuthal_average(csv_dir, write_intermediate=False, copy=True):
    """Take the azimuthal average of all the csv files within a given directory.

    Args:
        csv_dir (str): Path to directory containing """
    csv_list = dl.get_csv_files_list(csv_dir)
    total_slices = len(csv_list)
    slices_df_list = []
    for slice_number, csv_file in enumerate(csv_list):
        print("Transforming", csv_file)
        slice_df = dl.load_single_csv_file_as_df(csv_file)
        theta = (slice_number/total_slices)*np.pi*2
        cylindrical_slice_df = transform_df_to_cylindric_coordinates(slice_df,
                                                                     theta,
                                                                     copy)
        if write_intermediate:
            intermediate_dir = os.path.join(csv_dir, "intermediate")
            if not os.path.isdir(intermediate_dir):
                os.mkdir(intermediate_dir)
            intermediate_file = os.path.join(intermediate_dir, f"cyl_slice_{slice_number}.csv")
            cylindrical_slice_df.to_csv(intermediate_file, sep=",")
        slices_df_list.append(cylindrical_slice_df)
    concatenated_slices_df = pd.concat(
        slices_df_list,
        keys=[i for i in range(total_slices)])
    print(concatenated_slices_df)
    return concatenated_slices_df.groupby(level=1).mean()


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
    cart_pos_vectors = dt.CartesianVector(cart_pos_array)
    radial_pos = np.sqrt(cart_pos_vectors.x**2 + cart_pos_vectors.y**2)
    axial_pos = cart_pos_vectors.z
    return pd.DataFrame(data={"r": radial_pos, "z": axial_pos})


def transform_velocity_to_cylindrical(dataframe, theta):
    u_keys = common.umean_paraview_cart_coords()
    u_array = dataframe[u_keys].to_numpy()
    u_cart_vectors = dt.CartesianVector(vector_array=u_array)
    return (u_cart_vectors
            .convert_to_cylindrical(theta=theta)
            .as_dataframe(prefix="u_mean"))


def transform_stress_tensor_to_cylindrical(dataframe, theta):
    cart_stress_tensors = dt.SymmetricCartesianTensor(
        dataframe[common.r_stress_paraview_cart_coords()].to_numpy()
    )
    cyl_stress_tensors = cart_stress_tensors.convert_to_cylindrical(theta)
    return cyl_stress_tensors.as_dataframe(prefix="R")



