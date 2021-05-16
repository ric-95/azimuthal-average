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


r"""
/*---------------------------------*\
             VECTORS
\*---------------------------------*/
"""


class CartesianVector:
    """Class for vectors in cartesian coordinates.

    Instance attributes:

    self.vector -- numpy array containing the vector
    self.x -- radial component
    self.y -- azimuthal component
    self.z -- axial component
    """

    def __init__(self, vector_array):
        """Constructor for a CartesianVector instance.

        Args:
            vector_array (Numpy array): Array of cartesian row vectors
        """
        self.array = np.array(vector_array)
        self.comps = common.cartesian_vec_coords()
        for i, comp in enumerate(self.comps):
            setattr(self, comp, self.array[:, i])
        return

    def convert_to_cylindrical(self, theta):
        rotation_matrix = get_rotation_matrix(theta)
        print(rotation_matrix)
        cylindrical_vector_array = (rotation_matrix.dot(self.array.T)).T
        return CylindricalVector(cylindrical_vector_array)

    def __str__(self):
        return f"{self.array}"

    def as_dataframe(self, prefix=""):
        if prefix:
            columns = [f"{prefix}_{comp}" for comp in self.comps]
        else:
            columns = self.comps
        return pd.DataFrame(data=self.array, columns=columns)


class CylindricalVector:
    """Class for vectors in cylindrical coordinates.

    Instance attributes:

    self.vector -- numpy array containing the vector
    self.r -- radial component
    self.t -- azimuthal component
    self.z -- axial component
    """

    def __init__(self, vector_array):
        """Constructor for a CylindricalVector instance.

        Args:
            vector_array (Numpy array): Array of cartesian row vectors
        """
        self.array = np.array(vector_array)
        self.comps = common.cylindrical_vec_coords()
        for i, comp in enumerate(self.comps):
            setattr(self, comp, self.array[:, i])
        return

    def __str__(self):
        return f"{self.array}"

    def as_dataframe(self, prefix=""):
        if prefix:
            columns = [f"{prefix}_{comp}" for comp in self.comps]
        else:
            columns = self.comps
        return pd.DataFrame(data=self.array, columns=columns)



r"""
/*---------------------------------*\
             TENSORS
\*---------------------------------*/
"""


def rotate_tensor(tensor_stack, rotation_matrix):
    return np.matmul(rotation_matrix, np.matmul(tensor_stack, rotation_matrix.T))


class SymmetricCartesianTensor:

    list_of_components = ["xx", "yy", "zz", "xy", "yz", "xz"]

    def __init__(self, symm_tensor_array):
        for i, comp in enumerate(SymmetricCartesianTensor.list_of_components):
            setattr(self, comp, symm_tensor_array[:, i])
        tensor_row_index = np.array([
            [0, 3, 5],
            [3, 1, 4],
            [5, 4, 2]
        ])
        tensor_list = []
        for array_pos in range(len(symm_tensor_array)):
            tensor_row_i = symm_tensor_array[array_pos, :]
            tensor_i = tensor_row_i[tensor_row_index]
            tensor_list.append(tensor_i)
        self.tensor_stack = np.array(tensor_list)
        self.symm_array = symm_tensor_array
        return

    def convert_to_cylindrical(self, theta):
        rotation_matrix = get_rotation_matrix(theta)
        print(rotation_matrix)
        cylindrical_tensor_stack = rotate_tensor(self.tensor_stack, rotation_matrix)
        return SymmetricCylindricalTensor(
            np.array([cyl_tensor[np.triu_indices(3)] for cyl_tensor in cylindrical_tensor_stack])
        )

    def __str__(self):
        return f"{self.tensor_stack}"


class SymmetricCylindricalTensor:
    list_of_components = ["rr", "rt", "rz", "tt", "tz", "zz"]

    def __init__(self, symm_tensor_array):
        for i, comp in enumerate(SymmetricCylindricalTensor.list_of_components):
            setattr(self, comp, symm_tensor_array[:, i])
        tensor_row_index = np.array([
            [0, 1, 2],
            [1, 3, 4],
            [2, 4, 5]
        ])
        tensor_list = []
        for array_pos in range(len(symm_tensor_array)):
            tensor_row_i = symm_tensor_array[array_pos, :]
            tensor_i = tensor_row_i[tensor_row_index]
            tensor_list.append(tensor_i)
        self.symm_array = symm_tensor_array
        self.tensor_stack = np.array(tensor_list)

    def __str__(self):
        return f"{self.tensor_stack}"

    def as_dataframe(self, prefix=""):
        if prefix:
            columns = [f"{prefix}_{comp}"
                       for comp in SymmetricCylindricalTensor.list_of_components]
        else:
            columns = SymmetricCylindricalTensor.list_of_components
        return pd.DataFrame(data=self.symm_array, columns=columns)


def write_intermediate_file(dir_, slice_number, slice):
    common.check_create_dir(dir_)
    intermediate_file = f"cyl_clice_{slice_number}.csv"
    intermediate_file = os.path.join(dir_, intermediate_file)
    slice.to_csv(intermediate_file, sep=",")


def update_running_average(running_average, n, new_obs):
    """
    Parameters
    ----------
    cumulative_avg: value of the cumulative average so far
    n: Number of samples including new observation
    new_obs: New observation"""
    a = 1/n
    b = 1 - a
    return a*new_obs + b*running_average


def take_azimuthal_average(csv_dir, write_intermediate=False, copy=True):
    """Take the azimuthal average of all the csv files within a given directory.

    Args:
        csv_dir (str): Path to directory containing """
    csv_list = dl.get_csv_files_list(csv_dir)
    total_slices = len(csv_list)
    slices_df_list = []
    running_average_ = None
    for slice_number, csv_file in enumerate(csv_list):
        print("Transforming", csv_file)
        slice_df = dl.load_single_csv_file_as_df(csv_file)
        theta = (slice_number/total_slices)*np.pi*2
        cylindrical_slice_df = transform_df_to_cylindric_coordinates(slice_df,
                                                                     theta,
                                                                     copy)
        cylindrical_slice_arr = cylindrical_slice_df.to_numpy()
        if running_average is None:
            running_average = cylindrical_slice_arr
        running_average = update_running_average(running_average, slice_number+1,
                                                 cylindrical_slice_arr)
        if write_intermediate:
            intermediate_dir = os.path.join(csv_dir, "intermediate")
            write_intermediate_file(intermediate_dir,
                                     slice_number, cylindrical_slice_df)

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
