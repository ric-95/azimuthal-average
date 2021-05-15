import numpy as np
import pandas as pd
import azimuthal_average.common as common


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

