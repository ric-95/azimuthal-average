import azimuthal_average.common as common
from azimuthal_average.data_types import SymmetricCylindricalTensor
import numpy as np


def process(slice_df, length_ref, u_ref):
    coord_keys = common.normalized_cyl_coords()
    slice_df = coordinates(slice_df, coord_keys, length_ref)
    norm_u_keys = common.normalized_u_cyl_coords()
    norm_r_keys = common.normalized_r_stress_cyl_coords()
    norm_slice_df = flow_vars(slice_df,
                                   norm_u_keys=norm_u_keys,
                                   norm_r_keys=norm_r_keys,
                                   u_reference=u_ref)
    post_df = calculate_derived_properties(norm_slice_df, normalized=True)
    return post_df


def coordinates(avg_slice_df, norm_coord_keys, length_ref):
    """Normalize the coordinates by a reference length.

    Args:
        avg_slice_df (DataFrame): Averaged slice dataframe.
        norm_coord_keys (list): Keys to assign to normalized coordinate components.
        length_ref (float): Reference length used for normalization.
    """
    dataframe = avg_slice_df.copy()
    dataframe[norm_coord_keys] = dataframe[["r", "z"]]/length_ref
    return dataframe


def velocity(avg_slice_df, norm_u_keys, u_reference):
    """Normalize the flow velocity vector by a reference velocity.

    Args:
        avg_slice_df (DataFrame): Averaged slice dataframe.
        norm_u_keys (list): Keys to assign to normalized velocity components.
        u_reference (float): Reference velocity used for normalization.
    """
    dataframe = avg_slice_df.copy()
    dataframe[norm_u_keys] = dataframe[common.umean_cyl_coods()]/u_reference
    return dataframe


def reynolds(avg_slice_df, norm_r_keys, u_reference):
    """Normalize the Reynolds stress tensor.

    Args:
        avg_slice_df (DataFrame): Averaged slice dataframe.
        norm_r_keys (list): Keys to assign to normalized Reynolds stress tensor.
        u_reference (float): Reference velocity used for normalization.
    """
    dataframe = avg_slice_df.copy()
    dataframe[norm_r_keys] = dataframe[common.r_stress_cyl_coords()]/(u_reference**2)
    return dataframe


def flow_vars(avg_slice_df, norm_u_keys, norm_r_keys, u_reference):
    """Normalize velocity and Reynolds stress.

    Args:
        avg_slice_df (DataFrame): Averaged slice dataframe.
        norm_u_keys (list): Keys to assign to normalized velocity components.
        norm_r_keys (list): Keys to assign to normalized Reynolds stress tensor.
        u_reference (float): Reference velocity used for normalization.
    """
    dataframe = velocity(avg_slice_df, norm_u_keys, u_reference)
    dataframe = reynolds(dataframe, norm_r_keys, u_reference)
    return dataframe


def calculate_derived_properties(slice_df, normalized=False):
    if normalized:
        r_keys = common.normalized_r_stress_cyl_coords()
    else:
        r_keys = common.r_stress_cyl_coords()

    reynolds_stress = (
        SymmetricCylindricalTensor(
            slice_df[r_keys].to_numpy()
        ).tensor_stack
    )
    tke = calculate_TKE(reynolds_stress)
    anistropy_tensor = calculate_anisotropy_tensor(reynolds_stress)
    second_invariant = calculate_second_invariant(anistropy_tensor)*(-1)
    third_invariant = calculate_third_invariant(anistropy_tensor)
    df = slice_df.copy()
    derived_properties_keys = common.derived_properties()
    derived_properties_arrays = [tke, second_invariant, third_invariant]
    for key, array in zip(derived_properties_keys, derived_properties_arrays):
        df[key] = array
    return df


def calculate_TKE(tensor):
    return 1/2*np.trace(tensor, axis1=1, axis2=2)


def calculate_anisotropy_tensor(tensor):
    tke = calculate_TKE(tensor)+10**-7
    return tensor/(2*tke.reshape(tke.shape[0], 1, 1)) - np.identity(3)/3


def calculate_second_invariant(tensor):
    a2 = np.matmul(tensor, tensor)
    return 1/2*(np.trace(tensor, axis1=1, axis2=2)**2 - np.trace(a2, axis1=1, axis2=2))


def calculate_third_invariant(anisotropy_tensor):
    return np.linalg.det(anisotropy_tensor)
