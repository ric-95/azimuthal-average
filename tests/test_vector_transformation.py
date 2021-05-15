from azimuthal_average.data import data_types

velocity_cartesian_vector = data_types.CartesianVector([1, 2, 3])
r_vector = data_types.CartesianVector([3, -2, 1])
velocity_cylindrical_vector = velocity_cartesian_vector.convert_to_cylindrical(r_vector)
print(velocity_cylindrical_vector)