def cylindrical_vec_coords():
    return ["r", "t", "z"]


def cartesian_vec_coords():
    return ["x", "y", "z"]


def umean_cyl_coods():
    return [f"u_{coord}" for coord in cylindrical_vec_coords()]


def umean_cart_coors():
    return [f"u_{coord}" for coord in cartesian_vec_coords()]


def paraview_cart_coords():
    return [f"Points_{i}" for i in range(3)]


def umean_paraview_cart_coords():
    return [f"UMean_{i}" for i in range(3)]


def r_stress_paraview_cart_coords():
    return [f"UPrime2Mean_{i}" for i in range(6)]


def check_create_dir(dir_):
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    return dir_
