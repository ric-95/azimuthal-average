from paraview.simple import *
import sys
import math
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--case-dir", type=str)
parser.add_argument("--num-slices", type=int, default=16)
args = parser.parse_args()
paraview.simple._DisableFirstRenderCameraReset()

SCRIPTS_DIR = r"C:\Users\Testing\PycharmProjects\azimuthal_average\paraview_scripts"
sys.path.append(SCRIPTS_DIR)

import load_openfoam_case as lc
import extract_single_slice as ess
import export_slice_to_csv as esc

# import extract_single_slice as ess
# import export_slice_to_csv as esc
# import load_openfoam_case as lc

CASE_DIR = args.case_dir
NUMBER_OF_SLICES = args.num_slices
DELTA_RES = 0.00025
ORIGIN_VECTOR = [0, 0, 0]
RADIUS = 0.09
AXIS_LENGTH = 0.15

X_RESOLUTION = round(RADIUS / DELTA_RES)
Y_RESOLUTION = round(AXIS_LENGTH / DELTA_RES)

PI = math.pi


def export_all_slices_to_csv(case_dir,
                             number_of_slices,
                             origin_vector,
                             radius,
                             axis_length,
                             x_res,
                             y_res):
    """Exports ParaView azimuthal slices to separate csv files.

    Args:
        case_dir            (str): Path to OpenFOAM case directory.
        number_of_slices    (int): Number of azimuthal slices to extract to take an average.
        origin_vector       (list): Origin cartesian vector as list.
        radius              (float): Radius length.
        axis_length         (float): Length of the axis.
        x_res               (int): Number of points along X direction of plane.
        y_res               (int): Number of points along Y direction of plane.
        """
    print("Loading OpenFOAM")
    openfoam_source = lc.get_openfoam_source(case_dir)
    initial_radius_vector = [*ess.calculate_cartesian_radius_vector(radius, theta=0), 0]
    axial_vector = [0, 0, axis_length]
    print("Starting slice extraction")
    arbitrary_slice, render_view = ess.paraview_radial_slice_extraction(
        openfoam_source,
        origin_vector,
        initial_radius_vector,
        axial_vector,
        x_res=x_res,
        y_res=y_res
    )

    for slice_number in range(number_of_slices):
        "Calculate the angle and radius vector"
        theta = (slice_number / number_of_slices) * (2*PI)
        radius_xy = ess.calculate_cartesian_radius_vector(radius, theta)
        radius_vector = [*radius_xy, 0]
        arbitrary_slice.Point1 = radius_vector
        output_dir = os.path.join(case_dir, "slice_csv_files")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(
            output_dir, "slice_{slice_number}.csv".format(
                slice_number=str(slice_number).zfill(3)
            )
        )

        print("Outputting to csv files")
        render_view = esc.export_slice_to_csv(render_view=render_view,
                                              output_file=output_file)


def main():
    case_dir = CASE_DIR
    number_of_slices = NUMBER_OF_SLICES
    origin_vector = ORIGIN_VECTOR
    radius = RADIUS
    axis_length = AXIS_LENGTH
    x_res = X_RESOLUTION
    y_res = Y_RESOLUTION
    print(f"Extracting slices with {x_res}x{y_res} resolution")
    export_all_slices_to_csv(
        case_dir=case_dir,
        number_of_slices=number_of_slices,
        origin_vector=origin_vector,
        radius=radius,
        axis_length=axis_length,
        x_res=x_res,
        y_res=y_res
    )
    return


if __name__ == "__main__":
    main()
