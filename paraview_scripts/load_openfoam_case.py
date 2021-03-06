# trace generated using paraview version 5.9.0-RC2

#### import the simple module from the paraview
from paraview.simple import *
import json

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

LES_DEFAULT_ARRAYS = ('DESRegionMean',
                      'DESRegionPrime2Mean',
                      'UMean',
                      'UPrime2Mean',
                      'kMean',
                      'nutMean',
                      'pMean',
                      'pPrime2Mean',
                      'wallShearStressMean',
                      'yPlusMean')


def get_openfoam_source(case_dir, cell_arrays=LES_DEFAULT_ARRAYS):
    """Return a ParaView source of an OpenFOAM case.

     Args:
         case_dir (str): Path to OpenFOAM case.
         cell_arrays (list): Names of the cellArrays to load.
         """
    # create a new 'OpenFOAMReader'
    file_name = "{case_dir}/system/controlDict".format(case_dir=case_dir)
    case_source = OpenFOAMReader(registrationName='controlDict',
                                 FileName=file_name)
    case_source.MeshRegions = ['internalMesh']
    case_source.CellArrays = cell_arrays
    return case_source

