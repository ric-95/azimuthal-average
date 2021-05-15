# trace generated using paraview version 5.9.0-RC2

#### import the simple module from the paraview
from paraview.simple import *
import load_openfoam_case as lc
#### disable automatic camera reset on 'Show'
import math


def calculate_cartesian_radius_vector(radius, theta):
    """Calculates a cartesian radius vector given a radius and angle in radians

    Arguments:

    radius -- scalar
    theta -- angle in radians
    """
    x_component = radius*math.cos(theta)
    y_component = radius*math.sin(theta)
    return x_component, y_component


def add_vectors(u, v):
    return [a + b for a, b in zip(u,v)]



def paraview_radial_slice_extraction(openfoam_source,
                                     origin_vector,
                                     radius_vector,
                                     axial_vector,
                                     x_res=900,
                                     y_res=1500):
    """Extract a radial slice within ParaView.

    Args:
        openfoam_source: Source of OpenFOAM data.
        origin_vector (list): Origin cartesian vector as list
        radius_vector (list): Radius cartesian vector as list
        axial_vector (list): Axis of rotation vector as list
        x_res (int): Number of points in the XDirection of the plane in ParaView.
        y_res (int): Number of points in the YDirection of the plane in ParaView

        """
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'Plane'
    radial_plane_source = Plane(registrationName='radial_plane')

    # find source
    controlDict = openfoam_source

    # Properties modified on radial_plane_source
    point_1 = add_vectors(origin_vector, radius_vector)
    point_2 = add_vectors(origin_vector, axial_vector)
    radial_plane_source.Origin = origin_vector
    radial_plane_source.Point1 = point_1
    radial_plane_source.Point2 = point_2
    radial_plane_source.XResolution = x_res
    radial_plane_source.YResolution = y_res

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    plane1Display = Show(radial_plane_source, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    plane1Display.Representation = 'Surface'
    plane1Display.ColorArrayName = [None, '']
    plane1Display.SelectTCoordArray = 'TextureCoordinates'
    plane1Display.SelectNormalArray = 'Normals'
    plane1Display.SelectTangentArray = 'None'
    plane1Display.OSPRayScaleArray = 'Normals'
    plane1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    plane1Display.SelectOrientationVectors = 'None'
    plane1Display.ScaleFactor = 0.015000000596046448
    plane1Display.SelectScaleArray = 'None'
    plane1Display.GlyphType = 'Arrow'
    plane1Display.GlyphTableIndexArray = 'None'
    plane1Display.GaussianRadius = 0.0007500000298023224
    plane1Display.SetScaleArray = ['POINTS', 'Normals']
    plane1Display.ScaleTransferFunction = 'PiecewiseFunction'
    plane1Display.OpacityArray = ['POINTS', 'Normals']
    plane1Display.OpacityTransferFunction = 'PiecewiseFunction'
    plane1Display.DataAxesGrid = 'GridAxesRepresentation'
    plane1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    plane1Display.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    plane1Display.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Resample With Dataset'
    resampleWithDataset1 = ResampleWithDataset(registrationName='ResampleWithDataset1', SourceDataArrays=controlDict,
        DestinationMesh=radial_plane_source)
    resampleWithDataset1.CellLocator = 'Static Cell Locator'

    # show data in view
    resampleWithDataset1Display = Show(resampleWithDataset1, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'p'
    pLUT = GetColorTransferFunction('p')

    # trace defaults for the display properties.
    resampleWithDataset1Display.Representation = 'Surface'
    resampleWithDataset1Display.ColorArrayName = ['POINTS', 'p']
    resampleWithDataset1Display.LookupTable = pLUT
    resampleWithDataset1Display.SelectTCoordArray = 'None'
    resampleWithDataset1Display.SelectNormalArray = 'None'
    resampleWithDataset1Display.SelectTangentArray = 'None'
    resampleWithDataset1Display.OSPRayScaleArray = 'p'
    resampleWithDataset1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    resampleWithDataset1Display.SelectOrientationVectors = 'U'
    resampleWithDataset1Display.ScaleFactor = 0.015000000596046448
    resampleWithDataset1Display.SelectScaleArray = 'p'
    resampleWithDataset1Display.GlyphType = 'Arrow'
    resampleWithDataset1Display.GlyphTableIndexArray = 'p'
    resampleWithDataset1Display.GaussianRadius = 0.0007500000298023224
    resampleWithDataset1Display.SetScaleArray = ['POINTS', 'p']
    resampleWithDataset1Display.ScaleTransferFunction = 'PiecewiseFunction'
    resampleWithDataset1Display.OpacityArray = ['POINTS', 'p']
    resampleWithDataset1Display.OpacityTransferFunction = 'PiecewiseFunction'
    resampleWithDataset1Display.DataAxesGrid = 'GridAxesRepresentation'
    resampleWithDataset1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    resampleWithDataset1Display.ScaleTransferFunction.Points = [-30.801410675048828, 0.0, 0.5, 0.0, 21.050045013427734, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    resampleWithDataset1Display.OpacityTransferFunction.Points = [-30.801410675048828, 0.0, 0.5, 0.0, 21.050045013427734, 1.0, 0.5, 0.0]

    # hide data in view
    Hide(radial_plane_source, renderView1)

    # hide data in view
    Hide(controlDict, renderView1)

    # show color bar/color legend
    resampleWithDataset1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get opacity transfer function/opacity map for 'p'
    pPWF = GetOpacityTransferFunction('p')

    #================================================================
    # addendum: following script captures some of the application
    # state to faithfully reproduce the visualization during playback
    #================================================================

    # get layout
    layout1 = GetLayout()

    #--------------------------------
    # saving layout sizes for layouts

    # layout/tab size in pixels
    layout1.SetSize(990, 796)

    #-----------------------------------
    # saving camera placements for views

    # current camera placement for renderView1
    renderView1.CameraPosition = [0.0, 0.0, 1.3039457924347284]
    renderView1.CameraFocalPoint = [0.0, 0.0, 0.004999995231628418]
    renderView1.CameraParallelScale = 0.33619191087203887

    #--------------------------------------------
    # uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).
    return radial_plane_source, renderView1



