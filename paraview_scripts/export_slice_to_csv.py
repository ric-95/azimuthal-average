# trace generated using paraview version 5.9.0-RC2

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'


def export_slice_to_csv(render_view, output_file="slice.csv"):
    paraview.simple._DisableFirstRenderCameraReset()

    # get active view
    renderView1 = render_view

    # destroy renderView1
    Delete(renderView1)
    del renderView1

    # Create a new 'SpreadSheet View'
    spreadSheetView1 = CreateView('SpreadSheetView')
    spreadSheetView1.ColumnToSort = ''
    spreadSheetView1.BlockSize = 1024

    animationScene1 = GetAnimationScene()

    animationScene1.GoToLast()

    # get active source.
    resampleWithDataset1 = GetActiveSource()

    # show data in view
    resampleWithDataset1Display = Show(resampleWithDataset1, spreadSheetView1, 'SpreadSheetRepresentation')

    # get layout
    layout1 = GetLayoutByName("Layout #1")

    # assign view to a particular cell in the layout
    AssignViewToLayout(view=spreadSheetView1, layout=layout1, hint=0)

    # export view
    ExportView(output_file, view=spreadSheetView1)

    #================================================================
    # addendum: following script captures some of the application
    # state to faithfully reproduce the visualization during playback
    #================================================================

    #--------------------------------
    # saving layout sizes for layouts

    # layout/tab size in pixels
    layout1.SetSize(400, 400)

    #--------------------------------------------
    # uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).
    return spreadSheetView1
