# An example from scipy cookbook demonstrating the use of numpy arrays in vtk
import os
import numpy as np
import vtk
import tifffile as tf


def main():
    colors = vtk.vtkNamedColors()

    # path = "/Volumes/Kidbeat/data/collab/Mito-GFP in early nuclear cycles 11-13/Example data/DUP_Series 5.tif--OMERO ID:636290.tif"
    # path = "/Users/Fabio/Desktop/Sas6-GFP_Sqh-mCh_WT_PE_20210211.mvd2 - Series 4-1 t=18.tif"
    path = "/Volumes/AYDOGAN - DROPBOX/Cycles/Sas6-GFP_Sqh-mCh_WT_PE_20210211.mvd2 - Series 4-1 t=18.tif"
    with tf.TiffFile(path) as tif:
        # data_matrix = tif.asarray()[10, :]
        print(tif.asarray().shape)
        data_matrix = tif.asarray()[:, 0, :, :]
        # map the data range to 0 - 255
        # data_matrix = ((data_matrix - data_matrix.min()) / (data_matrix.ptp() / 255.0)).astype(np.uint8)

    ztot, row, col = data_matrix.shape

    # imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
    data_string = data_matrix.tostring()
    dataImporter.SetImportVoidPointer(data_string)
    dataImporter.SetDataScalarTypeToUnsignedShort()

    # Because the data that is imported only contains an intensity value
    # the importer must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)
    # The following two functions describe how the data is stored and the dimensions of the array it is stored in.
    dataImporter.SetDataExtent(0, row - 1, 0, col - 1, 0, ztot - 1)
    dataImporter.SetWholeExtent(0, row - 1, 0, col - 1, 0, ztot - 1)
    dataImporter.SetDataExtentToWholeExtent()
    dataImporter.SetDataSpacing(1.0, 1.0, 16.0)  # scale z with a high number!

    # Define transparency-values for use in the volume dataset.
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.01)
    alphaChannelFunc.AddPoint(np.iinfo(np.uint16).max, 0.5)

    # This class stores color data and can create color tables from a few color points.
    num = 20
    scale = np.linspace(0, np.iinfo(np.uint8).max * .5, num=num)
    colorFunc = vtk.vtkColorTransferFunction()
    for intensity, color in zip(np.linspace(0, np.iinfo(np.uint16).max, num=num), scale):
        print(f"{intensity:.2f}, {color:.2f}")
        colorFunc.AddRGBPoint(intensity, 0, color, 0)

    # Add color and aplha into volume properties.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)

    volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    # The class vtkVolume is used to pair the previously declared volume as well as
    # the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # Initialize the renderer and window
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)

    # Add the volume to the renderer
    renderer.AddVolume(volume)

    renderer.SetBackground(colors.GetColor3d("MistyRose"))
    renderWin.SetSize(1000, 1000)
    renderWin.SetPosition(2000, 100)

    # A simple function to be called when the user decides to quit the application.
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    # Tell the application to use the function as an exit check.
    renderWin.AddObserver("AbortCheckEvent", exitCheck)

    renderInteractor.Initialize()
    # Order the first render manually before control is handed over to the main-loop.
    renderWin.Render()
    renderInteractor.Start()


if __name__ == '__main__':
    main()
