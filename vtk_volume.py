import os

import numpy as np
import vtk
import tifffile as tf
import xml.dom.minidom

from cached import CachedImageFile, cached_step
from filters import polsby_popper
from segmentation.compartments import cluster_by_centroid, segment_zstack
from logger import get_logger

log = get_logger(name='volume')


def comparment_iterator(img_struct):
    comps_df = cached_step(f"c{0}t{8}-segmentation-dataframe.obj",
                           segment_zstack, img_struct, frame=8,
                           cache_folder=img_struct.cache_path)

    log.info("Computing geometric features.")
    comps_df.loc[:, 'area'] = comps_df['boundary'].apply(lambda c: c.area)
    comps_df.loc[:, 'radius'] = comps_df['area'].apply(lambda a: np.sqrt(a) / np.pi)

    comps_df = (comps_df
                # .pipe(lambda df: df[df['offset'] > 80])
                .pipe(lambda df: df[(df['area'] > 500) & (df['area'] < 10e4)])
                .pipe(polsby_popper, 'boundary', pp_threshold=0.7)
                .pipe(cluster_by_centroid, eps=0.13)
                .pipe(lambda df: df[df['cluster'] > 0])
                )
    for ix, row in comps_df.iterrows():
        boundary = row['boundary']
        # Create a vtkPoints object and store the points in it
        points = vtk.vtkPoints()
        for pt in boundary.exterior.coords:
            x, y = pt
            # flip y coordinate and construct point in 3D
            point = (x, y, row['z'])
            points.InsertNextPoint(point)
        # Create a cell array to store the lines in and add the lines to it
        lines = vtk.vtkCellArray()
        for i in range(points.GetNumberOfPoints() - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            lines.InsertNextCell(line)
        # Create a polydata to store everything in
        linesPolyData = vtk.vtkPolyData()
        # Add the points to the dataset
        linesPolyData.SetPoints(points)
        # Add the lines to the dataset
        linesPolyData.SetLines(lines)
        # Setup actor and mapper
        colors = vtk.vtkNamedColors()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(linesPolyData)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(4)
        actor.GetProperty().SetColor(colors.GetColor3d('Red'))
        actor.GetProperty().SetOpacity(0.4)

        yield actor


def main(img_struc):
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
    alphaChannelFunc.AddPoint(0, 0.005)
    alphaChannelFunc.AddPoint(np.iinfo(np.uint16).max, 0.03)

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

    for actor in comparment_iterator(img_struc):
        renderer.AddActor(actor)

    renderer.SetBackground(colors.GetColor3d("MistyRose"))
    renderWin.SetSize(1500, 1500)
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


def save_data(img_struc):
    # Write each grid into  a file
    filenames = list()
    for i, actor in enumerate(comparment_iterator(img_struc)):
        filename = f"boundary_{i:03d}.vtp"
        filenames.append(filename)
        print('Writing: ', filename)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(os.path.join(img_struc.cache_path, 'out', filename))
        writer.SetInputData(actor.GetMapper().GetInput())
        writer.Write()

        # --------------------------------------
        #  write pdv file
        # --------------------------------------
        pvd = xml.dom.minidom.Document()
        pvd_root = pvd.createElementNS("VTK", "VTKFile")
        pvd_root.setAttribute("type", "Collection")
        pvd_root.setAttribute("version", "0.1")
        pvd_root.setAttribute("byte_order", "LittleEndian")
        pvd.appendChild(pvd_root)

        collection = pvd.createElementNS("VTK", "Collection")
        pvd_root.appendChild(collection)

        for i, fname in enumerate(filenames):
            dataSet = pvd.createElementNS("VTK", "DataSet")
            dataSet.setAttribute("timestep", str(0))
            dataSet.setAttribute("group", "")
            dataSet.setAttribute("part", "0")
            dataSet.setAttribute("file", fname)
            collection.appendChild(dataSet)

        pdv_file = open(os.path.join(img_struc.cache_path, 'out', 'boundaries.pvd'), 'w')
        pvd.writexml(pdv_file, newl='\n')
        pdv_file.close()


if __name__ == '__main__':
    path = "/Volumes/AYDOGAN - DROPBOX/Cycles/" \
           "Sas6-GFP_Sqh-mCh_WT_PE_20210211/" \
           "Sas6-GFP_Sqh-mCh_WT_PE_20210211.mvd2"

    img_struc = CachedImageFile(path, image_series=3)

    save_data(img_struc)
    main(img_struc)

    # --------------------------------------
    #  Finish
    # --------------------------------------
    # if img_struc._jvm_on:
    import javabridge

    javabridge.kill_vm()
