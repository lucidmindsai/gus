"""The module holds general purpose python functions on data preperations."""
from functools import reduce
import numpy as np
import utm


def get_raster_data(
    dtwin,
    var="unique_id",
    probe=lambda x, y: x + y,
    predicate=lambda x: x,
    init=0,
    counts=False,
):
    """A general purpose function that collects celular data on the grid.

    Args:
            dtwin: (:obj:`Urban`): an urban forest model.
            var: (:obj:`string`): a valid memory variable of a tree.

    Returns:
            (:obj:numpy.ndarray`): raster like data on a 2D grid.
        Note:
            None
        Todo:
            None
    """
    if counts:
        probe = lambda x, y: x + 1
    raster = np.zeros((dtwin.grid.width, dtwin.grid.height))
    for cell in dtwin.grid.coord_iter():
        cell_content, x, y = cell
        cell_content = filter(
            lambda x: x.condition not in ["dead", "replaced"], cell_content
        )
        filtered = filter(predicate, [eval("a.{}".format(var)) for a in cell_content])
        raster[x][y] = reduce(probe, filtered, init)
    return raster


def latlng_to_xy(row):
    """A general purpose function that translates lat, lng data to x,y pos.

    Args:
            row: (pandas.DataFrame.row): a Pandas DataFrame row.

    Returns:
            row: (pandas.DataFrame.row): converted xpos and ypos added to the DataFrame row.
        Note:
            None
        Todo:
            None
    """
    coordinates = utm.from_latlon(row['lat'], row['lng'])
    row['xpos'], row['ypos'] = coordinates[0], coordinates[1]
    return row


def raster_grid(row, minx, miny, grid_width):
    """A general purpose function is to place the data on the grid with given sizes.

    Args:
            row: (pandas.DataFrame.row): a Pandas DataFrame row.
            minx: (:obj:`int`): minimum x pos value.
            miny: (:obj:`int`): minimum y pos value.
            grid_width: (:obj:`string`): width of the grid to be mapped at.
    Returns:
            row: (pandas.DataFrame.row): converted xpos and ypos added to the DataFrame row.
        Note:
            None
        Todo:
            None
    """
    row['gus_x'] = int((row['xpos'] - minx) // grid_width)
    row['gus_y'] = int((row['ypos'] - miny) // grid_width)
    return row

