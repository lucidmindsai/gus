# -*- coding: utf-8 -*-
from functools import reduce
import numpy as np

def get_raster_data(dtwin,
                    var = 'unique_id',
                    probe = lambda x,y: x + y,
                    predicate = lambda x: x,
                    init = 0,
                    counts = False):
    """A general purpose function that collects celular data on the grid.

    Args:
            dtwin: (:obj:`Urban`): an urban forest model.
            var: (:obj:`string`): a valid memory variable of a tree.

    Returns:
            (:obj:numpy.ndarray`): raster like data on a 2D grid.
        Note:
        Todo:
    """
    if counts: probe = lambda x,y: x + 1
    raster = np.zeros((dtwin.grid.width, dtwin.grid.height))
    for cell in dtwin.grid.coord_iter():
        cell_content, x, y = cell
        cell_content = filter(lambda x: x.condition not in ['dead' ,'replaced'], cell_content)
        filtered = filter(predicate, [eval('a.{}'.format(var)) for a in cell_content])
        raster[x][y] = reduce(probe, filtered, init)
    return raster






