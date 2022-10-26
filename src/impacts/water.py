# -*- coding: utf-8 -*-

# Water retention related implementation.
# Based on Notebook implemenation created by Marko Petrovic
#  

# Package for using operating system dependent functionality

# Numerical data processing
import pandas as pd
import numpy as np
from scipy import stats

# Time series operations
import glob
from datetime import datetime

class Calibration():
    """Water retention and impact estimate related parameters setting."""
    # Conversion coefficient meter to millimeter
    m_to_mm = 1

    # Extinction coefficient =0.7 for trees and 0.3 for shrubs (Wang et al. 2008)
    kappa = 0.7

    # Shade factor, is the percentage of sky covered by foliage and branches within
    # the perimeter of individual tree crowns, can vary by species from about
    # 60% to 95% when trees are in-leaf (McPherson, 1984).
    # The value below is set according to Glasgow mean
    avg_shade_factor = 0.85

    # Specific leaf storage of water (sl =0.0002 m).
    leaf_storage = 0.0002 * m_to_mm

    # Leaf-on leaf_off transition days (Wang_et_al_2008).
    leaf_transition_days = 28

    # Specific impervious cover storage of water (=0.0015 m).
    maximum_impervious_cover_storage = 0.0015 * m_to_mm

    # Specific pervious cover storage of water (=0.001 m).
    maximum_pervious_cover_storage = 0.001 * m_to_mm

    def __init__(
        self,
        leaf_transition_days = 28,
        leaf_storage = 0.0002,
        pervios_storage_max = 0.0010,
        impervios_storage_max = 0.0015
        ):
        """The constructor method.

        Args:
            leaf_transition_days: (:obj:`int`): Leaf-on leaf_off transition days
            leaf_storage: (:obj:`float`): Specific leaf storage of water.
            pervios_storage_max: (:obj:`float`): Specific pervious cover storage of water
            impervios_storage_max: (:obj:`float`): Specific impervious cover storage of water
        Returns:
            None
        Note:

        TODO:
            * pass optional settings as key word parameters.
        """
        self.leaf_storage = leaf_storage * Calibration.m_to_mm
        self.leaf_transition_days = leaf_transition_days
        self.maximum_impervious_cover_storage = impervios_storage_max * Calibration.m_to_mm
        self.maximum_pervious_cover_storage = pervios_storage_max * Calibration.m_to_mm

    def set_surface_storage_rates(
        self, leaf_storage=None,
        pervios_storage_max=None,
        impervios_storage_max=None):
        """Setting specific water storage rates. 

        Args:
            leaf_storage: (:obj:`float`): Specific leaf storage of water.
            pervios_storage_max: (:obj:`float`): Specific pervious cover storage of water
            impervios_storage_max: (:obj:`float`): Specific impervious cover storage of water
        Returns:
            None
        Note:

        Todo:

        """
        if leaf_storage:
            self.leaf_storage = leaf_storage * Calibration.m_to_mm
        if impervios_storage_max:
            self.maximum_impervious_cover_storage = impervios_storage_max * Calibration.m_to_mm
        if pervios_storage_max:
            self.maximum_pervious_cover_storage = pervios_storage_max * Calibration.m_to_mm
    

def compute_leaf_area_index(
    dbh,
    tree_height,
    crown_height,
    crown_width,
    crown_missing = 0,
    shade_factor = Calibration.avg_shade_factor):
    """The function given allometrics of a tree computes its leaf, bark and plant area indices. 

    Args:
        dbh: (:obj:`float`): the diameter in cm of the trunk usually measured at 1.3m from the ground.
        tree_height: (:obj:`float`): The tree height in meters.
        crown_height: (:obj:`float`): The vertical length of tree crown in meters.
        crown_width: (:obj:`float`): The horizontal length (diameter) of tree crown in meters.
        crown_missing: (:obj:`float`): The percentage loss of the crown.
        shade_factor: (:obj:`float`): The percentage of sky covered by foliage and branches.

    Returns:
        (:obj:`tuple`): the tuple returns the tree indices (LAI,BAI,PAI)
    Note:
        The beta multipliers and the main equation is based on Nowak (1996).

    TODO:
        Parametrize beta multipliers.
    """
    loss = crown_missing
    th = tree_height
    cw = crown_width
    ch = crown_height
    sf = shade_factor
    beta_0 = -4.3309
    beta_1 = 0.2942
    beta_2 = 0.7312
    beta_3 = 5.7217
    beta_4 = 0.0148

    def compute_under_canopy_area(crown_width):
        return pow((crown_width/2),2) * np.pi
    
    def compute_bark_area(dbh, tree_height, crown_height):
        # * 0.01 converts DBH(cm) into meter.
        return np.pi * (dbh * 0.01) * (tree_height - crown_height)

    # Outer surface area estimate below is based on Gacka-Grzesikiewicz (1980).
    under_canopy = compute_under_canopy_area(cw)
    crown_surface = np.pi  * crown_width * (crown_height + crown_width)/2
    bark_area = compute_bark_area(dbh,th,ch)
    leaf_area = (1-loss) * np.exp(beta_0 + beta_1 * th + beta_2 * cw + beta_3 * sf - beta_4 * crown_surface)
    leaf_area_index = leaf_area / under_canopy
    bark_area_index = bark_area / under_canopy
    plant_area_index = leaf_area_index + bark_area_index
    return (leaf_area_index, bark_area_index, plant_area_index)