""" The module implements water retention related implementation. """

# Based on Notebook implemenation created by Marko Petrovic

# Numerical data processing
import pandas as pd
import numpy as np

class Calibration:
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
        leaf_transition_days=28,
        leaf_storage=0.0002,
        pervios_storage_max=0.0010,
        impervios_storage_max=0.0015,
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
            None

        TODO:
            * pass optional settings as key word parameters.
        """
        self.leaf_storage = leaf_storage * Calibration.m_to_mm
        self.leaf_transition_days = leaf_transition_days
        self.maximum_impervious_cover_storage = (
            impervios_storage_max * Calibration.m_to_mm
        )
        self.maximum_pervious_cover_storage = pervios_storage_max * Calibration.m_to_mm

    def set_surface_storage_rates(
        self, leaf_storage=None, pervios_storage_max=None, impervios_storage_max=None
    ):
        """Setting specific water storage rates.

        Args:
            leaf_storage: (:obj:`float`): Specific leaf storage of water.
            pervios_storage_max: (:obj:`float`): Specific pervious cover storage of water
            impervios_storage_max: (:obj:`float`): Specific impervious cover storage of water
        Returns:
            None
        Note:
            None
        Todo:
            None
        """
        if leaf_storage:
            self.leaf_storage = leaf_storage * Calibration.m_to_mm
        if impervios_storage_max:
            self.maximum_impervious_cover_storage = (
                impervios_storage_max * Calibration.m_to_mm
            )
        if pervios_storage_max:
            self.maximum_pervious_cover_storage = (
                pervios_storage_max * Calibration.m_to_mm
            )