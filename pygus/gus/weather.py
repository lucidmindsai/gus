"""The module holds a simple weather simulation for the length of growth season."""

import numpy as np


class WeatherSim:
    """The environment objects that simulates the weather."""

    # Yearly mean frost free days at a given site.
    # Source for 153 iTree, Nowak et.al., 2020.
    ffdays_mean = 153
    # Yearly variance in frost free days for a given site.
    ffdays_var = 7

    def __init__(self, season_length=None, season_var=None):
        """The constructor method.

        Args:
            season_length: (:obj:`float`): Mean length of carbon sequestraion season.
            season_var: (:obj:`float`): variance in year by year change, assuming a stable climate condition.

        Returns:
            None
        Note:
            * The current model assumes a stable climate condition and draws growth season by site specific
            mean and varaiance to employ a normal distribution of expectations in the future.

        Todo:
            * Update the weather simulation based on trend and climate change.

        """
        if season_length:
            self.ffdays_mean = season_length
        if season_var:
            self.ffdays_var = season_var

        # Drawing frost free days for the given year from a normal distriburion
        # where the mean is based on iTree's proxy variable.
        self.frost_free_days_ref = np.random.normal(self.ffdays_mean, self.ffdays_var)

    def check_frost_free_days(self):
        """The method generates frost free days based on mean and variance of a given site.

        Args:
            None
        Returns:
            (:obj:`int`): Number of frost free days.
        Note:
            None
        Todo:
            None
        """
        return self.frost_free_days_ref
