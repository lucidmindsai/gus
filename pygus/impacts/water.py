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


def compute_leaf_area_index(
    dbh,
    tree_height,
    crown_height,
    crown_width,
    crown_missing=0,
    shade_factor=Calibration.avg_shade_factor,
):
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
        return pow((crown_width / 2), 2) * np.pi

    def compute_bark_area(dbh, tree_height, crown_height):
        # * 0.01 converts DBH(cm) into meter.
        return np.pi * (dbh * 0.01) * (tree_height - crown_height)

    # Outer surface area estimate below is based on Gacka-Grzesikiewicz (1980).
    under_canopy = compute_under_canopy_area(cw)
    crown_surface = np.pi * crown_width * (crown_height + crown_width) / 2
    bark_area = compute_bark_area(dbh, th, ch)
    leaf_area = (1 - loss) * np.exp(
        beta_0 + beta_1 * th + beta_2 * cw + beta_3 * sf - beta_4 * crown_surface
    )
    leaf_area_index = leaf_area / under_canopy
    bark_area_index = bark_area / under_canopy
    plant_area_index = leaf_area_index + bark_area_index
    return (leaf_area_index, bark_area_index, plant_area_index)


def pai_seasons(x, leaf_on_start, leaf_off_start, leaf_transition_days):
    """The method updates Plant Area Index (PAI) with respect to leaf on-off seasons.

    Args:
        A data frame "x" with following variables:
        Date_time: (:obj:`time`): date and time
        BAI: Bark Area Index
        LAI: Leaf Area Index
        conifers: it taks value "true" if conifers and "false" otherwise

        leaf_on_start: a day in the year when the leaf on season starts. In the case of Glasgow it is April 14 (day 105).
                        See also other sources: https://weatherspark.com/y/147740/Average-Weather-at-Glasgow-Airport-United-Kingdom-Year-Round
        leaf_off_start: a day in the year when the leaf off season starts. In the case of Glasgow it is November 2 (day 307).
        leaf_transition_days: the number of days that the leaf on-off transition last.


    Returns:
        An array of values of Plant Area Indexes (PAI) over time.

    Note:
        The function is built upon the following paper:
        Wang, Jun, Theodore A. Endreny, and David J. Nowak. “Mechanistic Simulation of Tree
        Effects in an Urban Water Balance Model 1.” JAWRA Journal of the American Water Resources
        Association 44, no. 1 (February 2008): 75–85. https://doi.org/10.1111/j.1752-1688.2007.00139.x

    Todo:
        None
    """
    x = x.assign(
        PAI=np.where(
            x["Conifers"],
            x["BAI"] + x["LAI"],
            np.where(
                x.Date_time.dt.day_of_year < leaf_on_start,
                x["BAI"],
                np.where(
                    (
                        (x.Date_time.dt.day_of_year >= leaf_on_start)
                        & (x.Date_time.dt.day_of_year < leaf_off_start)
                    ),
                    x["LAI"]
                    / (
                        1
                        + np.exp(
                            -0.37
                            * (
                                x.Date_time.dt.day_of_year
                                - (leaf_on_start + leaf_transition_days / 2)
                            )
                        )
                    )
                    + x["BAI"],
                    x["LAI"]
                    / (
                        1
                        + np.exp(
                            -0.37
                            * (
                                (leaf_off_start + leaf_transition_days / 2)
                                - x.Date_time.dt.day_of_year
                            )
                        )
                    )
                    + x["BAI"],
                ),
            ),
        )
    )
    return x["PAI"]


def lmbd(temperature):
    """The method calculates latent heat of vaporization.

    Args:
        Temperature in [C]

    Returns:
        latent heat of vaporization in [MJ/Kg]

    Note:
        The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    return 2.501 - 0.002361 * temperature


def e_s(temperature):
    """The method calculates saturated vapor pressure.

    Args:
        Temperature in [C]

    Returns:
        Saturated vapor pressure in [kPa]

    Note:
            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    return 0.6108 * np.exp(17.27 * temperature / (237.3 + temperature))


def e(dew_point_temperature):
    """The method calculates vapor pressure.

    Args:
        Dew point temperature in [C]

    Returns:
        Vapor pressure in [kPa]

    Note:
            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    return 0.6108 * np.exp(
        17.27 * dew_point_temperature / (237.3 + dew_point_temperature)
    )


def DELTA(temperature):
    """The method calculates the slope of vapor pressure temperature curve.

    Args:
        Temperature in [C]

    Returns:
        Slope of vapor pressure temperature curve in [kPa/C]

    Note:
        It includes in calculation saturated vapor pressure in [kPa] which is calculated with function e_s()!

        The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    return 4098 * e_s(temperature) / (237.3 + temperature) ** 2


def rho_a(temperature, surface_pressure):
    """The method calculates the density of air.

    Args:
        Temperature in [C]
        Surface preasure in [kPa]

    Returns:
        The density of air in [kg/m^3]

    Note:
            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    return 3.486 * surface_pressure / (275 + temperature)


def rho_w(temperature):
    """The method calculates the density of water.

    Args:
        Temperature in [C]

    Returns:
        The density of water in [kg/m^3]

    Note:
            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    return 999.88 + 0.018 * temperature - 0.0051 * temperature**2


def D(temperature, dew_point_temperature):
    """The method calculates vapor pressure deficit.

    Args:
        Temperature in [C]
        Dew point temperature in [C]

    Returns:
        Vapor pressure deficit in [kPa]

    Note:
            It includes in calculation vapor pressure as well as saturated vapor pressure, both in [kPa], which
            are calculated with functions e() e_s().

            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    return np.maximum(e_s(temperature) - e(dew_point_temperature), 0)


def U_t(wind_speed, wind_estimate_height):
    """The method estimates wind speed at the tree top.

    Args:
        Measured wind speed [m/s]
        Wind estimate height (tree height) [m]


    Returns:
        Wind speed at the tree top in [m/s]

    Note:
            The method is using two other parameters:
                Wind measurement height Z_u (usually 10m) and roughness height for water d_w which are set as constants.

            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    return wind_speed * np.log(wind_estimate_height / d_w) / np.log(Z_u / d_w)


def r_a(wind_speed, wind_estimate_height, roughness_height):
    """The method calculates aerodynamic resistance.

    Args:
        Measured wind speed [m/s]
        Wind estimate height (tree height) [m]
        Roughness_height [m]

    Returns:
        Aerodynamic resistance in [m/s]

    Note:
            It includes in calculation wind speed at the tree top U_t.
            If roughness height is negative, a new equation is applied to calculate aerodynamic resistance for
            the evapotranspiration from the soil.

            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    if roughness_height >= 0:
        return (
            4.72
            * np.log(wind_estimate_height / (Z_ov * roughness_height))
            / (1 + 0.53 * U_t(wind_speed, wind_estimate_height))
        )
    else:
        return 208 / U_t(wind_speed, wind_estimate_height)


def gamma(temperature, surface_pressure):
    """The method calculates psychrometric constant.

    Args:
        Temperature [C]
        Surface pressure [kPa]

    Returns:
        Psychrometric constant in [kPA/C]

    Note:
            It includes in calculation latent heat of vaporization lmbd().

            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    # return 10**(-3)*c_p*surface_pressure*10/(lmbd(temperature)*0.622)# pressure in mbar
    return (
        10 ** (-3) * c_p * surface_pressure / (lmbd(temperature) * 0.622)
    )  # pressure in kPa


def r_s(pai):
    """The method calculates stomatal resistance.

    Args:
        Plant area index (PAI)

    Returns:
        Stomatal resistance in [s/m]

    Note:
            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    return 200 / pai


def C_leaf(temperature):
    """The method calculates water vapor concentration at the evaporating surfaces within the leaf.

    Args:
        Temperature [C]

    Returns:
        Water vapor concentration at the evaporating surfaces within the leaf in [g/m^3]

    Note:
            Temerature in celsius [C] is converted to kelvins [K] such that T(K) = T(C)+273.15.
            The formula also includes saturated vapor pressure e_s() in [kPa].

            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    return 2165 * e_s(temperature) / (temperature + 273.15)


def C_air(dew_point_temperature, temperature):
    """The method calculates water vapor concentration in the air.

    Args:
        Temperature [C]
        Dew point temperature [C]

    Returns:
        Water vapor concentration in the air in [g/m^3]

    Note:
            Temerature in celsius [C] is converted to kelvins [K] such that T(K) = T(C)+273.15.
            The formula also includes vapor pressure e() in [kPa].

            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    return 2165 * e(dew_point_temperature) / (temperature + 273.15)


def potential_evaporation(
    temperature,
    dew_point,
    solar_radiation,
    sea_level_pressure,
    pai,
    wind_speed,
    wind_estimate_height,
    roughness_height,
):
    """The method calculate potential evaporation.

    Args: a data frame

    Returns: the updated ata frame

    Note:
        None

    Todo:
        None
    """
    potential_evaporation = (
        M_TO_MM
        * (1 / (lmbd(temperature) * rho_w(temperature)))
        * (
            DELTA(temperature) * solar_radiation
            + (rho_a(temperature, sea_level_pressure) * c_p * D(temperature, dew_point))
            / r_a(wind_speed, wind_estimate_height, roughness_height)
        )
        / (
            DELTA(temperature)
            + gamma(temperature, sea_level_pressure)
            * (1 + (r_s(pai) / r_a(wind_speed, wind_estimate_height, roughness_height)))
        )
    )

    return potential_evaporation


def potential_evapotranspiration(df):
    """The method calculate potential evapotranspiration.

    Args: a data frame

    Returns: the updated ata frame

    Note:
        None

    Todo:
        None
    """
    df = df.assign(
        Potential_evaporation_v=lambda x: potential_evaporation(
            x.Temperature,
            x.Dew_Point,
            x.Solar_Radiation,
            x.Sea_Level_Pressure,
            x.PAI,
            x.Wind_Speed,
            x.height,
            roughness_height_trees,
        ),
        Potential_evaporation_g=lambda x: potential_evaporation(
            x.Temperature,
            x.Dew_Point,
            x.Solar_Radiation,
            x.Sea_Level_Pressure,
            1,
            x.Wind_Speed,
            1,
            roughness_height_bare_soil,
        ),
        PET=lambda x: potential_evaporation(
            x.Temperature,
            x.Dew_Point,
            x.Solar_Radiation,
            x.Sea_Level_Pressure,
            x.PAI,
            x.Wind_Speed,
            x.height,
            -1,
        ),
        Transpiration=lambda x: 10 ** (-6)
        * (3600 / x.PAI)
        * np.maximum((C_leaf(x.Temperature) - C_air(x.Dew_Point, x.Temperature)), 0)
        / (r_s(x.PAI) + r_a(x.Wind_Speed, x.height, roughness_height_trees)),
    )
    df = df.assign(
        TF_average_ratio=lambda x: np.where(
            ((x.PET > x.Transpiration) & (x.PET > 0)), x.Transpiration / x.PET, np.nan
        )
    )

    df["TF_average_ratio"] = df.groupby(["AgentID", "Step"])[
        "TF_average_ratio"
    ].transform("mean")
    df = df.assign(
        Transpiration=lambda x: np.where(
            ((x.PAI < (x.LAI + x.BAI)) | (x.Transpiration > x.PET)),
            x.TF_average_ratio * x.PET,
            x.Transpiration,
        )
    )

    df = df.drop(columns="TF_average_ratio")
    return df


def ped1(Date_time, Precipitation, Potential_evaporation, Maximum_storage, evp):
    """The method computes precipitation-evaporation dynamics for one type at a time, which can be trees, pervious cover, impervious cover...

    Args:
        Date_time [time identifier]
        Precipitation [amount of water hitting the surface area of the type]
        Potential_evaporation [of the type for the given weather conditions]
        Maximum_storage [parameter: maximum storage of the type]
        evp [evaporation coefficient for the type]

    Returns:
        Precipitation storage of the type for each hour.

    Note:

            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    df = pd.DataFrame(
        {
            "Date_time": Date_time,
            "Precipitation": Precipitation,
            "Potential_evaporation": Potential_evaporation,
        }
    )
    df["Storage"] = 0
    df["Potential_evaporation_lag1"] = df.Potential_evaporation.shift(1).interpolate(
        limit_direction="backward"
    )
    x = 0

    def func2(row):
        # non local variable ==> will use pre_value from the ped function
        nonlocal x
        new_value = new_value = np.maximum(
            0,
            (
                np.minimum(Maximum_storage, x)
                + row["Precipitation"]
                - np.minimum(
                    np.minimum(Maximum_storage, x),
                    (
                        (np.minimum(Maximum_storage, x) / Maximum_storage) ** (evp)
                        * row["Potential_evaporation_lag1"]
                    ),
                )
            ),
        )
        x = new_value
        return new_value

    # This line might throw a SettingWithCopyWarning warning
    df.loc[0:, "Storage"] = df.loc[0:, :].apply(func2, axis=1)
    return df["Storage"]


def ped3(df, evp_v, evp_g):
    """The method computes precipitation-evaporation dynamics for each tree as well as the impervious and pervious covers.

    Args:
        A data frame including the following variables:
        Evaporation coefficient for trees: evp_v
        Evaporation coefficient for ground: evp_g

    Returns:
        A data frame with new variables: Vegetation_storage, Impervious_cover_storage_v, Pervious_cover_storage_v for each hour.

    Note:

            The function is built upon the following book:
            Maidment, David R and others. “Handbook of Hydrology, McGraw-Hil.” Inc., New York, NY, 1992.

            And a paper:
                Hirabayashi, Satoshi. “I-Tree Eco United States County-Based Hydrologic Estimates.”
                Washington, DC: US Department of Agriculture, Forest Service, 2015.

    Todo:
        None
    """
    df["Storage"] = 0
    df["Potential_evaporation_v_lag1"] = df.Potential_evaporation_v.shift(
        1
    ).interpolate(limit_direction="backward")
    df["Potential_evaporation_g_lag1"] = df.Potential_evaporation_g.shift(
        1
    ).interpolate(limit_direction="backward")
    x = 0
    y = 0
    z = 0

    def func2(row):
        # non local variable ==> will use pre_value from the ped function
        nonlocal x
        nonlocal y
        nonlocal z
        new_value_x = np.maximum(
            0,
            (
                np.minimum(row["Maximum_vegetation_storage"], x)
                + row["Canopy_interception"]
                - np.minimum(
                    np.minimum(row["Maximum_vegetation_storage"], x),
                    (
                        (
                            np.minimum(row["Maximum_vegetation_storage"], x)
                            / row["Maximum_vegetation_storage"]
                        )
                        ** (evp_v)
                        * row["Potential_evaporation_v_lag1"]
                    ),
                )
            ),
        )

        Precipitation_on_the_ground_with_vegetation = (
            np.maximum(0, (new_value_x - row["Maximum_vegetation_storage"]))
            + row["Through_canopy_precipitation"]
        )

        new_value_y = np.maximum(
            0,
            (
                np.minimum(MAXIMUM_IMPERVIOUS_COVER_STORAGE, y)
                + Precipitation_on_the_ground_with_vegetation
                - np.minimum(
                    np.minimum(MAXIMUM_IMPERVIOUS_COVER_STORAGE, y),
                    (
                        (
                            np.minimum(MAXIMUM_IMPERVIOUS_COVER_STORAGE, y)
                            / MAXIMUM_IMPERVIOUS_COVER_STORAGE
                        )
                        ** (evp_g)
                        * row["Potential_evaporation_g_lag1"]
                    ),
                )
            ),
        )

        new_value_z = np.maximum(
            0,
            (
                np.minimum(MAXIMUM_PERVIOUS_COVER_STORAGE, z)
                + Precipitation_on_the_ground_with_vegetation
                - np.minimum(
                    np.minimum(MAXIMUM_PERVIOUS_COVER_STORAGE, z),
                    (
                        (
                            np.minimum(MAXIMUM_PERVIOUS_COVER_STORAGE, z)
                            / MAXIMUM_PERVIOUS_COVER_STORAGE
                        )
                        ** (evp_g)
                        * row["Potential_evaporation_g_lag1"]
                    ),
                )
            ),
        )
        x = new_value_x
        y = new_value_y
        z = new_value_z
        return [new_value_x, new_value_y, new_value_z]

    # This line might throw a SettingWithCopyWarning warning
    df.loc[0:, "Storage"] = df.loc[0:, :].apply(func2, axis=1)
    df["Vegetation_storage"] = pd.DataFrame(df["Storage"].tolist())[0]
    df["Impervious_cover_storage_v"] = pd.DataFrame(df["Storage"].tolist())[1]
    df["Pervious_cover_storage_v"] = pd.DataFrame(df["Storage"].tolist())[2]
    df = df.drop(columns="Storage")
    return df


def ecosystem_services(i):
    """The method parallelize the computation of water retention benefits.

    Args: agent index "i"

    Returns: water retention benefits for each agent [data frame]

    Note:
        None

    Todo:
        None
    """
    output = df_scenario[tree_population.AgentID == AGENTS[i]]
    output = output[
        [
            "Step",
            "AgentID",
            "height",
            "BAI",
            "LA",
            "LAI",
            "PAI",
            "Conifers",
            "Under_canopy_area",
            "Total_under_canopy_area",
            "Scenario",
            "Precipitation_scale",
            "SAMPLE_AREA",
            "IMPERVIOUS_COVER_SHARE",
            "PERVIOUS_COVER_SHARE",
            "POPULATION_AREA",
            "POPULATION_SAMPLE_TREE_RATIO",
        ]
    ]
    output = pd.merge(output, weather_forcast, on=["Step"], how="left").sort_values(
        ["Step", "AgentID"]
    )
    # Calculate plant area index (PAI) in leaf-on and leaf-off seasons
    output["PAI"] = pai_seasons(
        output[["Date_time", "BAI", "LAI", "Conifers"]],
        Leaf_on_transition_day_start,
        Leaf_off_transition_day_start,
        LEAF_TRANSITION_DAYS,
    )

    # Calculate potential evapotranspiration over the vegetation and ground areas
    output = potential_evapotranspiration(output)

    output = output.assign(
        Canopy_cover_fraction=lambda x: 1 - np.exp(-KAPPA * x.PAI),
        Maximum_vegetation_storage=lambda x: SL * x.PAI,
        Through_canopy_precipitation=lambda x: x.Precipitation
        * (1 - x.Canopy_cover_fraction),
        Canopy_interception=lambda x: x.Precipitation - x.Through_canopy_precipitation,
    )

    output = ped3(output, evp_v=2 / 3, evp_g=1)
    output = output.assign(
        Canopy_drip=lambda x: np.maximum(
            0, (x.Vegetation_storage - x.Maximum_vegetation_storage)
        ),
        Evaporation_from_vegetation=lambda x: np.maximum(
            0, (x.Canopy_interception - x.Canopy_drip)
        ),
        Precipitation_on_the_ground_with_vegetation=lambda x: x.Canopy_drip
        + x.Through_canopy_precipitation,
        Vegetation_storage=lambda x: np.minimum(
            x.Vegetation_storage, x.Maximum_vegetation_storage
        ),
        Run_off_v=lambda x: np.maximum(
            0, (x.Impervious_cover_storage_v - MAXIMUM_IMPERVIOUS_COVER_STORAGE)
        ),
        Evaporation_from_impervious_cover_v=lambda x: np.maximum(
            0, (x.Precipitation_on_the_ground_with_vegetation - x.Run_off_v)
        ),
        Impervious_cover_storage_v=lambda x: np.minimum(
            x.Impervious_cover_storage_v, MAXIMUM_IMPERVIOUS_COVER_STORAGE
        ),
        Infiltration_v=lambda x: np.maximum(
            0, (x.Pervious_cover_storage_v - MAXIMUM_PERVIOUS_COVER_STORAGE)
        ),
        Evaporation_from_pervious_cover_v=lambda x: np.maximum(
            0, (x.Precipitation_on_the_ground_with_vegetation - x.Infiltration_v)
        ),
        Pervious_cover_storage_v=lambda x: np.minimum(
            x.Pervious_cover_storage_v, MAXIMUM_PERVIOUS_COVER_STORAGE
        ),
    )

    # Annual aggregation
    output = (
        output.groupby(["Scenario", "Precipitation_scale", "AgentID", "Step"])
        .agg(
            Leaf_area=pd.NamedAgg(column="LA", aggfunc=np.mean),
            Under_canopy_area=pd.NamedAgg(column="Under_canopy_area", aggfunc=np.mean),
            SAMPLE_AREA=pd.NamedAgg(column="SAMPLE_AREA", aggfunc=np.mean),
            IMPERVIOUS_COVER_SHARE=pd.NamedAgg(
                column="IMPERVIOUS_COVER_SHARE", aggfunc=np.mean
            ),
            PERVIOUS_COVER_SHARE=pd.NamedAgg(
                column="PERVIOUS_COVER_SHARE", aggfunc=np.mean
            ),
            POPULATION_AREA=pd.NamedAgg(column="POPULATION_AREA", aggfunc=np.mean),
            POPULATION_SAMPLE_TREE_RATIO=pd.NamedAgg(
                column="POPULATION_SAMPLE_TREE_RATIO", aggfunc=np.mean
            ),
            Annual_precipitation=pd.NamedAgg(column="Precipitation", aggfunc=np.sum),
            Annual_canopy_interception_loss=pd.NamedAgg(
                column="Evaporation_from_vegetation", aggfunc=np.sum
            ),
            Annual_run_off_v=pd.NamedAgg(column="Run_off_v", aggfunc=np.sum),
            Annual_evaporation_from_impervious_cover_v=pd.NamedAgg(
                column="Evaporation_from_impervious_cover_v", aggfunc=np.sum
            ),
            Annual_infiltration_v=pd.NamedAgg(column="Infiltration_v", aggfunc=np.sum),
            Annual_evaporation_from_pervious_cover_v=pd.NamedAgg(
                column="Evaporation_from_pervious_cover_v", aggfunc=np.sum
            ),
            Annual_transpiration=pd.NamedAgg(column="Transpiration", aggfunc=np.sum),
            Total_under_canopy_area=pd.NamedAgg(
                column="Total_under_canopy_area", aggfunc=np.mean
            ),
        )
        .reset_index()
    )
    output["Annual_stormwater_retention"] = (
        output.Annual_transpiration
        + output.Annual_canopy_interception_loss
        + output.Annual_evaporation_from_impervious_cover_v
        * output.IMPERVIOUS_COVER_SHARE
        + output.Annual_evaporation_from_pervious_cover_v * output.PERVIOUS_COVER_SHARE
    )

    return output
