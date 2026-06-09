import numpy as np
import pandas as pd
from uncertainties import ufloat

def broken_law_muller(dataset, uncertainty=False):
    """
    This function follows the broken power law described in Müller, 2024.
    
    M < 4.4 M_earth -> R ∝ M^{0.27}
    4.4 < M < 127 M_earth -> R ∝ M^{0.67}
    127 < M_earth -> R ∝ M^{-0.06}
    
    INPUTS:
    dataset: Pandas Dataframe from the load_dataset function in bem
    uncertainty: Boolean, if True, returns radius predictions with uncertainties. Default is False.
    
    OUTPUTS:
    radius_pred: List of predicted radii in Earth radii, either with or without uncertainties
    """
    mass = dataset['mass']
    radius_pred = []

    if not uncertainty:
        for m in mass:
            if m < 4.4:
                radius_pred.append(m**0.27 * 1.02)
            elif m < 127:
                radius_pred.append(m**0.67 * 0.56)
            else:
                radius_pred.append(m**-0.06 * 18.6)

    else:
        M_low_const  = ufloat(1.02,  0.03)
        M_low_pwr    = ufloat(0.27,  0.04)
        M_mdm_const  = ufloat(0.56,  0.03)
        M_mdm_pwr    = ufloat(0.67,  0.05)
        M_high_const = ufloat(18.6,  6.7)
        M_high_pwr   = ufloat(-0.06, 0.07)

        for m in mass:
            if m <= 4.4:
                radius_pred.append(m**M_low_pwr  * M_low_const)
            elif m < 127:
                radius_pred.append(m**M_mdm_pwr  * M_mdm_const)
            else:
                radius_pred.append(m**M_high_pwr * M_high_const)

    return radius_pred

def mousavi_sadr(dataset, uncertainty=False):
    """
    Function follows the broken power law described in Mousavi-Sadr, 2026.
    
    INPUTS:
    dataset: Pandas Dataframe from the load_dataset function in bem
    uncertainty: Boolean, if True, returns radius predictions with uncertainties. Default is False.
    
    OUTPUTS:
    radius_pred: List of predicted radii in Earth radii, either with or without uncertainties
    """
    dataset, radius_planet = dataset
    #Only use mass of planet and star.
    mass_planet = dataset['mass']
    mass_star = dataset['star_mass']
    radius_pred = []

    if not uncertainty:
        for m_planet, m_star, r_planet in zip(mass_planet, mass_star, radius_planet):
            if m_planet <= 52.48 and r_planet <= 8.13:
                radius_pred.append(10**(0.497 * np.log10(m_planet) - 0.050))
            else:
                radius_pred.append(10**(0.480 * np.log10(m_star) + 1.109))
    else:
        M_low_const = ufloat(0.497, 0.023)
        M_low_value = ufloat(0.05, 0.024)
        M_high_const = ufloat(0.480, 0.0365)
        M_high_value = ufloat(1.109, 0.004)
        
        for m_planet, m_star, r_planet in zip(mass_planet, mass_star, radius_planet):
            if m_planet <= 52.48 and r_planet <= 8.13:
                radius_pred.append(10**(M_low_const*np.log10(m_planet) - M_low_value))
            else:
                radius_pred.append(10**(M_high_const*np.log10(m_star) + M_high_value))
            
    return radius_pred