import pandas as pd
from astropy.units import earthMass, jupiterMass, earthRad, jupiterRad, solMass, solRad, AU
from .teq_planet import getTeqpl, getTeqpl_error
import numpy as np
from uncertainties import ufloat


def read_file_pandas(csv_file, radius=True):
    # Read the CSV file with Pandas
    # in a dataset
    # with specific parameters
    dataset = pd.read_csv(csv_file)
    if radius is True:
        dataset = dataset[['mass', 'mass_error_max', 'semi_major_axis',
                           'orbital_period', 'eccentricity',
                           'star_radius', 'star_teff', 'star_mass',
                           'radius', 'radius_error_max']]
    else:
        dataset = dataset[['mass', 'mass_error_max', 'semi_major_axis',
                           'orbital_period', 'eccentricity',
                           'star_radius', 'star_teff', 'star_mass']]
    return dataset


def get_semi_amplitude_k(ecc, m_star, m_p, a, inc):
    # Compute the velocity semi amplitude K
    # ecc    : eccentricity
    # m_star : star mass(solar mass)
    # m_p    : planet mass(jupiter mass)
    # a      : semi major axis
    # inc    : inclination
    # Return k in m.s-1
    # -------------------------------------------------------
    sqrt_g = 28.4329  # m.s-1
    m_p_solar = m_p * jupiterMass.to('solMass')
    # Compute the semi_amplitude K
    k = ((sqrt_g / np.sqrt(1 - ecc**2)) *
         (m_p * np.sin(inc)) *
         ((m_star + m_p_solar)**(-1/2)) * a**(-1/2))
    return abs(k)


def add_k_dataset(dataset):
    # Add the velocity semi amplitude to dataset
    k_planet = [get_semi_amplitude_k(ecc, m_star, m_p, a, inc)
                for ecc, m_star, m_p, a, inc
                in zip(dataset.eccentricity, dataset.star_mass,
                       dataset.mass, dataset.semi_major_axis,
                       dataset.inclination)]
    dataset.insert(2, 'k', k_planet)
    return dataset


def add_temp_eq_dataset(dataset):
    semi_major_axis = dataset.semi_major_axis * AU.to('solRad')
    teq_planet = [getTeqpl(teff, a/rad, ecc)
                  for teff, a, rad, ecc,
                  in zip(dataset.star_teff, semi_major_axis,
                         dataset.star_radius, dataset.eccentricity)]
    dataset.insert(2, 'temp_eq', teq_planet)
    return dataset


def add_temp_eq_error_dataset(dataset):
    semi_major_axis = dataset.semi_major_axis * AU.to('solRad')
    semi_major_axis_error = dataset.semi_major_axis_error * AU.to('solRad')
    teq_planet = [getTeqpl_error(ufloat(teff, abs(teff_e)),
                                 ufloat(a, abs(a_e))/ufloat(rad, abs(rad_e)),
                                 ufloat(ecc, abs(ecc_e)))
                  for teff, teff_e, a, a_e, rad, rad_e, ecc, ecc_e
                  in zip(dataset.star_teff, dataset.star_teff_error,
                         semi_major_axis, semi_major_axis_error,
                         dataset.star_radius, dataset.star_radius_error,
                         dataset.eccentricity, dataset.eccentricity_error)]
    teq_planet_value = [teq.nominal_value for teq in teq_planet]
    teq_planet_error = [teq.s for teq in teq_planet]
    dataset.insert(2, 'temp_eq_error', teq_planet_error)
    dataset.insert(2, 'temp_eq', teq_planet_value)
    return dataset


def add_star_luminosity_dataset(dataset):
    # Compute the stellar luminosity
    # L_star/L_sun = (R_star/R_sun)**2 * (Teff_star / Teff_sun)**4
    # Radius star is already expressed in Sun radii in the dataset
    # lum_sun    = 3.828 * 10**26   # Watt
    # radius_sun = 6.95508 * 10**8  # meters
    Teff_sun = 5777.0                 # Kelvin
    L_star = [R_star**2 * (Teff_star / Teff_sun)**4
              for R_star, Teff_star
              in zip(dataset.star_radius, dataset.star_teff)]
    dataset.insert(2, 'star_luminosity', L_star)
    return dataset


def add_star_luminosity_error_dataset(dataset):
    # Compute the stellar luminosity
    # L_star/L_sun = (R_star/R_sun)**2 * (Teff_star / Teff_sun)**4
    # Radius star is already expressed in Sun radii in the dataset
    # lum_sun    = 3.828 * 10**26   # Watt
    # radius_sun = 6.95508 * 10**8  # meters
    Teff_sun = 5778                 # Kelvin
    L_star = [ufloat(R_star, abs(R_star_error))**2 *
              (ufloat(Teff_star, abs(Teff_star_error)) / Teff_sun)**4
              for R_star, R_star_error, Teff_star, Teff_star_error
              in zip(dataset.star_radius, dataset.star_radius_error,
                     dataset.star_teff, dataset.star_teff_error)]
    L_star_value = [ls.nominal_value for ls in L_star]
    L_star_error = [ls.s for ls in L_star]
    dataset.insert(2, 'star_luminosity_error', L_star_error)
    dataset.insert(2, 'star_luminosity', L_star_value)
    return dataset


def add_insolation_dataset(dataset):
    # Compute the insolation flux
    # S / S_earth = (L_star / L_sun) * (AU / a)**2
    insolation_earth = 1.37 * 10**3   # Watts/m2
    # insolation = [insolation_earth * (l_star * (1 / a))
    #               for l_star, a
    #               in zip(dataset.star_luminosity, dataset.semi_major_axis)]
    # Insolation expressed in Solar insolation
    insolation = [(l_star * (1 / a))
                  for l_star, a
                  in zip(dataset.star_luminosity, dataset.semi_major_axis)]
    dataset.insert(2, 'insolation', insolation)
    return dataset


def jupiter_to_earth_mass(dataset, column_name):
    df = dataset[column_name].apply(lambda x:
                                    (x*jupiterMass).to('earthMass').value)
    new_df = pd.DataFrame({column_name: df})
    dataset.update(new_df)
    return dataset


def jupiter_to_earth_radius(dataset, column_name):
    df = dataset[column_name].apply(lambda x:
                                    (x*jupiterRad).to('earthRad').value)
    new_df = pd.DataFrame({column_name: df})
    dataset.update(new_df)
    return dataset
