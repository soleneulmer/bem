import pandas as pd
from astropy.units import earthMass, jupiterMass, earthRad, jupiterRad, solMass, solRad, AU
import numpy as np
import teq_planet
from uncertainties import ufloat
import importlib
importlib.reload(teq_planet)
from teq_planet import getTeqpl, getTeqpl_error

def read_files_pandas(csv_file, radius=True):
    """
    Read the csv file with Pandsa in a dataset with specific parameters.
    """
    
    dataset = pd.read_csv(csv_file)
    if radius is True:
        dataset = dataset[['mass', 'mass_error_max', 'semi_major_axis',
                           'orbital_period', 'eccentricity',
                           'star_radius', 'star_teff', 'star_mass',
                           'radius', 'radius_error_max']]
    else:
        dataset  = dataset[['mass', 'mass_error_max', 'semi_major_axis',
                           'orbital_period', 'eccentricity',
                           'star_radius', 'star_teff', 'star_mass']]
        
    return dataset

def add_temp_eq_dataset(dataset):
    """
    Function add the equilibrium temperature to the dataset.
    """
    semi_major_axis = dataset.semi_major_axis * AU.to('solRad')
    
    teq_planet = [getTeqpl(teff, a/rad, ecc) for teff, a, rad, ecc in
                  zip(dataset.star_teff, semi_major_axis, dataset.star_radius,
                      dataset.eccentricity)]
    
    dataset.insert(2, 'temp_eq', teq_planet)
    
    return dataset

def add_temp_eq_error_dataset(dataset):
    """
    Function adds the errorbar of the equi temp to the dataset.
    """
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
    
    teq_planet_value = [teq.n for teq in teq_planet]
    teq_planet_error = [teq.s for teq in teq_planet]
    
    dataset.insert(2, 'temp_eq_error', teq_planet_error)
    dataset.insert(2, 'temp_eq', teq_planet_value)
    
    return dataset

def add_star_luminosity_dataset(dataset):
    """
    Compute the stellar luminosity.
    L_star / L_sun = (R_star / R_sun)^2 * (Teff_star / Teff_sun)^4
    
    Radius star is already expressed in Sun radii in the dataset from catalogue
    lum_sun = 3.828e26 # Watt
    R_sun = 6.95508 * 10**8  # meters
    """
    Teff_sun = 5770 # Kelvin
    L_star = [R_star**2  *(Teff_star / Teff_sun)**4 for R_star, Teff_star
              in zip(dataset.star_radius, dataset.star_teff)]
    dataset.insert(2, 'star_luminosity', L_star)
    
    return dataset 

def add_star_luminosity_error_dataset(dataset):
    """
    Compute the stellar luminosity uncertainty.
    """
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

def jupiter_to_earth_mass(dataset, column_name):
    """
    Convert respective column jupiter mass to earth mass. 
    """
    df = dataset[column_name].apply(lambda x: (x* jupiterMass).to('earthMass').value)
    new_df = pd.DataFrame({column_name: df})
    dataset.update(new_df)
    return dataset

def jupiter_to_earth_radius(dataset, column_name):
    """
    Convert resp. column from jupiter to earth radius.
    """
    
    df = dataset[column_name].apply(lambda x: (x* jupiterRad).to('earthRad').value)
    new_df = pd.DataFrame({column_name: df})
    dataset.update(new_df)
    return dataset
 
    
    
    
    
    
    
