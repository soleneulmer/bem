import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import multivariate_normal as mvn
import pandas as pd
import datetime
import os
import joblib
import format_dataset as fd
import matplotlib.pyplot as plt
from matplotlib import cm 
import lime
import lime.lime_tabular
import importlib
importlib.reload(fd)

saved_pickle_model = "bem_output/rf_r2_0.87_2026-04-30_15.pkl"

def load_dataset(cat_exoplanet='data/exoplanet.eu_catalog_20-01-26_15_03_11.csv', 
                cat_solar="data/solar_system_planets_catalog.csv", 
                feature_names=['mass', 
                               'semi_major_axis',
                               'eccentricity',
                               'star_radius', 
                               'star_teff',
                               'star_mass',
                               'radius'],
                solar=True):
    """
    Select exoplanet in the catalogue which have mass and radius measurements
    as well as stellar parameters. This dataset will be used to train and 
    test the RF.

    Inputs: 
    cat_exoplanet: CSV file from exoplanet.eu
    cat_solar: CSV file from Planetary sheet
    feature_names: list of features to select in the dataset.
    
    Returns:
    dataset_exo = pandas dataframe with exoplanets with mass & radius 
                  measurements. mass/radii are in Earth mass/radii
                  
    """

    # Importing exoplanet dataset
    dataset_exo = pd.read_csv(cat_exoplanet, index_col = 0)
    # Importing Solar system dataset
    dataset_solar_system = pd.read_csv(cat_solar, index_col = 0)
    
    # Choosing features/data
    if not feature_names:
        print("\nNo features selected, loading default features")
    else: 
        dataset_exo = dataset_exo[feature_names]
        dataset_solar_system = dataset_solar_system[feature_names]

    # Remove planets with NaN's
    dataset_exo = dataset_exo.dropna(axis=0, how='any')

    # Converting from Jupiter to Earth mass/radius
    print("\nConvert planets mass/radius from Jupiter to Earth.")
    dataset_exo = fd.jupiter_to_earth_mass(dataset_exo, 'mass')
    dataset_exo = fd.jupiter_to_earth_radius(dataset_exo, 'radius')

    # Add the solar system planets with the exoplanets
    if solar:
        dataset = pd.concat([dataset_exo, dataset_solar_system])
    else:
        dataset = dataset_exo


    # Add observables
    dataset = fd.add_temp_eq_dataset(dataset)
    dataset = fd.add_star_luminosity_dataset(dataset)

    
    # Returning the dataset with selected features
    select_features = ['mass',
                       'semi_major_axis',
                       'temp_eq',
                       'star_luminosity',
                       'star_radius', 'star_teff',
                       'star_mass',
                       'radius']

    dataset = dataset[select_features]
    # Number of planets in dataset
    print('\nNumber of planets loaded: ', len(dataset))


    return dataset


def load_dataset_errors(cat_exoplanet='data/exoplanet.eu_catalog_20-01-26_15_03_11.csv',
                        cat_solar="data/solar_system_planets_catalog.csv",
                        remove_bad_planets=True,
                        solar=True):
    """
    Select exoplanet in the catalogue which have uncertainty measurements as 
    well as stellar parameters. If there is no uncertainty measurement, the 
    uncertainty is set to the 0.9 quantile of the distribution of uncertainties.
    
    If the uncertainty is higher then the value, the planet will be removed.
    
    This dataset will be used to compute error bars for the test set.
    
    Input:
    cat_exoplanet = CSV file from exoplanet.eu
    cat_solar = CSV file from planetary sheet.
    
    Returns:
    dataset_exo = pandas dataframe with exoplanets with mass & radius measurements
                  the mass/radius are in Earth mass/radius.
    """

    # Importing exoplanet dataset
    dataset_exo = pd.read_csv(cat_exoplanet, index_col=0)

    dataset_exo = dataset_exo[['mass', 
                               'mass_error_min', 'mass_error_max',
                               'radius', 
                               'radius_error_min', 'radius_error_max',
                               'semi_major_axis', 
                               'semi_major_axis_error_min', 'semi_major_axis_error_max',
                               'eccentricity', 
                               'eccentricity_error_min', 'eccentricity_error_max',
                               'star_mass', 
                               'star_mass_error_min', 'star_mass_error_max',
                               'star_radius', 
                               'star_radius_error_min','star_radius_error_max',
                               'star_teff',
                               'star_teff_error_min', 'star_teff_error_max'
                               ]]

    dataset_solar_system = pd.read_csv(cat_solar, index_col=0)
    

    dataset_solar_system = dataset_solar_system[['mass',            'mass_error', 
                                                 'semi_major_axis', 'semi_major_axis_error', 
                                                 'eccentricity',    'eccentricity_error',
                                                 'star_mass',       'star_mass_error',
                                                 'star_radius',     'star_radius_error',
                                                 'star_teff',       'star_teff_error', 
                                                 'radius',          'radius_error']]
    
    

    #Remove NaN's in features only
    dataset_exo = dataset_exo.dropna(subset=['mass',
                                            'semi_major_axis',
                                            'eccentricity',
                                            'star_radius',
                                            'star_mass',
                                            'star_teff',
                                            'radius'])

                            
    dataset_solar_system = dataset_solar_system.dropna(subset=['mass',
                                                               'semi_major_axis',
                                                               'eccentricity',
                                                               'star_radius',
                                                               'star_mass',
                                                               'star_teff',
                                                               'radius'])
    
    print("\nNumber of planets loaded including error columns, excluding solar system planets: ", len(dataset_exo))

    # Replace inf by NaN
    dataset_exo = dataset_exo.replace([np.inf, -np.inf], np.nan)

    # Replace NaN values in the error features by the 0.9 quantile value
    error_columns = ['mass_error_min', 'mass_error_max',
                     'radius_error_min', 'radius_error_max',
                     'semi_major_axis_error_min', 'semi_major_axis_error_max',
                     'eccentricity_error_min', 'eccentricity_error_max',
                     'star_mass_error_min', 'star_mass_error_max',
                     'star_radius_error_min', 'star_radius_error_max',
                     'star_teff_error_min', 'star_teff_error_max']

    for error_col in error_columns:
        # Find the 0.9 quantile value of the error column
        max_error = dataset_exo[error_col].quantile(0.9)
        

        # Replace NaN by the 0.9 error value
        dataset_exo[error_col] = dataset_exo[error_col].replace(np.nan, max_error)

    # After filling error NaNs, drop rows missing any core feature
    core_features = ['mass', 'semi_major_axis', 'eccentricity', 
                 'star_mass', 'star_radius', 'star_teff', 'radius']
    dataset_exo = dataset_exo.dropna(subset=core_features)

    # Convert from Jupiter to Earth
    print("Converting planets mass/radius to Earth masses/radii")
    dataset_exo = fd.jupiter_to_earth_mass(dataset_exo, 'mass')
    dataset_exo = fd.jupiter_to_earth_mass(dataset_exo, 'mass_error_max')
    dataset_exo = fd.jupiter_to_earth_mass(dataset_exo, 'mass_error_min')
    dataset_exo = fd.jupiter_to_earth_radius(dataset_exo, 'radius')
    dataset_exo = fd.jupiter_to_earth_radius(dataset_exo, 'radius_error_max')
    dataset_exo = fd.jupiter_to_earth_radius(dataset_exo, 'radius_error_min')

    # Computes the average error column
    dataset_exo['mass_error'] = dataset_exo[['mass_error_min', 'mass_error_max']].mean(axis=1).abs()
    dataset_exo['radius_error'] = dataset_exo[['radius_error_min', 'radius_error_max']].mean(axis=1).abs()
    dataset_exo['semi_major_axis_error'] = dataset_exo[['semi_major_axis_error_min', 'semi_major_axis_error_max']].mean(axis=1).abs()
    dataset_exo['eccentricity_error'] = dataset_exo[['eccentricity_error_min', 'eccentricity_error_max']].mean(axis=1).abs()
    dataset_exo['star_mass_error'] = dataset_exo[['star_mass_error_min', 'star_mass_error_max']].mean(axis=1).abs()
    dataset_exo['star_radius_error'] = dataset_exo[['star_radius_error_min', 'star_radius_error_max']].mean(axis=1).abs()
    dataset_exo['star_teff_error'] = dataset_exo[['star_teff_error_min', 'star_teff_error_max']].mean(axis=1).abs()

    dataset_exo = dataset_exo[['mass',              'mass_error',
                               'semi_major_axis',   'semi_major_axis_error',
                               'eccentricity',      'eccentricity_error',
                               'star_mass',         'star_mass_error',
                               'star_radius',       'star_radius_error',
                               'star_teff',         'star_teff_error',
                               'radius',            'radius_error']]
    
    if remove_bad_planets:
        bad_mask = (
            (dataset_exo['mass_error'] >= dataset_exo['mass']) |
            (dataset_exo['radius_error'] >= dataset_exo['radius']) |
            (dataset_exo['semi_major_axis_error'] >= dataset_exo['semi_major_axis']) |
            (dataset_exo['star_mass_error'] >= dataset_exo['star_mass']) |
            (dataset_exo['star_radius_error'] >= dataset_exo['star_radius']) |
            (dataset_exo['star_teff_error'] >= dataset_exo['star_teff'])
        )

        removed_planets = dataset_exo.index[bad_mask]

        print("\nRemoving", len(removed_planets), "planets with uncertainty >= value")

        dataset_exo = dataset_exo.loc[~bad_mask]


    # Add the solar system planets with the exoplanets
    if solar:
        dataset = pd.concat([dataset_exo, dataset_solar_system])
    else:
        dataset = dataset_exo

    # Add observables
    dataset = fd.add_temp_eq_error_dataset(dataset)
    dataset = fd.add_star_luminosity_error_dataset(dataset)


    # Select the same features as the original dataset
    dataset = dataset[['mass',              'mass_error',
                       'star_luminosity',   'star_luminosity_error',
                       'temp_eq',           'temp_eq_error',
                       'semi_major_axis',   'semi_major_axis_error',
                       'star_mass',         'star_mass_error',
                       'star_radius',       'star_radius_error',
                       'star_teff',         'star_teff_error',
                       'radius',            'radius_error']]

    print("\nNumber of planets loaded including error columns: ", len(dataset))

    return dataset    
   
def load_dataset_RV(cat_exoplanet="data/exoplanet.eu_catalog_20-01-26_15_03_11.csv", 
                    feature_names=['mass', 'mass_error_min', 'mass_error_max',
                                   'semi_major_axis',
                                   'eccentricity',
                                   'star_metallicity',
                                   'star_radius',
                                   'star_teff', 'star_mass']):
    """
    Select exoplanets in the catalog which are detected using RV and do not 
    have mass measurement. This dataset will be used to llater predict their
    masses.
    
    INPUTS:
    cat_exoplanet = CSV file from exoplanet.eu.
    features = list of features to select in the datsaset.
    
    OUTPUTS:
    dataset_radial = pd struct with exoplanets detected using RV without radius
                     measurements. Mass in in earth mass.
    """

    print("\nLoading exoplanet dataset found with RVs:")
    dataset = pd.read_csv(cat_exoplanet, index_col=0)

    # Select detected by RV
    dataset_radial = dataset[dataset.detection_type == "Radial Velocity"]

    # the radius column in Null = NaN
    dataset_radial = dataset_radial[pd.isnull(dataset_radial['radius'])]

    # Choosing features/data
    if not feature_names:
        print("No features selected, loading all features")
        pass
    else:
        print("Selected features:")
        print(feature_names)

        dataset_radial = dataset_radial[feature_names]

        # Excluding exoplanets with missing data
        dataset_radial = dataset_radial.dropna(subset=['mass', 'semi_major_axis',
                                                       'eccentricity',
                                                       'star_metallicity',
                                                       'star_radius', 'star_teff',
                                                       'star_mass'])

    # Replace inf by NaN
    dataset_radial = dataset_radial.replace([np.inf, -np.inf], np.nan)

    # Replace NaN values in the error features by the 0.9 quantile value
    error_columns = ['mass_error_min', 'mass_error_max']

    for error_col in error_columns:
        # Find the 0.9 quantile value of the error columns
        # max_error = dataset_radial[error_col].replace(np.nan, max_error)
        max_error = 0.0
        print(error_col, max_error)
        # replace NaN by the 0.9 error value
        dataset_radial[error_col] = dataset_radial[error_col].replace(np.nan,
                                                                      max_error)

    # Converting from Jupiter to Earth masses/radii
    print("Converting planet's mass/radius in Earth masses/radii")
    dataset_radial = fd.jupiter_to_earth_mass(dataset_radial, 'mass')
    dataset_radial = fd.jupiter_to_earth_mass(dataset_radial, 'mass_error_max')
    dataset_radial = fd.jupiter_to_earth_mass(dataset_radial, 'mass_error_min')

    # Computes the average error column
    dataset_radial['mass_error'] = dataset_radial[['mass_error_min',
                                                   'mass_error_max']].mean(axis=1).abs()

    # Adding observables
    print('Computing planet\'s equilibrium temperature')
    dataset_radial = fd.add_temp_eq_dataset(dataset_radial)
    print('Computing stellar luminosity')
    dataset_radial = fd.add_star_luminosity_dataset(dataset_radial)

    print('\nNumber of planets: ', len(dataset_radial))

    # Remove the mass error column for Random forest
    dataset_radial = dataset_radial[['mass',
                                     'semi_major_axis',
                                     'temp_eq',
                                     'star_luminosity',
                                     'star_radius', 'star_teff',
                                     'star_mass']]

    return dataset_radial


def split_data(dataset):
    dataset = dataset.copy()

    features_needed = [
        'mass',
        'semi_major_axis',
        'temp_eq',
        'star_luminosity',
        'star_radius',
        'star_teff',
        'star_mass'
    ]

    features = dataset[features_needed]
    label = dataset['radius']

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        label,
        test_size=0.25,
        random_state=9
    )
    forced_for_train = [
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    'Kepler-12 b', 'WASP-94 Ab', 'HAT-P-47 b',
    'kappa And b', 'HD 984 b', 'Kepler-16 (AB)b', 'Kepler-34 (AB)b', 'KELT-9 Ab'
    ]
    forced_for_test = [
    #     # uncomment for the LIME plots
    #         "TOI-561 b",
    #         "HATS-35 b",
    #         "CoRoT-13 b",
    #         "Kepler-75 b",
    #         "WASP-17 b",
    #         "Kepler-20 Ac",
    'HAT-P-32 Ab', 'TOI-3071 b', 'WASP-122 Ab', 'TOI-3976 Ab', 'Kepler-4 b',
    'LHS 1140 b', 'Gliese 12 b', 'TOI-1231 b',
    'TOI-201 c', 'TOI-561 f', 'HIP 41378 f',
    ]
    for name in forced_for_train:
        if name in X_test.index and name not in X_train.index:
            X_train = pd.concat([X_train, X_test.loc[[name]]])
            y_train = pd.concat([y_train, y_test.loc[[name]]])
            X_test = X_test.drop(index=name)
            y_test = y_test.drop(index=name)
            swap_candidates = [
                idx for idx in X_train.index
                if idx not in forced_for_train and idx != name
            ]
            if len(swap_candidates) > 0:
                swap_name = swap_candidates[0]
                X_test = pd.concat([X_test, X_train.loc[[swap_name]]])
                y_test = pd.concat([y_test, y_train.loc[[swap_name]]])
                X_train = X_train.drop(index=swap_name)
                y_train = y_train.drop(index=swap_name)

    for name in forced_for_test:
        if name in X_train.index and name not in X_test.index:
            X_test = pd.concat([X_test, X_train.loc[[name]]])
            y_test = pd.concat([y_test, y_train.loc[[name]]])
            X_train = X_train.drop(index=name)
            y_train = y_train.drop(index=name)
            swap_candidates = [
                idx for idx in X_test.index
                if idx not in forced_for_test and idx != name
            ]
            if len(swap_candidates) > 0:
                swap_name = swap_candidates[0]
                X_train = pd.concat([X_train, X_test.loc[[swap_name]]])
                y_train = pd.concat([y_train, y_test.loc[[swap_name]]])
                X_test = X_test.drop(index=swap_name)
                y_test = y_test.drop(index=swap_name)

    
    train_test_values = [
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values
    ]
    train_test_sets = [X_train, X_test, y_train, y_test]
    return features_needed, X_train, X_test, y_train, y_test, train_test_values, train_test_sets


def random_forest_regression(dataset,
                             model=saved_pickle_model, fit=False):
    """
    Random Forest regressor

    Returns:
        regr, y_test_predict, train_test_values, train_test_sets
    """
    # Load the dataset and split into train/test sets
    features_needed, X_train, X_test, y_train, y_test, train_test_values, train_test_sets = split_data(dataset)

    print('Dataset loaded and split into train/test sets. Starting Random Forest regression...')

    if fit:
        params_grid_rf = [
            {
                "bootstrap": [True],
                "n_estimators": [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500],
                "max_depth": [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None],
                "max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "sqrt", "log2"],
                "min_samples_split": [2, 3, 4, 6, 8, 10, 15, 20, 30],
                "min_samples_leaf": [1, 2, 3, 4, 5, 7, 10, 15],
                "max_samples": [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, None],
                "min_impurity_decrease": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                "max_leaf_nodes": [None, 20, 30, 50, 75, 100, 150, 200],
                "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            },
            {
                "bootstrap": [False],
                "n_estimators": [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500],
                "max_depth": [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None],
                "max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "sqrt", "log2"],
                "min_samples_split": [2, 3, 4, 6, 8, 10, 15, 20, 30],
                "min_samples_leaf": [1, 2, 3, 4, 5, 7, 10, 15],
                "max_samples": [None],
                "min_impurity_decrease": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                "max_leaf_nodes": [None, 20, 30, 50, 75, 100, 150, 200],
                "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            },
        ]

        rf_cv = RandomizedSearchCV(
            RandomForestRegressor(random_state=9),
            param_distributions=params_grid_rf,
            n_iter=40,
            cv=5,
            scoring='r2',
            verbose=1,
            n_jobs=-1,
            random_state=9,
            return_train_score=True
        )

        rf_cv.fit(X_train, y_train)
        print("\nBest paramaters Random Forest:")
        for param, value in rf_cv.best_params_.items():
            print(f"{param}: {value}")

        regr = rf_cv.best_estimator_

        outdir = 'bem_output'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        name_rf = 'rf_r2_' + str(round(rf_cv.best_score_, 2)) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H")) + '.pkl'
        name_rf = os.path.join(outdir, name_rf)

        print('\nRandom Forest model will be saved in:', name_rf)

        print("\nTraining Random Forest regressor...")
        regr.fit(X_train, y_train)
        joblib.dump(regr, name_rf)

    else:
        print("\nLoading Random Forest model:", model)
        regr = joblib.load(model)

    # Predict the radius for the train and test sets
    y_train_predict = regr.predict(X_train)
    y_test_predict = regr.predict(X_test)

    #feature importance 
    print('\nFeature importance')
    importances = regr.feature_importances_
    importances_percent = 100 * importances / importances.sum()

    for name, value in zip(features_needed, importances_percent):
        print(f'{name}: {value:.2f}%')

    # Evaluate
    print("\nEvaluation of Random Forest regressor:")
    print(f"Train R²:   {r2_score(y_train, y_train_predict):.4f}")
    print(f"Test  R²:   {r2_score(y_test,  y_test_predict):.4f}")
    print(f"Train RMSE: {root_mean_squared_error(y_train, y_train_predict):.4f}")
    print(f"Test  RMSE: {root_mean_squared_error(y_test,  y_test_predict):.4f}")
    print(f"Mean ratio: {np.mean(y_test_predict / y_test):.4f}")

    print('\nFeature importance')
    importances = regr.feature_importances_
    importances_percent = 100 * importances / importances.sum()

    for name, value in zip(features_needed, importances_percent):
        print(f'{name}: {value:.2f}%')  

    metrics = {
    "r2_test": r2_score(y_test, y_test_predict),
    "r2_train": r2_score(y_train, y_train_predict),
    "rmse_train": root_mean_squared_error(y_train, y_train_predict),
    "rmse_test": root_mean_squared_error(y_test, y_test_predict),
    "mean_ratio": np.mean(y_test_predict / y_test)}

    

    return regr, y_test_predict, train_test_values, train_test_sets, metrics


def computing_errorbars(regr, dataset_errors, train_test_set):
    """
    INPUTS:
    regr = random forest regression model.
    dataset_errors = pandas df with each feature and their uncertainty.
    train_test_sets = pd df with exoplanets and their feature name incl the values.
    
    OUTPUTS:
    radii_test_output_error = error on the predicted radius for the test set.
    radii_test_input_error = original uncertainty on the radius measurements.
    """

    # Original train and test sets
    X_train, X_test, y_train, y_test = train_test_set

    # Cross matching the test set with the dataset with errors to compute error
    # bars for the exoplanet which have input errors.
    dataset_errors = dataset_errors.loc[X_test.index.values.tolist()]

    # Remove an exoplanet in case there is still a NaN in one of the features
    dataset_errors = dataset_errors.dropna(axis=0, how='any')

    # Matrix with all the erros on the different features
    features_errors = dataset_errors.iloc[:, :-2].values

    # Radius vector
    radii_test = dataset_errors.iloc[:, -2].values

    #Error on the radius vector
    radii_test_input_error = dataset_errors.iloc[:, -1].values

    # Empty vector to store the error bars
    radii_test_output_error = np.zeros_like(radii_test_input_error)
    for i in range(radii_test.size):
        rerr = regr.predict(mvn(features_errors[i, ::2], np.diag(features_errors[i, 1::2]), allow_singular=True).rvs(1000)).std()

        radii_test_output_error[i] = rerr

    # Save the errorbars in a txt file
    outdir = 'bem_output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    filename = 'bem_output/test_radius_RF_errorbars.dat'
    print("Error bars of the test set are savid in: ", filename)
    np.savetxt(filename, radii_test_output_error)

    return radii_test_output_error, radii_test_input_error

def predict_radius(my_planet=np.array([[1, 1, 0, 1, 5777, 1]]),
                   my_name=np.array(['My planet b']),
                   regr=None,
                   jupiter_mass=False,
                   error_bar=False):
    """
    Predict the radius of a planet given the planetary mass, semi major axis,
    eccentricity, stellar radius, star effective temperature and stellar mass.
    
    Inputs: my_planet = array with shape (1, 6)
                        np.array([[planetary mass, 
                                   semi major axis,
                                   eccentricity,
                                   star_radius,
                                   star_teff,
                                   star_mass]])
            my_name = array with shape (1, )
                      np.array(['my planet b'])
            regr = random forest regression model
            jupiter_mass = bool, True if the planet's mass is given in Jupiter 
                           mass.
            error_bar = bool, True if an error is provided for each parameter
                        such as:
                        my_planet = np.array([[planetary mass, planetary mass error,
                                   semi major axis, semi major axis error,
                                   eccentricity, eccentricity error,
                                   star_radius, star_radius error,
                                   star_teff, star_teff error,
                                   star_mass, star_mass error]]) 
                                   
    OUTPUTS: radius = planet's radius predicting with the RF model
             my_pred_planet = pandas dataframe with the input features used by 
                              the random forest model.
                              Can be used as input in plot_LIME_predictions()
                              The features are now:
                              'mass', 'semi_major_axis',
                              'temp_eq', 'star_luminosity',
                              'star_radius', 'star_teff',
                              'star_mass'    
    """

    if regr is None:
        # Loading the random forest model saved
        print("Loading random forest model: ", regr)
        regr = joblib.load(regr)
    else:
        pass

    if error_bar:
        print("\nPredicting radius for planet:\n")
        my_planet = pd.DataFrame(data=my_planet,
                                 index=my_name,
                                 columns=np.array(['mass','mass_error',
                                                   'semi_major_axis', 'semi_major_axis_error',
                                                   'eccentricity', 'eccentricity_error', 
                                                   'star_radius', 'star_radius_error',
                                                   'star_teff', 'star_teff_error',
                                                   'star_mass', 'star_mass_error']))
        if jupiter_mass:   
            # Changing mass units to Earth mass
            my_planet = fd.jupiter_to_earth_mass(my_planet, 'mass')
            my_planet = fd.jupiter_to_earth_radius(my_planet, 'mass_error')

        else:
            print("Planetary mass is given in Earth")

        # Computing equi temperature
        my_planet = fd.add_temp_eq_dataset(my_planet)
        # Computing stellar luminosity
        my_planet = fd.add_star_luminosity_dataset(my_planet)

        # Planet with error bars
        print("Planet with error bars\n", my_planet.iloc[0])

        # Radius error prediction
        my_pred_planet = my_planet[['mass', 'mass_error', 'semi_major_axis',
                                    'semi_major_axis_error', 'temp_eq', 
                                    'temp_eq_error', 'star_luminosity', 'star_luminosity_error',
                                        'star_radius', 'star_radius_error',
                                        'star_teff', 'star_teff_error',
                                        'star_mass', 'star_mass_error']]

        # Feature / feature error
        features_with_errors = my_pred_planet.iloc[0].values.reshape(1, -1)
        radius_error = regr.predict(mvn(features_with_errors[0, ::2],
                                        np.diag(features_with_errors[0, 1::2]),
                                        allow_singular=True).rvs(1000)).std()

        # Radius prediction
        my_pred_planet = my_planet[['mass', 'semi_major_axis', 'temp_eq', 
                                    'star_luminosity', 'star_radius', 'star_teff',
                                    'star_mass']]

        radius = regr.predict(my_pred_planet.iloc[0].values.reshape(1, -1))

        # Print pred radius
        print("Predicted radius (R_earth): ", radius, '±', radius_error)   
        return [radius, radius_error], my_pred_planet

    else:
        print('\nPredicting radius for planet:\n')
        my_planet = pd.DataFrame(data=my_planet,
                                 index=my_name,
                                 columns=np.array(['mass', 'semi_major_axis',
                                                   'eccentricity',
                                                   'star_radius',
                                                   'star_teff', 'star_mass']))
        # Changing mass units to Earth mass
        if jupiter_mass:
            my_planet = fd.jupiter_to_earth_mass(my_planet, 'mass')
        else:
            print('Planetary mass is given in Earth mass')

        # Computing equilibrium temperature
        my_planet = fd.add_temp_eq_dataset(my_planet)
        # Computing stellar luminosity
        my_planet = fd.add_star_luminosity_dataset(my_planet)
        # Select features
        my_pred_planet = my_planet[['mass', 'semi_major_axis',
                                    'temp_eq', 'star_luminosity',
                                    'star_radius', 'star_teff',
                                    'star_mass']]
        # Radius prediction
        print(my_pred_planet.iloc[0])
        radius = regr.predict(my_pred_planet.iloc[0].values.reshape(1, -1))
        print('Predicted radius (Rearth): ', radius)

        return radius, my_pred_planet

def plot_dataset(dataset, predicted_radii=[], rv=False):
    """
    Function plots the dataset.
    """
    if not rv:
        #Plot the original dataset
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')

        size = dataset.temp_eq
        plt.scatter(dataset.mass, dataset.radius, c=size, cmap=cm.magma_r,
                    s=4, label='Verification sample')
        plt.colorbar(label=r"Equilibrium temperature (K)")
        plt.xlabel(r"Mass ($M_\oplus$)")
        plt.ylabel(r"Radius ($R_\oplus$)")
        plt.legend(loc='lower right', markerscale=0, handletextpad=0, handlelength=0)
        plt.savefig("figures/dataset_radius.pdf", bbox_inches="tight")
        plt.show()

    if rv:
        # Plot the radial velocity dataset
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')

        size = dataset.temp_eq
        plt.scatter(dataset.mass, predicted_radii, c=size,
                    cmap=cm.magma_r, s=4, label='RV sample')
        plt.colorbar(label=r'Equilibrium temperature (K)')
        plt.xlabel(r'Mass ($M_\oplus$)')
        plt.ylabel(r'Radius ($R_\oplus$)')
        plt.legend(loc='lower right', markerscale=0,
                   handletextpad=0.0, handlelength=0)




    return None

def plot_true_predicted(train_test_sets, radii_test_RF,
                        radii_test_output_error):
    """
    Plot the residuals on the test set between true radius and RF.
    """
    X_train, X_test, y_train, y_test = train_test_sets

    plt.figure()
    plt.errorbar(radii_test_RF, y_test.values, xerr=radii_test_output_error,
                 fmt='.', c='C1', elinewidth=0.5, label="Random forest")

    # 1:1 line and labels
    plt.plot(np.sort(y_test.values), np.sort(y_test.values), 'k-', lw=0.25)
    plt.ylabel(r'True radius ($R_\oplus$)')
    plt.ylabel(r'Predicted radius ($R_\oplus$)')
    plt.legend(loc='lower right')
    plt.show()
    return None

def plot_learning_curve(regr, dataset, save=False, fit=False):
    """
    Function plots the learning curve of the random forest regression model.
    Cross validation with 100 iterations to get smoother mean test and train
    test score curves, each time with 20% data randomly selected as a 
    validation set.
    
    INPUTS:
    regr = random forest regression model.
    dataset = pandas dataframe with features and labels.
    save = bool, writes (True) or not (False) the scores.
    fit = bool, computes the score if True
    
    OUTPUTS:
    written files
    """

    features = dataset.drop(['radius'], axis=1).values
    label = dataset.radius.values #radius

    outdir = 'bem_output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if fit:
        cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=9)
        train_sizes, train_scores, test_scores= learning_curve(regr, 
                                                                X = features,
                                                                y = label,
                                                                cv=cv,
                                                                train_sizes=np.linspace(0.1, 1, 10),
                                                                n_jobs=-1,
                                                                verbose=1)

    else:
        train_sizes = np.loadtxt("bem_output/lc_train_sizes.dat")
        train_scores = np.loadtxt("bem_output/lc_train_scores.dat")
        test_scores = np.loadtxt("bem_output/lc_test_scores.dat")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color = 'hotpink')

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', 
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', 
             label="Cross-validation score")
    plt.legend(loc='lower right')
    plt.show()

    if save:
        np.savetxt(os.path.join(outdir, "lc_train_sizes.dat"), train_sizes)
        np.savetxt(os.path.join(outdir, 'lc_train_scores.dat'), train_scores)
        np.savetxt(os.path.join(outdir, 'lc_test_scores.dat'), test_scores)
    return plt 

def plot_validation_curves(regr, dataset, name='features', save=False,
                           fit=False):
    """
    INPUTS: 
    regr = random forest regression model
    dataset = pandas dataframe with features and labels
    name = str, can be 'features', 'tree', 'depth'
    save = bool, writes (True) or not (False) the scores
    fit = bool, computes the score if True 
    
    OUTPUTS:
    Written files
    """
    features = dataset.drop(["radius"], axis=1).values
    label = dataset.radius.values

    outdir = 'bem_output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if name == 'features':
        param_range = np.arange(features.shape[1]) + 1
        # param_range = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  
        param_name = 'max_features'
    elif name == 'tree':
        param_range = np.array([10, 20, 35, 50, 100, 1000, 5000, 10000])
        param_name = 'n_estimators'
    elif name == 'depth':
        param_range = np.array([1, 2, 3, 4, 5, 6, 7,
                                8, 9, 10, 50, 100, 1000])  
        param_name = 'max_depth'
    elif name == 'min_samples_split':
        param_range = np.array([2, 3, 4, 6, 8, 10, 15, 20, 30])
        param_name = 'min_samples_split'
    elif name == 'min_samples_leaf':
        param_range = np.array([1, 2, 3, 4, 5, 7, 10, 15])
        param_name = 'min_samples_leaf'
    elif name == 'max_samples':
        param_range = np.array([0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                                0.8, 0.85, 0.9, 0.95])  
        param_name = 'max_samples'
    elif name == 'min_impurity_decrease':
        param_range = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
        param_name = 'min_impurity_decrease'
    elif name == 'max_leaf_nodes':
        param_range = np.array([20, 30, 50, 75, 100, 150, 200])  
        param_name = 'max_leaf_nodes'
    elif name == 'ccp_alpha':
        param_range = np.array([0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
        param_name = 'ccp_alpha'
    elif name == "learning_rate":
        param_range = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1])
        param_name = "learning_rate"

    else:
        print("Error the parameter of the validation curve is incorrect")
        # print("Names can be features, tree, depth")
        return None

    if fit:
        train_scores, test_scores = validation_curve(regr, features, label,
                                                     param_name=param_name,
                                                     param_range=param_range,
                                                     cv=3, scoring='r2', 
                                                     n_jobs=-1, verbose=1)
    else:
        if name == 'features':
            train_scores = np.loadtxt("bem_output/vc_features_train_scores.dat")
            test_scores = np.loadtxt("bem_output/vc_features_test_scores.dat")
        elif name == 'tree':
            train_scores = np.loadtxt("bem_output/vc_tree_train_scores.dat")
            test_scores = np.loadtxt("bem_output/vc_tree_test_scores.dat")
        elif name == 'depth':
            train_scores = np.loadtxt("bem_output/vc_depth_train_scores.dat")
            test_scores = np.loadtxt("bem_output/vc_depth_test_scores.dat")
        else:
            pass

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Validation Curve with Random Forest regressor")
    plt.xlabel(param_name)
    plt.ylabel("Score")

    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

    if save:
        if name == 'features':
            np.savetxt(os.path.join(outdir, "vc_features_train_scores.dat"),
                       train_scores)
            np.savetxt(os.path.join(outdir, "vc_features_test_scores.dat"),
                       test_scores)
        elif name == 'tree':
            np.savetxt(os.path.join(outdir, "vc_tree_train_scores.dat"),
                       train_scores)
            np.savetxt(os.path.join(outdir, "vc_tree_test_scores.dat"),
                       test_scores)
        elif name == 'depth':
            np.savetxt(os.path.join(outdir, "vc_depth_train_scores.dat"),
                       train_scores)
            np.savetxt(os.path.join(outdir, "vc_depth_test_scores.dat"),
                       test_scores)
        else:
            pass
    return None


def plot_valll_curves(regr, dataset, name='features', model='rf', save=False, 
                      fit=False):
    features = dataset.drop(["radius"], axis=1).values
    label = dataset.radius.values
    
    outdir = "bem_output"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    param_config = {
        'rf': {
            'features': ('max_features', np.arange(features.shape[1]) + 1, True),
            'tree': ('n_estimators', np.array([10, 20, 35, 50, 100, 1000, 5000, 10000]), True),
            'depth': ('max_depth', np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 1000]), True),
            'min_samples_split': ('min_samples_split', np.array([2, 3, 4, 6, 8, 10, 15, 20, 30]), False),
            'min_samples_leaf': ('min_samples_leaf', np.array([1, 2, 3, 4, 5, 7, 10, 15]), False),
            'max_samples': ('max_samples', np.array([0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]), False),
            'min_impurity_decrease':('min_impurity_decrease', np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]), False),
            'max_leaf_nodes': ('max_leaf_nodes', np.array([20, 30, 50, 75, 100, 150, 200]), False),
            'ccp_alpha': ('ccp_alpha', np.array([0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]), False),
        },
        'lgbm': {
            'n_estimators': ('n_estimators', np.array([100, 150, 200, 300, 500]), True),
            'learning_rate': ('learning_rate', np.array([0.01, 0.02, 0.03, 0.05, 0.1]), False),
            'depth': ('max_depth', np.array([2, 3, 4, 5]), False),
            'num_leaves': ('num_leaves', np.array([3, 5, 7, 10, 15]), False),
            'min_child_samples': ('min_child_samples', np.array([5, 10, 20, 40]), False),
            'subsample': ('subsample', np.array([0.1, 0.2, 0.7, 0.8, 0.9, 1.0, 2.0]), False),
            'colsample_bytree': ('colsample_bytree', np.array([0.7, 0.8, 0.9, 1.0]), False),
            'reg_lambda': ('reg_lambda', np.array([0, 0.01, 0.1, 1.0]), True),
        },
        'xgb': {
            'n_estimators': ('n_estimators', np.array([50, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500]), True),
            'learning_rate': ('learning_rate', np.array([0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]), False),
            'depth': ('max_depth', np.array([2, 3, 4, 5, 6]), False),
            'subsample': ('subsample', np.array([0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]), False),
            'colsample_bytree': ('colsample_bytree', np.array([0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]), False),
            'reg_lambda': ('reg_lambda', np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]), True),
            'reg_alpha': ('reg_alpha', np.array([0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]), True),
            'min_child_weight': ('min_child_weight', np.array([3, 5, 7, 10, 15, 20, 30, 50]), False),
            'gamma': ('gamma', np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]), True),
            'colsample_bylevel': ('colsample_bylevel', np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), False),
        },   
    }

    if model not in param_config:
        print(f"Error, {model} is not a valid model")
        return None
    
    model_cofig = param_config[model]
    
    if name not in model_cofig:
        print(f"Error, {model} not a valid param for this model")
        return None
    
    param_name, param_range, use_log_scale = model_cofig[name]
    
    scores_path_train = os.path.join(outdir, f"vc_{model}_{name}_train_scores.dat")
    scores_path_test = os.path.join(outdir, f"vc_{model}_{name}_test_scores.dat")
    
    if fit:
        train_scores, test_scores = validation_curve(regr, features, label,
                                                     param_name=param_name,
                                                     param_range=param_range,
                                                     cv=3, scoring='r2', 
                                                     n_jobs=-1, verbose=1)
        
    else:
        train_scores = np.loadtxt(scores_path_train)
        test_scores = np.loadtxt(scores_path_test)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plot_fn = plt.semilogx if use_log_scale else plt.plot

    plt.figure()
    # plt.title(f"Validation Curve — {model.upper()} — {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("R² Score")

    lw = 2
    plot_fn(param_range, train_mean, label="Training score", color="darkorange", lw=lw)
    plot_fn(param_range, test_mean,  label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color="darkorange")
    plt.fill_between(param_range, test_mean  - test_std,  test_mean  + test_std,
                     alpha=0.2, color="navy")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    if save:
        np.savetxt(scores_path_train, train_scores)
        np.savetxt(scores_path_test,  test_scores)

    return None
        
def plot_LIME_predictions(regr, dataset, train_test_sets,
                          planets=None,
                          my_pred_planet=None,
                          my_true_radius=None,
                          feature_name=None, 
                          model_name = None):
    """
    Compute and plot the LIME explanation for one or several radius predictions
    made by the random forest model
    INPUTS: REGR = the random forest model, lightgbm or xgboost regression model for which we want to compute the LIME explanation
            DATASET = the input dataset from which the model is built
            TRAIN_TEST_SET = the training and test sets

            PLANETS = list of indexes of the planets in the Test set,
                      for which we want an LIME explanation
                      Contains maximum 6 numbers
    or
            MY_PRED_PLANET = pandas dataset with the input features
                        used by the regression model
                        > mass, semi_major_axis, temp_eq, star_luminosity,
                          star_radius, star_teff, star_mass
            The my_pred_planet output of predict_radius() can be used as
            my_pred_planet input for this function

            FEATURE_NAME = list of input features used by the regression model

            MODEL_NAME = str, name of the regression model used for the prediction, to be displayed in the title of the plot

    OUTPUTS: EXP = LIME explainer, contains the LIME radius prediction
    """
    if planets is None:
        planets = []

    # Data
    X_train, X_test, y_train, y_test = train_test_sets

    if feature_name is None:
        feature_name = list(X_train.columns)

    #stop running if no model name provided, to avoid confusion in the plot title
    if model_name is None:
        print("Error, model name must be provided for the plot title")
        return None 
    
    features = dataset.iloc[:, :-1].to_numpy()

    # makes sure pandas object with correct column names are used for LIME
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_name)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_name)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    # Check if some features are non continuous
    nb_unique_obj_in_features = np.array(
        [pd.Series(features[:, x]).nunique(dropna=False) for x in range(features.shape[1])]
    )

    # In our case the list of categorical features is empty
    cat_features = np.argwhere(nb_unique_obj_in_features <= 10).flatten().tolist()

    # LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_name,
        class_names=['radius'],
        categorical_features=cat_features,
        verbose=True,
        mode='regression'
    )
    # Prediction function for LIME
    def predict_fn(x):
        x_df = pd.DataFrame(x, columns=feature_name)
        return regr.predict(x_df)

    # Select planets to explain with LIME
    if my_pred_planet is not None:
        if not isinstance(my_pred_planet, pd.DataFrame):
            my_pred_planet = pd.DataFrame(my_pred_planet, columns=feature_name)

        exp = explainer.explain_instance(
            my_pred_planet.values[0],
            predict_fn,
            num_features=len(feature_name)
        )

        if my_true_radius is not None:
            print('True radius: ', my_true_radius)
        else:
            print('True radius was not provided')

        lime_radius = float(np.ravel(exp.local_pred)[0])
        model_radius = float(exp.predicted_value)

        # My plot of exp_as_pyplot()
        exp_list = exp.as_list()
        vals = [x[1] for x in exp_list]
        names = [
            x[0].replace("<=", r'$\leq$')
                .replace('_', ' ')
                .replace('.00', '')
                .replace("<", "$<$")
                .replace(">", "$>$")
            for x in exp_list
        ]
        vals.reverse()
        names.reverse()
        colors = ['C2' if x > 0 else 'C3' for x in vals]
        pos = np.arange(len(exp_list)) + .5

        # Plotting
        plt.figure()
        plt.xlabel('Weight')
        plt.ylabel('Feature')
        if len(my_pred_planet.index) > 0:
            plt.title(str(my_pred_planet.index[0]), loc='right')
        rects = plt.barh(pos, vals, align='center', color=colors, alpha=0.5)
        for i, rect in enumerate(rects):
            plt.text(plt.xlim()[0] + 0.03, rect.get_y() + 0.2, str(names[i]))

        # Text box
        if my_true_radius is not None:
            textstr = '\n'.join((
                r'True radius=%.2f$R_\oplus$' % (my_true_radius,),

                f'{model_name} radius={model_radius:.2f}$R_\\oplus$',
                r'LIME radius=%.2f$R_\oplus$' % (lime_radius,)))
        else:
                textstr = '\n'.join((
                f'{model_name} radius={model_radius:.2f}$R_\\oplus$',
                r'LIME radius=%.2f$R_\oplus$' % (lime_radius,)))
        # place a text box in bottom right in axes coords
        ax = plt.gca()
        ax.text(
            0.98, 0.02, textstr,
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9}
        )
        return exp
    
    elif not planets: 
        default_names = [
                'TOI-561 b',
                'HATS-35 b',
                'CoRoT-13 b',
                'Kepler-75 b',
                'WASP-17 b',
                'Kepler-20 Ac'
            ]

        for name in default_names:
            matches = np.where(X_test.index == name)[0]
            if len(matches) > 0:
                planets.append(matches[0])

        if len(planets) == 0:
            print("None of the default LIME planets are in the test set.")
            planets = list(range(min(6, len(X_test))))
    else: 

        pass
   
    # keep maximum 6 planets to match the 3x2 subplot grid
    planets = planets[:6]

    # Plotting
    fig, axs = plt.subplots(3, 2, constrained_layout=True, 
                            figsize=(15, 7.5))
    
    axs = axs.flatten()
    for j, planet in enumerate(planets):
        print('\n', X_test.iloc[planet])
        print('True radius: ', y_test.iloc[planet] if hasattr(y_test, 'iloc') else y_test[planet])
        exp = explainer.explain_instance(
            X_test.values[planet],
            predict_fn,
            num_features=len(feature_name)
        )
        lime_radius = float(np.ravel(exp.local_pred)[0])
        model_radius = float(exp.predicted_value)

        # My plot of exp_as_pyplot()
        exp_list = exp.as_list()
        vals = [x[1] for x in exp_list]
        names = [
            x[0].replace("<=", r'$\leq$')
                .replace('_', ' ')
                .replace('.00', '')
                .replace("<", "$<$")
                .replace(">", "$>$")
            for x in exp_list
        ]
        vals.reverse()
        names.reverse()
        colors = ['C2' if x > 0 else 'C3' for x in vals]
        pos = np.arange(len(exp_list)) + .5
        
        # Plotting
        axs[j].get_yaxis().set_visible(False)
        axs[j].set_xlabel('Weight')
        axs[j].set_ylabel('Feature')
        axs[j].set_title(X_test.iloc[planet].name, loc='right')
        rects = axs[j].barh(pos, vals, align='center', color=colors, alpha=0.5)
        for i, rect in enumerate(rects):
            axs[j].text(axs[j].get_xlim()[0] + 0.03, rect.get_y() + 0.2, str(names[i]))

        # Text box
        true_radius = y_test.iloc[planet] if hasattr(y_test, 'iloc') else y_test[planet]
        textstr = '\n'.join((
            r'True radius=%.2f$R_\oplus$' % (true_radius,),
            f'{model_name} radius={model_radius:.2f}$R_\\oplus$',
            r'LIME radius=%.2f$R_\oplus$' % (lime_radius,)))
        # place a text box in upper left in axes coords
        axs[j].text(0.68, 0.1, textstr,
                    bbox={'boxstyle': 'round', 'facecolor': 'white'},
                    transform=axs[j].transAxes)
        
        
    plt.savefig("figures/LIME_predictions_radius.pdf", bbox_inches="tight")

    # Plot the Mass Radius Temp eq relation
    # with LIME predicted planets in circles
    # Extra plot: mass-radius relation with LIME planets circled
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Mass-radius relation of the test set with LIME predicted planets")

    size = X_test["temp_eq"].values

    plt.scatter(X_test.mass.values, y_test.values,
                c=size, cmap=cm.magma_r)

    plt.colorbar(label=r"Equilibrium temperature (K)")
    plt.xlabel(r"Mass ($M_\oplus$)")
    plt.ylabel(r"Radius ($R_\oplus$)")

    for planet in planets:
        plt.plot(X_test.iloc[planet].mass,
                 y_test.values[planet], 'o',
                 mfc='none', ms=12,
                 label=X_test.iloc[planet].name)

    if len(planets) > 0:
        plt.legend()
    plt.savefig("figures/LIME_planets_mass.pdf", bbox_inches="tight")
   
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_title("Mass-Radius relation of the test set with LIME predicted planets")
    # size = X_test.temp_eq.values
    # plt.scatter(X_test.mass.values, y_test.values,
    #             c=size, cmap=cm.magma_r)
    # plt.colorbar(label=r'Equilibrium temperature (K)')
    # plt.xlabel(r'Mass ($M_\oplus$)')
    # plt.ylabel(r'Radius ($R_\oplus$)')
    # for planet in planets:
    #     plt.plot(X_test.iloc[planet].mass,
    #              y_test.values[planet], 'o',
    #              mfc='none', ms=12,
    #              label=X_test.iloc[planet].name)
    # if len(planets) > 0:
    #     plt.legend()

    # plt.savefig("figures/LIME_planets_radius.pdf", bbox_inches = "tight")

    return exp
