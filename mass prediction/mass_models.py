#import packages 
import os
import datetime
import joblib
from matplotlib import cm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import format_dataset as fd

import lime
import lime.lime_tabular

from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def load_dataset(cat_exoplanet='data/exoplanet.eu_catalog_20-01-26_15_03_11.csv', 
                cat_solar="data/solar_system_planets_catalog.csv", 
                feature_names=['mass',
                                'radius',
                                'orbital_period',
                                'semi_major_axis',
                                'eccentricity',
                                'star_radius',
                                'star_teff',
                                'star_mass',
                                'star_metallicity',
                                'number_of_planets'],
                solar=True):

    """
    Select exoplanet in the catalogue which have mass and radius measurements
    as well as stellar parameters. This dataset will be used to train and 
    test the regression model.
                                
    Inputs: 
    cat_exoplanet: CSV file from exoplanet.eu
    cat_solar: CSV file from Planetary sheet
    feature_names: list of features to select in the dataset.
    
    Returns:
    dataset_exo = pandas dataframe with exoplanets with mass & radius 
                  measurements. mass/radii are in Earth mass/radii
                  
    """
    print("\nLoading the exoplanet dataset and solar system planets: ")

    # Importing exoplanet dataset
    dataset_exo = pd.read_csv(cat_exoplanet, index_col = 0)

    # Add the number of planets per host star as feature
    if "star_name" in dataset_exo.columns:
        dataset_exo["number_of_planets"] = (
            dataset_exo.groupby("star_name")["star_name"].transform("count")
        )     
        
    else:
        dataset_exo["number_of_planets"] = 1

    # Importing Solar system dataset
    dataset_solar_system = pd.read_csv(cat_solar, index_col = 0)

    # Solar system planets all orbit the Sun
    dataset_solar_system["star_name"] = "Sun"

    # There are 8 planets in the solar system file
    dataset_solar_system["number_of_planets"] = 8

    # Add solar metallicity. The Sun has a metallicity of 0.0 by definition.
    dataset_solar_system["star_metallicity"] = 0.0

    # Choosing features/data
    if not feature_names:
        print("No features selected, loading all features")
    else: 
        dataset_exo = dataset_exo[feature_names]
        dataset_solar_system = dataset_solar_system[feature_names]



    # Converting from Jupiter to Earth mass/radius
    print("Convert planets mass/radius from Jupiter to Earth.")
    dataset_exo = fd.jupiter_to_earth_mass(dataset_exo, 'mass')
    dataset_exo = fd.jupiter_to_earth_radius(dataset_exo, 'radius')

    # Add the solar system planets with the exoplanets
    if solar:
        dataset = pd.concat([dataset_exo, dataset_solar_system])
    else:
        dataset = dataset_exo

    # add the equilibrium temperature and its error
    dataset = fd.add_temp_eq_dataset(dataset)
    # add the stellar luminosity and its error
    dataset = fd.add_star_luminosity_dataset(dataset)

    

    # Number of planets in dataset
    print('\nNumber of planets before NaN removal: ', len(dataset))

    # Returning the dataset with selected features
    select_features = [
        'mass',
        'radius',
        'orbital_period',
        'eccentricity',
        'temp_eq',
        'star_mass',
        'star_metallicity',
        'number_of_planets'
    ]

    print('Selecting features:')
    print(select_features)
    dataset = dataset[select_features]

    # Remove planets with NaN's
    dataset = dataset.dropna(axis=0, how='any')
    print('Number of planets after final NaN removal: ', len(dataset))

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
    cat_exoplanet =          CSV file from exoplanet.eu

    cat_solar =              CSV file from planetary sheet.

    remove_bad_planets =     txt file with the names of the exoplanets we have left
                             out of the dataset. The reason for removing these 
                             planets are based on their uncertainties.

    Returns:
    dataset_exo =            pandas dataframe with exoplanets with mass & radius measurements
                             the mass/radius are in Earth mass/radius.
    """

    # load the dataset 
    dataset_exo = pd.read_csv(cat_exoplanet, index_col=0)

    # Add number of planets before selecting columns
    if "star_name" in dataset_exo.columns:
        dataset_exo["number_of_planets"] = (
            dataset_exo.groupby("star_name")["star_name"].transform("count")
        )
    else:
        dataset_exo["number_of_planets"] = 1

    mass_model_features = [
        'mass',
        'radius',
        'orbital_period',
        'eccentricity',
        'star_mass',
        'star_metallicity',
        'number_of_planets'
    ]

    # select the features and their errors
    dataset_exo = dataset_exo[[
                'mass', 'mass_error_min', 'mass_error_max',
                'orbital_period',
                'semi_major_axis', 'semi_major_axis_error_min', 'semi_major_axis_error_max',
                'eccentricity', 'eccentricity_error_min', 'eccentricity_error_max',
                'star_mass', 'star_mass_error_min', 'star_mass_error_max',
                'star_radius', 'star_radius_error_min', 'star_radius_error_max',
                'star_teff', 'star_teff_error_min', 'star_teff_error_max',
                'star_metallicity',
                'radius', 'radius_error_min', 'radius_error_max',
                'number_of_planets'
            ]]
    
    # load solar system dataset 
    dataset_solar_system = pd.read_csv(cat_solar, index_col=0)
    dataset_solar_system["number_of_planets"] = 8
    dataset_solar_system["star_metallicity"] = 0.0

    # select the features and their errors
    dataset_solar_system = dataset_solar_system[[
                'mass', 'mass_error', 
                'orbital_period',
                'semi_major_axis', 'semi_major_axis_error',
                'eccentricity', 'eccentricity_error',
                'star_mass', 'star_mass_error',
                'star_radius', 'star_radius_error',
                'star_teff', 'star_teff_error', 
                'star_metallicity',
                'radius', 'radius_error',
                'number_of_planets'
            ]]
    
    #Remove NaN's in features only
    dataset_exo = dataset_exo.dropna(subset=mass_model_features)                                       
    dataset_solar_system = dataset_solar_system.dropna(subset=mass_model_features)

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
        # print(error_col, max_error)

        #Replace NaN by the 0.9 error value
        dataset_exo[error_col] = dataset_exo[error_col].replace(np.nan, max_error)



    # After filling error NaNs, drop rows missing any core feature
    dataset_exo = dataset_exo.dropna(subset=mass_model_features)

    #Convert from Jupiter to Earth mass/radius
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
    
    
    dataset_exo = dataset_exo[['mass', 'mass_error',
                           'orbital_period',
                           'semi_major_axis',
                           'semi_major_axis_error',
                           'eccentricity',
                           'eccentricity_error',
                           'star_mass',
                           'star_mass_error',
                           'star_radius',
                           'star_radius_error',
                           'star_teff',
                           'star_teff_error',
                           'star_metallicity',
                           'radius', 'radius_error',
                           'number_of_planets']]
    
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

        print("Removing", len(removed_planets), "planets with uncertainty >= value")

        dataset_exo = dataset_exo.loc[~bad_mask]

        print("\nRemoved planets:")
        for planet in removed_planets:
            print(planet)

    # Add the solar system planets with the exoplanets
    if solar:
        dataset = pd.concat([dataset_exo, dataset_solar_system])
    else:
        dataset = dataset_exo

    
    # add the equilibrium temperature and its error
    dataset = fd.add_temp_eq_error_dataset(dataset)
    # add the stellar luminosity and its error
    dataset = fd.add_star_luminosity_error_dataset(dataset)


    # Select the same features as the original dataset
    dataset = dataset.dropna(subset=[
        'mass', 'mass_error',
        'radius', 'radius_error',
        'orbital_period',
        'temp_eq',
        'star_mass',
        'eccentricity',
        'star_metallicity',
        'number_of_planets'
    ])
    
    print("\nNumber of planets including error columns: ", len(dataset))
    return dataset    

def split_data_mass(dataset):
    """
    Create one consistent train/test split for mass prediction.

    Force a few named planets into the training and test set, 
    and put all solar-system planets into the train set.

    Target:
        mass

    Features:
        radius
        orbital_period
        temp_eq
        star_mass
        number_of_planets
        eccentricity
        star_metallicity
        
    Returns:
        features_needed
        X_train, X_test, y_train, y_test
        train_test_values
        train_test_sets
    """

    
    dataset = dataset.copy()

    # Remove rows where log10 would be impossible
    positive_columns = [
        'mass',
        'radius',
        'orbital_period',
        'temp_eq',
        'star_mass'
    ]

    for col in positive_columns:
        dataset = dataset[dataset[col] > 0]

    # Create log-transformed columns (based on Lalande et al. 2024)
    dataset['log_radius'] = np.log10(dataset['radius'])
    dataset['log_orbital_period'] = np.log10(dataset['orbital_period'])
    dataset['log_temp_eq'] = np.log10(dataset['temp_eq'])
    dataset['log_star_mass'] = np.log10(dataset['star_mass'])

    # select features and label for the exoplanet dataset
    features_needed = [
        'log_radius',
        'log_orbital_period',
        'log_temp_eq',
        'log_star_mass',
        'number_of_planets',
        'eccentricity', 
        'star_metallicity'
    ]

    features = dataset[features_needed]
    label = dataset['mass']

    # split into train and test sets with a fixed random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        label,
        test_size=0.25,
        random_state=9
    )

    # assign planets to force into the train and test sets, starting with the solar-system planets 
    forced_for_train = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
        'Kepler-12 b', 'WASP-94 Ab', 'HAT-P-47 b',
        'kappa And b', 'HD 984 b', 'Kepler-16 (AB)b', 'Kepler-34 (AB)b']
    
    forced_for_test = [
        # # uncomment for the LIME plots
        #     "TOI-561 b",
        #     "HATS-35 b",
        #     "CoRoT-13 b",
        #     "Kepler-75 b",
        #     "WASP-17 b",
        #     "Kepler-20 Ac",
        'HAT-P-32 Ab', 'TOI-3071 b', 'WASP-122 Ab', 'TOI-3976 Ab', 'Kepler-4 b',
        'LHS 1140 b', 'Gliese 12 b', 'TOI-1231 b',
        'TOI-201 c', 'TOI-561 f', 'HIP 41378 f']

    # Force selected planets into train set, while keeping the train/test split balanced by swapping with other planets if needed
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

    # Force selected planets into test set
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

    train_test_sets = [
        X_train,
        X_test,
        y_train,
        y_test
    ]

    return features_needed, X_train, X_test, y_train, y_test, train_test_values, train_test_sets

def lightgbm_mass(dataset, model=None, fit=False):
    """
    LightGBM regressor to predict log10 planetary mass.
    Predictions are converted back to normal mass.

    Target returned by split:
        mass

    Target used by LightGBM:
        log10(mass)
    """

    features_needed, X_train, X_test, y_train, y_test, train_test_values, train_test_sets = split_data_mass(dataset)

    # Remove planets with zero/negative mass, because log10 is impossible there
    train_mask = y_train > 0
    test_mask = y_test > 0

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    # Use log10 mass as target
    y_train_log = np.log10(y_train)
    y_test_log = np.log10(y_test)

    print('\nDataset loaded and split into train/test sets. Starting LightGBM regression for log mass...')

    if fit:
        params_grid = {
            'n_estimators': [100, 150, 200, 300, 500],
            'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1],
            'max_depth': [2, 3, 4, 5],
            'num_leaves': [3, 5, 7, 10, 15],
            'min_child_samples': [5, 10, 20, 40],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 1.0]
        }

        lgbm_cv = RandomizedSearchCV(
            LGBMRegressor(verbose=-1, random_state=9),
            param_distributions=params_grid,
            n_iter=80,
            scoring='neg_root_mean_squared_error',
            cv=3,
            verbose=1,
            n_jobs=-1,
            random_state=9,
        )

        # Fit on log mass
        lgbm_cv.fit(X_train, y_train_log)
        print("\nBest parameters LightGBM:")
        for param, value in lgbm_cv.best_params_.items():
            print(f"{param}: {value}")
        

        lgbm = lgbm_cv.best_estimator_

        outdir = 'bem_output'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        name_lgbm = 'lgbm_log_mass_rmse_' + str(round(-lgbm_cv.best_score_, 2)) + '_' + \
                    str(datetime.datetime.now().strftime("%Y-%m-%d_%H")) + '.pkl'
        name_lgbm = os.path.join(outdir, name_lgbm)

        print('\nLightGBM log mass model will be saved in:', name_lgbm)

        print("\nTraining LightGBM regressor...")
        lgbm.fit(X_train, y_train_log)
        joblib.dump(lgbm, name_lgbm)

    else:
        if model is None:
            raise ValueError("Please provide a saved model path when fit=False.")
        print("\nLoading LightGBM model:", model)
        lgbm = joblib.load(model)

    # Predict log mass
    y_train_predict_log = lgbm.predict(X_train)
    y_test_predict_log = lgbm.predict(X_test)

    # Convert predictions back to normal mass
    y_train_predict = 10 ** y_train_predict_log
    y_test_predict = 10 ** y_test_predict_log

    
    print('\nFeature importance:')
    importances = lgbm.feature_importances_
    importances_percent = 100 * importances / importances.sum()

    for name, value in zip(features_needed, importances_percent):
        print(f'{name}: {value:.2f}%')

    # Update returned values so they match the filtered data
    train_test_values = [
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values
    ]

    train_test_sets = [X_train, X_test, y_train, y_test]

    metrics = {
    #normal mass metrics
    "r2_test": r2_score(y_test, y_test_predict),
    "r2_train": r2_score(y_train, y_train_predict),
    "rmse_train": root_mean_squared_error(y_train, y_train_predict),
    "rmse_test": root_mean_squared_error(y_test, y_test_predict),
    "mean_ratio": np.mean(y_test_predict / y_test),
    "epsilon": root_mean_squared_error(y_test, y_test_predict)* np.log(10),

    # log mass metrics
    "r2_test_log": r2_score(y_test_log, y_test_predict_log),
    "r2_train_log": r2_score(y_train_log, y_train_predict_log),
    "rmse_test_log": root_mean_squared_error(y_test_log, y_test_predict_log),
    "rmse_train_log": root_mean_squared_error(y_train_log, y_train_predict_log),
    "mean_ratio_log": np.mean(y_test_predict_log / y_test_log),
    "epsilon_log": root_mean_squared_error(y_test_log, y_test_predict_log)* np.log(10)
    }

    print("\nEvaluation of LightGBM regressor for normal mass:")
    print(f"Train R²:   {r2_score(y_train, y_train_predict):.4f}")
    print(f"Test  R²:   {r2_score(y_test,  y_test_predict):.4f}")
    print(f"Train RMSE: {root_mean_squared_error(y_train, y_train_predict):.4f}")
    print(f"Test  RMSE: {root_mean_squared_error(y_test,  y_test_predict):.4f}")
    print(f"Mean ratio: {np.mean(y_test_predict / y_test):.4f}")
    print(f"Epsilon: {metrics['epsilon']:.4f}")

    print("\nEvaluation of LightGBM regressor for log mass:")
    print(f"Train R² log:   {r2_score(y_train_log, y_train_predict_log):.4f}")
    print(f"Test  R² log:   {r2_score(y_test_log,  y_test_predict_log):.4f}")
    print(f"Train RMSE log: {root_mean_squared_error(y_train_log, y_train_predict_log):.4f}")
    print(f"Test RMSE log:   {root_mean_squared_error(y_test_log, y_test_predict_log):.4f}")
    print(f"Mean ratio log: {np.mean(y_test_predict_log / y_test_log):.4f}")
    print(f"Epsilon log: {metrics['epsilon_log']:.4f}")

    return lgbm, y_test_predict, train_test_values, train_test_sets, metrics

def xgboost_mass(dataset, model=None, fit=False):
    """
    XGBoost regressor to predict log10 planetary mass.

    Target returned by split:
        mass

    Target used by XGBoost:
        log10(mass)

    Returns:
        xgb
        y_test_predict_log
        train_test_values
        train_test_sets
        metrics
    """

    features_needed, X_train, X_test, y_train, y_test, train_test_values, train_test_sets = split_data_mass(dataset)

    # Remove planets with zero/negative mass, because log10 is impossible there
    train_mask = y_train > 0
    test_mask = y_test > 0

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    # Use log10 mass as target
    y_train_log = np.log10(y_train)
    y_test_log = np.log10(y_test)

    print('\nDataset loaded and split into train/test sets. Starting XGBoost regression for log mass...')

    if fit:
        params_grid = {
            "n_estimators": [50, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500],
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
            "subsample": [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            "colsample_bytree": [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            "reg_lambda": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0],
            "reg_alpha": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
            "min_child_weight": [3, 5, 7, 10, 15, 20, 30, 50],
            "gamma": [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0],
            "colsample_bylevel": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }

        xgb_cv = RandomizedSearchCV(
            XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=9
            ),
            param_distributions=params_grid,
            n_iter=40,
            cv=5,
            scoring="neg_root_mean_squared_error",
            verbose=1,
            n_jobs=-1,
            random_state=9,
            return_train_score=True
        )
        # Fit on log mass
        xgb_cv.fit(X_train, y_train_log)
        print("\nBest parameters XGBoost:")
        for param, value in xgb_cv.best_params_.items():
            print(f"{param}: {value}")

        xgb = xgb_cv.best_estimator_

        outdir = 'bem_output'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        name_xgb = (
            'xgb_log_mass_rmse_' +
            str(round(-xgb_cv.best_score_, 2)) + '_' +
            str(datetime.datetime.now().strftime("%Y-%m-%d_%H")) +
            '.pkl'
        )
        name_xgb = os.path.join(outdir, name_xgb)

        print('\nXGBoost log mass model will be saved in:', name_xgb)

        print("\nTraining XGBoost regressor...")
        xgb.fit(X_train, y_train_log)
        joblib.dump(xgb, name_xgb)

    else:
        if model is None:
            raise ValueError("Please provide a saved model path when fit=False.")
        print("\nLoading XGBoost model:", model)
        xgb = joblib.load(model)

    # Predict log mass
    y_train_predict_log = xgb.predict(X_train)
    y_test_predict_log = xgb.predict(X_test)

    # Convert predictions back to normal mass only for normal-space metrics
    y_train_predict = 10 ** y_train_predict_log
    y_test_predict = 10 ** y_test_predict_log

    print('\nFeature importance:')
    importances = xgb.feature_importances_
    importances_percent = 100 * importances / importances.sum()

    for name, value in zip(features_needed, importances_percent):
        print(f'{name}: {value:.2f}%')

    train_test_values = [
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values
    ]

    train_test_sets = [
        X_train,
        X_test,
        y_train,
        y_test
    ]

    metrics = {
    #normal mass metrics
    "r2_test": r2_score(y_test, y_test_predict),
    "r2_train": r2_score(y_train, y_train_predict),
    "rmse_train": root_mean_squared_error(y_train, y_train_predict),
    "rmse_test": root_mean_squared_error(y_test, y_test_predict),
    "mean_ratio": np.mean(y_test_predict / y_test),
    "epsilon": root_mean_squared_error(y_test, y_test_predict)* np.log(10),

    # log mass metrics
    "r2_test_log": r2_score(y_test_log, y_test_predict_log),
    "r2_train_log": r2_score(y_train_log, y_train_predict_log),
    "rmse_test_log": root_mean_squared_error(y_test_log, y_test_predict_log),
    "rmse_train_log": root_mean_squared_error(y_train_log, y_train_predict_log),
    "mean_ratio_log": np.mean(y_test_predict_log / y_test_log),
    "epsilon_log": root_mean_squared_error(y_test_log, y_test_predict_log)* np.log(10)
    }

    print("\nEvaluation of XGBoost regressor for normal mass:")
    print(f"Train R²:   {r2_score(y_train, y_train_predict):.4f}")
    print(f"Test  R²:   {r2_score(y_test,  y_test_predict):.4f}")
    print(f"Train RMSE: {root_mean_squared_error(y_train, y_train_predict):.4f}")
    print(f"Test  RMSE: {root_mean_squared_error(y_test,  y_test_predict):.4f}")
    print(f"Mean ratio: {np.mean(y_test_predict / y_test):.4f}")
    print(f"Epsilon: {metrics['epsilon']:.4f}")

    print("\nEvaluation of XGBoost regressor for log mass:")
    print(f"Train R² log:   {r2_score(y_train_log, y_train_predict_log):.4f}")
    print(f"Test  R² log:   {r2_score(y_test_log,  y_test_predict_log):.4f}")
    print(f"Train RMSE log: {root_mean_squared_error(y_train_log, y_train_predict_log):.4f}")
    print(f"Test RMSE log:   {root_mean_squared_error(y_test_log, y_test_predict_log):.4f}")
    print(f"Mean ratio log: {np.mean(y_test_predict_log / y_test_log):.4f}")
    print(f"Epsilon log: {metrics['epsilon_log']:.4f}")

    

    return xgb, y_test_predict, train_test_values, train_test_sets, metrics

def random_forest_mass(dataset, model=None, fit=False):
    """
    Random Forest regressor to predict log10 planetary mass.
    Predictions are converted back to normal mass.

    Target returned by split:
        mass

    Target used by Random Forest:
        log10(mass)

    Returns:
        regr
        y_test_predict
        train_test_values
        train_test_sets
        metrics
    """

    features_needed, X_train, X_test, y_train, y_test, train_test_values, train_test_sets = split_data_mass(dataset)

    # Remove planets with zero/negative mass, because log10 is impossible there
    train_mask = y_train > 0
    test_mask = y_test > 0

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    # Use log10 mass as target
    y_train_log = np.log10(y_train)
    y_test_log = np.log10(y_test)

    print('\nDataset loaded and split. Starting Random Forest regression for log mass...')

    if fit:
        params_grid_rf = [
            {
                "bootstrap": [True],
                "n_estimators": [50, 100, 200, 300, 400, 500, 700, 1000, 1200, 1500],
                "max_depth": [3, 4, 5, 6, 8, 10, 12, 15, 20, None],
                "max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, "sqrt", "log2"],
                "min_samples_split": [2, 3, 4, 6, 8, 10, 15, 20],
                "min_samples_leaf": [1, 2, 3, 4, 5, 7, 10],
                "max_samples": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None],
                "min_impurity_decrease": [0.0, 0.005, 0.01, 0.02, 0.05],
                "max_leaf_nodes": [None, 20, 30, 50, 75, 100, 150, 200],
                "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
            },
            {
                "bootstrap": [False],
                "n_estimators": [50, 100, 200, 300, 400, 500, 700, 1000, 1200, 1500],
                "max_depth": [3, 4, 5, 6, 8, 10, 12, 15, 20, None],
                "max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, "sqrt", "log2"],
                "min_samples_split": [2, 3, 4, 6, 8, 10, 15, 20],
                "min_samples_leaf": [1, 2, 3, 4, 5, 7, 10],
                "max_samples": [None],
                "min_impurity_decrease": [0.0, 0.005, 0.01, 0.02, 0.05],
                "max_leaf_nodes": [None, 20, 30, 50, 75, 100, 150, 200],
                "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
            }
        ]

        rf_cv = RandomizedSearchCV(
            RandomForestRegressor(random_state=9),
            param_distributions=params_grid_rf,
            n_iter=40,
            cv=5,
            scoring="neg_root_mean_squared_error",
            verbose=1,
            n_jobs=-1,
            random_state=9,
            return_train_score=True
        )

        # Fit on log mass
        rf_cv.fit(X_train, y_train_log)

        print("\nBest paramaters Random Forest:")
        for param, value in rf_cv.best_params_.items():
            print(f"{param}: {value}")

        # Use the actual best estimator
        regr = rf_cv.best_estimator_

        outdir = 'bem_output'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        name_rf = (
            'rf_log_mass_rmse_' +
            str(round(-rf_cv.best_score_, 2)) + '_' +
            str(datetime.datetime.now().strftime("%Y-%m-%d_%H")) +
            '.pkl'
        )

        name_rf = os.path.join(outdir, name_rf)

        print('\nRandom Forest log mass model will be saved in:', name_rf)

        # Refit best estimator on full training data
        regr.fit(X_train, y_train_log)

        joblib.dump(regr, name_rf)

    else:
        if model is None:
            raise ValueError("Please provide a saved model path when fit=False.")
        print('\nLoading Random Forest model:', model)
        regr = joblib.load(model)

    # Predict log mass
    y_train_predict_log = regr.predict(X_train)
    y_test_predict_log = regr.predict(X_test)

    # Convert predictions back to normal mass
    y_train_predict = 10 ** y_train_predict_log
    y_test_predict = 10 ** y_test_predict_log

    
    print('\nFeature importance:')
    importances = regr.feature_importances_
    importances_percent = 100 * importances / importances.sum()

    for name, value in zip(features_needed, importances_percent):
        print(f'{name}: {value:.2f}%')

    train_test_values = [
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values
    ]

    train_test_sets = [
        X_train,
        X_test,
        y_train,
        y_test
    ]
    
    metrics = {
    #normal mass metrics
    "r2_test": r2_score(y_test, y_test_predict),
    "r2_train": r2_score(y_train, y_train_predict),
    "rmse_train": root_mean_squared_error(y_train, y_train_predict),
    "rmse_test": root_mean_squared_error(y_test, y_test_predict),
    "mean_ratio": np.mean(y_test_predict / y_test),
    "epsilon": root_mean_squared_error(y_test, y_test_predict)* np.log(10),

    # log mass metrics
    "r2_test_log": r2_score(y_test_log, y_test_predict_log),
    "r2_train_log": r2_score(y_train_log, y_train_predict_log),
    "rmse_test_log": root_mean_squared_error(y_test_log, y_test_predict_log),
    "rmse_train_log": root_mean_squared_error(y_train_log, y_train_predict_log),
    "mean_ratio_log": np.mean(y_test_predict_log / y_test_log),
    "epsilon_log": root_mean_squared_error(y_test_log, y_test_predict_log)* np.log(10)
    }

    print("\nEvaluation of Random Forest regressor for normal mass:")
    print(f"Train R²:   {r2_score(y_train, y_train_predict):.4f}")
    print(f"Test  R²:   {r2_score(y_test,  y_test_predict):.4f}")
    print(f"Train RMSE: {root_mean_squared_error(y_train, y_train_predict):.4f}")
    print(f"Test  RMSE: {root_mean_squared_error(y_test,  y_test_predict):.4f}")
    print(f"Mean ratio: {np.mean(y_test_predict / y_test):.4f}")
    print(f"Epsilon: {metrics['epsilon']:.4f}")

    print("\nEvaluation of Random Forest regressor for log mass:")
    print(f"Train R² log:   {r2_score(y_train_log, y_train_predict_log):.4f}")
    print(f"Test  R² log:   {r2_score(y_test_log,  y_test_predict_log):.4f}")
    print(f"Train RMSE log: {root_mean_squared_error(y_train_log, y_train_predict_log):.4f}")
    print(f"Test RMSE log:   {root_mean_squared_error(y_test_log, y_test_predict_log):.4f}")
    print(f"Mean ratio log: {np.mean(y_test_predict_log / y_test_log):.4f}")
    print(f"Epsilon log: {metrics['epsilon_log']:.4f}")

    return regr, y_test_predict, train_test_values, train_test_sets, metrics


def plot_LIME_mass_predictions(regr, train_test_sets,
                               planets=None,
                               feature_name=None,
                               model_name="Model"):
    """
    Compute and plot LIME explanations for mass predictions.

    The model predicts log10 planetary mass, so LIME weights are in
    log10(M/M_earth) space.
    """

    if planets is None:
        planets = []

    X_train, X_test, y_train, y_test = train_test_sets

    if feature_name is None:
        feature_name = list(X_train.columns)

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_name)

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_name)

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)

    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, index=X_test.index)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_name,
        class_names=["log_mass"],
        verbose=True,
        mode="regression"
    )

    def predict_fn(x):
        x_df = pd.DataFrame(x, columns=feature_name)
        return regr.predict(x_df)

    # Use the same default planets as the radius LIME plot
    if not planets:
        default_names = [
            "TOI-561 b",
            "HATS-35 b",
            "CoRoT-13 b",
            "Kepler-75 b",
            "WASP-17 b",
            "Kepler-20 Ac"
        ]

        for name in default_names:
            matches = np.where(X_test.index == name)[0]
            if len(matches) > 0:
                planets.append(matches[0])

        if len(planets) == 0:
            print("\nNone of the default LIME planets are in the test set.")
            planets = list(range(min(6, len(X_test))))

    planets = planets[:6]

    fig, axs = plt.subplots(
        3, 2,
        constrained_layout=True,
        figsize=(15, 7.5)
        

    
    )

    axs = axs.flatten()

    exp = None

    for j, planet in enumerate(planets):
        x_planet = X_test.iloc[planet]
        true_mass = y_test.iloc[planet]
        true_log_mass = np.log10(true_mass)

        model_log_mass = float(regr.predict(X_test.iloc[[planet]])[0])
        model_mass = 10 ** model_log_mass

        exp = explainer.explain_instance(
            x_planet.values,
            predict_fn,
            num_features=len(feature_name)
        )

        lime_log_mass = float(np.ravel(exp.local_pred)[0])
        lime_mass = 10 ** lime_log_mass

        print("\nPlanet:", X_test.index[planet])
        print("True mass:", true_mass)
        print("Model mass:", model_mass)
        print("True log mass:", true_log_mass)
        print("LIME log mass:", lime_log_mass)

        exp_list = exp.as_list()
        vals = [x[1] for x in exp_list]
        names = [
            x[0].replace("<=", r"$\leq$")
                .replace("_", " ")
                .replace(".00", "")
                .replace("<", "$<$")
                .replace(">", "$>$")
            for x in exp_list
        ]

        vals.reverse()
        names.reverse()

        colors = ["C2" if x > 0 else "C3" for x in vals]
        pos = np.arange(len(exp_list)) + 0.5

        axs[j].get_yaxis().set_visible(False)
        axs[j].set_xlabel(r"Weight in $\log_{10}(M/M_\oplus)$")
        axs[j].set_ylabel("Feature")
        axs[j].set_title(X_test.index[planet], loc="right")

        rects = axs[j].barh(
            pos,
            vals,
            align="center",
            color=colors,
            alpha=0.5
        )

        for i, rect in enumerate(rects):
            axs[j].text(
                axs[j].get_xlim()[0] + 0.03,
                rect.get_y() + 0.2,
                str(names[i])
            )

        textstr = "\n".join((
            r"True mass=%.2f$M_\oplus$" % (true_mass,),
            f"{model_name} mass={model_mass:.2f}$M_\\oplus$",
            r"LIME mass=%.2f$M_\oplus$" % (lime_mass,)
        ))

        axs[j].text(
            0.68,
            0.10,
            textstr,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.8
            },
            transform=axs[j].transAxes
        )

    for k in range(len(planets), len(axs)):
        axs[k].axis("off")

    plt.savefig("figures/LIME_mass_predictions.pdf", bbox_inches="tight")

    # Extra plot: mass-radius relation with LIME planets circled
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Mass-radius relation of the test set with LIME predicted planets")

    if "log_temp_eq" in X_test.columns:
        size = 10 ** X_test["log_temp_eq"].values
    else:
        size = np.ones(len(X_test))

    plt.scatter(y_test.values, 10 ** X_test["log_radius"].values,   
        c=size,cmap=cm.magma_r)

    plt.colorbar(label=r"Equilibrium temperature (K)")
    plt.xlabel(r"Mass ($M_\oplus$)")
    plt.ylabel(r"Radius ($R_\oplus$)")

    for planet in planets:
        plt.plot(
            y_test.values[planet],
            10 ** X_test.iloc[planet]["log_radius"], "o",
            mfc="none",ms=12,
            label=X_test.index[planet])

    if len(planets) > 0:
        plt.legend()
    plt.savefig("figures/LIME_planets_mass.pdf", bbox_inches="tight")
   

    return exp
