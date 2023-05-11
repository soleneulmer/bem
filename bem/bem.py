import datetime
import pandas as pd
from . import format_dataset as fd
from pprint import pprint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
# from sklearn.externals import joblib
import joblib
from scipy.stats import binned_statistic
from scipy.stats import multivariate_normal as mvn
import lime
import lime.lime_tabular
from astropy.units import earthMass, earthRad, jupiterMass, jupiterRad
import os
import sys
import pickle

__all__ = [
    'load_dataset', 'load_dataset_errors', 'load_dataset_RV',
    'random_forest_regression', 'computing_errorbars',
    'predict_radius', 'plot_dataset', 'plot_true_predicted',
    'plot_learning_curve', 'plot_validation_curves',
    'plot_LIME_predictions'
]

here = os.path.abspath(os.path.dirname(__file__))
published_dir = os.path.join(here, '/home/antonin/Documents/1-Master/Laboratory/APLII/bem', 'published_output')
if os.path.exists(os.path.join(published_dir, 'r2_0.89_2023-04-17_14:32.pkl')):
    pass
else:
    published_dir = os.path.join(sys.prefix, 'published_output')

saved_pickle_model = os.path.join(published_dir, 'r2_0.89_2023-04-17_14:32.pkl')


def load_dataset(
        cat_exoplanet='/home/antonin/Documents/1-Master/Laboratory/APLII/bem/published_output/exoplanet.eu_catalog_27-02-23.csv',
        cat_solar='/home/antonin/Documents/1-Master/Laboratory/APLII/bem/published_output/solar_system_planets_catalog.csv',
        feature_names_input=None,
        feature_names_output=None,
        solar=True,
        rm_ecc=False,
        rm_outliers=None
):
    '''
    Select exoplanet in the catalog which have mass and radius measurements
    as well as desired stellar parameters
    This dataset will be used to train and test the Random Forest
    :param cat_exoplanet: CSV file from exoplanet.eu
    :param cat_solar: CSV file from Planetary sheet
    :param feature_names_input:  list of features to select in the dataset
    :param feature_names_output: list of features to return in the dataset
    :param rm_outliers: if True: remove outliers planets. if False: don't.
                        Requires to have a pandas df containing the outliers saved in a pickle file
                        Or a list containing the outliers
                        cf. 'rm_outliers' in 'format_dataset.py' for more informations
    :param rm_ecc: if True: remove the planets with 'eccentricity' = NaN. if False: set it to 0.0 if NaN
    :param solar: if True: add solar system planets to dataset. if False: don't
    :return: Pandas struct with columns=feature_names_output of exoplanets.
                          Exoplanets have mass & radius measurements
                          the masses/radii are in Earth mass/radius
    '''

    if feature_names_input is None:
        feature_names_input = ['mass', 'radius', 'semi_major_axis', 'orbital_period', 'eccentricity',
                               'star_mass', 'star_radius', 'star_teff', 'star_age', 'star_metallicity']
    if feature_names_output is None:
        feature_names_output = ['mass', 'radius', 'semi_major_axis', 'orbital_period', 'temp_eq', 'insolation',
                                'star_mass', 'star_radius', 'star_age']

    print('\n #----- Start load_dataset -----#')
    print('\nLoading exoplanet dataset and solar system planets:')
    # Importing exoplanet dataset
    cat_exoplanet = os.path.join(published_dir, cat_exoplanet)
    dataset_exo = pd.read_csv(cat_exoplanet, index_col=0)
    # adding the number of planets per system to the dataset
    print('Adding the number of planets in every system to the dataset')
    dataset_exo = fd.add_n_planets_syst_dataset(dataset_exo)
    # Removing the masses detected with Theoretical (Chen&Kipping 2017 MR relation) & TTV, Timing
    print('Removing the planets whose masses where detected with Theoretical, TTV or Timing')
    dataset_exo = dataset_exo[
        dataset_exo.mass_detection_type.isin(['Radial Velocity', np.nan, 'Astrometry', 'Spectrum'])]

    if solar:
        # Importing Solar system dataset
        # Masses and Radii already in Earth metrics
        cat_solar = os.path.join(published_dir, cat_solar)
        dataset_solar_system = pd.read_csv(cat_solar, index_col=0)

    # Choosing features/data
    if not feature_names_input:
        print('No features selected, loading all features')
        pass
    else:
        print('Following features selected, loading them:')
        print(feature_names_input)
        dataset_exo = dataset_exo[feature_names_input]
        if solar:
            dataset_solar_system = dataset_solar_system[feature_names_input]

    # Choose if you want to remove planets with NaN eccentricity or set it to 0
    # True if you want to remove and False if you want to replace it by 0
    if not rm_ecc:
        print('Setting the eccentricity to 0.0 if NaN:')
        dataset_exo.fillna(value={'eccentricity': 0.}, inplace=True)
    else:
        print('Removing planets for which the eccentricity is NaN:')
    # Removes the planets with NaN values
    dataset_exo = dataset_exo.dropna(axis=0, how='any')

    # Converting from Jupiter to Earth masses/radii - exoplanets
    print('Converting planet\'s mass/radius in Earth masses/radii')
    dataset_exo = fd.jupiter_to_earth_mass(dataset_exo, 'mass')
    dataset_exo = fd.jupiter_to_earth_radius(dataset_exo, 'radius')

    # # Changing the mass of Kepler-10 c
    # print('\nKepler 10 c changing mass')
    # print(dataset_exo.loc['Kepler-10 c'].mass)
    # dataset_exo.loc['Kepler-10 c'].mass = 17.2
    # print(dataset_exo.loc['Kepler-10 c'].mass, '\n')

    # Add the Solar system planets with the Exoplanets
    if solar:
        print('Adding the Solar system planets with the Exoplanets dataset')
        dataset = pd.concat([dataset_exo, dataset_solar_system], axis=0)
    else:
        dataset = dataset_exo
    # Removes the planets with NaN values
    dataset.dropna(axis=0, how='any', inplace=True)

    # Removes outliers
    # Set it to None to first obtain dataset with wanted parameters.
    # Then go to notebook 'Find Outliers.ipynb', load the dataset and run/modify to find the outliers
    # The notebook will save the outliers to the file 'outliers.pkl' that can be now used by rerunning the script with rm_outliers=True
    if rm_outliers:
        print('Removing outliers')
        dataset = fd.rm_outliers(dataset, outliers_pkl=rm_outliers)
    else:
        print('Not removing outliers')

    # Add observables
    if 'temp_eq' in feature_names_output:
        print('Computing planet\'s equilibrium temperature')
        dataset = fd.add_temp_eq_dataset(dataset)
    if 'star_luminosity' in feature_names_output:
        print('Computing stellar luminosity')
        dataset = fd.add_star_luminosity_dataset(dataset)
    if 'insolation' in feature_names_output:
        print('Computing insolation')
        dataset = fd.add_insolation_dataset(dataset)
    # Convert some columns to float (because they are objects for some reason)
    if 'star_age' in feature_names_output:
        dataset['star_age'] = pd.to_numeric(dataset['star_age'], errors='coerce')

    # Returning the dataset with selected features
    print('Selected features for the dataset:')
    print(feature_names_output)
    # Number of planets in dataset
    print('\nNumber of planets in the dataset: ', len(dataset))
    print('\n', dataset.head())

    dataset = dataset[feature_names_output]

    print('Writing dataset to a pickle file')
    write_list(dataset, 'published_output/', 'filtered_dataset.pkl')
    print('\n #----- End load_dataset -----#')

    return dataset


def load_dataset_errors(
        cat_exoplanet='/home/antonin/Documents/1-Master/Laboratory/APLII/bem/published_output/exoplanet.eu_catalog_27-02-23.csv',
        cat_solar='/home/antonin/Documents/1-Master/Laboratory/APLII/bem/published_output/solar_system_planets_catalog.csv',
        features_input=None,
        features_ss_input=None,
        features_output=None,
        solar=True):
    '''
    Select exoplanet in the catalog which have uncertainty measurements as well as stellar parameters.
    If there is no uncertainty measurement, the uncertainty is set to the 0.9 quantile of the distribution of uncertainties.
    This dataset will be used to compute error bars for the test set.

    :param cat_exoplanet: CSV file from exoplanet.eu
    :param cat_solar: CSV file from Planetary sheet
    :param features_input: input features of the exoplanets dataframe
                            Needs to be formatted the following way:
                            ['param', 'param_error_min', 'param_error_max', ...]
                            Otherwise dataset_exo = dataset_exo.dropna(subset=features_input[::3]) will not work
    :param features_ss_input: input features of the solar system dataframe
                             Needs to be formatted the following way:
                             ['param', 'param_error', ...]
                             Otherwise dataset_solar_system = dataset_solar_system.dropna(subset=features_ss_input[::2]) will not work
    :param features_output: output parameters
                            Needs to be formatted the following way:
                            ['param', 'param_error', ...]
    :param solar: list of features to select in the dataset

    :return: pandas struct with exoplanets
                          with mass & radius measurements
                          the mass/radius are in Earth massses/radii
    '''

    if features_output is None:
        features_output = ['mass', 'mass_error',
                           'star_luminosity', 'star_luminosity_error',
                           'temp_eq', 'temp_eq_error',
                           'semi_major_axis', 'semi_major_axis_error',
                           'star_mass', 'star_mass_error',
                           'star_radius', 'star_radius_error',
                           'star_teff', 'star_teff_error',
                           'radius', 'radius_error']
    if features_ss_input is None:
        features_ss_input = ['mass', 'mass_error',
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
                             'radius', 'radius_error']
    if features_input is None:
        features_input = ['mass', 'mass_error_min', 'mass_error_max',
                          'radius', 'radius_error_min', 'radius_error_max',
                          'semi_major_axis', 'semi_major_axis_error_min',
                          'semi_major_axis_error_max',
                          'eccentricity', 'eccentricity_error_min',
                          'eccentricity_error_max',
                          'star_mass',
                          'star_mass_error_min', 'star_mass_error_max',
                          'star_radius', 'star_radius_error_min',
                          'star_radius_error_max',
                          'star_teff',
                          'star_teff_error_min', 'star_teff_error_max']

    print('\nLoading exoplanet dataset and solar system planets:')

    # Importing exoplanet dataset
    cat_exoplanet = os.path.join(published_dir, cat_exoplanet)
    dataset_exo = pd.read_csv(cat_exoplanet, index_col=0)
    dataset_exo = dataset_exo[features_input]

    # Importing Solar system dataset
    cat_solar = os.path.join(published_dir, cat_solar)
    dataset_solar_system = pd.read_csv(cat_solar, index_col=0)

    dataset_solar_system = dataset_solar_system[features_ss_input]
    # Remove NaNs in features only
    dataset_exo = dataset_exo.dropna(subset=features_input[::3])
    dataset_solar_system = dataset_solar_system.dropna(subset=features_ss_input[::2])

    # Replace inf by NaN
    dataset_exo = dataset_exo.replace([np.inf, -np.inf], np.nan)

    # Replace NaN values in the error features by the 0.9 quantile value
    error_columns = features_input[1::3] + features_input[2::3]

    for error_col in error_columns:
        # find the 0.9 quantile value of the error columns
        max_error = dataset_exo[error_col].quantile(0.9)
        print(error_col, max_error)
        # replace NaN by the 0.9 error value
        dataset_exo[error_col] = dataset_exo[error_col].replace(np.nan,
                                                                max_error)
        # Converting from Jupiter to Earth masses/radii - exoplanets
    print('Converting planet\'s mass/radius in Earth masses/radii')
    convert_features = ['mass', 'mass_error_max', 'mass_error_min',
                        'radius', 'radius_error_max', 'radius_error_min']
    for feature in convert_features:
        dataset_exo = fd.jupiter_to_earth_mass(dataset_exo, feature)

    # Computes the average error column
    err_min_max = {}
    for i in range(0, int(len(error_columns) / 2), 1):
        err_min_max[features_output[int(2 * i + 1)]] = [error_columns[i],
                                                        error_columns[int(i + len(error_columns) / 2)]]

    err = features_output[::2]
    for error in err:
        dataset_exo[error] = dataset_exo[err_min_max[error]].mean(axis=1).abs()

    dataset_exo = dataset_exo[features_output]

    # # Changing the mass of Kepler-10 c
    # print('\nKepler 10 c changing mass and mass error')
    # print(dataset_exo.loc['Kepler-10 c'].mass)
    # print(dataset_exo.loc['Kepler-10 c'].mass_error)
    # dataset_exo.loc['Kepler-10 c'].mass = 17.2
    # dataset_exo.loc['Kepler-10 c'].mass_error = 1.9
    # print(dataset_exo.loc['Kepler-10 c'].mass)
    # print(dataset_exo.loc['Kepler-10 c'].mass_error, '\n')

    # If error on radius is > 10% or error on mass is > 25%
    # remove the planet if rm=True
    rm = False
    if not rm:
        pass
    else:
        dataset_exo = dataset_exo[dataset_exo.radius_error <= 0.1 * dataset_exo.radius]
        dataset_exo = dataset_exo[dataset_exo.mass_error <= 0.25 * dataset_exo.mass]

    # Add the Solar system planets with the Exoplanets
    if solar:
        dataset = pd.concat([dataset_exo, dataset_solar_system], axis=0)
    else:
        dataset = dataset_exo

    # Add observables
    if 'temp_eq' in features_output:
        print('Computing planet\'s equilibrium temperature')
        dataset = fd.add_temp_eq_error_dataset(dataset)
    if 'star_luminosity' in features_output:
        print('Computing stellar luminosity')
        dataset = fd.add_star_luminosity_error_dataset(dataset)

    # Number of planets in dataset
    print('\nNumber of planets: ', len(dataset))

    # Select the same features as the original dataset
    dataset = dataset[features_output]

    print('The selected features can be change in [load_dataset_errors]')
    print('\n', dataset.head())

    return dataset


def load_dataset_RV(
        catalog_exoplanet='/home/antonin/Documents/1-Master/Laboratory/APLII/bem/published_output/exoplanet.eu_catalog_27-02-23.csv',
        features_names_input=None,
        features_names_output=None
):
    '''
    Select exoplanet in the catalog which are detected by RV and do not have mass measurement.
    This dataset will be used to later predict their masses

    :param catalog_exoplanet: CSV file from exoplanet.eu
    :param features_names_input: list of features to select in the dataset
    :param features_names_output: list of features to select in the output dataset

    :return: dataset_radial: pandas struct with exoplanets detected by RV without radius measurements.
                             the mass is in Earth masses
    '''
    if features_names_output is None:
        features_names_output = ['mass', 'semi_major_axis', 'orbital_period', 'temp_eq', 'insolation',
                                 'star_mass', 'star_radius', 'star_age']
    if features_names_input is None:
        features_names_input = ['mass', 'mass_error_min', 'mass_error_max',
                                'semi_major_axis', 'orbital_period', 'eccentricity',
                                'star_mass', 'star_radius', 'star_teff', 'star_age', 'star_metallicity']

    print('\n #----- Start load_dataset_RV -----#')

    print('\nLoading exoplanet dataset found with RVs:')
    catalog_exoplanet = os.path.join(published_dir, catalog_exoplanet)
    dataset = pd.read_csv(catalog_exoplanet, index_col=0)
    # adding the number of planets per system to the dataset
    print('Adding the number of planets in every system to the dataset')
    dataset = fd.add_n_planets_syst_dataset(dataset)
    # Select detected by RV
    dataset_radial = dataset[dataset.detection_type == 'Radial Velocity']
    # the radius column in Null (=NaN)
    dataset_radial = dataset_radial[pd.isnull(dataset_radial['radius'])]

    # Choosing features/data
    if not features_names_input:
        print('No features selected, loading all features')
        pass
    else:
        print('Selecting features:')
        print(features_names_input)
        dataset_radial = dataset_radial[features_names_input]
        print('length dataset radial before rm nan', len(dataset_radial))
        # Excluding exoplanets with missing data
        # dataset_radial = dataset_radial.dropna(subset=['mass', 'semi_major_axis',
        #                                                'eccentricity',
        #                                                'star_metallicity',
        #                                                'star_radius', 'star_teff',
        #                                                'star_mass'])
    # Replace inf by NaN
    dataset_radial.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Replace NaN values in the error features by the 0.9 quantile value
    error_columns = ['mass_error_min', 'mass_error_max']

    for error_col in error_columns:
        # find the 0.9 quantile value of the error columns
        # max_error = dataset_radial[error_col].quantile(0.9)
        max_error = 0.0
        print(error_col, max_error)
        # replace NaN by the 0.9 error value
        dataset_radial[error_col] = dataset_radial[error_col].replace(np.nan, max_error)

    # Removing NaN values
    dataset_radial.dropna(axis=0, how='any', inplace=True)
    print('length dataset radial after rm nan', len(dataset_radial))


    # Converting from Jupiter to Earth masses/radii - exoplanets
    print('Converting planet\'s mass/radius in Earth masses/radii')
    convert_features = ['mass', 'mass_error_max', 'mass_error_min']
    for feature in convert_features:
        dataset_radial = fd.jupiter_to_earth_mass(dataset_radial, feature)

    # Computes the average error column
    dataset_radial['mass_error'] = dataset_radial[['mass_error_min',
                                                   'mass_error_max']].mean(axis=1).abs()

    # Adding observables
    if 'temp_eq' in features_names_output:
        print('Computing planet\'s equilibrium temperature')
        dataset_radial = fd.add_temp_eq_dataset(dataset_radial)
    if 'star_luminosity' in features_names_output:
        print('Computing stellar luminosity')
        dataset_radial = fd.add_star_luminosity_dataset(dataset_radial)
    if 'insolation' in features_names_output:
        print('Computing insolation')
        dataset_radial = fd.add_insolation_dataset(dataset_radial)
    # Convert some columns to float (because they are objects for some reason)
    if 'star_age' in features_names_output:
        dataset_radial['star_age'] = pd.to_numeric(dataset['star_age'], errors='coerce')

    # Remove the mass error column for Random forest
    dataset_radial = dataset_radial[features_names_output]
    print('Selected features for the dataset:')
    print(features_names_output)

    print('\nNumber of planets in the dataset: ', len(dataset_radial))
    print('\n', dataset_radial.head())

    print('\n #----- End load_dataset_RV -----#')

    return dataset_radial


def random_forest_regression(dataset,
                             model=saved_pickle_model,
                             fit=False):
    '''
    Split the dataset into a training (75%) and testing set (25%)
    Removing 3 outliers planets from both sets

    If fit is True:
    Fitting the hyperparameters of the random forest regression
    otherwise loading a already fitted model

    :param dataset: pandas dataframe with all the exoplanets and their planetary and stellar parameters as features
    :param model: random forest model with best fit hyperparameters
    :param fit: boolean, to do the fitting (True) or not (False)
    :return: regr: the random forest regression model
            y_test_predict = radius predictions of the test set
            train_test_values = arrays with the values of the train and test sets
            train_test_sets = pandas dataframes with exoplanets and features names as well as the values
    '''
    print('\n #----- Start random_forest_regression -----#')
    # Preparing the training and test sets
    # ------------------------------------
    # Exoplanet and Solar system dataset
    # The solar system planets are the last 8 planets of the dataframe
    dataset_exo = dataset[:-8]
    dataset_sol = dataset[-8:]

    # Separating the data into dependent and independent variables
    # Implies that radius must be at the end in features given
    features = dataset_exo.iloc[:, :-1]  # mass, teq, etc
    labels = dataset_exo.iloc[:, -1]  # radius

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.25,
                                                        random_state=0)
    features_sol = dataset_sol.iloc[:, :-1]
    labels_sol = dataset_sol.iloc[:, -1]

    X_train_sol, X_test_sol, y_train_sol, y_test_sol = train_test_split(features_sol,
                                                                        labels_sol,
                                                                        test_size=0.25,
                                                                        random_state=0)

    X_train = pd.concat([X_train, X_train_sol], axis=0)
    y_train = pd.concat([y_train, y_train_sol], axis=0)
    X_test = pd.concat([X_test, X_test_sol], axis=0)
    y_test = pd.concat([y_test, y_test_sol], axis=0)

    try:
        # Outliers in the sample
        # Remove HATS-12 b from the training set
        X_test = X_test.drop(['HATS-12 b'])
        y_test = y_test.drop(labels=['HATS-12 b'])
        print('\nHATS-12 b removes from test set\n')
    except KeyError:
        pass
    try:
        # Remove K2-95 b from the training set
        X_train = X_train.drop(['K2-95 b'])
        y_train = y_train.drop(labels=['K2-95 b'])
        print('\nK2-95 b removes from training set\n')
    except KeyError:
        pass
    try:
        # Remove Kepler-11 g from the training set
        X_train = X_train.drop(['Kepler-11 g'])
        y_train = y_train.drop(labels=['Kepler-11 g'])
        print('\nKepler-11 g removes from training set\n')
    except KeyError:
        pass

    train_test_values = [X_train.values, X_test.values,
                         y_train.values, y_test.values]
    train_test_sets = [X_train, X_test, y_train, y_test]

    # Fitting the hyperparameters of the random forest model
    # with the grid search method
    # ------------------------------------------------------
    if fit:
        # Setting up the grid of hyperparameters
        rf = GridSearchCV(RandomForestRegressor(),
                          param_grid={'n_estimators': np.arange(80, 200),
                                      'max_depth': np.arange(4, 10),
                                      'max_features': np.arange(3, 6),
                                      'min_samples_split': np.arange(4, 5)},
                          cv=3, verbose=1, n_jobs=-1)

        # Fitting training set - finding best hyperparameters
        rf.fit(X_train, y_train)

        # Best hyperparameters found by the grid search
        print(rf.best_params_)

        # Random forest model with the best hyperparameters
        regr = RandomForestRegressor(n_estimators=rf.best_params_['n_estimators'],
                                     max_depth=rf.best_params_['max_depth'],
                                     max_features=rf.best_params_['max_features'],
                                     min_samples_split=rf.best_params_['min_samples_split'],
                                     random_state=0, oob_score=True)

        # Saving the random forest model in a file
        outdir = 'bem_output'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        name_Rf = 'r2_' + str(round(rf.best_score_, 2)) + '_' + str(
            datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")) + '.pkl'
        name_Rf = os.path.join(outdir, name_Rf)

        joblib.dump(regr, name_Rf)
        print('RF model save in : ', name_Rf)

    else:
        # Loading the random forest model saved
        print('Loading random forest model: ', model)
        regr = joblib.load(model)

    # Fit the best random forest model to the training set
    # ----------------------------------------------------
    regr.fit(X_train, y_train)

    # Predict the radius for the training and testing sets
    y_train_predict = regr.predict(X_train)
    y_test_predict = regr.predict(X_test)

    # Scores of the random forest
    test_score = r2_score(y_test, y_test_predict)
    pearson = pearsonr(y_test, y_test_predict)
    print(f'Test set, R-2 score: {test_score:>5.3}')
    print(f'\nTest set, Pearson correlation: {pearson[0]:.3}')

    # Mean squared errors of the train and test set
    print('Root mean squared errors')
    print('Train set: ', np.sqrt(np.mean((y_train - y_train_predict) ** 2)),
          '\nTest set:  ', np.sqrt(np.mean((y_test - y_test_predict) ** 2)))

    # Feature importance
    name_features = dataset.columns.tolist()
    print('\nFeature importance')
    _ = [print(name, ':  \t', value)
         for name, value
         in zip(name_features, regr.feature_importances_)]

    print('\n #----- End random_forest_regression -----#')

    return regr, y_test_predict, train_test_values, train_test_sets


def computing_errorbars(regr, dataset_errors, train_test_sets):
    '''

    :param regr: random forest regression model
    :param dataset_errors: pandas dataframe with each feature and their associated uncertainties
    :param train_test_sets: pandas dataframes with exoplanets and features names as well as the values
    :return: radii_test_output_error: error on the predicted radius for the Test set
             radii_test_input_error: original uncertainty on the radius measurements
    '''

    # Original train and test sets
    X_train, X_test, y_train, y_test = train_test_sets

    # Cross matching the Test set with the dataset with errors
    # to compute error bars for the exoplanets which have input errors
    dataset_errors = dataset_errors.loc[X_test.index.values.tolist()]
    # Remove an exoplanet in case there is still a NaN
    # in one of the feature
    dataset_errors = dataset_errors.dropna(axis=0, how='any')

    # Matrix with all the errors on the different features
    features_errors = dataset_errors.iloc[:, :-2].values
    # Radius vector
    radii_test = dataset_errors.iloc[:, -2].values
    # Error on the radius vector
    radii_test_input_error = dataset_errors.iloc[:, -1].values

    # Empty vector to store the error bars
    radii_test_output_error = np.zeros_like(radii_test_input_error)
    for i in range(radii_test.size):
        # print(i)
        # from each line in X_train generate new values for all parameters
        # with a multivariate gaussian which has
        # a vector of mean with the value columns and std with the error column
        # mean_values_0 = features_errors[i,0:-1:2]
        # >> takes the features : [mass0, temp_eq0, ...]
        # std_errors_0 = features_errors[0,1:-1:2]
        # >> takes the errors : [mass_err0, temp_eq_err0, ...]

        rerr = regr.predict(mvn(features_errors[i, ::2],
                                np.diag(features_errors[i, 1::2]),
                                allow_singular=True).rvs(1000)).std()
        radii_test_output_error[i] = rerr
        # print(radii_test_output_error[i], radii_test_input_error[i])

    # Save the errorbars in a txt file
    outdir = 'bem_output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    filename = 'bem_output/test_radius_RF_errorbars.dat'
    print('Error bars of the test set are saved in: ', filename)
    np.savetxt(filename, radii_test_output_error)

    return radii_test_output_error, radii_test_input_error


def predict_radius(my_name=np.array(['My planet b']),
                   my_param_name=['mass',
                                  'semi major axis',
                                  'eccentricity',
                                  'star_radius',
                                  'star_teff',
                                  'star_mass'],
                   my_param=np.array([[1,
                                       1,
                                       0,
                                       1,
                                       5777,
                                       1]]),
                   regr=None,
                   jupiter_mass=False,
                   error_bar=False):
    '''

    :param my_name: string array with a shape (1,) containing the name of the planet one wants to predict the radius of.
                        Example: np.array(['my planet b'])

    :param my_param_name: String array containing the name of the parameters one wants to use

        If no error_bar:
                String array with a shape (1,#parameters)
                    Example: np.array([[
                                        planetary mass,
                                        semi major axis,
                                        eccentricity,
                                        star_radius,
                                        star_teff,
                                        star_mass
                                ]])

        If error_bar:
            String array with a shape (1,2 * #parameters)
            Where 2 * #parameters = #parameters + #errors
            It is compulsory to give the parameters the following way:
            [param1, param1 error, param2, param2 error, ...]
                Example: np.array([[
                                    planetary mass, planetary mass error,
                                    semi major axis, semi major axis error,
                                    eccentricity, eccentricity error,
                                    star_radius, star_radius error,
                                    star_teff, star_teff error,
                                    star_mass, star_mass error
                                ]])

    :param my_param: Same as my_param_name but with the corresponding values

    :param regr: random forest regression model

    :param jupiter_mass: bool, True is the planet's mass is given in Jupiter mass

    :param error_bar: bool, True if an error is provided for EVERY parameter

                        Example: my_param = np.array([[
                                                planetary mass, planetary mass error,
                                                semi major axis, semi major axis error,
                                                eccentricity, eccentricity error,
                                                star_radius, star_radius error,
                                                star_teff, star_teff error,
                                                star_mass, star_mass error
                                            ]])


    :return: radius: planet's radius predicting with the RF model

             my_pred_planet: pandas dataframe with the input features used by the random forest model
                             Can be used as input in plot_LIME_predictions()
                                Example: The features are now:
                                          'mass', 'semi_major_axis',
                                          'temp_eq', 'star_luminosity',
                                          'star_radius', 'star_teff',
                                          'star_mass
    '''

    if regr is None:
        # Loading the random forest model saved
        print('Loading random forest model: ', saved_pickle_model)
        regr = joblib.load(saved_pickle_model)
    else:
        pass

    if error_bar:
        print('\nPredicting radius for planet:\n')
        my_param = pd.DataFrame(data=my_param,
                                index=my_name,
                                columns=np.array(my_param_name))
        # Changing mass units to Earth mass
        if jupiter_mass:
            my_param = fd.jupiter_to_earth_mass(my_param, 'mass')
            my_param = fd.jupiter_to_earth_mass(my_param, 'mass_error')
        else:
            print('Planetary mass is given in Earth mass')

        # Computing equilibrium temperature
        if 'temp_eq' in my_param:
            my_param_name = fd.add_temp_eq_error_dataset(my_param)
        # Computing stellar luminosity
        if 'star_luminosity' in my_param:
            my_param_name = fd.add_star_luminosity_error_dataset(my_param)
        # Computing insolation
        if 'insolation' in my_param:
            my_param_name = fd.add_insolation_dataset(my_param)

        # Planet with error bars
        print('Planet with error bars\n', my_param.iloc[0])

        # Radius error prediction
        my_pred_planet = my_param[my_param_name]

        # Feature / feature error
        features_with_errors = my_pred_planet.iloc[0].values.reshape(1, -1)
        radius_error = regr.predict(mvn(features_with_errors[0, ::2],
                                        np.diag(features_with_errors[0, 1::2]),
                                        allow_singular=True).rvs(1000)).std()

        # Radius prediction
        my_pred_planet = my_param[my_param_name[::2]]
        radius = regr.predict(my_pred_planet.iloc[0].values.reshape(1, -1))

        # Print
        print('Predicted radius (Rearth): ', radius, '+-', radius_error)
        return [radius, radius_error], my_pred_planet

    else:
        print('\nPredicting radius for planet:\n')
        my_param = pd.DataFrame(data=my_param,
                                index=my_name,
                                columns=np.array(my_param_name))
        # Changing mass units to Earth mass
        if jupiter_mass:
            my_param = fd.jupiter_to_earth_mass(my_param, 'mass')
        else:
            print('Planetary mass is given in Earth mass')

        # Computing equilibrium temperature
        if 'temp_eq' in my_param:
            my_param = fd.add_temp_eq_dataset(my_param)
        # Computing stellar luminosity
        if 'star_luminosity' in my_param:
            my_param = fd.add_star_luminosity_dataset(my_param)
        # Select features
        my_pred_planet = my_param[my_param_name]
        # Radius prediction
        print(my_pred_planet.iloc[0])
        radius = regr.predict(my_pred_planet.iloc[0].values.reshape(1, -1))
        print('Predicted radius (Rearth): ', radius)

        return radius, my_pred_planet


def plot_dataset(dataset, predicted_radii=[], rv=False):
    if not rv:
        try:
            # Remove outlier planets
            dataset = dataset.drop(['Kepler-11 g'])
            dataset = dataset.drop(['K2-95 b'])
            dataset = dataset.drop(['HATS-12 b'])
        except KeyError:
            pass

        # Plot the original dataset
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')

        if 'temp_eq' in dataset.columns:
            plt.scatter(dataset.mass, dataset.radius, c=dataset.temp_eq,
                        cmap=cm.magma_r, s=4, label='Verification sample')
            plt.colorbar(label=r'Equilibrium temperature (K)')
        elif 'star_teff' in dataset.columns:
            plt.scatter(dataset.mass, dataset.radius, c=dataset.star_teff,
                        cmap=cm.magma_r, s=4, label='Verification sample')
            plt.colorbar(label=r'Star Effective Temperature (K)')
        else:
            plt.scatter(dataset.mass, dataset.radius, c='k', alpha=0.5,
                        s=4, label='Verification sample')
        plt.xlabel(r'Mass ($M_\oplus$)')
        plt.ylabel(r'Radius ($R_\oplus$)')
        plt.legend(loc='lower right', markerscale=0,
                   handletextpad=0.0, handlelength=0)

    if rv:
        # Plot the radial velocity dataset
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')

        if 'temp_eq' in dataset.columns:
            plt.scatter(dataset.mass, predicted_radii, c=dataset.temp_eq,
                        cmap=cm.magma_r, s=4, label='RV sample')
            plt.colorbar(label=r'Equilibrium temperature (K)')
        elif 'star_teff' in dataset.columns:
            plt.scatter(dataset.mass, predicted_radii, c=dataset.star_teff,
                        cmap=cm.magma_r, s=4, label='RV sample')
            plt.colorbar(label=r'Star Effective Temperature (K)')
        else:
            plt.scatter(dataset.mass, predicted_radii, c='k', alpha=0.5,
                        s=4, label='RV sample')

        plt.xlabel(r'Mass ($M_\oplus$)')
        plt.ylabel(r'Radius ($R_\oplus$)')
        plt.legend(loc='lower right', markerscale=0,
                   handletextpad=0.0, handlelength=0)

    return None


def plot_true_predicted(train_test_sets, radii_test_RF,
                        radii_test_output_error):
    """Plot the residuals on the test set
    between True radius and Random forest"""

    X_train, X_test, y_train, y_test = train_test_sets
    plt.figure()
    plt.errorbar(radii_test_RF, y_test.values,
                 xerr=radii_test_output_error,
                 fmt='.', c='C1', elinewidth=0.5,
                 label='Random forest')
    # 1:1 line and labels
    plt.plot(np.sort(y_test.values), np.sort(y_test.values), 'k-', lw=0.25)

    plt.ylabel(r'True radius ($R_\oplus$)')
    plt.ylabel(r'Predicted radius ($R_\oplus$)')
    plt.legend(loc='lower right')
    return None


def plot_learning_curve(regr, dataset, save=False, fit=False):
    '''
    Cross validation with 100 iterations to get smoother mean test and train score curves,
    each time with 20% data randomly selected as a validation set.

    :param regr: random forest regression model
    :param dataset: pandas dataframe with features and labels
    :param save: bool, writes (True) or not (False) the scores
    :param fit: bool, computes the score if True
    :return: Written files
    '''

    features = dataset.iloc[:, :-1].values  # mass, teq, etc
    labels = dataset.iloc[:, -1].values  # radius

    outdir = 'bem_output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if fit:
        cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=11)

        train_sizes, train_scores, test_scores = learning_curve(regr,
                                                                features,
                                                                labels,
                                                                cv=cv,
                                                                n_jobs=-1,
                                                                train_sizes=np.linspace(.1,
                                                                                        1.0,
                                                                                        10),
                                                                verbose=1)
    else:
        train_sizes = np.loadtxt(os.path.join(published_dir, 'lc_train_sizes.dat'))
        train_scores = np.loadtxt(os.path.join(published_dir, 'lc_train_scores.dat'))
        test_scores = np.loadtxt(os.path.join(published_dir, 'lc_test_scores.dat'))

    plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="lower right")

    if save:
        np.savetxt(os.path.join(outdir, 'lc_train_sizes.dat'), train_sizes)
        np.savetxt(os.path.join(outdir, 'lc_train_scores.dat'), train_scores)
        np.savetxt(os.path.join(outdir, 'lc_test_scores.dat'), test_scores)
    return plt


def plot_validation_curves(regr, dataset, name='features',
                           save=False, fit=False):
    '''

    :param regr: random forest regression model
    :param dataset: pandas dataframe with features and labels
    :param name: str, can be 'features', 'tree', 'depth'
    :param save: bool, writes (True) or not (False) the scores
    :param fit: bool, computes the score if True

    :return: Written files
    '''

    features = dataset.iloc[:, :-1].values  # mass, teq, etc
    labels = dataset.iloc[:, -1].values  # radius

    outdir = 'bem_output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if name == 'features':
        param_range = np.arange(features.shape[1]) + 1
        param_name = 'max_features'
    elif name == 'tree':
        param_range = np.array([10, 20, 35, 50, 100, 1000, 5000, 10000])
        param_name = 'n_estimators'
    elif name == 'depth':
        param_range = np.array([1, 2, 3, 4, 5, 6, 7,
                                8, 9, 10, 50, 100, 1000])
        param_name = 'max_depth'
    else:
        print('Error the parameter of the validation curve is incorrect')
        print('Names can be features, tree, depth')
        return None

    # Need to catch the user warning:
    # Some inputs do not have OOB scores.
    # This probably means too few trees were used
    # to compute any reliable oob estimates.
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    if fit:
        train_scores, test_scores = validation_curve(regr, features, labels,
                                                     param_name=param_name,
                                                     param_range=param_range,
                                                     cv=3, scoring="r2",
                                                     n_jobs=-1, verbose=1)
    else:
        if name == 'features':
            train_scores = np.loadtxt(os.path.join(published_dir, 'vc_features_train_scores.dat'))
            test_scores = np.loadtxt(os.path.join(published_dir, 'vc_features_test_scores.dat'))
        elif name == 'tree':
            train_scores = np.loadtxt(os.path.join(published_dir, 'vc_tree_train_scores.dat'))
            test_scores = np.loadtxt(os.path.join(published_dir, 'vc_tree_test_scores.dat'))
        elif name == 'depth':
            train_scores = np.loadtxt(os.path.join(published_dir, 'vc_depth_train_scores.dat'))
            test_scores = np.loadtxt(os.path.join(published_dir, 'vc_depth_test_scores.dat'))
        else:
            pass
    # Averaging
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

    if save:
        if name == 'features':
            np.savetxt(os.path.join(outdir, 'vc_features_train_scores.dat'), train_scores)
            np.savetxt(os.path.join(outdir, 'vc_features_test_scores.dat'), test_scores)
        elif name == 'tree':
            np.savetxt(os.path.join(outdir, 'vc_tree_train_scores.dat'), train_scores)
            np.savetxt(os.path.join(outdir, 'vc_tree_test_scores.dat'), test_scores)
        elif name == 'depth':
            np.savetxt(os.path.join(outdir, 'vc_depth_train_scores.dat'), train_scores)
            np.savetxt(os.path.join(outdir, 'vc_depth_test_scores.dat'), test_scores)
        else:
            pass

    return None


def plot_LIME_predictions(regr, dataset, train_test_sets,
                          planets=[],
                          my_pred_planet=pd.DataFrame(),
                          my_true_radius=None,
                          feature_name=['mass',
                                        'semi_major_axis',
                                        'orbital_period',
                                        'temp_eq',
                                        'star_luminosity',
                                        'insolation',
                                        'star_age',
                                        'star_radius',
                                        'star_teff',
                                        'star_mass',
                                        'radius']):
    '''
    Compute and plot the LIME explanation for one or several radius predictions
    made by the random forest model

    :param regr: the random forest model
    :param dataset: the input dataset from which the RF is built
    :param train_test_sets: the training and test sets
    :param planets: list of indexes of the planets in the Test set,
                      for which we want an LIME explanation
                      Contains maximum 6 numbers
    :param my_pred_planet: pandas dataset with the input features
                        used by the random forest model
                        > mass, semi_major_axis, temp_eq, star_luminosity,
                          star_radius, star_teff, star_mass
            The my_pred_planet output of predict_radius() can be used as
            my_pred_planet input for this function
    :param my_true_radius:
    :param feature_name: list of input features used by the random forest
    :return: exp: LIME explainer, contains the LIME radius prediction
    '''

    # Data
    X_train, X_test, y_train, y_test = train_test_sets
    features = dataset.iloc[:, :-1].values  # mass, teq, etc
    labels = dataset.iloc[:, -1].values  # radius

    # Check if some features are non continuous
    nb_unique_obj_in_features = np.array([len(set(features[:, x]))
                                          for x in range(features.shape[1])])
    # In our case the list of categorical features is empty
    cat_features = np.argwhere(nb_unique_obj_in_features <= 10).flatten()

    # LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                       feature_names=feature_name,
                                                       class_names=['radius'],
                                                       categorical_features=cat_features,
                                                       verbose=True,
                                                       mode='regression')

    # Select planets to explain with LIME
    if not my_pred_planet.empty:
        exp = explainer.explain_instance(my_pred_planet.values[0],
                                         regr.predict, num_features=7)
        if my_true_radius:
            print('True radius: ', my_true_radius)
        else:
            print('True radius was not provided')

        lime_radius = exp.local_pred
        rf_radius = exp.predicted_value

        # My plot of exp_as_pyplot()
        exp = exp.as_list()
        vals = [x[1] for x in exp]
        names = [
            x[0].replace("<=", r'$\leq$').replace('_', ' ').replace('.00', '').replace("<", "$<$").replace(">", "$>$")
            for x in exp]
        # print(names)
        vals.reverse()
        names.reverse()
        colors = ['C2' if x > 0 else 'C3' for x in vals]
        pos = np.arange(len(exp)) + .5
        # Plotting
        plt.figure()
        plt.xlabel('Weight')
        plt.ylabel('Feature')
        plt.title(my_pred_planet.index[0], loc='right')
        rects = plt.barh(pos, vals, align='center', color=colors, alpha=0.5)
        for i, rect in enumerate(rects):
            # if rf_radius > 12.0:
            plt.text(plt.xlim()[0] + 0.03, rect.get_y() + 0.2, str(names[i]))

        # Text box
        if my_true_radius:
            textstr = '\n'.join((
                r'True radius=%.2f$R_\oplus$' % (my_true_radius,),
                r'RF radius=%.2f$R_\oplus$' % (rf_radius,),
                r'LIME radius=%.2f$R_\oplus$' % (lime_radius,)))
        else:
            textstr = '\n'.join((
                r'RF radius=%.2f$R_\oplus$' % (rf_radius,),
                r'LIME radius=%.2f$R_\oplus$' % (lime_radius,)))
        # place a text box in upper left in axes coords
        plt.text(-4, 0.1, textstr,
                 bbox={'boxstyle': 'round', 'facecolor': 'white'})
        return exp

    elif not planets:
        pass
        # planets.append(np.where(X_test.index == 'TRAPPIST-1 g')[0][0])
        # planets.append(np.where(X_test.index == 'HATS-35 b')[0][0])
        # planets.append(np.where(X_test.index == 'CoRoT-13 b')[0][0])
        # planets.append(np.where(X_test.index == 'Kepler-75 b')[0][0])
        # planets.append(np.where(X_test.index == 'WASP-17 b')[0][0])
        # planets.append(np.where(X_test.index == 'Kepler-20 c')[0][0])
    # else:
    #     pass

    # Plotting
    fig, axs = plt.subplots(3, 2, constrained_layout=True, figsize=(15, 7.2712643025))
    axs = axs.flatten()
    for j, planet in enumerate(planets):
        print('\n', X_test.iloc[planet])
        print('True radius: ', y_test[planet])
        exp = explainer.explain_instance(X_test.values[planet],
                                         regr.predict, num_features=7)
        lime_radius = exp.local_pred
        rf_radius = exp.predicted_value
        # pprint(exp.as_list())

        # My plot of exp_as_pyplot()
        exp = exp.as_list()
        vals = [x[1] for x in exp]
        names = [
            x[0].replace("<=", r'$\leq$').replace('_', ' ').replace('.00', '').replace("<", "$<$").replace(">", "$>$")
            for x in exp]
        # print(names)
        vals.reverse()
        names.reverse()
        colors = ['C2' if x > 0 else 'C3' for x in vals]
        pos = np.arange(len(exp)) + .5
        # Plotting
        axs[j].get_yaxis().set_visible(False)
        axs[j].set_xlabel('Weight')
        axs[j].set_ylabel('Feature')
        axs[j].set_title(X_test.iloc[planet].name, loc='right')
        rects = axs[j].barh(pos, vals, align='center', color=colors, alpha=0.5)
        for i, rect in enumerate(rects):
            # if rf_radius > 12.0:
            axs[j].text(axs[j].get_xlim()[0] + 0.03, rect.get_y() + 0.2, str(names[i]))

        # Text box
        textstr = '\n'.join((
            r'True radius=%.2f$R_\oplus$' % (y_test[planet],),
            r'RF radius=%.2f$R_\oplus$' % (rf_radius,),
            r'LIME radius=%.2f$R_\oplus$' % (lime_radius,)))
        # place a text box in upper left in axes coords
        axs[j].text(0.68, 0.1, textstr,
                    bbox={'boxstyle': 'round', 'facecolor': 'white'},
                    transform=axs[j].transAxes)

    # Plot the Mass Radius Temp eq relation
    # with LIME predicted planets in circles
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if 'temp_eq' in X_test.columns:
        size = X_test.temp_eq.values
        plt.scatter(X_test.mass.values, y_test.values,
                    c=size, cmap=cm.magma_r)
        plt.colorbar(label=r'Equilibrium temperature (K)')
    else:
        plt.scatter(X_test.mass.values, y_test.values, c='k', alpha=0.5)
    plt.xlabel(r'Mass ($M_\oplus$)')
    plt.ylabel(r'Radius ($R_\oplus$)')
    plt.legend()
    for planet in planets:
        plt.plot(X_test.iloc[planet].mass,
                 y_test.values[planet], 'o',
                 mfc='none', ms=12,
                 label=X_test.iloc[planet].name)
        plt.legend()
    # return exp


def write_list(dataset, path_to_file, file_name):
    '''
    Prints a pandas dataframe to a binary file.
    It checks if the file already exists or not. Creates it if not and overwrite if does.
    :param dataset: dataset of which you want to print the indices
    :param path_to_file: path to the folder in which the file is
    :param file_name: name of the file
    :param file_name: name of the file
    :return: Nothing
    '''
    file_loc = path_to_file + file_name
    # store list in binary file so 'wb' mode
    print('Start writing list into a binary file')
    with open(file_loc, 'wb') as fp:
        pickle.dump(dataset, fp)
        print('Done writing list into a binary file')


# Read list to memory
def read_list(file_name):
    # for reading also binary mode is important
    with open(file_name, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


def write_data(my_features, my_params, my_planets, file_path='', file_name='parameters.txt'):
    """
    Write the features, parameters such as solar, rm_ecc, ...
    Write the number of planets in dataset
    Write the results of the radius prediction <- NOT DONE
    """

    file_name_path = file_path + file_name

    with open(file_name_path, 'w') as f:

        # Write the parameters used by the functions
        for key, value in my_features.items():
            f.write('\nParameters for {}:\n'.format(key))
            f.write(str(value))
            f.write('\n')

        f.write('\n')
        # Write some parameters
        for key, value in my_params.items():
            f.write('{}: {}\n'.format(key, value))

        f.write('\n')
        # Write the number of planets in the datasets (normal and RV)
        f.write('# planets in the dataset: {}\n'.format(my_planets))
        # f.write('# planets in the RV dataset: {}\n'.format(nplanets_RV))

        f.close

    return None


# def create_dict(my_list):
#     my_dict = {}
#     for i in my_list:
#         my_dict[locals()[i]] = i
#     return my_dict
