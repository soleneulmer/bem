from bem import bem
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------ CHOOSING THE FEATURES ------------------------------ #
#
# I put the planet parameters on the first line and the star parameters on the second line
# radius must be at the end because feature names must be in the same order as they were in fit
# cf. beginning of 'random_forest_regression' in 'bem.py'
#
#
# Features wanted as parameters and features needed to do calculations to create new features in load_dataset
# For the features needed to do calculations check format_dataset.py
features_load_dataset_input = ['mass', 'semi_major_axis', 'eccentricity',
                               'star_mass', 'star_radius', 'star_teff', 'star_metallicity',
                               'radius']

# Features wanted as parameters in load_dataset:
features_load_dataset_output = ['mass', 'semi_major_axis', 'temp_eq',
                                'star_mass', 'star_radius', 'star_teff', 'star_luminosity',
                                'radius']

# Features wanted as parameters and features needed to do calculations to create new features in load_dataset_RV
# For the features needed to do calculations check format_dataset.py
features_load_dataset_RV_input = ['mass_sini', 'mass_sini_error_min', 'mass_sini_error_max', 'semi_major_axis', 'eccentricity',
                                  'star_mass', 'star_radius', 'star_teff', 'star_metallicity']

# Features wanted as parameters in load_dataset_RV:
features_load_dataset_RV_output = ['mass', 'semi_major_axis', 'temp_eq',
                                   'star_mass', 'star_radius', 'star_teff', 'star_luminosity']

# Features wanted as parameters and features needed to do calculations to create new features in load_dataset_error
# Needs to be formatted the following way:
# ['param', 'param_error_min', 'param_error_max', ...]
# Otherwise dataset_exo = dataset_exo.dropna(subset=features_input[::3]) will not work
features_load_dataset_errors_input = ['mass', 'mass_error_min', 'mass_error_max', 'semi_major_axis',
                                      'semi_major_axis_error_min', 'semi_major_axis_error_max',
                                      'eccentricity', 'eccentricity_error_min', 'eccentricity_error_max',
                                      'star_mass', 'star_mass_error_min', 'star_mass_error_max', 'star_radius',
                                      'star_radius_error_min', 'star_radius_error_max',
                                      'star_teff', 'star_teff_error_min', 'star_teff_error_max',
                                      'radius', 'radius_error_min', 'radius_error_max']

# Features wanted as parameters for the solar system and features needed to do calculations to create new features in
# load_dataset_error
# Needs to be formatted the following way:
# ['param', 'param_error', ...]
# Otherwise dataset_solar_system = dataset_solar_system.dropna(subset=features_ss_input[::2]) will not work
features_load_dataset_errors_ss_input = ['mass', 'mass_error',
                                         'semi_major_axis', 'semi_major_axis_error',
                                         'eccentricity', 'eccentricity_error',
                                         'star_mass', 'star_mass_error',
                                         'star_radius', 'star_radius_error',
                                         'star_teff', 'star_teff_error',
                                         'radius', 'radius_error']

# Features wanted as exit parameters load_dataset_error
# Needs to be formatted the following way:
# ['param', 'param_error', ...]
features_load_dataset_errors_output = ['mass', 'mass_error',
                                       'semi_major_axis', 'semi_major_axis_error',
                                       'temp_eq', 'temp_eq_error',
                                       'star_mass', 'star_mass_error',
                                       'star_radius', 'star_radius_error',
                                       'star_teff', 'star_teff_error',
                                       'star_luminosity', 'star_luminosity_error',
                                       'radius', 'radius_error']

# Features wanted as parameters in plot_LIME_predictions:
features_plot_LIME_predictions = ['mass', 'semi_major_axis', 'temp_eq',
                                  'star_mass', 'star_radius', 'star_teff', 'star_luminosity',
                                  'radius']

features = {'features_load_dataset_input': features_load_dataset_input,
            'features_load_dataset_output': features_load_dataset_output,
            'features_load_dataset_RV_input': features_load_dataset_RV_input,
            'features_load_dataset_RV_output': features_load_dataset_RV_output,
            'features_load_dataset_errors_input': features_load_dataset_errors_input,
            'features_load_dataset_errors_ss_input': features_load_dataset_errors_ss_input,
            'features_load_dataset_errors_output': features_load_dataset_errors_output,
            'features_plot_LIME_predictions': features_plot_LIME_predictions}

# ------------------------------ END CHOOSING THE FEATURES ------------------------------ #
# ------------------------------ CHOOSING PARAMETERS ------------------------------ #
solar = True
rm_ecc = True
rm_outliers = 'published_output/Errorbars/Errorbars_outliers.pkl'
parameters = {'solar': solar,
              'rm_ecc': rm_ecc,
              'rm_outliers': rm_outliers}

# ------------------------------ END CHOOSING PARAMETERS ------------------------------ #
# ------------------------------ CHOOSING PLANET TO PREDICT RADIUS ------------------------------ #
name = ['GJ 357 b']

param_name = ['mass', 'mass_error',
              # 'n_planets', 'n_planets_error',
              # 'eccentricity', 'eccentricity_error',
              'semi_major_axis', 'semi_major_axis_error',
              # 'star_metallicity', 'star_metallicity_error',
              'star_teff', 'star_teff_error',
              'star_radius', 'star_radius_error',
              'star_mass', 'star_mass_error']

param = [[0.006566, 0.00101,
          # 3, 1,
          # 0.047, 0.059,
          0.033, 0.001,
          # -0.12, 0.16,
          3505.0, 51.0,
          0.337, 0.015,
          0.342, 0.011]]

jupiter_mass = True,
error_bar = True
# ------------------------------ END CHOOSING PLANET TO PREDICT RADIUS ------------------------------ #

# Load exoplanet and solar system planets dataset
dataset = bem.load_dataset(feature_names_input=features_load_dataset_input,
                           feature_names_output=features_load_dataset_output,
                           solar=solar,
                           rm_ecc=rm_ecc,
                           rm_outliers=rm_outliers)

# write some data to a file
bem.write_data(features, parameters, len(dataset), file_path='published_output/Errorbars/',
               file_name='Parameters.txt')

# Plot the dataset radius as a function of mass and equilibrium temperature
bem.plot_dataset(dataset)
# Build the random forest model and predict radius of the dataset
regr, y_test_predict, _, train_test_sets = bem.random_forest_regression(dataset, fit=False)
print('regr', np.shape(regr))

# Load exoplanet and solar system planets dataset with uncertainties
dataset_errors = bem.load_dataset_errors(features_input=features_load_dataset_errors_input,
                                         features_ss_input=features_load_dataset_errors_ss_input,
                                         features_output=features_load_dataset_errors_output)
print('\ndataset_errors:\n', dataset_errors.columns[1::2])
print(np.shape(dataset_errors))
print('\ntrain_test_sets', train_test_sets[0].columns)

# Compute the error bars for the test set planets
radii_test_output_error, _ = bem.computing_errorbars(regr,
                                                     dataset_errors,
                                                     train_test_sets)
# Plot the test set, true radius versus RF predicted radius
bem.plot_true_predicted(train_test_sets,
                        y_test_predict,
                        radii_test_output_error)

# Load the radial velocity dataset
dataset_rv = bem.load_dataset_RV(features_names_input=features_load_dataset_RV_input,
                                 features_names_output=features_load_dataset_RV_output)
# Predict the radius of the RV dataset
radii_RV_RF = regr.predict(dataset_rv)
# Plot the predictions of the RV dataset
bem.plot_dataset(dataset_rv, predicted_radii=radii_RV_RF, rv=True)

# Plot the learning curve
bem.plot_learning_curve(regr, dataset, save=True, fit=True)
# Plot the validation curves
bem.plot_validation_curves(regr, dataset, name='features', save=True, fit=True)
bem.plot_validation_curves(regr, dataset, name='tree', save=True, fit=True)
bem.plot_validation_curves(regr, dataset, name='depth', save=True, fit=True)

# Explain the RF predictions
exp = bem.plot_LIME_predictions(regr, dataset, train_test_sets,
                                feature_name=features_load_dataset_output)

# Predict a new radius
# example given with GJ 357 b
# radius, my_pred_planet = bem.predict_radius(my_name=np.array(name),
#                                             my_param_name=np.array(param_name),
#                                             my_param=np.array(param),
#                                             regr=regr,
#                                             jupiter_mass=jupiter_mass,
#                                             error_bar=error_bar)
# If error_bar is True
# print('Radius: ', radius[0][0], '+-', radius[1])

# Plot LIME explanation for the newly predicted radius
# exp = bem.plot_LIME_predictions(regr, dataset, train_test_sets,
#                                 my_pred_planet=my_pred_planet,
#                                 my_true_radius=1.166)

plt.show()
