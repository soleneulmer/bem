from bem import bem
import numpy as np
import matplotlib.pyplot as plt
import corner

# Load exoplanet and solar system planets dataset
dataset = bem.load_dataset(rm_ecc=False, solar=True)
bem.print_to_file(dataset, 'published_output/', 'filtered_dataset.pkl')

figure = corner.corner(dataset,
                       labels=list(dataset.columns),
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True,
                       title_kwargs={"fontsize": 12},)
# Plot the dataset radius as a function of mass and equilibrium temperature
bem.plot_dataset(dataset)
# Build the random forest model and predict radius of the dataset
regr, y_test_predict, _, train_test_sets = bem.random_forest_regression(dataset, fit=False)

# # Load exoplanet and solar system planets dataset with uncertainties
# dataset_errors = load_dataset_errors()
# # Compute the error bars for the test set planets
# radii_test_output_error, _ = computing_errorbars(regr,
#                                                      dataset_errors,
#                                                      train_test_sets)

# setting errors to 0.1, because load_dataset errors is not using the latest exoplanet databse
radii_test_output_error = y_test_predict * 0 + 0.1

# Plot the test set, true radius versus RF predicted radius
bem.plot_true_predicted(train_test_sets,
                        y_test_predict,
                        radii_test_output_error)

# Load the radial velocity dataset
dataset_rv = bem.load_dataset_RV()
# Predict the radius of the RV dataset
radii_RV_RF = regr.predict(dataset_rv)
# Plot the predictions of the RV dataset
bem.plot_dataset(dataset_rv, predicted_radii=radii_RV_RF, rv=True)


# Plot the learning curve
bem.plot_learning_curve(regr, dataset, save=False, fit=False)
# Plot the validation curves
bem.plot_validation_curves(regr, dataset, name='features', save=False, fit=False)
bem.plot_validation_curves(regr, dataset, name='tree', save=False, fit=False)
bem.plot_validation_curves(regr, dataset, name='depth', save=False, fit=False)

# Explain the RF predictions
exp = bem.plot_LIME_predictions(regr, dataset, train_test_sets)

# Predict a new radius
# example given with GJ 357 b
radius, my_pred_planet = bem.predict_radius(my_name=np.array(['GJ 357 b']),
                                            my_param_name=np.array(['mass', 'mass_error',
                                                                    'orbital_period', 'orbital_period_error',
                                                                    'eccentricity', 'eccentricity_error',
                                                                    'semi_major_axis', 'semi_major_axis_error',
                                                                    'star_teff', 'star_teff_error',
                                                                    'star_radius', 'star_radius_error',
                                                                    'star_age', 'star_age_error',
                                                                    'star_mass', 'star_mass_error']),
                                            my_param=np.array([[0.006566, 0.00101,
                                                                3.93086, 0.00004,
                                                                0.047, 0.059,
                                                                0.033, 0.001,
                                                                3505.0, 51.0,
                                                                0.337, 0.015,
                                                                5, 1,
                                                                0.342, 0.011]]),
                                            regr=regr,
                                            jupiter_mass=True,
                                            error_bar=True)
# If error_bar is True
# print('Radius: ', radius[0][0], '+-', radius[1])

# Plot LIME explanation for the newly predicted radius
exp = bem.plot_LIME_predictions(regr, dataset, train_test_sets,
                                my_pred_planet=my_pred_planet,
                                my_true_radius=1.166)

plt.show()
