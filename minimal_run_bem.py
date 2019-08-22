import bem
import numpy as np
import matplotlib.pyplot as plt

# Load exoplanet and solar system planets dataset
dataset = bem.load_dataset()
# Plot the dataset radius as a function of mass and equilibrium temperature
bem.plot_dataset(dataset)
# Build the random forest model and predict radius of the dataset
regr, y_test_predict, _, train_test_sets = bem.random_forest_regression(dataset)

# Load exoplanet and solar system planets dataset with uncertainties
dataset_errors = bem.load_dataset_errors()
# Compute the error bars for the test set planets
radii_test_output_error, _ = bem.computing_errorbars(regr,
                                                     dataset_errors,
                                                     train_test_sets)
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
bem.plot_learning_curve(regr, dataset)
# Plot the validation curves
bem.plot_validation_curves(regr, dataset, name='features')
bem.plot_validation_curves(regr, dataset, name='tree')
bem.plot_validation_curves(regr, dataset, name='depth')

# Explain the RF predictions
bem.plot_LIME_predictions(regr, dataset, train_test_sets)

# Predict a new radius
radius = bem.predict_radius(my_planet=np.array([[1.63,
                                                 0.034,
                                                 0.02,
                                                 0.337,
                                                 3505.0,
                                                 0.342]]),
                            my_name=np.array(['GJ 357 b']),
                            regr=regr,
                            jupiter_mass=False)
plt.show()
