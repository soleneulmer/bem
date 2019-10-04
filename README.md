## BEM :  beyond the exoplanet mass-radius relation with random forest
Predicting the radius of exoplanets based on its planetary and stellar parameters

<img src="https://github.com/soleneulmer/bem/raw/master/figures/Bem.png" width="200">

[![Build Status](https://travis-ci.org/soleneulmer/bem.svg?branch=master)](https://travis-ci.org/soleneulmer/bem)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/soleneulmer/bem/LICENSE)
[![PyPI version](https://badge.fury.io/py/bem.svg)](https://badge.fury.io/py/bem)
[![arXiv](https://img.shields.io/badge/arXiv-1909.07392-%23B31B1B)](https://arxiv.org/abs/1909.07392)

### Branca Edmée Marques
A portuguese scientist who worked on nuclear physics in France with Marie Curie


### To install bem
```bash
pip install bem
```
or
```bash
git clone https://github.com/soleneulmer/bem.git
cd bem
python setup.py install
```

### A simple decision tree
#### to predict exoplanet radius

<img src="https://github.com/soleneulmer/bem/raw/master/figures/decision_tree.png" width="200">

### How to run bem:
#### 1. Load dataset and model
```bash
# Load exoplanet and solar system planets dataset
dataset = bem.load_dataset()
# Plot the dataset radius as a function of mass and equilibrium temperature
bem.plot_dataset(dataset)
```
```bash
# Build the random forest model and predict radius of the dataset
regr, y_test_predict, _, train_test_sets = bem.random_forest_regression(dataset)
```
#### 2. Predict the radius of your planet

my_planet = [planetary_mass,
             semi major axis,
             eccentricity,
             stellar radius,
             stellar effective temperature,
             stellar mass]

or with errors

my_planet = [planetary_mass, error,
             semi major axis, error
             eccentricity, error,
             stellar radius, error,
             stellar effective temperature, error,
             stellar mass, error]
```bash
# Predict a new radius
radius, my_pred_planet = bem.predict_radius(my_planet=np.array([[1.63,
								 0.034,
                                                 		 0.02,
                                                 		 0.337,
                                                 		 3505.0,
                                                 		 0.342]]),
                        		    my_name=np.array(['GJ 357 b']),
                            		    regr=regr,
                            		    jupiter_mass=False,
					    error_bar=False)
# If error_bar is True
# print('Radius: ', radius[0][0], '+-', radius[1])
```

#### 3. Compute error bars for the radius predictions
```bash
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

```

#### 4. Radial velocity dataset
```bash
# Load the radial velocity dataset
dataset_rv = bem.load_dataset_RV()
# Predict the radius of the RV dataset
radii_RV_RF = regr.predict(dataset_rv)
# Plot the predictions of the RV dataset
bem.plot_dataset(dataset_rv, predicted_radii=radii_RV_RF, rv=True)
```

#### 5. Diagnostic plots
```bash
# Plot the learning curve
bem.plot_learning_curve(regr, dataset)
# Plot the validation curves
bem.plot_validation_curves(regr, dataset, name='features')
bem.plot_validation_curves(regr, dataset, name='tree')
bem.plot_validation_curves(regr, dataset, name='depth')
```

#### 6. LIME explanations 
see their [github](https://github.com/marcotcr/lime)
```bash
# Explain the RF predictions
# of the exoplanets from the test set
bem.plot_LIME_predictions(regr, dataset, train_test_sets)
# LIME explanation for your planet
# in this case GJ 357 b
bem.plot_LIME_predictions(regr, dataset, train_test_sets,
                          my_pred_planet=my_pred_planet,
                          my_true_radius=1.166)
```
