# BEM: Beyond the Exoplanet Mass-Radius Relation with Machine Learning

Predicting exoplanet radii and masses from planetary and stellar parameters using machine learning.

<img src="https://github.com/soleneulmer/bem/raw/master/figures/Bem.png" width="200">

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/soleneulmer/bem/LICENSE)

---

## Bachelor Thesis
### *Exploring the Properties of Exoplanets using Machine Learning*

**Donna Aardoom** and **Sebastiaan Louwerse**

Leiden Observatory, Leiden University

---

## Overview

This repository contains the code used in our Bachelor Research Project (BRP), which investigates whether machine learning models can predict exoplanet radii and masses from a combination of planetary and stellar parameters.

Three regression algorithms are implemented and compared:

- Random Forest Regression (RFR)
- XGBoost
- LightGBM

The models are trained on confirmed exoplanets from the Extrasolar Planets Encyclopaedia and are accompanied by diagnostic plots and explainability methods using LIME.

---

## Features

- Radius prediction
- Mass prediction
- Random Forest, XGBoost and LightGBM models
- Feature importance analysis
- Learning curves and validation curves
- Residual analysis
- LIME explanations
- Uncertainty propagation
- Outlier detection using Local Outlier Factor (LOF)

---

# Radius Prediction

## 1. Train a model

Build a machine learning model and predict exoplanet radii:

```python
regr, y_test_predict, _, train_test_sets = bem.random_forest_regression(dataset)
```

---

## 2. Predict the radius of a new planet

```python
radius, my_pred_planet = bem.predict_radius(
    my_planet=np.array([[1.63,
                         0.034,
                         0.02,
                         0.337,
                         3505.0,
                         0.342]]),
    my_name=np.array(['GJ 357 b']),
    regr=regr,
    jupiter_mass=False,
    error_bar=False
)
```

If `error_bar=True`, uncertainties on the predicted radius are computed.

---

## 3. Compute prediction uncertainties

```python
# Load exoplanet dataset with uncertainties
dataset_errors = bem.load_dataset_errors()

# Compute the error bars for the test-set planets
radii_test_output_error, _ = bem.computing_errorbars(
    regr,
    dataset_errors,
    train_test_sets
)

# Plot true radius versus predicted radius
bem.plot_true_predicted(
    train_test_sets,
    y_test_predict,
    radii_test_output_error
)
```

---

## 4. Diagnostic plots

```python
# Plot the learning curve
bem.plot_learning_curve(regr, dataset)

# Plot validation curves
bem.plot_validation_curves(regr, dataset, name='features')
bem.plot_validation_curves(regr, dataset, name='tree')
bem.plot_validation_curves(regr, dataset, name='depth')
```

---

## 5. LIME explanations

LIME (Local Interpretable Model-Agnostic Explanations) is used to explain individual model predictions.

See the original repository:

https://github.com/marcotcr/lime

Explain the predictions for planets in the test set:

```python
bem.plot_LIME_predictions(
    regr,
    dataset,
    train_test_sets
)
```

Generate a LIME explanation for a specific planet:

```python
bem.plot_LIME_predictions(
    regr,
    dataset,
    train_test_sets,
    my_pred_planet=my_pred_planet,
    my_true_radius=1.166
)
```

---

# Mass Prediction

Mass prediction is implemented analogously to radius prediction. Models are trained in logarithmic mass space and evaluated using:

- Test-set coefficient of determination ($R^2$)
- Mean fractional error ($\epsilon$)
- Mean predicted-to-actual ratio

Experiments include:

- Full dataset training
- Two-regime mass split
- Three-regime mass split

---

# Explainability and Analysis

The repository also includes:

- Feature importance analysis
- Residual plots
- Learning curves
- Validation curves
- LIME explanations
- Local Outlier Factor (LOF) outlier detection
- Comparison with previous work

---

# Dataset

Confirmed exoplanets are obtained from:

**The Extrasolar Planets Encyclopaedia**

http://exoplanet.eu

The data are filtered based on observational uncertainties before being used for training and testing.

---

# Dependencies

The project uses:

- numpy
- pandas
- matplotlib
- scikit-learn
- xgboost
- lightgbm
- lime
- scipy

Install the required packages with:

```bash
pip install -r requirements.txt
```

---

# Reference

This work was performed as part of the Bachelor Research Project at Leiden University.

> **Exploring the Properties of Exoplanets using Machine Learning**  
> Donna Aardoom and Sebastiaan Louwerse  
> Leiden Observatory, Leiden University (2026)

---

