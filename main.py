import pandas as pd
from bem import load_dataset, load_dataset_errors, random_forest_regression
from matplotlib import pyplot as plt
import time
import numpy as np
import format_dataset as fd
import bem
import lgbm_xgb 
time1 = time.time()


dataset_all = "data/exoplanet.eu_catalog_20-01-26_15_03_11.csv"
cat_solar = 'data/solar_system_planets_catalog.csv'


dataset = bem.load_dataset(remove_bad_planets=False)
dataset = bem.load_dataset_errors(remove_bad_planets=False, reference_dataset=dataset)
# otegi threshold data selection
selection_uncertainty = (
    (dataset['mass_error'] / dataset['mass'] < 0.25) &
    (dataset['radius_error'] / dataset['radius'] < 0.08)
)
dataset = dataset[selection_uncertainty]

#remove K2-123 b 
planets_to_remove = ['K2-123 b']
dataset = dataset.drop(index=planets_to_remove, errors='ignore')

# remove planets with uncertainties larger than the value of the parameter itself
bad_planets = set()
for col in dataset.columns:
    if col.endswith("_error"):
        base_col = col[:-6]
        if base_col in dataset.columns:
            mask = dataset[col] >= dataset[base_col]
            bad_planets.update(dataset.index[mask])

# save to a txt file
with open("data/bad_planets.txt", "w", encoding="utf-8") as f:
    for planet in sorted(bad_planets):
        f.write(planet + "\n")

for planet in bad_planets:
    if planet in dataset.index:
        dataset = dataset.drop(labels=planet)

# bem.plot_dataset(dataset=dataset)

# #select planet with lowest masses 
# dataset = dataset.sort_values(by='mass').head(20)
# print(dataset)

regr, y_test_predict, train_test_values, train_test_sets = bem.random_forest_regression(
    dataset=dataset,
    model=None,
    fit=True
)

regr_lgbm, y_test_pred_lgbm, train_test_values_lgbm, train_test_sets_lgbm = lgbm_xgb.lightgbm(
    dataset, 
    model = None,
    fit=True)


regr_xgb, y_test_pred_xgb, train_test_values_xgb, train_test_sets_xgb = lgbm_xgb.xgboost(
    dataset,  
    model = None,      
    fit=True)


# Explain the models predictions
bem.plot_LIME_predictions(regr_lgbm, dataset, train_test_sets_lgbm, model_name="LightGBM")
bem.plot_LIME_predictions(regr_xgb, dataset, train_test_sets_xgb, model_name="XGBoost")
bem.plot_LIME_predictions(regr, dataset, train_test_sets, model_name="Random Forest")
plt.show()

