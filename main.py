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



dataset = load_dataset(cat_exoplanet=dataset_all,
    cat_solar=cat_solar,remove_shit_planets=True)


# Remove planets where error >= measured value and save in txt file
bad_planets = []

for col in dataset.columns:
    if col.endswith("_error"):
        base_col = col[:-6]
        if base_col in dataset.columns:
            mask = dataset[col] >= dataset[base_col]
            for planet in dataset.index[mask]:
                if planet not in bad_planets:
                    bad_planets.append(planet)

with open("data/shit_planets.txt", "w", encoding="utf-8") as f:
    for planet in bad_planets:
        f.write(f"{planet}\n")

print(f"The length of the bad_planets list is: {len(bad_planets)}")
dataset = dataset.drop(index=bad_planets)





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


