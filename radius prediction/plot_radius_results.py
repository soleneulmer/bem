from bem import random_forest_regression
from matplotlib import pyplot as plt
import time
import bem
import lgbm_xgb 
import numpy as np
time1 = time.time()


dataset_all = "data/exoplanet.eu_catalog_20-01-26_15_03_11.csv"
cat_solar = 'data/solar_system_planets_catalog.csv'

dataset = bem.load_dataset_errors(remove_bad_planets=True)

# otegi threshold data selection
selection_uncertainty = (
    (dataset['mass_error'] / dataset['mass'] < 0.25) &
    (dataset['radius_error'] / dataset['radius'] < 0.08)
)
print("Number of removed planets due to Otegi uncertainty selection: ", len(dataset[~selection_uncertainty])  )
dataset = dataset[selection_uncertainty]

# Remove manually excluded planet
planets_to_remove = ["K2-123 b"]
dataset = dataset.drop(index=planets_to_remove, errors="ignore")

print("Number of planets after cleaning: ", len(dataset) )





# radius predictions full dataset on all three models
regr_lgbm, y_test_pred_lgbm, train_test_values_lgbm, train_test_sets_lgbm, lgbm_metrics = lgbm_xgb.lightgbm(
    dataset,  
    model = None,
    fit=True
)

regr_xgb, y_test_pred_xgb, train_test_values_xgb, train_test_sets_xgb, xgb_metrics = lgbm_xgb.xgboost(
    dataset,
    model = None,
    fit=True
)
regr, y_test_pred, train_test_values, train_test_sets, rf_metrics = random_forest_regression(
    dataset,    
    model = None,
    fit=True
)



def plot_pred_true(models, log_scale=False, relative_residuals=False):
    """
    Plot true vs predicted radius and residuals.
    """

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.7]}
    )

    all_true = []
    all_pred = []
    markers = ["*", "o", "^"]

    for (name, train_test_sets, y_test_pred, metrics), marker in zip(models, markers):
        _, _, _, y_test = train_test_sets

        y_test = np.asarray(y_test)
        y_test_pred = np.asarray(y_test_pred)

        label = (
            rf"{name}: "
            rf"RMSE$_{{\rm test}}$={metrics['rmse_test']:.3f}, "
            rf"$R^2_{{\rm train}}$={metrics['r2_train']:.3f}, "
            rf"$R^2_{{\rm test}}$={metrics['r2_test']:.3f}"
        )

        ax1.scatter(
            y_test,
            y_test_pred,
            alpha=0.7,
            label=label,
            marker=marker,
            s=30
        )

        if relative_residuals:
            residuals = (y_test_pred - y_test) / y_test
        else:
            residuals = y_test - y_test_pred

        ax2.scatter(
            y_test,
            residuals,
            alpha=0.7,
            marker=marker,
            s=30
        )

        all_true.extend(y_test)
        all_pred.extend(y_test_pred)

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)

    min_value = min(all_true.min(), all_pred.min())
    max_value = max(all_true.max(), all_pred.max())

    ax1.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Perfect prediction"
    )

    ax2.axhline(
        0,
        color="red",
        linestyle="--",
        linewidth=2
    )

    if log_scale:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")

    ax1.set_ylabel(r"Predicted radius [$R_\oplus$]")
    ax1.set_title("Radius Prediction Comparison: RF vs LightGBM vs XGBoost")
    ax1.legend(loc="upper left", fontsize="small")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    ax2.set_xlabel(r"True radius [$R_\oplus$]")

    if relative_residuals:
        ax2.set_ylabel("Relative residual")
    else:
        ax2.set_ylabel(r"Residual [$R_\oplus$]")

    ax2.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/results_radius.pdf", bbox_inches="tight")
    

plot_pred_true([
    ("Random Forest", train_test_sets, y_test_pred, rf_metrics),
    ("LightGBM", train_test_sets_lgbm, y_test_pred_lgbm, lgbm_metrics),
    ("XGBoost", train_test_sets_xgb, y_test_pred_xgb, xgb_metrics),
])

# # Explain the models predictions
# bem.plot_LIME_predictions(regr_lgbm, dataset, train_test_sets_lgbm, model_name="LightGBM")
# bem.plot_LIME_predictions(regr_xgb, dataset, train_test_sets_xgb, model_name="XGBoost")
# bem.plot_LIME_predictions(regr, dataset, train_test_sets, model_name="Random Forest")




# radius predictions using LightGBM in three regimes based on Otegi et al. 2020 
dataset_small = dataset[dataset["mass"] < 4.4]
dataset_intermediate = dataset[(dataset["mass"] >= 4.4) & (dataset["mass"] < 127)]
dataset_giant = dataset[dataset["mass"] >= 127]

print("Number of small planets: ", len(dataset_small))
regr_small, y_test_pred_small, train_test_values_small, train_test_sets_small, lgbm_metrics_small = lgbm_xgb.lightgbm(
    dataset_small,  
    model = None,
    fit=True
)


print("Number of intermediate planets: ", len(dataset_intermediate))
regr_intermediate, y_test_pred_intermediate, train_test_values_intermediate, train_test_sets_intermediate, lgbm_metrics_intermediate = lgbm_xgb.lightgbm(
    dataset_intermediate,
    model = None,
    fit=True
)

print("Number of giant planets: ", len(dataset_giant))
regr_giant, y_test_pred_giant, train_test_values_giant, train_test_sets_giant, lgbm_metrics_giant = lgbm_xgb.lightgbm(
    dataset_giant,
    model = None,
    fit=True
)   

# radius predictions using LightGBM in two regimes based on Mousavi et al. 2020
dataset_small_mousavi = dataset[dataset["mass"] < 52.48]
dataset_giants_mousavi = dataset[dataset["mass"] >= 52.48]

print("Number of small planets: ", len(dataset_small_mousavi))
regr_small_mousavi, y_test_pred_small_mousavi, train_test_values_small_mousavi, train_test_sets_small_mousavi, lgbm_metrics_small = lgbm_xgb.lightgbm(
    dataset_small_mousavi,     
    model = None,
    fit=True
)   
print("Number of giant planets: ", len(dataset_giants_mousavi))
regr_giants_mousavi, y_test_pred_giants_mousavi, train_test_values_giants_mousavi, train_test_sets_giants_mousavi, lgbm_metrics_giants = lgbm_xgb.lightgbm(
    dataset_giants_mousavi, 
    model = None,
    fit=True
)

#now for XGBoost
# radius predictions using LightGBM in three regimes based on Otegi et al. 2020 
dataset_small = dataset[dataset["mass"] < 4.4]
dataset_intermediate = dataset[(dataset["mass"] >= 4.4) & (dataset["mass"] < 127)]
dataset_giant = dataset[dataset["mass"] >= 127]

print("Number of small planets: ", len(dataset_small))
regr_small, y_test_pred_small, train_test_values_small, train_test_sets_small, lgbm_metrics_small = lgbm_xgb.xgboost(
    dataset_small,  
    model = None,
    fit=True
)

print("Number of intermediate planets: ", len(dataset_intermediate))
regr_intermediate, y_test_pred_intermediate, train_test_values_intermediate, train_test_sets_intermediate, lgbm_metrics_intermediate = lgbm_xgb.xgboost(
    dataset_intermediate,
    model = None,
    fit=True
)

print("Number of giant planets: ", len(dataset_giant))
regr_giant, y_test_pred_giant, train_test_values_giant, train_test_sets_giant, lgbm_metrics_giant = lgbm_xgb.xgboost(
    dataset_giant,
    model = None,
    fit=True
)   

# radius predictions using LightGBM in two regimes based on Mousavi et al. 2020
dataset_small_mousavi = dataset[dataset["mass"] < 52.48]
dataset_giants_mousavi = dataset[dataset["mass"] >= 52.48]

print("Number of small planets: ", len(dataset_small_mousavi))
regr_small_mousavi, y_test_pred_small_mousavi, train_test_values_small_mousavi, train_test_sets_small_mousavi, lgbm_metrics_small = lgbm_xgb.xgboost(
    dataset_small_mousavi,     
    model = None,
    fit=True
)   
print("Number of giant planets: ", len(dataset_giants_mousavi))
regr_giants_mousavi, y_test_pred_giants_mousavi, train_test_values_giants_mousavi, train_test_sets_giants_mousavi, lgbm_metrics_giants = lgbm_xgb.xgboost(
    dataset_giants_mousavi, 
    model = None,
    fit=True
)

#now for random forest
# radius predictions using LightGBM in three regimes based on Otegi et al. 2020 
dataset_small = dataset[dataset["mass"] < 4.4]
dataset_intermediate = dataset[(dataset["mass"] >= 4.4) & (dataset["mass"] < 127)]
dataset_giant = dataset[dataset["mass"] >= 127]

print("Number of small planets: ", len(dataset_small))
regr_small, y_test_pred_small, train_test_values_small, train_test_sets_small, rf_metrics_small = bem.random_forest_regression(
    dataset_small,  
    model = None,
    fit=True
)

print("Number of intermediate planets: ", len(dataset_intermediate))
regr_intermediate, y_test_pred_intermediate, train_test_values_intermediate, train_test_sets_intermediate, rf_metrics_intermediate = bem.random_forest_regression(
    dataset_intermediate,
    model = None,
    fit=True
)

print("Number of giant planets: ", len(dataset_giant))
regr_giant, y_test_pred_giant, train_test_values_giant, train_test_sets_giant, rf_metrics_giant = bem.random_forest_regression( 
    dataset_giant,
    model = None,
    fit=True
)   

# radius predictions using LightGBM in two regimes based on Mousavi et al. 2020
dataset_small_mousavi = dataset[dataset["mass"] < 52.48]
dataset_giants_mousavi = dataset[dataset["mass"] >= 52.48]

print("Number of small planets: ", len(dataset_small_mousavi))
regr_small_mousavi, y_test_pred_small_mousavi, train_test_values_small_mousavi, train_test_sets_small_mousavi, rf_metrics_small = bem.random_forest_regression(
    dataset_small_mousavi,     
    model = None,
    fit=True
)   
print("Number of giant planets: ", len(dataset_giants_mousavi))
regr_giants_mousavi, y_test_pred_giants_mousavi, train_test_values_giants_mousavi, train_test_sets_giants_mousavi, rf_metrics_giants = bem.random_forest_regression(
    dataset_giants_mousavi, 
    model = None,
    fit=True
)


plt.show()