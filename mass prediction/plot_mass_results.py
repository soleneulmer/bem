
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mass_models


# Load datasets including error columns
dataset_all = "data/exoplanet.eu_catalog_20-01-26_15_03_11.csv"
cat_solar = 'data/solar_system_planets_catalog.csv'

dataset = mass_models.load_dataset_errors(remove_bad_planets=False)

# otegi threshold data selection
selection_uncertainty = (
    (dataset['mass_error'] / dataset['mass'] < 0.25) &
    (dataset['radius_error'] / dataset['radius'] < 0.08)
)
print("\nNumber of removed planets due to Otegi uncertainty selection: ", len(dataset[~selection_uncertainty])  )
dataset = dataset[selection_uncertainty]

#remove K2-123 b 
planets_to_remove = ['K2-123 b']
dataset = dataset.drop(index=planets_to_remove, errors='ignore')

def plot_predicted_vs_true(
    rf_results,
    lgbm_results,
    xgb_results,
    log_scale=True,
    relative_residuals=True, 
    titel = "Mass Prediction Comparison; RF vs LightGBM vs XGBoost"):

    rf_model, rf_pred, rf_values, rf_sets, rf_metrics = rf_results
    lgbm_model, lgbm_pred, lgbm_values, lgbm_sets, lgbm_metrics = lgbm_results
    xgb_model, xgb_pred, xgb_values, xgb_sets, xgb_metrics = xgb_results

    y_test_rf = np.asarray(rf_sets[3])
    y_test_lgbm = np.asarray(lgbm_sets[3])
    y_test_xgb = np.asarray(xgb_sets[3])

    rf_pred = np.asarray(rf_pred)
    lgbm_pred = np.asarray(lgbm_pred)
    xgb_pred = np.asarray(xgb_pred)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1.7]}
    )

    models = [
        ("Random Forest", y_test_rf, rf_pred, rf_metrics),
        ("LightGBM", y_test_lgbm, lgbm_pred, lgbm_metrics),
        ("XGBoost", y_test_xgb, xgb_pred, xgb_metrics)
    ]

    markers = ['*', 'o', '^']

    all_true = []
    all_pred = []

    for (name, y_true, y_pred, metrics), marker in zip(models, markers):
        label = (
            rf"{name}: "
            rf"$\epsilon$={metrics['epsilon_log']:.3f}, "
            rf"$R^2_{{\rm train}}$={metrics['r2_train_log']:.3f}, "
            rf"$R^2_{{\rm test}}$={metrics['r2_test_log']:.3f}"
        )

        ax1.scatter(
            y_true,
            y_pred,
            alpha=0.7,
            label=label,
            marker=marker,
            s=30
        )

        if relative_residuals:
            residuals = (y_pred - y_true) / y_true
        else:
            residuals = y_true - y_pred

        ax2.scatter(
            y_true,
            residuals,
            alpha=0.7,
            label=name,
            marker=marker,
            s=30
        )

        all_true.extend(y_true)
        all_pred.extend(y_pred)

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

    ax1.set_ylabel(r"Predicted mass [$M_\oplus$]")
    ax1.set_title(titel)
    ax1.legend(loc="upper left", fontsize="small")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    ax2.set_xlabel(r"True mass [$M_\oplus$]")

    if relative_residuals:
        ax2.set_ylabel("Relative residual")
    else:
        ax2.set_ylabel(r"Residual [$M_\oplus$]")

    ax2.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/" + titel.replace(" ", "_").lower() + ".pdf", bbox_inches="tight")
    



if __name__ == "__main__":
    #### FULL DATA ###
    print("\nNumber of planets after cleaning: ", len(dataset) )
    
    # Train all three mass-prediction models.
    rf_results = mass_models.random_forest_mass(dataset, fit=True)
    lgbm_results = mass_models.lightgbm_mass(dataset, fit=True)
    xgb_results = mass_models.xgboost_mass(dataset, fit=True)

    # Plot predicted mass versus true mass for all three models.
    plot_predicted_vs_true(
        rf_results,
        lgbm_results,
        xgb_results,
        log_scale=True
    )

    # # Plot LIME explanations for all models
    # mass_models.plot_LIME_mass_predictions(
    #     lgbm_results[0],
    #     lgbm_results[3],
    #     model_name="LightGBM"
    # )       
    mass_models.plot_LIME_mass_predictions( 
        rf_results[0],
        rf_results[3],
        model_name="Random Forest"
    )
    

    # mass_models.plot_LIME_mass_predictions(
    #     xgb_results[0],
    #     xgb_results[3],
    #     model_name="XGBoost"
    # )

    ### MOUSAVI SPLIT ###

    # Train all three mass-prediction models on the Mousavi small and giant subsets.
    dataset_small_mousavi = dataset[dataset["mass"] < 52.48]
    dataset_giants_mousavi = dataset[dataset["mass"] >= 52.48]
    print("\nNumber of small planets: ", len(dataset_small_mousavi))
    print("Number of giant planets: ", len(dataset_giants_mousavi))

    rf_results_small_ = mass_models.random_forest_mass(dataset_small_mousavi, fit=True)
    rf_results_giants_ = mass_models.random_forest_mass(dataset_giants_mousavi, fit=True)
    lgbm_results_small_ = mass_models.lightgbm_mass(dataset_small_mousavi, fit=True)
    lgbm_results_giants_ = mass_models.lightgbm_mass(dataset_giants_mousavi, fit=True)
    xgb_results_small_ = mass_models.xgboost_mass(dataset_small_mousavi, fit=True)
    xgb_results_giants_ = mass_models.xgboost_mass(dataset_giants_mousavi, fit=True)

    # plot predicted mass versus true mass for all three models on the Mousavi small and giant subsets.
    plot_predicted_vs_true(
        rf_results_small_,
        lgbm_results_small_,
        xgb_results_small_,
        log_scale=True, 
        titel = "Mass Prediction Comparison on Mousavi Small Planets; RF vs LightGBM vs XGBoost"
    )   
    plot_predicted_vs_true(
        rf_results_giants_,
        lgbm_results_giants_,
        xgb_results_giants_,
        log_scale=True, 
        titel = "Mass Prediction Comparison on Mousavi Giant Planets; RF vs LightGBM vs XGBoost"
    )   

    # #plot LIME explanations for all three models on the Mousavi small and giant subsets.
    # mass_models.plot_LIME_mass_predictions(
    #     lgbm_results_small_[0],
    #     lgbm_results_small_[3], 
    #     planets=[0, 1, 2, 3, 4, 5]
    # )
    # mass_models.plot_LIME_mass_predictions(     
    #     rf_results_small_[0],
    #     rf_results_small_[3],   
    #     planets=[0, 1, 2, 3, 4, 5]
    # )

    # mass_models.plot_LIME_mass_predictions(
    #     xgb_results_small_[0],  
    #     xgb_results_small_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )       

    # mass_models.plot_LIME_mass_predictions(
    #     lgbm_results_giants_[0],    
    #     lgbm_results_giants_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )
    # mass_models.plot_LIME_mass_predictions( 
    #     rf_results_giants_[0],
    #     rf_results_giants_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )   
    # mass_models.plot_LIME_mass_predictions(
    #     xgb_results_giants_[0], 
    #     xgb_results_giants_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )   









    ### OTEGI SPLIT ###

    # Train three mass prediction models on the Otegi small, intermediate, and giant subsets.
    dataset_small = dataset[dataset["mass"] < 4.4]
    dataset_intermediate = dataset[(dataset["mass"] >= 4.4) & (dataset["mass"] < 127)]
    dataset_giants = dataset[dataset["mass"] >= 127]

    print("Number of small planets: ", len(dataset_small))
    print("Number of intermediate planets: ", len(dataset_intermediate))
    print("Number of giant planets: ", len(dataset_giants))

    rf_results_small = mass_models.random_forest_mass(dataset_small, fit=True)
    rf_results_intermediate = mass_models.random_forest_mass(dataset_intermediate, fit=True)
    rf_results_giants = mass_models.random_forest_mass(dataset_giants, fit=True)
    lgbm_results_small = mass_models.lightgbm_mass(dataset_small, fit=True)
    lgbm_results_intermediate = mass_models.lightgbm_mass(dataset_intermediate, fit=True)
    lgbm_results_giants = mass_models.lightgbm_mass(dataset_giants, fit=True)
    xgb_results_small = mass_models.xgboost_mass(dataset_small, fit=True)   
    xgb_results_intermediate = mass_models.xgboost_mass(dataset_intermediate, fit=True)
    xgb_results_giants = mass_models.xgboost_mass(dataset_giants, fit=True)

    # plot predicted mass versus true mass for all three models on the Otegi subsets.
    plot_predicted_vs_true(
        rf_results_small,
        lgbm_results_small,
        xgb_results_small,
        log_scale=True,
        titel="Mass Prediction Comparison on Otegi Small Planets; RF vs LightGBM vs XGBoost"
    )
    plot_predicted_vs_true(
        rf_results_intermediate,
        lgbm_results_intermediate,
        xgb_results_intermediate,
        log_scale=True,
        titel="Mass Prediction Comparison on Otegi Intermediate Planets; RF vs LightGBM vs XGBoost"
    )
    plot_predicted_vs_true(
        rf_results_giants,
        lgbm_results_giants,
        xgb_results_giants,
        log_scale=True,
        titel="Mass Prediction Comparison on Otegi Giant Planets; RF vs LightGBM vs XGBoost"
    )

    # #plot LIME explanations for all three models on the Mousavi small and giant subsets.
    # mass_models.plot_LIME_mass_predictions(
    #     lgbm_results_small_[0],
    #     lgbm_results_small_[3], 
    #     planets=[0, 1, 2, 3, 4, 5]
    # )
    # mass_models.plot_LIME_mass_predictions(     
    #     rf_results_small_[0],
    #     rf_results_small_[3],   
    #     planets=[0, 1, 2, 3, 4, 5]
    # )

    # mass_models.plot_LIME_mass_predictions(
    #     xgb_results_small_[0],  
    #     xgb_results_small_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )       

    # mass_models.plot_LIME_mass_predictions(
    #     lgbm_results_intermediate[0],
    #     lgbm_results_intermediate[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )   

    # mass_models.plot_LIME_mass_predictions(
    #     rf_results_intermediate[0],     
    #     rf_results_intermediate[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )       

    # mass_models.plot_LIME_mass_predictions(
    #     xgb_results_intermediate[0],    
    #     xgb_results_intermediate[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )   

    # mass_models.plot_LIME_mass_predictions(
    #     lgbm_results_giants_[0],    
    #     lgbm_results_giants_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )
    # mass_models.plot_LIME_mass_predictions( 
    #     rf_results_giants_[0],
    #     rf_results_giants_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )   
    # mass_models.plot_LIME_mass_predictions(
    #     xgb_results_giants_[0], 
    #     xgb_results_giants_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )   


    # plt.show()