# Light GBM and XGBoost regression functions for the BEM project

#import packages 
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import datetime
import joblib
import os
import pandas as pd
import numpy as np



# split dataset fundtion to split into train and test sets, while forcing a few named planets into the test set

def split_data(dataset):
    dataset = dataset.copy()

    features_needed = [
        'mass',
        'semi_major_axis',
        'temp_eq',
        'star_luminosity',
        'star_radius',
        'star_teff',
        'star_mass'
    ]

    features = dataset[features_needed]
    label = dataset['radius']

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        label,
        test_size=0.25,
        random_state=9
    )

    forced_for_train = [
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    'Kepler-12 b', 'WASP-94 Ab', 'HAT-P-47 b',
    'kappa And b', 'HD 984 b', 'Kepler-16 (AB)b', 'Kepler-34 (AB)b', 'KELT-9 Ab'
    ]
    forced_for_test = [ 
        # # uncomment for the LIME plots
        #     "TOI-561 b",
        #     "HATS-35 b",
        #     "CoRoT-13 b",
        #     "Kepler-75 b",
        #     "WASP-17 b",
        #     "Kepler-20 Ac",
    'HAT-P-32 Ab', 'TOI-3071 b', 'WASP-122 Ab', 'TOI-3976 Ab', 'Kepler-4 b',
    'LHS 1140 b', 'Gliese 12 b', 'TOI-1231 b',
    'TOI-201 c', 'TOI-561 f', 'HIP 41378 f',
    ]
    for name in forced_for_train:
        if name in X_test.index and name not in X_train.index:
            X_train = pd.concat([X_train, X_test.loc[[name]]])
            y_train = pd.concat([y_train, y_test.loc[[name]]])
            X_test = X_test.drop(index=name)
            y_test = y_test.drop(index=name)
            swap_candidates = [
                idx for idx in X_train.index
                if idx not in forced_for_train and idx != name
            ]
            if len(swap_candidates) > 0:
                swap_name = swap_candidates[0]
                X_test = pd.concat([X_test, X_train.loc[[swap_name]]])
                y_test = pd.concat([y_test, y_train.loc[[swap_name]]])
                X_train = X_train.drop(index=swap_name)
                y_train = y_train.drop(index=swap_name)

    for name in forced_for_test:
        if name in X_train.index and name not in X_test.index:
            X_test = pd.concat([X_test, X_train.loc[[name]]])
            y_test = pd.concat([y_test, y_train.loc[[name]]])
            X_train = X_train.drop(index=name)
            y_train = y_train.drop(index=name)
            swap_candidates = [
                idx for idx in X_test.index
                if idx not in forced_for_test and idx != name
            ]
            if len(swap_candidates) > 0:
                swap_name = swap_candidates[0]
                X_train = pd.concat([X_train, X_test.loc[[swap_name]]])
                y_train = pd.concat([y_train, y_test.loc[[swap_name]]])
                X_test = X_test.drop(index=swap_name)
                y_test = y_test.drop(index=swap_name)

    
    train_test_values = [
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values
    ]
    train_test_sets = [X_train, X_test, y_train, y_test]
    return features_needed, X_train, X_test, y_train, y_test, train_test_values, train_test_sets

def lightgbm(dataset, model=None, fit=False):
    """
    LightGBM regressor

    Returns:
        lgbm, y_test_predict, train_test_values, train_test_sets
    """
    # Load the dataset and split into train/test sets
    features_needed, X_train, X_test, y_train, y_test, train_test_values, train_test_sets = split_data(dataset)

    print('\nDataset loaded and split into train/test sets. Starting LightGBM regression...')
    # load the model if it exists, otherwise fit a new one 
    if fit:
        # set a wide range of hyperparameters to search over for the LightGBM regressor
        params_grid = {
            'n_estimators': [100, 150, 200, 300, 500],
            'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1],
            'max_depth': [2, 3, 4, 5],
            'num_leaves': [3, 5, 7, 10, 15],
            'min_child_samples': [5, 10, 20, 40],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 1.0]
        }
        # use randomized search to find the best hyperparameters for the LightGBM regressor
        lgbm_cv = RandomizedSearchCV(
            LGBMRegressor(verbose=-1, random_state=9),
            param_distributions=params_grid,
            n_iter=80,
            scoring='neg_root_mean_squared_error',
            cv=3,
            verbose=1,
            n_jobs=-1,
            random_state=9,
        )
        
        # fit the model to the training data and print the best hyperparameters
        lgbm_cv.fit(X_train, y_train)
        print("\nBest paramaters LightGBM:")
        for param, value in lgbm_cv.best_params_.items():
            print(f"{param}: {value}")

        # train the LightGBM regressor with the best hyperparameters
        lgbm = lgbm_cv.best_estimator_
        
        # save the LightGBM model in a file with the R-2 score and the date in the name
        outdir = 'bem_output'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        name_lgbm = 'lgbm_r2_' +str(round(lgbm_cv.best_score_, 2)) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H")) + '.pkl'
        name_lgbm = os.path.join(outdir, name_lgbm)
        
        print('\nLightGBM model will be saved in:', name_lgbm)

        # train the regressor on the training data and save it in a file
        print("\nTraining LightGBM regressor...")
        lgbm.fit(X_train, y_train)
        joblib.dump(lgbm, name_lgbm)
        
    else:
        # if no model is provided, use the default saved model
        print("\nLoading LightGBM model: ", model)
        lgbm = joblib.load(model)


    # predict the radius for the train and test sets
    y_train_predict = lgbm.predict(X_train)
    y_test_predict = lgbm.predict(X_test)

    # Evaluate
    print("\nEvaluation of LightGBM regressor:")
    print(f"Train R²:   {r2_score(y_train, y_train_predict):.4f}")
    print(f"Test  R²:   {r2_score(y_test,  y_test_predict):.4f}")
    print(f"Train RMSE: {root_mean_squared_error(y_train, y_train_predict):.4f}")
    print(f"Test  RMSE: {root_mean_squared_error(y_test,  y_test_predict):.4f}")
    print(f"Mean ratio: {np.mean(y_test_predict / y_test):.4f}")

    print('\nFeature importance')
    importances = lgbm.feature_importances_
    importances_percent = 100 * importances / importances.sum()

    for name, value in zip(features_needed, importances_percent):
        print(f'{name}: {value:.2f}%')
        
    
    metrics = {
        "r2_test": r2_score(y_test, y_test_predict),
        "r2_train": r2_score(y_train, y_train_predict),
        "rmse_train": root_mean_squared_error(y_train, y_train_predict),
        "rmse_test": root_mean_squared_error(y_test, y_test_predict),
        "mean_ratio": np.mean(y_test_predict / y_test)}

    return lgbm, y_test_predict, train_test_values, train_test_sets, metrics

def xgboost(dataset, model=None, fit=False):
    '''
    XGBoost regressor

    Returns:
        regr, y_test_predict, train_test_values, train_test_sets
    '''
    # Load the dataset and split into train/test sets
    features_needed, X_train, X_test, y_train, y_test, train_test_values, train_test_sets = split_data(dataset)

    print('\nDataset loaded and split into train/test sets. Starting XGBoost regression...')

     # load the model if it exists, otherwise fit a new one 
    if fit:
        # set a wide range of hyperparameters to search over for the XGBoost regressor
        params_grid = {
            "n_estimators": [50, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500],
            "max_depth": [2, 3, 4, 5, 6],  
            "learning_rate": [0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
            "subsample": [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],  
            "colsample_bytree": [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],  
            "reg_lambda": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0],
            "reg_alpha": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],  
            "min_child_weight": [3, 5, 7, 10, 15, 20, 30, 50],  
            "gamma": [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0],  
            "colsample_bylevel": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  
        }
        # use RandomizedSearchCV to find the best hyperparameters for the XGBoost regressor
        xgb_cv = RandomizedSearchCV(
            XGBRegressor(objective="reg:absoluteerror", tree_method="hist"),
            param_distributions=params_grid,
            n_iter=40,
            cv=5,
            scoring="r2",
            verbose=1,
            n_jobs=-1,
            random_state=9,
            return_train_score=True
        )

        # fit the model to the training data and print the best hyperparameters
        xgb_cv.fit(X_train, y_train)
        print("\nBest parameters XGBoost:")
        for param, value in xgb_cv.best_params_.items():
            print(f"{param}: {value}")

        # train the XGBoost regressor with the best hyperparameters
        xgb = xgb_cv.best_estimator_
        
        # save the XGBoost model in a file with the R-2 score and the date in the name
        outdir = 'bem_output'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        name_xgb = 'xgb_r2_' +str(round(xgb_cv.best_score_, 2)) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H")) + '.pkl'
        name_xgb = os.path.join(outdir, name_xgb)
        
        print('\nXGBoost model will be saved in:', name_xgb)

        # train the regressor on the training data and save it in a file
        print("\nTraining XGBoost regressor...")
        xgb.fit(X_train, y_train)
        joblib.dump(xgb, name_xgb)

    else:
        # if no model is provided, use the default saved model
        print("\nLoading XGBoost model: ", model)
        xgb = joblib.load(model)

    
    # predict the radius for the train and test sets
    y_train_predict = xgb.predict(X_train)
    y_test_predict = xgb.predict(X_test)

    # Evaluate
    print("\nEvaluation of XGBoost regressor:")
    print(f"Train R²:   {r2_score(y_train, y_train_predict):.4f}")
    print(f"Test  R²:   {r2_score(y_test,  y_test_predict):.4f}")
    print(f"Train RMSE: {root_mean_squared_error(y_train, y_train_predict):.4f}")
    print(f"Test  RMSE: {root_mean_squared_error(y_test,  y_test_predict):.4f}")
    print(f"Mean ratio: {np.mean(y_test_predict / y_test):.4f}")

    print('\nFeature importance')
    importances = xgb.feature_importances_
    importances_percent = 100 * importances / importances.sum()

    for name, value in zip(features_needed, importances_percent):
        print(f'{name}: {value:.2f}%')  

    metrics = {"r2_test": r2_score(y_test, y_test_predict),
    "r2_train": r2_score(y_train, y_train_predict),
    "rmse_train": root_mean_squared_error(y_train, y_train_predict),
    "rmse_test": root_mean_squared_error(y_test, y_test_predict),
    "mean_ratio": np.mean(y_test_predict / y_test)}

    return xgb, y_test_predict, train_test_values, train_test_sets, metrics
