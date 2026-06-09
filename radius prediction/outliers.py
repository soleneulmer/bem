
"""
This file contains the code that removes outliers in different ways;

- LOF (Local Outlier Factor) in two ways: on the whole dataset and in three mass regimes (small, intermediate and giant planets)
- Otegi qualitative selection including K2-123 b
- Handpicked outlier selection based on the mass-radius diagram 
"""
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import os
import bem 

print("##### Now is LOF code #####")

# Load data
dataset_all = "data/exoplanet.eu_catalog_20-01-26_15_03_11.csv"
cat_solar = "data/solar_system_planets_catalog.csv"


dataset = bem.load_dataset(
    cat_exoplanet=dataset_all,
    cat_solar=cat_solar,  
    solar=True
)

dataset_error = bem.load_dataset_errors(
    cat_exoplanet=dataset_all,
    cat_solar=cat_solar, 
    remove_bad_planets=False,
    solar=True
)
dataset_error = dataset_error.loc[dataset.index]



print(f"Dataset columns: {dataset.columns.tolist()}, with length of {len(dataset)}")
print(f"The amount of columns we use is: {len(dataset.columns.tolist())}")

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---------------------------
# LOF on whole dataset
# ---------------------------
model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred = model.fit_predict(dataset)
lof = model.negative_outlier_factor_
print( "##### Now is LOF code for the whole dataset #####")
print(f"The mean LOF score for the outliers is: {np.mean(lof[y_pred == -1] * -1)}")
print(f"The mean LOF score for the inliers is: {np.mean(lof[y_pred == 1] * -1)}")

LOF_outliers = dataset[y_pred == -1]
LOF_inliers = dataset[y_pred == 1]

print(f"Number of outliers detected: {len(LOF_outliers)}")
print(f"Number of inliers detected: {len(LOF_inliers)}")

som = len(LOF_outliers) + len(LOF_inliers)
assert som == len(dataset), f"Total number of points should be {len(dataset)}, but got {som}"

fig, ax = plt.subplots()
scatter0 = ax.scatter(
    LOF_inliers["mass"],
    LOF_inliers["radius"],
    label="Normal points",
    alpha=0.5,
    zorder=2
)
scatter1 = ax.scatter(
    LOF_outliers["mass"],
    LOF_outliers["radius"],
    label="Outliers",
    zorder=3,
    edgecolor="black"
)
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Mass ($M_\oplus$)")
ax.set_ylabel(r"Radius ($R_\oplus$)")
ax.set_title("LOF Outlier Detection excluding uncertainties: Mass vs Radius")
plt.tight_layout()
plt.savefig("figures/LOF_outliers_excl_uncer.pdf", dpi=300)

# ---------------------------
# LOF on whole dataset INCLUDING uncertainties
# ---------------------------
model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred = model.fit_predict(dataset_error)
lof = model.negative_outlier_factor_
print("\n##### Now is LOF code for the whole dataset including uncertainties #####")

print(f"The mean LOF score for the outliers is : {np.mean(lof[y_pred == -1] * -1)}")
print(f"The mean LOF score for the inliers is: {np.mean(lof[y_pred == 1] * -1)}")

LOF_outliers = dataset_error[y_pred == -1]
LOF_inliers = dataset_error[y_pred == 1]

print(f"Number of outliers detected: {len(LOF_outliers)}")
print(f"Number of inliers detected: {len(LOF_inliers)}")

som = len(LOF_outliers) + len(LOF_inliers)
assert som == len(dataset_error), f"Total number of points should be {len(dataset_error)}, but got {som}"

fig, ax = plt.subplots()
scatter0 = ax.scatter(
    LOF_inliers["mass"],
    LOF_inliers["radius"],
    label="Normal points",
    alpha=0.5,
    zorder=2
)
scatter1 = ax.scatter(
    LOF_outliers["mass"],
    LOF_outliers["radius"],
    label="Outliers",
    zorder=3,
    edgecolor="black"
)
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Mass ($M_\oplus$)")
ax.set_ylabel(r"Radius ($R_\oplus$)")
ax.set_title("LOF Outlier Detection including uncertainties: Mass vs Radius")
plt.tight_layout()
plt.savefig("figures/LOF_outliers_incl_uncer.pdf", dpi=300)


# ---------------------------
# LOF in three mass regimes
# ---------------------------
dataset_small = dataset[dataset["mass"] < 4.4]
dataset_intermediate = dataset[(dataset["mass"] >= 4.4) & (dataset["mass"] < 127)]
dataset_giant = dataset[dataset["mass"] >= 127]

# small planets
model_small = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred_small = model_small.fit_predict(dataset_small)
lof_small = model_small.negative_outlier_factor_

print("\n##### Now is LOF code for small planets #####")
print(f"Number of outliers detected: {len(dataset_small[y_pred_small == -1])}")
print(f"Number of inliers detected: {len(dataset_small[y_pred_small == 1])}")
print(f"The mean LOF score for the outliers is: {np.mean(lof_small[y_pred_small == -1] * -1)}")
print(f"The mean LOF score for the inliers is: {np.mean(lof_small[y_pred_small == 1] * -1)}")

# intermediate planets
model_intermediate = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred_intermediate = model_intermediate.fit_predict(dataset_intermediate)
lof_intermediate = model_intermediate.negative_outlier_factor_

print("\n##### Now is LOF code for intermediate planets #####")
print(f"Number of outliers detected: {len(dataset_intermediate[y_pred_intermediate == -1])}")
print(f"Number of inliers detected: {len(dataset_intermediate[y_pred_intermediate == 1])}")
print(f"The mean LOF score for the outliers is: {np.mean(lof_intermediate[y_pred_intermediate == -1] * -1)}")
print(f"The mean LOF score for the inliers is: {np.mean(lof_intermediate[y_pred_intermediate == 1] * -1)}")

# giant planets
model_giant = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred_giant = model_giant.fit_predict(dataset_giant)
lof_giant = model_giant.negative_outlier_factor_

print("\n##### Now is LOF code for giant planets #####")
print(f"Number of outliers detected: {len(dataset_giant[y_pred_giant == -1])}")
print(f"Number of inliers detected: {len(dataset_giant[y_pred_giant == 1])}")
print(f"The mean LOF score for the outliers is: {np.mean(lof_giant[y_pred_giant == -1] * -1)}")
print(f"The mean LOF score for the inliers is: {np.mean(lof_giant[y_pred_giant == 1] * -1)}")

inliers = pd.concat([
    dataset_small[y_pred_small == 1],
    dataset_intermediate[y_pred_intermediate == 1],
    dataset_giant[y_pred_giant == 1]
])

outliers_small = dataset_small[y_pred_small == -1]
outliers_intermediate = dataset_intermediate[y_pred_intermediate == -1]
outliers_giant = dataset_giant[y_pred_giant == -1]

fig, ax = plt.subplots()
scatter0 = ax.scatter(
    inliers["mass"],
    inliers["radius"],
    label="Inliers",
    alpha=0.5,
    zorder=2
)
scatter1 = ax.scatter(
    outliers_small["mass"],
    outliers_small["radius"],
    label="Outliers small planets",
    zorder=3,
    edgecolor="black"
)
scatter2 = ax.scatter(
    outliers_intermediate["mass"],
    outliers_intermediate["radius"],
    label="Outliers intermediate planets",
    zorder=3,
    edgecolor="black"
)
scatter3 = ax.scatter(
    outliers_giant["mass"],
    outliers_giant["radius"],
    label="Outliers giant planets",
    zorder=3,
    edgecolor="black"
)

ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Mass ($M_\oplus$)")
ax.set_ylabel(r"Radius ($R_\oplus$)")
ax.set_title("LOF Outlier Detection excluding uncertainties: Mass vs Radius")
plt.tight_layout()
plt.savefig("figures/LOF_outliers_excl_uncer_all.pdf", dpi=300)



# ---------------------------
# LOF in three mass regimes INCLUDING uncertainties
# ---------------------------

common_index = dataset.index.intersection(dataset_error.index)
dataset = dataset.loc[common_index]
dataset_error = dataset_error.loc[common_index]

dataset_small = dataset[dataset["mass"] < 4.4]
dataset_intermediate = dataset[(dataset["mass"] >= 4.4) & (dataset["mass"] < 127)]
dataset_giant = dataset[dataset["mass"] >= 127]

dataset_error_small = dataset_error[dataset["mass"] < 4.4]
dataset_error_intermediate = dataset_error[(dataset["mass"] >= 4.4) & (dataset["mass"] < 127)]
dataset_error_giant = dataset_error[dataset["mass"] >= 127]

# small planets
model_small = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred_small = model_small.fit_predict(dataset_error_small)
lof_small = model_small.negative_outlier_factor_

print("\n##### Now is LOF code for small planets including uncertainties #####")
print(f"Number of outliers detected: {len(dataset_small[y_pred_small == -1])}")
print(f"Number of inliers detected: {len(dataset_small[y_pred_small == 1])}")
print(f"The mean LOF score for the outliers is: {np.mean(lof_small[y_pred_small == -1] * -1)}")
print(f"The mean LOF score for the inliers is: {np.mean(lof_small[y_pred_small == 1] * -1)}")

# intermediate planets
model_intermediate = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred_intermediate = model_intermediate.fit_predict(dataset_error_intermediate)
lof_intermediate = model_intermediate.negative_outlier_factor_

print("\n##### Now is LOF code for intermediate planets including uncertainties #####")
print(f"Number of outliers detected: {len(dataset_intermediate[y_pred_intermediate == -1])}")
print(f"Number of inliers detected: {len(dataset_intermediate[y_pred_intermediate == 1])}")
print(f"The mean LOF score for the outliers is: {np.mean(lof_intermediate[y_pred_intermediate == -1] * -1)}")
print(f"The mean LOF score for the inliers is: {np.mean(lof_intermediate[y_pred_intermediate == 1] * -1)}")

# giant planets
model_giant = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred_giant = model_giant.fit_predict(dataset_error_giant)
lof_giant = model_giant.negative_outlier_factor_

print("\n#### Now is LOF code for giant planets including uncertainties #####")
print(f"Number of outliers detected: {len(dataset_giant[y_pred_giant == -1])}")
print(f"Number of inliers detected: {len(dataset_giant[y_pred_giant == 1])}")
print(f"The mean LOF score for the outliers is: {np.mean(lof_giant[y_pred_giant == -1] * -1)}")
print(f"The mean LOF score for the inliers is: {np.mean(lof_giant[y_pred_giant == 1] * -1)}")

inliers = pd.concat([
    dataset_small[y_pred_small == 1],
    dataset_intermediate[y_pred_intermediate == 1],
    dataset_giant[y_pred_giant == 1]
])

outliers_small = dataset_small[y_pred_small == -1]
outliers_intermediate = dataset_intermediate[y_pred_intermediate == -1]
outliers_giant = dataset_giant[y_pred_giant == -1]

fig, ax = plt.subplots()
scatter0 = ax.scatter(
    inliers["mass"],
    inliers["radius"],
    label="Inliers",
    alpha=0.5,
    zorder=2
)
scatter1 = ax.scatter(
    outliers_small["mass"],
    outliers_small["radius"],
    label="Outliers small planets",
    zorder=3,
    edgecolor="black"
)
scatter2 = ax.scatter(
    outliers_intermediate["mass"],
    outliers_intermediate["radius"],
    label="Outliers intermediate planets",
    zorder=3,
    edgecolor="black"
)
scatter3 = ax.scatter(
    outliers_giant["mass"],
    outliers_giant["radius"],
    label="Outliers giant planets",
    zorder=3,
    edgecolor="black"
)

ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Mass ($M_\oplus$)")
ax.set_ylabel(r"Radius ($R_\oplus$)")
ax.set_title("LOF Outlier Detection including uncertainties: Mass vs Radius")
plt.tight_layout()
plt.savefig("figures/LOF_outliers_incl_uncer_all.pdf", dpi=300)


# ---------------------------
# Otegi qualitative selection
# ---------------------------
print("\n#### Now is Otegi qualitative selection code #####")
otegi_mask = (
    (dataset_error['mass_error'] / dataset_error['mass'] < 0.25) &
    (dataset_error['radius_error'] / dataset_error['radius'] < 0.08)
)

dataset_qualitative = dataset[otegi_mask].copy()
outliers_qualitative = dataset[~otegi_mask].copy()
print(f"Number of Otegi outliers detected: {len(outliers_qualitative)}")
print(f"Length of dataset AFTER Otegi selection: {len(dataset_qualitative)}")

# remove k2-123b  in a separate plot
k2_123b_mask = dataset.index == "K2-123 b"
k2_123b_outlier = dataset[k2_123b_mask]

#total outliers after Otegi selection and adding K2-123 b
outliers_qualitative = pd.concat([outliers_qualitative, k2_123b_outlier])

fig, ax = plt.subplots()
scatter0 = ax.scatter(
    dataset_qualitative["mass"],
    dataset_qualitative["radius"],
    label="Inliers",
    alpha=0.5,
    zorder=2
)
scatter1 = ax.scatter(
    outliers_qualitative["mass"],
    outliers_qualitative["radius"],
    label="Outliers",
    zorder=3,
    edgecolor="black"
)
scatter2 = ax.scatter(
        k2_123b_outlier["mass"],
        k2_123b_outlier["radius"],
        label="K2-123 b",
        zorder=4,
        edgecolor="blue",
        facecolor = "none",
        s=180,
        linewidth=1.5)

ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Mass ($M_\oplus$)")
ax.set_ylabel(r"Radius ($R_\oplus$)")
ax.set_title("Otegi Outlier Selection: Mass vs Radius")
plt.tight_layout()
plt.savefig("Figures/Otegi_outliers.pdf", dpi=300)


# ---------------------------
# Handpicked outlier selection
# ---------------------------
print("\n##### Now is handpicked outlier selection code #####")
outliers = []
for i in range(len(dataset)):
    if (
        (dataset.radius.loc[dataset.index[i]] <= 6e0 and dataset.mass.loc[dataset.index[i]] >= 1e2)
        or (dataset.radius.loc[dataset.index[i]] <= 5e-1 and dataset.mass.loc[dataset.index[i]] >= 1e0)
        or (dataset.radius.loc[dataset.index[i]] <= 1e-1 and dataset.mass.loc[dataset.index[i]] >= 1e-2)
        or (dataset.radius.loc[dataset.index[i]] <= 7e-1 and dataset.mass.loc[dataset.index[i]] >= 2e-1)
        or (dataset.radius.loc[dataset.index[i]] <= 1.5e0 and dataset.mass.loc[dataset.index[i]] <= 4e1 and dataset.mass.loc[dataset.index[i]] >= 6e0)
        or (dataset.radius.loc[dataset.index[i]] <= 9e-1 and dataset.mass.loc[dataset.index[i]] <= 1e1 and dataset.mass.loc[dataset.index[i]] >= 1e0)
        or (dataset.radius.loc[dataset.index[i]] <= 1e1 and dataset.radius.loc[dataset.index[i]] >= 4.5e0 and dataset.mass.loc[dataset.index[i]] >= 1e0 and dataset.mass.loc[dataset.index[i]] <= 1e1)
        or (dataset.radius.loc[dataset.index[i]] <= 3e0 and dataset.radius.loc[dataset.index[i]] >= 2e0 and dataset.mass.loc[dataset.index[i]] <= 3e0 and dataset.mass.loc[dataset.index[i]] >= 1e0)
        or (dataset.radius.loc[dataset.index[i]] <= 2e0 and dataset.radius.loc[dataset.index[i]] >= 1e0 and dataset.mass.loc[dataset.index[i]] <= 1e2 and dataset.mass.loc[dataset.index[i]] >= 3e1)
        or (dataset.radius.loc[dataset.index[i]] <= 3e1 and dataset.radius.loc[dataset.index[i]] >= 2e1 and dataset.mass.loc[dataset.index[i]] <= 1e4 and dataset.mass.loc[dataset.index[i]] >= 7e3)
        or (dataset.radius.loc[dataset.index[i]] >= 6e-1 and dataset.mass.loc[dataset.index[i]] <= 1e-1)
        or dataset.temp_eq.loc[dataset.index[i]] >= 4000
    ):
        outliers.append(dataset.loc[dataset.index[i]])

outliers_df = pd.DataFrame(outliers)

print(f"Number of handpicked outliers detected: {len(outliers_df)}")

fig, ax = plt.subplots()
inliers_handpicked = dataset[~dataset.index.isin(outliers_df.index)]

scatter0 = ax.scatter(
    inliers_handpicked["mass"],
    inliers_handpicked["radius"],
    label="Inliers",
    alpha=0.5,
    zorder=2
)
scatter1 = ax.scatter(
    outliers_df["mass"],
    outliers_df["radius"],
    label="Outliers",
    zorder=3,
    edgecolor="black"
)

ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Mass ($M_\oplus$)")
ax.set_ylabel(r"Radius ($R_\oplus$)")
ax.set_title("Handpicked Outlier Detection excluding uncertainties: Mass vs Radius")
plt.tight_layout()
plt.savefig("figures/Handpicked_outliers_excl_uncer.pdf", dpi=300)

# plt.show()

