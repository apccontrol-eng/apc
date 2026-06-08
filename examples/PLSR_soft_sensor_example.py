import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from apc.calibration.PCA_eigen import scale_with_reference
from apc.calibration.PCA_eigen import column_stats
from apc.calibration.PCA_eigen import standardize_matrix
from apc.calibration.PCA_eigen import plot_column_distributions_with_stats
from apc.calibration.PLS_NIPALS import pls_nipals
from matplotlib.patches import Ellipse
from scipy.stats import f as f_
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score

'''
===============================================================================
PLSR monitoring tools

References:

    Kadlec, P., Gabrys, B., & Strandt, S. (2009). 
    Data-driven soft sensors in the process industry. 
    Computers & Chemical Engineering, 33 (4), 795–814. 
    https://doi.org/10.1016/j.compchemeng.2008.12.012
    
    Kadlec, P., Grbić, R., & Gabrys, B. (2011). 
    Review of adaptation mechanisms for data-driven soft sensors. 
    Computers & Chemical Engineering, 35 (1), 1–24. 
    https://doi.org/10.1016/j.compchemeng.2010.07.034
    
    Qin, S. J. (1998). 
    Recursive PLS algorithms for adaptive data modeling. 
    Computers & Chemical Engineering, 22 (4), 503–514. 
    https://doi.org/10.1016/S0098-1354(97)00262-7
    
    Geladi, P., & Kowalski, B. R. (1986). 
    Partial least-squares regression: A tutorial. 
    Analytica Chimica Acta, 185, 1–17. 
    https://doi.org/10.1016/0003-2670(86)80028-9  

    Wise, B. M., & Gallagher, N. B. (1996). 
    The process chemometrics approach to process monitoring and fault detection. 
    Journal of Process Control, 6 (6), 329–348. 
    https://doi.org/10.1016/0959-1524(96)00009-1

===============================================================================
'''

def q_residual_threshold_PLS(X, T, P, alpha=0.95):
    """
    Compute SPE (Q-statistic) control limit for PLS using Jackson's chi-square 
    approximation.

    Parameters:
        X : np.ndarray (n_samples, n_features)
            Original data matrix (centered/scaled as in model training)
        T : np.ndarray (n_samples, n_components)
            Score matrix from PLS
        P : np.ndarray (n_features, n_components)
            X-loading matrix
        alpha : float
            Confidence level (e.g., 0.95, 0.99)

    Returns:
        Q_crit : float
            SPE control limit
    """
    # =========================================================================
    # reconstruct X from latent structure
    X_hat = T @ P.T

    # =========================================================================
    # residual matrix (THIS is the key step)
    E = X - X_hat

    # =========================================================================
    # eigenvalues of residual covariance matrix
    # equivalent to variance along each residual variable
    residual_eigvals = np.linalg.eigvalsh(np.cov(E, rowvar=False))

    # removing miniscule negatives from numerical noise
    residual_eigvals = np.maximum(residual_eigvals, 0)

    # =========================================================================
    # Jackson moments
    theta1 = np.sum(residual_eigvals)
    theta2 = np.sum(residual_eigvals**2)
    theta3 = np.sum(residual_eigvals**3)

    # =========================================================================
    # avoid divide-by-zero
    if theta2 < 1e-12:
        return 0.0
    h0 = 1 - (2 * theta1 * theta3) / (3 * theta2**2)
    h0 = max(h0, 1e-6)

    # =========================================================================
    # normal quantile
    z = norm.ppf(alpha)

    # =========================================================================
    # Jackson Q-statistic threshold
    term1 = (z * np.sqrt(2 * theta2) * h0) / theta1
    term2 = (theta2 * h0 * (h0 - 1)) / (theta1**2)
    Q_crit = theta1 * (1 + term1 + term2) ** (1 / h0)
    return Q_crit

def hotelling_t2_threshold(n_samples, n_components, alpha=0.95):
    """
    ===========================================================================
    Hotelling's T² critical value using F-distribution
    ===========================================================================
    """
    F_crit = f_.ppf(alpha, dfn=n_components, dfd=n_samples - n_components)
    T2_crit = (n_components * (n_samples**2 - 1)) / (n_samples * (n_samples - n_components)) * F_crit
    return T2_crit

def pls_biplot_with_t2(scores, loadings, T2_thresh,
                       labels=None, feature_names=None, scale_scores=1.0, scale_loadings=1.0):
    """
    ===========================================================================
    PLS biplot including Hotelling's T² threshold as an ellipse
    ===========================================================================
    """
    plt.figure(figsize=(8, 6))
    # =========================================================================
    # plotting scores
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            plt.scatter(scores[idx, 0] * scale_scores, scores[idx, 1] * scale_scores, label=f"Class {label}", alpha=0.7)
        plt.legend()
    else:
        plt.scatter(scores[:, 0] * scale_scores, scores[:, 1] * scale_scores, alpha=0.7, color='gray')

    # =========================================================================
    # plotting loadings as red arrows
    for i in range(loadings.shape[0]):
        plt.arrow(0, 0,
                  loadings[i, 0] * scale_loadings,
                  loadings[i, 1] * scale_loadings,
                  color='red', alpha=0.8, head_width=0.03)
        name = feature_names[i] if feature_names is not None else f"Var{i+1}"
        plt.text(loadings[i, 0] * scale_loadings * 1.1,
                 loadings[i, 1] * scale_loadings * 1.1,
                 name, color='red', ha='center', va='center', fontsize=9)

    plt.xlabel(f"LV1")
    plt.ylabel(f"LV2")

    # =========================================================================
    # adding T² confidence ellipse (assuming LV scores are uncorrelated & standardized)
    lambda1 = np.var(scores[:, 0])  # ≈ eigenvalue 1
    lambda2 = np.var(scores[:, 1])  # ≈ eigenvalue 2
    width = 2 * np.sqrt(T2_thresh * lambda1)
    height = 2 * np.sqrt(T2_thresh * lambda2)
    ellipse = Ellipse(xy=(0, 0), width=width, height=height,
                      edgecolor='red', fc='None', lw=2, linestyle='--', label="T² calibration threshold")
    plt.gca().add_patch(ellipse)

    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title("PLS Biplot with Hotelling's T² Ellipse")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_PLS_t2_q(data, W_star, loadings, num_components):

    """
    ===========================================================================
    Hotelling's T² and Q residual statistics for PLS.

    Parameters:
        data (np.ndarray): Standardized data (Z-scores).
        scores (np.ndarray): PLS scores (samples x LVs).
        loadings (np.ndarray): PLS loadings (features x LVs).
        num_components (int): Number of LVs to include in model.

    Returns:
        T2 (np.ndarray): Hotelling’s T² statistic per sample.
        Q (np.ndarray): Q residual statistic per sample.
    ===========================================================================
    """
    # =========================================================================
    # truncating to selected LVs
    T = data @ W_star
    T = T[:, :num_components]
    
    P = loadings[:, :num_components]

    # =========================================================================
    # Hotelling's T² = sum of squared standardized scores
    T2 = np.sum((T / np.std(T, axis=0))**2, axis=1)

    # =========================================================================
    # reconstruct data from selected LVs
    X_hat = T @ P.T
    residuals = data - X_hat
    Q = np.sum(residuals**2, axis=1)

    return T2, Q


sys.path.append('/Users/emil/Documents/GitHub/apc')

# data source: https://www.kaggle.com/datasets/jorgecote/distillation-column?resource=download
df = pd.read_csv("../apc/data/dataset_distill.csv", sep=';')
print(df.head())
df = df.replace(',', '.', regex=True).astype(float)

'''
1 col: Pressure of column (bar)
2-15 cols: Temperature at each tray (K)
16 col: Liquid flowrate (Kg mol/hour)
17 col: Vapor flowrate (Kg mol/hour)
18 col: Distillate flowrate (Kg mol/hour)
19 col: Bottoms flowrate (Kg mol/hour)
20 col: Feed flowrate (Kg mol/hour)
21 col: Molar concentration of ethanol (output)
'''

cols = (
    ["pressure"] +
    [f"T{i}" for i in range(1,15)] +
    ["liquid_flow","vapor_flow","distillate_flow",
     "bottoms_flow","feed_flow","ethanol_conc"]
)

df.columns = cols


# =============================================================================
# DISTRIBUTIONS (HISTOGRAMS)

plt.figure(figsize=(6,4))
sns.histplot(df["liquid_flow"], bins=30, kde=False)
plt.title("Liquid Flow Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["vapor_flow"], bins=30, kde=False)
plt.title("Vapor Flow Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["distillate_flow"], bins=30, kde=False)
plt.title("Distillate Flow Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["bottoms_flow"], bins=30, kde=False)
plt.title("Bottoms Flow Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["feed_flow"], bins=30, kde=False)
plt.title("Feed Flow Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["ethanol_conc"], bins=30, kde=False)
plt.title("Ethanol Concentration Distribution")
plt.show()

temp_cols = [f"T{i}" for i in range(1, 15)]

for col in temp_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=30, kde=False)
    plt.title(f"{col} Distribution")
    plt.xlabel("Temperature")
    plt.ylabel("Count")
    plt.show()


# =============================================================================
# TIME SERIES PLOTS

features = [
    "liquid_flow",
    "vapor_flow",
    "distillate_flow",
    "bottoms_flow",
    "feed_flow",
    "ethanol_conc"
]
    
for col in features:
    plt.figure(figsize=(10,4))
    sns.lineplot(x=df.index, y=df[col])
    plt.xlabel("Time step")
    plt.ylabel(col)
    plt.title(f"{col} Time Series")
    plt.show()

temp_cols = [f"T{i}" for i in range(1, 15)]

for col in temp_cols:
    plt.figure(figsize=(10, 3))
    sns.lineplot(x=df.index, y=df[col])
    plt.xlabel("Time step")
    plt.ylabel("Temperature")
    plt.title(f"{col} Time Series")
    plt.show()



df_for_modeling = df.drop(columns=["pressure"])
#print(df_for_modeling)

# Features: first 20 columns but only the column temperature profile is selected
X = df_for_modeling.iloc[:,0:13].values

#print(df_for_modeling.iloc[:,0:13])

# Target: ethanol concentration
y = df_for_modeling.iloc[:,19].values

#####
#X_train = X[500:600:,0:13]
#y_train = y[500:600]
#X_test = X[600:1050,0:13]
#y_test = y[600:1050]

#####
#X_train = X[600:2250:,0:13]
#y_train = y[600:2250]
#X_test = X[2250:,0:13]
#y_test = y[2250:]

#####
X_train = X[600:2250:,0:13]
y_train = y[600:2250]
X_test = X[2250:,0:13]
y_test = y[2250:]


# =============================================================================
## calibration with own PLSR implementation
n_components=9

X_calibration_autoscaled, X_calibration_means, X_calibration_stds = standardize_matrix(X_train)
y_calibration_autoscaled, y_calibration_means, y_calibration_stds = standardize_matrix(y_train)
T_calibration, U_calibration, P_calibration, Q, W, B, W_star = pls_nipals(X_calibration_autoscaled, y_calibration_autoscaled, n_components)
## predictions with own PLSR implementation
X_monitored_autoscaled = scale_with_reference(X_test, X_calibration_means, X_calibration_stds)
y_pred = X_monitored_autoscaled@W_star@B@Q.T * y_calibration_stds + y_calibration_means

#model = LinearRegression()
#model = PLSRegression( n_components = 9 )
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = round(np.sqrt(mse),3)
r2 = round(r2_score(y_test, y_pred),3)
print("RMSE:", rmse)
print("R²:", r2)

#C_PLS = model.coef_
#print("C_PLS : ", C_PLS)

C_PLSR_own = W_star@B@Q.T
#print("C_PLSR_own : ", C_PLSR_own)

# =============================================================================
## PLOTTING RESULTS
os.makedirs("figures", exist_ok=True)
def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")

show_start = 200
show_end = 900

actual = np.asarray(y_test)[show_start:show_end].ravel()
pred = np.asarray(y_pred)[show_start:show_end].ravel()

df = pd.DataFrame({
    "Actual": actual,
    "Predicted": pred
})

plt.figure(figsize=(6,6))
sns.scatterplot(data=df, x="Actual", y="Predicted")
plt.grid(False)
plt.xlabel("Actual ethanol concentration")
plt.ylabel("Predicted ethanol concentration")
plt.title("Regression Model Performance")
save_fig("regression_model_performance")
plt.show()

# =============================================================================

plt.figure(figsize=(6,6))
ax = sns.scatterplot(data=df, x="Actual", y="Predicted")

plt.grid(False)
plt.xlabel("Actual ethanol concentration")
plt.ylabel("Predicted ethanol concentration")
plt.title("Regression Model Performance")

# adding RMSE and R²
ax.text(
    0.05, 0.95,
    f"RMSE = {rmse:.2f}\n$R^2$ = {r2:.2f}",
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

save_fig("regression_model_performance")
plt.show()

# =============================================================================

df = pd.DataFrame({
    "Sample": np.arange(len(actual)),
    "Actual": actual,
    "Predicted": pred
})

df_long = df.melt(
    id_vars="Sample",
    value_vars=["Actual", "Predicted"],
    var_name="Type",
    value_name="Ethanol concentration"
)

plt.figure(figsize=(10,4))
sns.lineplot(data=df_long, x="Sample", y="Ethanol concentration", hue="Type")
plt.grid(False)

plt.title("Model Prediction vs Actual")
plt.show()


# =============================================================================
# saving as a gif


import matplotlib.animation as animation
import imageio as imageio
import os

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 4))

ax.set_xlim(0, len(actual) - 1)

ymin = min(np.min(actual), np.min(pred))
ymax = max(np.max(actual), np.max(pred))
padding = 0.05 * (ymax - ymin)

ax.set_ylim(ymin - padding, ymax + padding)

ax.set_title("Model Prediction vs Actual")
ax.set_xlabel("Sample")
ax.set_ylabel("Ethanol concentration")
ax.grid(False)


actual_line, = ax.plot([], [], label="Actual", lw=2)
pred_line,   = ax.plot([], [], label="Predicted", lw=2)

ax.legend()


def update(frame):
    x = np.arange(frame + 1)

    actual_line.set_data(x, actual[:frame + 1])
    pred_line.set_data(x, pred[:frame + 1])

    return actual_line, pred_line


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(actual),
    interval=50,
    blit=True,
    repeat=False
)

gif_filename = os.path.join("figures", "PLSR_model_prediction_vs_actual.gif")


ani.save(
    gif_filename,
    writer=animation.PillowWriter(fps=20)
)

plt.close(fig)


def gif_to_mp4(gif_filename, mp4_filename, fps=20):
    gif = imageio.get_reader(gif_filename)

    writer = imageio.get_writer(
        mp4_filename,
        fps=fps,
        codec="libx264"
    )

    for frame in gif:
        writer.append_data(frame)

    writer.close()

mp4_filename = os.path.join("figures", "PLSR_model_prediction_vs_actual.mp4")

gif_to_mp4(
    gif_filename,
    mp4_filename,
    fps=20
)

print("Saved:")
print(gif_filename)
print(mp4_filename)



# =============================================================================
# PLOTTING PLSR MONITORING STATISTICS

calibration_T2, calibration_Q = compute_PLS_t2_q(data = X_calibration_autoscaled, 
                                                 W_star = W_star, 
                                                 loadings = P_calibration, 
                                                 num_components = n_components)

#print("calibration_T2 : ", calibration_T2)
#print("calibration_Q : ", calibration_Q)

n = X_calibration_autoscaled.shape[0]

calibration_T2_thresh = hotelling_t2_threshold(n_samples = n, n_components = n_components)

calibration_Q_thresh = q_residual_threshold_PLS(X_calibration_autoscaled, T_calibration, P_calibration, alpha=0.95)


print("calibration_T2_thresh : ", calibration_T2_thresh)
print("calibration_Q_thresh : ", calibration_Q_thresh)



def plot_t2_q(T2, Q, threshold_T2=None, threshold_Q=None, name=""):
    """
    Plots Hotelling's T² and Q residuals with optional thresholds.
    """
    
    os.makedirs("figures", exist_ok=True)
    def save_fig(name):
        plt.tight_layout()
        plt.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].scatter(range(len(T2)), T2, color='steelblue')
    axs[0].axhline(threshold_T2, color='red', linestyle='--', label='Threshold' if threshold_T2 else None)
    axs[0].set_title("Hotelling's T²")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("T²")
    axs[0].grid(True)

    axs[1].scatter(range(len(Q)), Q, color='darkgreen')
    axs[1].axhline(threshold_Q, color='red', linestyle='--', label='Threshold' if threshold_Q else None)
    axs[1].set_title("Q Residuals (SPE)")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("Q")
    axs[1].grid(False)

    plt.tight_layout()
    save_fig(name)
    plt.show()

plot_t2_q(calibration_T2, calibration_Q, threshold_T2 = calibration_T2_thresh, threshold_Q = calibration_Q_thresh)


pls_biplot_with_t2(
    scores = T_calibration,
    loadings = P_calibration,
    T2_thresh = calibration_T2_thresh,
    feature_names = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13"]
)


#### next on the line is implementing the monitoring statistics for new data and for biplot adding the labels how much variance LV1 and LV2 capture in the calibration data!

