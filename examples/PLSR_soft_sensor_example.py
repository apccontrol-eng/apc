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

# Add local project path
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
#df.pressure.hist(bins=30)

'''
# -------------------------
# 1) PRESSURE OVER TIME
# -------------------------
plt.figure(figsize=(10,4))
sns.lineplot(x=df.index, y=df["pressure"])
plt.xlabel("Time step")
plt.ylabel("Pressure")
plt.title("Pressure Over Time")
plt.show()
'''

'''
# -------------------------
# 2) DISTRIBUTIONS (HISTOGRAMS)
# -------------------------

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


# -------------------------
# 3) TIME SERIES PLOTS
# -------------------------
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

'''

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


## calibration with own PLSR implementation
X_calibration_autoscaled, X_calibration_means, X_calibration_stds = standardize_matrix(X_train)
y_calibration_autoscaled, y_calibration_means, y_calibration_stds = standardize_matrix(y_train)
T_calibration, U_calibration, P_calibration, Q, W, B, W_star = pls_nipals(X_calibration_autoscaled, y_calibration_autoscaled, n_components=9)
## predictions with own PLSR implementation
X_monitored_autoscaled = scale_with_reference(X_test, X_calibration_means, X_calibration_stds)
y_pred = X_monitored_autoscaled@W_star@B@Q.T * y_calibration_stds + y_calibration_means

#model = LinearRegression()
model = PLSRegression( n_components = 9 )
model.fit(X_train, y_train)
#y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = round(np.sqrt(mse),4)
r2 = round(r2_score(y_test, y_pred),4)
print("RMSE:", rmse)
print("R²:", r2)

#C_PLS = model.coef_
#print("C_PLS : ", C_PLS)

C_PLSR_own = W_star@B@Q.T
#print("C_PLSR_own : ", C_PLSR_own)

## PLOTTING RESULTS
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
plt.xlabel("Actual ethanol concentration")
plt.ylabel("Predicted ethanol concentration")
plt.title("Regression Model Performance")
plt.show()




df = pd.DataFrame({
    "Index": np.arange(len(actual)),
    "Actual": actual,
    "Predicted": pred
})

df_long = df.melt(
    id_vars="Index",
    value_vars=["Actual", "Predicted"],
    var_name="Type",
    value_name="Value"
)

plt.figure(figsize=(10,4))
sns.lineplot(data=df_long, x="Index", y="Value", hue="Type")

plt.title("Model Prediction vs Actual")
plt.show()






