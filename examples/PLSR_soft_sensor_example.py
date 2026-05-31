import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Add local project path
sys.path.append('/Users/emil/Documents/GitHub/apc')

from apc.calibration.PCA_eigen import scale_with_reference
from apc.calibration.PCA_eigen import column_stats
from apc.calibration.PCA_eigen import standardize_matrix
from apc.calibration.PCA_eigen import plot_column_distributions_with_stats
from apc.calibration.PLS_NIPALS import pls_nipals


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

#print(df.pressure)

df.pressure.hist(bins=30)



plt.figure(figsize=(10,4))
plt.plot(df["pressure"])
#plt.ylim(0, 2)   # adjust limits
plt.xlabel("Time step")
plt.ylabel("pressure ")
plt.title("pressure Over Time")
plt.show()


df.liquid_flow.hist(bins=30)
plt.figure(figsize=(10,4))
plt.plot(df["liquid_flow"])
#plt.ylim(0, 2)   # adjust limits
plt.xlabel("Time step")
plt.ylabel("pressure ")
plt.title("liquid_flow Over Time")
plt.show()

df.vapor_flow.hist(bins=30)
plt.figure(figsize=(10,4))
plt.plot(df["vapor_flow"])
#plt.ylim(0, 2)   # adjust limits
plt.xlabel("Time step")
plt.ylabel("pressure ")
plt.title("vapor_flow Over Time")
plt.show()

df.distillate_flow.hist(bins=30)
plt.figure(figsize=(10,4))
plt.plot(df["distillate_flow"])
#plt.ylim(0, 2)   # adjust limits
plt.xlabel("Time step")
plt.ylabel("pressure ")
plt.title("distillate_flow Over Time")
plt.show()

df.bottoms_flow.hist(bins=30)
plt.figure(figsize=(10,4))
plt.plot(df["bottoms_flow"])
#plt.ylim(0, 2)   # adjust limits
plt.xlabel("Time step")
plt.ylabel("pressure ")
plt.title("bottoms_flow Over Time")
plt.show()


df.feed_flow.hist(bins=30)
plt.figure(figsize=(10,4))
plt.plot(df["feed_flow"])
#plt.ylim(0, 2)   # adjust limits
plt.xlabel("Time step")
plt.ylabel("pressure ")
plt.title("feed_flow Over Time")
plt.show()


df.ethanol_conc.hist(bins=30)
plt.figure(figsize=(10,4))
plt.plot(df["ethanol_conc"])
#plt.ylim(0, 2)   # adjust limits
plt.xlabel("Time step")
plt.ylabel("pressure ")
plt.title("ethanol_conc Over Time")
plt.show()

df_for_modeling = df.drop('pressure', axis=1)

print(df_for_modeling)


# Features: first 20 columns
X = df_for_modeling.iloc[:,0:13].values

#print(df_for_modeling.iloc[:,0:13])

# Target: ethanol concentration
y = df_for_modeling.iloc[:,19].values



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9 )

#####
#X_train = X[500:600:,0:13]
#y_train = y[500:600]
#X_test = X[600:1050,0:13]
#y_test = y[600:1050]

#####
X_train = X[600:2250:,0:13]
y_train = y[600:2250]
X_test = X[2250:,0:13]
y_test = y[2250:]

'''
df_for_modeling.iloc[:,0:19].head()
print("X:",X[:,0:19])
'''


## calibration with own PLSR implementation
X_calibration_autoscaled, X_calibration_means, X_calibration_stds = standardize_matrix(X_train)
y_calibration_autoscaled, y_calibration_means, y_calibration_stds = standardize_matrix(y_train)
T_calibration, U_calibration, P_calibration, Q, W, B, W_star = pls_nipals(X_calibration_autoscaled, y_calibration_autoscaled, n_components=9)
## predictions with own PLSR implementation
X_monitored_autoscaled = scale_with_reference(X_test, X_calibration_means, X_calibration_stds)

#y_pred_PLSR = X_monitored_autoscaled@W_star@B@Q.T * y_calibration_stds + y_calibration_means
y_pred = X_monitored_autoscaled@W_star@B@Q.T * y_calibration_stds + y_calibration_means

#model = LinearRegression()
#model = PLSRegression( n_components = 9 )
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R²:", r2)

show_start = 0
show_end = 2000

plt.figure(figsize=(6,6))
plt.scatter(y_test[show_start:show_end], y_pred[show_start:show_end])
plt.xlabel("Actual ethanol concentration")
plt.ylabel("Predicted ethanol concentration")
plt.title("Regression Model Performance")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(y_test[show_start:show_end], label="Actual")
plt.plot(y_pred[show_start:show_end], label="Predicted")
plt.legend()
plt.title("Model Prediction vs Actual")
plt.show()

C_PLS = model.coef_
print(C_PLS)
