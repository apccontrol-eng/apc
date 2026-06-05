import sys
sys.path.insert(0, "/Users/emil/Documents/GitHub/apc")
from apc.subspace_identification.MOESP import MOESP
from apc.subspace_identification.N4SID import N4SID
import numpy as np

np.random.seed(0)

# ----------------------------
# TRUE SYSTEM
# ----------------------------
n = 3
m = 1
p = 1

A = np.array([
    [0.9, 0.1, 0.0],
    [0.0, 0.8, 0.2],
    [0.0, 0.0, 0.7]
])

B = np.array([[0.1], [0.05], [0.02]])
C = np.array([[1.0, 0.0, 0.0]])
D = np.array([[0.0]])


# ----------------------------
# INPUT (PRBS) or rich step test input
# ----------------------------
N = 2000
#u = np.random.choice([-1, 1], size=(N, 1))

u = np.zeros(N)
step_size = 100
num_steps = N // step_size
# choose levels in a reasonable range
levels = np.linspace(-1.0, 1.0, num_steps)
# shuffle to avoid monotonic pattern
np.random.shuffle(levels)

for i in range(num_steps):
    u[i*step_size:(i+1)*step_size] = levels[i]
u = u.reshape(-1, 1)


# ----------------------------
# SIMULATE DETERMINISTIC SYSTEM
# ----------------------------
x = np.zeros((N, n))
y = np.zeros((N, p))

for k in range(N - 1):
    y[k] = C @ x[k] + D @ u[k]
    x[k+1] = A @ x[k] + B @ u[k]

y[-1] = C @ x[-1] + D @ u[-1]

# ----------------------------
# ADD NOISE
# ----------------------------
noise_std = 0.01
y_noisy = y + noise_std * np.random.randn(N, p)

y_id = y_noisy

# ----------------------------
# CENTER DATA
# ----------------------------
u_mean = np.mean(u)
y_mean = np.mean(y_id)

u_c = u - u_mean
y_c = y_id - y_mean

# ----------------------------
# IDENTIFICATION
# ----------------------------
A_id, B_id, C_id, D_id = MOESP(u_c, y_c, i=10, n=3)
#A_id, B_id, C_id, D_id = N4SID(u_c, y_c, i=10, n=3)

# ----------------------------
# LINEAR SIMULATION FUNCTION
# ----------------------------
def simulate_linear(A, B, C, D, u):
    N = len(u)
    n = A.shape[0]

    x = np.zeros((N, n))
    y = np.zeros((N, C.shape[0]))

    for k in range(N - 1):
        y[k] = C @ x[k] + D @ u[k]
        x[k+1] = A @ x[k] + B @ u[k]

    y[-1] = C @ x[-1] + D @ u[-1]
    return y

# ----------------------------
# Simulation in deviation (centered) form
# ----------------------------
y_id_model_dev = simulate_linear(A_id, B_id, C_id, D_id, u_c)

# add operating point back
y_id_model = y_id_model_dev + y_mean

# ----------------------------
# FIT R^2 METRIC
# ----------------------------
fit = 100 * (1 - np.linalg.norm(y - y_id_model) / np.linalg.norm(y - np.mean(y)))
print(f"FIT: {fit:.2f}%")
print("Eigenvalues:", np.linalg.eigvals(A_id))


##########################################################################################
##########################################################################################
##########################################################################################


import matplotlib.pyplot as plt

t = np.arange(len(y))  # time index

plt.figure(figsize=(14, 8))  # 👈 bigger and clearer

# ----------------------------
# OUTPUT COMPARISON
# ----------------------------

plt.subplot(2,1,1)

plt.plot(t, y, label="True system", linewidth=2)
plt.plot(t, y_id_model, '--', label="Identified model", linewidth=2)

plt.title("True vs Identified System Output", fontsize=14)
plt.ylabel("Output", fontsize=12)
plt.legend()
plt.grid(False)

# ----------------------------
# INPUT SIGNAL
# ----------------------------
plt.subplot(2,1,2)

plt.plot(t, u, label="Input (u)", linewidth=2)

plt.xlabel("Time step", fontsize=12)
plt.ylabel("Input", fontsize=12)
plt.title("Input Signal", fontsize=14)
plt.grid(False)

plt.tight_layout()
plt.show()


##########################################################################################
##########################################################################################
##########################################################################################


plt.figure(figsize=(14,5))

start, end = 200, 600

plt.plot(t[start:end], y[start:end], label="True", linewidth=2)
plt.plot(t[start:end], y_id_model[start:end], '--', label="Identified", linewidth=2)

plt.title("Zoomed Output Comparison")
plt.xlabel("Time step")
plt.ylabel("Output")
plt.legend()
plt.grid(False)

plt.show()


##########################################################################################
##########################################################################################
##########################################################################################


print("A_id : ", A_id)
print("\nB_id : ", B_id)
print("\nC_id : ", C_id)
print("\nD_id : ", D_id)

print("\nA : ", A)
print("\nB : ", B)
print("\nC : ", C)
print("\nD : ", D)

print("Eigenvalues of A:", np.linalg.eigvals(A))

print("Eigenvalues of A_id:", np.linalg.eigvals(A_id))


##########################################################################################
##########################################################################################
##########################################################################################



