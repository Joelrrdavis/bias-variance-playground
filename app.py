import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bias–Variance Playground", layout="wide")

st.title("Bias–Variance Playground")
st.write("Tweak the sliders to see underfitting, overfitting, noise, outliers, and regularization.")

# -----------------------------
# Controls
# -----------------------------
colA, colB, colC = st.columns(3)

with colA:
    n = st.slider("Number of points", 40, 500, 120, 10)
    noise = st.slider("Noise (σ)", 0.0, 3.0, 0.8, 0.1)

with colB:
    degree = st.slider("Model complexity (polynomial degree)", 1, 15, 3, 1)
    ridge_alpha = st.slider("Regularization (Ridge α)", 0.0, 10.0, 0.0, 0.1)

with colC:
    outlier_frac = st.slider("Outliers (% of points)", 0, 40, 5, 5)
    outlier_scale = st.slider("Outlier severity (multiplier)", 1.0, 8.0, 3.0, 0.5)

seed = st.number_input("Random seed", min_value=0, max_value=99999, value=1955, step=1)

rng = np.random.default_rng(int(seed))

# -----------------------------
# Data generation
# -----------------------------
X = np.linspace(-3, 3, n).reshape(-1, 1)

def f(x):
    # Nonlinear truth so bias is visible with low-degree models
    return 0.7*x - 0.5*x**2 + 0.1*x**3

y = f(X).ravel() + rng.normal(0, noise, size=n)

# Inject outliers
m = int(outlier_frac / 100 * n)
if m > 0:
    idx = rng.choice(n, m, replace=False)
    # Larger noise for outliers; include +1e-6 to avoid all-zero when noise=0
    y[idx] += rng.normal(0, outlier_scale * (10*noise + 1e-6), size=m)

# -----------------------------
# Helpers: polynomial design matrix, ridge fit, metrics, CV
# -----------------------------
def poly_design(x, deg):
    """Return [1, x, x^2, ..., x^deg] for a 1D vector x."""
    x = np.asarray(x).reshape(-1)
    # increasing=True gives [x^0, x^1, ..., x^deg]
    return np.vander(x, N=deg + 1, increasing=True)

def ridge_fit(Phi, y_vec, alpha):
    """Closed-form ridge regression with intercept included in Phi."""
    y_vec = np.asarray(y_vec).reshape(-1, 1)
    d = Phi.shape[1]
    I = np.eye(d)
    # Do not regularize the intercept term (column 0)
    I[0, 0] = 0.0
    A = Phi.T @ Phi + alpha * I
    b = Phi.T @ y_vec
    w = np.linalg.solve(A, b)
    return w  # (d, 1)

def predict(Phi, w):
    return (Phi @ w).reshape(-1)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def kfold_indices(n_obs, k_folds, rng_local):
    idx = np.arange(n_obs)
    rng_local.shuffle(idx)
    folds = np.array_split(idx, k_folds)
    for i in range(k_folds):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k_folds) if j != i])
        yield train, val

# -----------------------------
# Train/test split
# -----------------------------
perm = rng.permutation(n)
split = int(0.7 * n)
tr, te = perm[:split], perm[split:]
Xtr, Xte = X[tr], X[te]
ytr, yte = y[tr], y[te]

Phi_tr = poly_design(Xtr, degree)
Phi_te = poly_design(Xte, degree)

w = ridge_fit(Phi_tr, ytr, ridge_alpha)
pred_tr = predict(Phi_tr, w)
pred_te = predict(Phi_te, w)

rmse_tr = rmse(ytr, pred_tr)
rmse_te = rmse(yte, pred_te)

# -----------------------------
# Cross-validation
# -----------------------------
k = 5
cv_errs = []
rng_cv = np.random.default_rng(int(seed) + 1)
for tr_idx, va_idx in kfold_indices(n, k, rng_cv):
    Phi_k_tr = poly_design(X[tr_idx], degree)
    Phi_k_va = poly_design(X[va_idx], degree)
    w_k = ridge_fit(Phi_k_tr, y[tr_idx], ridge_alpha)
    pred_va = predict(Phi_k_va, w_k)
    cv_errs.append(rmse(y[va_idx], pred_va))
cv_rmse = float(np.mean(cv_errs))

# -----------------------------
# Plots
# -----------------------------
xx = np.linspace(-3.2, 3.2, 400).reshape(-1, 1)
yy_true = f(xx).ravel()
Phi_xx = poly_design(xx, degree)
yy_hat = predict(Phi_xx, w)

col1, col2 = st.columns(2, gap="large")

with col1:
    fig, ax = plt.subplots()
    ax.scatter(Xtr, ytr, alpha=0.7, label="Train")
    ax.scatter(Xte, yte, alpha=0.7, marker="x", label="Test")
    ax.plot(xx, yy_true, linewidth=2, label="True function")
    ax.plot(xx, yy_hat, linewidth=2, linestyle="--", label="Model fit")
    ax.set_title("Data, truth, and fitted curve")
    ax.legend()
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.bar(["Train RMSE", "Test RMSE", "5-fold CV RMSE"], [rmse_tr, rmse_te, cv_rmse])
    ax2.set_title("Errors (lower is better)")
    st.pyplot(fig2)

st.markdown(
    """
**Teaching tip:**  
- Start with degree=1 and noise=0 to show *bias* (underfit).  
- Increase degree until train error falls but test/CV rises to show *variance* (overfit).  
- Add outliers/noise to see instability; increase Ridge α to stabilize the fit.
"""
)
