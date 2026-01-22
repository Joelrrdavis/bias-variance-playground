# Bias–Variance Playground 

An interactive, in-browser simulator for teaching regression, 

runs **entirely on GitHub Pages** using **streamlit-lite** (Python in the browser via Pyodide). No server required.

---

## What students can do

- Add **noise**
- Inject **outliers**
- Change **polynomial degree** (model complexity)
- Add **Ridge regularization (α)**
- View **train/test RMSE** and **5-fold CV RMSE**
- See the fitted curve update instantly

---

## Files

- `index.html` – boots streamlit-lite and loads `app.py`
- `app.py` – Streamlit app (NumPy + Matplotlib only for broad Pyodide compatibility)
- `.nojekyll` – avoids GitHub Pages/Jekyll processing

---



## notes

Suggested sequence:

1. `noise=0`, `outliers=0`, `degree=1` → underfit (bias)
2. Increase degree → fit improves
3. Keep increasing degree → train RMSE ↓, test/CV RMSE ↑ (variance)
4. Add outliers → instability increases
5. Increase Ridge α → curve stabilizes, CV error often improves
