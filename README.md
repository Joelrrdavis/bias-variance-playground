# Bias–Variance Playground (GitHub Pages)

An interactive, in-browser simulator for teaching regression, under/overfitting, and the bias–variance tradeoff.

It runs **entirely on GitHub Pages** using **streamlit-lite** (Python in the browser via Pyodide). No server required.

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
- `PAGES_SETTINGS.md` – quick checklist for GitHub Pages configuration

---

## Deploy on GitHub Pages

1. Create a new repo (e.g., `bias-variance-playground`) and push these files to the **repo root**.
2. In GitHub, go to **Settings → Pages**
3. Under **Build and deployment**:
   - **Source:** Deploy from a branch
   - **Branch:** `main`
   - **Folder:** `/ (root)`
4. Click **Save**

Your site will appear at:

`https://<your-username>.github.io/<repo-name>/`

---

## Local preview (optional)

```bash
python -m http.server 8000
```

Then open:

`http://localhost:8000`

---

## Classroom notes

Suggested demo sequence:

1. `noise=0`, `outliers=0`, `degree=1` → underfit (bias)
2. Increase degree → fit improves
3. Keep increasing degree → train RMSE ↓, test/CV RMSE ↑ (variance)
4. Add outliers → instability increases
5. Increase Ridge α → curve stabilizes, CV error often improves
