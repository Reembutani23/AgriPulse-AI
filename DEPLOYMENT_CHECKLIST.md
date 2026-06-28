# Deployment & Monitoring Checklist

This checklist ensures a smooth transition from development to production for the **AgriPulse AI** project.

## ✅ Production Deployment Checklist
- [ ] **All tests passing** – `pytest src/testing` runs without failures.
- [ ] **Code reviewed** – PR approved by at least one reviewer.
- [ ] **Documentation complete** – API docs, README, and user guides updated.
- [ ] **Models saved and versioned** – Serialized model artifacts stored in `models/` with version tags.
- [ ] **Scalers serialized** – Scaler metadata (`models/scaling_metadata.json`) present.
- [ ] **Environment variables configured** – Sensitive values (e.g., API keys) set in Streamlit Cloud secrets.
- [ ] **API keys secured** – No hard‑coded secrets in code; use `os.getenv`.
- [ ] **Monitoring setup** – Logging, health‑check endpoint, and performance dashboards.
- [ ] **Error handling implemented** – Graceful fallback for missing artefacts or prediction errors.
- [ ] **Logging configured** – Structured logs written to `logs/` and forwarded to Streamlit Cloud.
- [ ] **Continuous Integration (CI) passing** – GitHub Actions workflow runs on each PR/commit.

## 📦 Deployment Steps
1. **Commit final version**
   ```bash
   git add .
   git commit -m "Final deployment version"
   git push origin main
   ```
2. **Push to GitHub** – Ensure repository is linked to Streamlit Cloud.
3. **Configure Streamlit Cloud**
   - Connect the GitHub repo.
   - Set the **main** branch as the deployment source.
   - Add required secrets (e.g., `STREAMLIT_SECRETS`, `DATABASE_URL`).
4. **Deploy** – Streamlit Cloud will automatically build and launch the app.
5. **Verify** – Open the deployed URL, run a few predictions, and check logs.

---
*Last Updated: 2024*
*Master Roadmap Version: 1.0*
