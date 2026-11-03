# Deployment: free live demo on Streamlit Community Cloud

The result is a public URL like `https://recruitai-<you>.streamlit.app`, so reviewers can use the app without installing anything.

## Steps (about 5 minutes)

1. Push this repository to GitHub. `.env` is gitignored, so the key never leaves your machine.
2. Go to **https://share.streamlit.io** and click **Continue with GitHub**, then authorise.
3. Click **Create app → Deploy a public app from GitHub**:
   - Repository: `<your-username>/recruiter-chatbot`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: choose e.g. `recruitai-demo`
4. Open **Advanced settings → Python version: 3.12**, then paste the following into **Secrets**:
   ```toml
   GEMINI_API_KEY = "your-gemini-key"
   # Optional but recommended for a public link - reviewers type this code to enter:
   APP_PASSWORD = "choose-a-code"
   # Optional
   GEMINI_MODEL = "gemini-flash-lite-latest"
   RATE_LIMIT_PER_MIN = "30"
   ```
5. Click **Deploy**. The first build takes about 3 to 5 minutes while it installs `requirements.txt`.

## Notes
- Secrets are stored server-side by Streamlit. They're read through `st.secrets` (`src/config.py`) and never reach the browser.
- With `APP_PASSWORD` set, strangers who find the URL can't use up your Gemini quota.
- The free tier sleeps after inactivity. Open the link once before your interview to wake it up.
- To update the app, `git push`; it redeploys automatically.

## Running locally instead
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # add GEMINI_API_KEY
streamlit run app.py
```
