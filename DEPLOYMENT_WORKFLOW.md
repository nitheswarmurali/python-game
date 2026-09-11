# Vision Forge Deployment Workflow

This project uses two services:

- **Render** runs the Python Flask API with OpenCV and YOLO.
- **Vercel** hosts the browser frontend.

Deploy the API first. Then connect the Vercel frontend to the API URL.

## 1. Push the Project to GitHub

From the project folder:

```powershell
git status
git add .
git commit -m "Prepare deployment"
git push origin main
```

The GitHub repository should contain these files:

```text
webapp.py
requirements-web.txt
render.yaml
vercel.json
pyproject.toml
package.json
scripts/build.mjs
static/
templates/
```

## 2. Deploy the Python API on Render

1. Open [https://render.com](https://render.com) and sign in with GitHub.
2. Select **New** and then **Web Service**.
3. Select the GitHub repository `python-game`.
4. Use these settings:

```text
Name: vision-forge-api
Runtime: Python
Build Command: pip install -r requirements-web.txt
Start Command: gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 webapp:app
Health Check Path: /api/status
```

5. Select the free plan or the plan you need.
6. Create the Web Service.
7. Wait until the service is live.
8. Open this URL in a browser:

```text
https://YOUR-RENDER-SERVICE.onrender.com/api/status
```

A successful response looks similar to:

```json
{"haar":true,"yolo":true}
```

Copy the Render service URL. Do not include `/api/status` when using it as the API base URL.

Example:

```text
https://vision-forge-api.onrender.com
```

The repository already includes `render.yaml`, so Render may detect these settings automatically.

## 3. Deploy the Frontend on Vercel

1. Open [https://vercel.com](https://vercel.com) and sign in with GitHub.
2. Select **Add New Project**.
3. Import the same GitHub repository.
4. Use these project settings:

```text
Framework Preset: Other
Root Directory: ./
Build Command: npm run build
Output Directory: public
Install Command: leave the default
```

5. Before deploying, open **Environment Variables**.
6. Add this variable:

```text
Name: VISION_API_URL
Value: https://YOUR-RENDER-SERVICE.onrender.com
```

7. Enable it for **Production**, **Preview**, and **Development** if those options are shown.
8. Select **Deploy**.

The repository already includes `vercel.json`, `package.json`, `scripts/build.mjs`, and `pyproject.toml`.

## 4. Configure CORS on Render

After Vercel finishes deploying, copy the Vercel URL. It will look similar to:

```text
https://python-game.vercel.app
```

In Render:

1. Open the `vision-forge-api` service.
2. Open **Environment**.
3. Add or update:

```text
Name: CORS_ORIGIN
Value: https://YOUR-VERCEL-PROJECT.vercel.app
```

4. Save the environment variable.
5. Select **Manual Deploy** and then **Deploy latest commit**.

Do not add a trailing slash to `CORS_ORIGIN`.

## 5. Test the Deployed Application

Open the Vercel URL and test these features:

1. Upload an image.
2. Analyze a public image URL.
3. Start the camera and allow camera permission.
4. Confirm that face and person counts appear.
5. Download an analyzed result.
6. Export the CSV report.

The camera requires an HTTPS URL. The Vercel URL provides HTTPS automatically.

## 6. Environment Variable Summary

### Render

```text
CORS_ORIGIN=https://YOUR-VERCEL-PROJECT.vercel.app
```

### Vercel

```text
VISION_API_URL=https://YOUR-RENDER-SERVICE.onrender.com
```

Never put private keys or passwords in the frontend environment variables.

## 7. Local Development

Activate the project environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the local Flask API:

```powershell
python webapp.py
```

Open:

```text
http://127.0.0.1:5000
```

For a local frontend build:

```powershell
npm run build
```

Leave `VISION_API_URL` empty for local development. The browser then calls the local Flask API.

## 8. Troubleshooting

### Vercel says `main.py` is not a Flask app

Confirm that `pyproject.toml` contains:

```toml
[tool.vercel]
entrypoint = "webapp:app"
```

`main.py` starts the desktop Tkinter application. `webapp.py` contains the Flask application.

### Vercel says `No project table found`

Confirm that `pyproject.toml` contains a `[project]` section. The current project file includes the required dependencies and Python version.

### Browser shows `Backend offline`

Check the following:

- Render service is running.
- `VISION_API_URL` contains the Render URL, not the Vercel URL.
- The Render URL uses `https://`.
- Render was redeployed after changing `CORS_ORIGIN`.
- Open `/api/status` on the Render URL and confirm it returns JSON.

### Browser reports a CORS error

Set Render's `CORS_ORIGIN` to the exact Vercel URL. Do not add a trailing slash. Redeploy Render after saving it.

### Render build fails while installing OpenCV or Ultralytics

Check the Render logs and use a service with enough memory. The first YOLO model startup can take several minutes. Keep the Render worker count at `1` for the free or smallest instance.

### YOLO is unavailable

The Haar face detector can still work. Check the `/api/status` response and Render logs. YOLO model startup may take longer on the first request.

## 9. Redeploying After Code Changes

Push changes to GitHub:

```powershell
git add .
git commit -m "Update Vision Forge"
git push origin main
```

Vercel and Render will normally redeploy automatically when connected to the `main` branch.

## 10. Final URLs

Record these two URLs after deployment:

```text
Frontend: https://YOUR-VERCEL-PROJECT.vercel.app
API:      https://YOUR-RENDER-SERVICE.onrender.com
```

Share the **Frontend** URL with users. Keep the API URL available for troubleshooting.
