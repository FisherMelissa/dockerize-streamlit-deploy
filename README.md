# dockerize-streamlit-deploy
Bayesian Network do-intervention Demo
# cto.new
Ai-service

## Bayesian Network do-intervention Demo

This repository includes a Bayesian Network demonstration for modeling risk/protection factors and their impact on school discipline outcomes using do-intervention analysis.

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the Bayesian Network do-intervention demo:

```bash
python scripts/bn_do_intervention.py
```

### Features

- Bayesian Network with nodes: ParentingStyle, PeerRisk, LawEdu, NightNet, PsychReg, SchoolDiscipline
- Synthetic data generation consistent with the DAG structure
- Do-intervention analysis to evaluate causal effects
- Comparison of baseline vs intervention probabilities
- Student trajectory sampling

### Network Structure

The Bayesian Network models the following relationships:
- ParentingStyle → PsychReg
- ParentingStyle → PeerRisk  
- PeerRisk → SchoolDiscipline
- NightNet → SchoolDiscipline
- LawEdu → SchoolDiscipline
- PsychReg → SchoolDiscipline

### Interventions Analyzed

- `do(ParentingStyle=protect)`: Setting parenting style to protective
- `do(LawEdu=high)`: Setting law education to high level
- `do(NightNet=low)`: Setting night net usage to low level

The script compares the probability of `SchoolDiscipline=high` under baseline conditions vs each intervention.

## ZINB Simulation

This repository includes a Zero-Inflated Negative Binomial (ZINB) simulation script for modeling monthly incident counts with many zeros.

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the ZINB simulation:

```bash
python scripts/zinb_simulation.py
```

Optional parameters:

```bash
python scripts/zinb_simulation.py --runs 2000 --seed 123 --plot --months 24
```

- `--runs N`: Number of Monte Carlo simulations (default: 1000)
- `--seed N`: Random seed for reproducibility (default: 42)
- `--plot`: Generate plots if matplotlib is available
- `--months N`: Number of months to simulate (default: 24)

### Features

- Synthetic dataset generation with zero inflation
- ZINB model fitting with logit inflation model
- Forward simulation (deterministic + Monte Carlo)
- Intervention vs no-intervention scenario comparison
- Coefficient summary and in-sample expected counts
- Optional plot generation

## Streamlit Dashboard

This repository includes an interactive Streamlit dashboard that provides a web-based interface for both the Bayesian Network and ZINB simulations.

### Local Development

Run the Streamlit app locally:

```bash
streamlit run app/app.py
```

The app will be available at `http://localhost:8501`

### Docker Deployment

Build and run the app using Docker:

```bash
# Build the Docker image
docker build -t risk-sim-app .

# Run the container
docker run -p 8501:8501 risk-sim-app
```

Access the app at `http://localhost:8501`

### Deployment Options

#### 1. Streamlit Community Cloud

Streamlit Community Cloud offers free hosting for Streamlit apps.

**Steps:**
1. Push your code to a GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository, branch (`main` or `feat/dockerize-streamlit-deploy`), and file path (`app/app.py`)
6. Click "Deploy"

**Configuration:**
- No environment variables needed
- The app will automatically use `.streamlit/config.toml` settings
- Streamlit Community Cloud automatically installs dependencies from `requirements.txt`

#### 2. Hugging Face Spaces

Hugging Face Spaces supports both Docker and Streamlit SDK deployments.

**Option A: Using Streamlit SDK**

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select "Streamlit" as the SDK
3. Clone the Space repository:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   ```
4. Copy the required files:
   ```bash
   cp -r app/* YOUR_SPACE_NAME/
   cp requirements.txt YOUR_SPACE_NAME/
   cp -r .streamlit YOUR_SPACE_NAME/
   ```
5. Create a `README.md` in your Space with the following header:
   ```yaml
   ---
   title: Risk Simulation Dashboard
   emoji: 📊
   colorFrom: blue
   colorTo: green
   sdk: streamlit
   sdk_version: 1.28.0
   app_file: app.py
   pinned: false
   ---
   ```
6. Push to Hugging Face:
   ```bash
   cd YOUR_SPACE_NAME
   git add .
   git commit -m "Initial commit"
   git push
   ```

**Option B: Using Docker**

1. Create a new Space with Docker SDK
2. Copy the entire repository to your Space
3. Ensure the `Dockerfile` is in the root directory
4. Push to Hugging Face (the Space will automatically build from the Dockerfile)

**Environment Variables:**
- No special environment variables required
- PORT is automatically set by Hugging Face (default: 7860)

#### 3. Render

Render provides easy deployment with automatic builds from Git.

**Steps:**
1. Push your code to a GitHub repository
2. Visit [render.com](https://render.com) and sign up/login
3. Click "New +" and select "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - **Name:** risk-sim-app (or your preferred name)
   - **Environment:** Docker
   - **Region:** Choose your preferred region
   - **Branch:** `main` or `feat/dockerize-streamlit-deploy`
   - **Build Command:** (leave empty, uses Dockerfile)
   - **Start Command:** (leave empty, uses Dockerfile CMD)

**Alternative: Using Buildpack (without Docker)**

If you prefer not to use Docker:
1. Select "Python 3" as the environment instead of Docker
2. Set the following:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0`

**Environment Variables:**
- `PORT` - Automatically set by Render
- `PYTHON_VERSION` - Set to `3.11.0` (optional, if using buildpack method)

**Note:** Render automatically sets the `$PORT` environment variable, which is used by both the Dockerfile and the Procfile.

### Environment Variables

The application uses the following environment variables:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Port for the Streamlit server | 8501 | No |

### Platform-Specific Notes

- **Streamlit Community Cloud**: Automatically handles all configuration. Just deploy and go!
- **Hugging Face Spaces**: Works with both Streamlit SDK and Docker. The Streamlit SDK option is simpler and faster to deploy.
- **Render**: Supports both Docker and buildpack deployments. Docker is recommended for consistency.
- **Railway**: Similar to Render. Use the Procfile for buildpack deployment or Dockerfile for Docker deployment.

### Troubleshooting

**Port Issues:**
- Ensure the `PORT` environment variable is properly set
- The app should listen on `0.0.0.0` (all interfaces) not `localhost`

**Memory Issues:**
- The ZINB simulation can be memory-intensive with large numbers of Monte Carlo runs
- Consider reducing the default number of runs for cloud deployments with limited memory

**Dependencies:**
- All required dependencies are listed in `requirements.txt`
- Ensure `statsmodels>=0.14` is installed for ZINB functionality

### Features

The Streamlit dashboard includes:
- **Bayesian Network Tab**: Interactive do-intervention analysis with visualization
- **ZINB Simulation Tab**: Zero-inflated negative binomial modeling with Monte Carlo simulation
- Adjustable parameters for both simulations
- Interactive visualizations using Plotly
- Export capabilities for results
