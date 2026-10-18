# Pet Breed Classifier - MLOps Pipeline (Track C)

This repository contains a complete end-to-end Machine Learning Operations (MLOps) pipeline for a Pet Breed Classification model, fulfilling all Track C requirements.

##  Project Overview
- **Model**: ResNet18 trained on a custom pet breed dataset.
- **Serving**: BentoML API with Docker containerization.
- **CI/CD**: GitHub Actions for linting, ONNX logit agreement tests, and baseline quality gates.
- **Monitoring**: Evidently (Data Drift) + Prometheus & Grafana (System/API Metrics).
- **Optimization**: Post-Training Quantization (PTQ) to INT8 using ONNX Runtime.

### 0. Environment Setup
This project uses `pyproject.toml` for modern dependency management. Before running anything, install the required packages:

```bash
pip install .
```

### 1. Download Model Weights
Due to recent Google Drive Security policies blocking the default DVC OAuth app, we provided a fallback script to fetch the models directly from GitHub Releases:

```bash
bash pull_models.sh
```

### 2. Run Quality Gates (Optional)
To verify that the model weights are loaded correctly and the ONNX model matches the PyTorch baseline, run the tests:

```bash
python test_logits.py
```

### 3. Build and Containerize the Model Service
Pack the model and the API into a Bento, then build the Docker image so it can be picked up by Docker Compose:

```bash
bentoml build
bentoml containerize pet_breed_classifier:latest -t pet-breed:latest
```
*(Note to Reviewer: If you run this in GitHub Codespaces and encounter a "No space left on device" error, you can free up space by running `docker system prune -a -f --volumes` before running the containerize command).*

### 4. Start the API and Monitoring Stack
Run the following command to spin up the containers:

```bash
docker-compose up -d
```

### 5. Access the Services
Once the stack is running, you can access:
- **BentoML API (Swagger UI):** [http://localhost:8000](http://localhost:8000) *(Use this to test image classification)*
- **Prometheus:** [http://localhost:9090](http://localhost:9090)
- **Grafana:** [http://localhost:3000](http://localhost:3000) (User: `admin`, Password: `admin`)

### ⚠️ Troubleshooting: Constrained Environments
If you are reviewing this project in a constrained environment (e.g., GitHub Codespaces) and the Docker containerization in Step 3 fails , you can bypass Docker to quickly test the model. 

Run the lightweight BentoML server directly:
```bash
bentoml serve pet_breed_classifier:latest
```
*The API will then be available at [http://localhost:3000](http://localhost:3000).*

### 6. Data Drift Detection
To run the drift detector on the corrupted data and generate a new report:
    python monitor_drift.py

*Note: this option is not available now because we can no access data from DVC Due to Google drive security -> A pre-generated HTML report is already available in the reports/ directory.*

##  Repository Structure
- service.py: BentoML API definition.
- bentofile.yaml: BentoML build configuration.
- monitor_drift.py: Evidently drift detection script.
- .github/workflows/ci.yml: Automated CI/CD pipeline.
- test_baseline.py & test_logits.py: Quality gate tests.
- docker-compose.yml: Orchestration for API, Prometheus, and Grafana.
- prometheus.yml: Prometheus scraping configuration.
- reports/: HTML reports for data drift detection.
