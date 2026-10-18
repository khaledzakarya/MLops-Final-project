# Pet Breed Classifier - MLOps Pipeline (Track C)

This repository contains a complete end-to-end Machine Learning Operations (MLOps) pipeline for a Pet Breed Classification model, fulfilling all Track C requirements.

##  Project Overview
- **Model**: ResNet18 trained on a custom pet breed dataset.
- **Serving**: BentoML API with Docker containerization.
- **CI/CD**: GitHub Actions for linting, ONNX logit agreement tests, and baseline quality gates.
- **Monitoring**: Evidently (Data Drift) + Prometheus & Grafana (System/API Metrics).
- **Optimization**: Post-Training Quantization (PTQ) to INT8 using ONNX Runtime.

##  How to Run

### 1. Start the API and Monitoring Stack
Run the following command:
    docker-compose up -d

This single command will spin up:
- **BentoML API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (User: admin, Password: admin)

### 2. Data Drift Detection
To run the drift detector on the corrupted data and generate a new report:
    python monitor_drift.py

*Note: A pre-generated HTML report is already available in the reports/ directory.*

##  Repository Structure
- service.py: BentoML API definition.
- bentofile.yaml: BentoML build configuration.
- monitor_drift.py: Evidently drift detection script.
- .github/workflows/ci.yml: Automated CI/CD pipeline.
- test_baseline.py & test_logits.py: Quality gate tests.
- docker-compose.yml: Orchestration for API, Prometheus, and Grafana.
- prometheus.yml: Prometheus scraping configuration.
- reports/: HTML reports for data drift detection.
