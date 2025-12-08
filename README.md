# 🩺 Kidney Disease Classification using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Latest-green.svg)
![DVC](https://img.shields.io/badge/DVC-Enabled-blueviolet.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.2.2-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-yellow.svg)
![AWS](https://img.shields.io/badge/AWS-Deployed-orange.svg)

**An end-to-end deep learning solution for automated kidney disease classification from CT scan images**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Deployment](#-deployment)

</div>

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Use Cases](#-use-cases)
- [Features](#-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Installation & Setup](#-installation--setup)
  - [Prerequisites](#prerequisites)
  - [Local Development Setup](#local-development-setup)
  - [DVC Setup](#dvc-setup)
  - [MLflow Setup](#mlflow-setup)
  - [AWS Configuration](#aws-configuration)
- [Usage](#-usage)
- [ML Pipeline Stages](#-ml-pipeline-stages)
- [Model Training](#-model-training)
- [Model Evaluation](#-model-evaluation)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [AWS Deployment](#-aws-deployment)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Introduction

The **Kidney Disease Classification** project is a production-ready, end-to-end deep learning application designed to automatically classify kidney CT scan images into different categories. This project leverages transfer learning with VGG16 architecture, implements MLOps best practices using DVC and MLflow, and provides a user-friendly web interface for real-time predictions.

This system aims to assist medical professionals in early detection and diagnosis of kidney diseases by providing accurate, fast, and automated analysis of CT scan images.

### 🎓 Key Highlights

- **Transfer Learning**: Utilizes pre-trained VGG16 model fine-tuned on kidney CT scans
- **MLOps Integration**: Complete experiment tracking with MLflow and version control with DVC
- **Production Ready**: Dockerized application with CI/CD pipeline using GitHub Actions
- **Cloud Deployment**: Automated deployment on AWS EC2 with ECR for container registry
- **Modular Design**: Clean, maintainable code following software engineering best practices
- **REST API**: Flask-based API for easy integration with other systems


---

## ✨ Features

### 🤖 Machine Learning Features

- ✅ **Transfer Learning with VGG16**: Pre-trained ImageNet weights fine-tuned for kidney disease classification
- ✅ **Data Augmentation**: Robust training with image augmentation techniques
- ✅ **Automated Training Pipeline**: End-to-end automated ML pipeline with DVC
- ✅ **Experiment Tracking**: Complete experiment tracking and model versioning with MLflow
- ✅ **Model Evaluation**: Comprehensive evaluation metrics and performance monitoring

### 🛠️ Engineering Features

- ✅ **Modular Architecture**: Clean separation of concerns with components, entities, and pipelines
- ✅ **Configuration Management**: YAML-based configuration for easy parameter tuning
- ✅ **Logging System**: Comprehensive logging for debugging and monitoring
- ✅ **Error Handling**: Robust error handling and validation
- ✅ **Type Hints**: Full type annotations for better code quality

### 🌐 Web Application Features

- ✅ **REST API**: Flask-based RESTful API for predictions
- ✅ **Web Interface**: User-friendly HTML interface for image upload and prediction
- ✅ **CORS Support**: Cross-origin resource sharing enabled
- ✅ **Real-time Predictions**: Instant classification results
- ✅ **Base64 Image Support**: Direct image upload via API

### 🚀 DevOps Features

- ✅ **Dockerization**: Complete Docker containerization for consistent deployments
- ✅ **CI/CD Pipeline**: Automated testing, building, and deployment with GitHub Actions
- ✅ **AWS Integration**: Deployment on AWS EC2 with ECR container registry
- ✅ **Version Control**: Git-based version control with DVC for data and models
- ✅ **Environment Management**: Secure environment variable management

---

## 🎬 Demo

### Web Interface
The application provides an intuitive web interface where users can:
1. Upload kidney CT scan images
2. Get instant classification results
3. View confidence scores
4. Access training functionality

### API Usage Example

```bash
# Health check
curl http://localhost:8080/

# Trigger training
curl -X POST http://localhost:8080/train

# Make prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_string"}'
```

---

## 🔧 Tech Stack

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8 | Core programming language |
| **TensorFlow** | 2.12.0 | Deep learning framework |
| **Flask** | Latest | Web framework for API |
| **DVC** | Latest | Data version control |
| **MLflow** | 2.2.2 | Experiment tracking & model registry |

### ML & Data Science

- **TensorFlow/Keras**: Model building and training
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **SciPy**: Scientific computing

### DevOps & Cloud

- **Docker**: Containerization
- **GitHub Actions**: CI/CD automation
- **AWS EC2**: Cloud compute
- **AWS ECR**: Container registry
- **AWS CLI**: AWS management

### Development Tools

- **python-box**: Configuration management
- **PyYAML**: YAML parsing
- **python-dotenv**: Environment management
- **tqdm**: Progress bars
- **gdown**: Google Drive downloads
- **Flask-CORS**: Cross-origin support

---

## 🏗️ Project Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                    (Web App / API Client)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Flask REST API                              │
│                    (app.py - Port 8080)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Prediction Pipeline                            │
│              (Real-time Inference Engine)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Trained ML Model                              │
│              (VGG16 Transfer Learning)                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Data Ingestion    → Download & Extract Dataset        │
│  Stage 2: Base Model Prep   → Load & Configure VGG16            │
│  Stage 3: Model Training    → Fine-tune on Kidney Data          │
│  Stage 4: Model Evaluation  → Validate & Log to MLflow          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Infrastructure                          │
├─────────────────────────────────────────────────────────────────┤
│  DVC: Data & Model Versioning                                   │
│  MLflow: Experiment Tracking & Model Registry                   │
│  GitHub Actions: CI/CD Automation                               │
│  Docker: Containerization                                       │
│  AWS: Cloud Deployment (EC2 + ECR)                              │
└─────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
src/cnnClassifier/
├── components/          # Core ML components
│   ├── data_ingestion.py
│   ├── prepare_base_model.py
│   ├── model_training.py
│   └── model_evaluation_mlflow.py
├── config/             # Configuration management
├── entity/             # Data classes and entities
├── pipeline/           # Training and prediction pipelines
├── utils/              # Utility functions
└── constants/          # Constants and paths
```

---

## 📊 Dataset

The project uses kidney CT scan images categorized into different classes. The dataset is automatically downloaded from Google Drive during the data ingestion stage.

### Dataset Structure
```
kidney-ct-scan-image/
├── Normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Tumor/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Dataset Specifications
- **Image Size**: 224x224x3 (RGB)
- **Classes**: 2 (Normal, Tumor)
- **Format**: JPEG images
- **Source**: Medical CT scans

---

## 🚀 Installation & Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8** or higher
- **Git** for version control
- **pip** package manager
- **virtualenv** or **conda** for environment management
- **Docker** (optional, for containerized deployment)
- **AWS CLI** (optional, for cloud deployment)

### Local Development Setup

#### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Adiaparmar/Kidney-Disease-Classification.git

# Navigate to project directory
cd Kidney-Disease-Classification
```

#### 2. Create Virtual Environment

**Using virtualenv:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

**Using conda:**
```bash
# Create conda environment
conda create -n kidney-classifier python=3.8 -y

# Activate environment
conda activate kidney-classifier
```

#### 3. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

#### 4. Setup Environment Variables

Create a `.env` file in the project root:

```bash
# .env file
MLFLOW_TRACKING_URI=https://dagshub.com/Adiaparmar/Kidney-Disease-Classification.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
```

**Note**: Never commit the `.env` file to version control. It's already included in `.gitignore`.

---

### DVC Setup

DVC (Data Version Control) is used for managing datasets and model versions.

#### 1. Initialize DVC

```bash
# DVC is already initialized in this project
# To verify DVC installation
dvc version
```

#### 2. Configure DVC Remote Storage (Optional)

If you want to use remote storage for DVC:

```bash
# Add remote storage (e.g., AWS S3)
dvc remote add -d myremote s3://your-bucket-name/path

# Configure AWS credentials
dvc remote modify myremote access_key_id 'your-access-key'
dvc remote modify myremote secret_access_key 'your-secret-key'
```

#### 3. Pull Data and Models

```bash
# Pull data and models from remote storage
dvc pull
```

#### 4. Reproduce Pipeline

```bash
# Run the entire ML pipeline
dvc repro

# Run specific stage
dvc repro -s data_ingestion
dvc repro -s prepare_base_model
dvc repro -s training
dvc repro -s evaluation
```

#### 5. DVC Commands Reference

```bash
# Check pipeline status
dvc status

# Show pipeline DAG
dvc dag

# Track new data
dvc add data/new_dataset

# Push changes to remote
dvc push

# View metrics
dvc metrics show

# Compare experiments
dvc metrics diff
```

---

### MLflow Setup

MLflow is used for experiment tracking, model versioning, and model registry.

#### 1. Local MLflow Setup

```bash
# Start MLflow UI locally
mlflow ui

# Access at http://localhost:5000
```

#### 2. DagsHub Integration (Recommended)

This project uses DagsHub for remote MLflow tracking:

1. **Create DagsHub Account**
   - Go to [dagshub.com](https://dagshub.com)
   - Sign up for a free account

2. **Create Repository**
   - Create a new repository or connect existing GitHub repo
   - Enable MLflow tracking

3. **Configure Credentials**
   - Get your tracking URI from DagsHub
   - Add credentials to `.env` file:
   ```bash
   MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
   MLFLOW_TRACKING_USERNAME=your_username
   MLFLOW_TRACKING_PASSWORD=your_token
   ```

4. **Verify Connection**
   ```bash
   # Run evaluation to test MLflow logging
   python src/cnnClassifier/pipeline/stage_04_model_evaluation_mlflow.py
   ```

#### 3. MLflow Features Used

- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version and manage trained models
- **Artifact Storage**: Store model files and plots
- **Metric Visualization**: Compare experiments and visualize metrics

#### 4. MLflow Commands Reference

```bash
# View experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-id 0

# Serve model
mlflow models serve -m "models:/kidney-classifier/Production" -p 5001

# Compare runs
mlflow ui --backend-store-uri ./mlruns
```

---

### AWS Configuration

#### 1. Install AWS CLI

**Windows:**
```bash
# Download and install from AWS website
# Or use chocolatey
choco install awscli
```

**macOS:**
```bash
brew install awscli
```

**Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

#### 2. Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Enter your credentials when prompted:
# AWS Access Key ID: your_access_key
# AWS Secret Access Key: your_secret_key
# Default region name: us-east-1
# Default output format: json
```

#### 3. Verify AWS Configuration

```bash
# Test AWS connection
aws sts get-caller-identity

# List S3 buckets (if you have any)
aws s3 ls
```

#### 4. Create AWS Resources

**Create ECR Repository:**
```bash
# Create ECR repository for Docker images
aws ecr create-repository --repository-name kidney-classifier --region us-east-1
```

**Create EC2 Instance:**
1. Go to AWS Console → EC2
2. Launch Instance
3. Choose Ubuntu Server 22.04 LTS
4. Instance type: t2.medium or higher
5. Configure security group:
   - Allow SSH (port 22)
   - Allow HTTP (port 80)
   - Allow Custom TCP (port 8080)
6. Create or select key pair
7. Launch instance

**Setup EC2 Instance:**
```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
sudo apt-get install docker.io -y
sudo usermod -aG docker ubuntu
newgrp docker

# Install AWS CLI
sudo apt-get install awscli -y

# Configure as self-hosted runner (see CI/CD section)
```

#### 5. Configure GitHub Secrets

Add the following secrets to your GitHub repository:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `ECR_REPOSITORY_NAME`
- `AWS_ECR_LOGIN_URI`

Go to: Repository → Settings → Secrets and variables → Actions → New repository secret

---

## 📖 Usage

### Running the Application Locally

#### 1. Train the Model

**Option A: Using main.py (All stages)**
```bash
# Run complete training pipeline
python main.py
```

**Option B: Using DVC**
```bash
# Run pipeline with DVC
dvc repro
```

**Option C: Individual stages**
```bash
# Stage 1: Data Ingestion
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py

# Stage 2: Prepare Base Model
python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py

# Stage 3: Model Training
python src/cnnClassifier/pipeline/stage_03_model_training.py

# Stage 4: Model Evaluation
python src/cnnClassifier/pipeline/stage_04_model_evaluation_mlflow.py
```

#### 2. Start the Web Application

```bash
# Run Flask application
python app.py

# Application will start at http://localhost:8080
```

#### 3. Make Predictions

**Using Web Interface:**
1. Open browser and go to `http://localhost:8080`
2. Upload a kidney CT scan image
3. Click "Predict"
4. View classification results

**Using API:**
```python
import requests
import base64

# Read and encode image
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make prediction request
response = requests.post(
    "http://localhost:8080/predict",
    json={"image": image_data}
)

print(response.json())
```

**Using cURL:**
```bash
# Encode image to base64
base64 test_image.jpg > encoded_image.txt

# Make prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(cat encoded_image.txt)\"}"
```

---

## 🔄 ML Pipeline Stages

### Stage 1: Data Ingestion

**Purpose**: Download and prepare the dataset

**Process**:
1. Downloads dataset from Google Drive
2. Extracts ZIP file
3. Organizes data into training structure

**Configuration** (`config/config.yaml`):
```yaml
data_ingestion:
  root_dir: artifacts/data_ingestion
  source_url: https://drive.google.com/file/d/...
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion
```

**Run**:
```bash
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
```

---

### Stage 2: Prepare Base Model

**Purpose**: Load and configure VGG16 base model

**Process**:
1. Loads pre-trained VGG16 with ImageNet weights
2. Removes top layers
3. Adds custom classification layers
4. Freezes base model layers
5. Compiles model with optimizer

**Configuration** (`params.yaml`):
```yaml
IMAGE_SIZE: [224, 224, 3]
INCLUDE_TOP: False
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.02
```

**Run**:
```bash
python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
```

---

### Stage 3: Model Training

**Purpose**: Train the model on kidney CT scan data

**Process**:
1. Loads prepared base model
2. Sets up data generators with augmentation
3. Trains model with specified epochs
4. Saves trained model

**Configuration** (`params.yaml`):
```yaml
AUGMENTATION: True
EPOCHS: 2
BATCH_SIZE: 16
```

**Features**:
- Data augmentation (rotation, flip, zoom)
- Batch processing
- Progress tracking
- Model checkpointing

**Run**:
```bash
python src/cnnClassifier/pipeline/stage_03_model_training.py
```

---

### Stage 4: Model Evaluation

**Purpose**: Evaluate model and log metrics to MLflow

**Process**:
1. Loads trained model
2. Evaluates on validation set
3. Calculates metrics (loss, accuracy)
4. Logs to MLflow
5. Saves metrics to JSON

**Metrics Tracked**:
- Loss
- Accuracy
- Model parameters
- Training configuration

**Run**:
```bash
python src/cnnClassifier/pipeline/stage_04_model_evaluation_mlflow.py
```

**View Results**:
```bash
# View metrics file
cat scores.json

# View in MLflow UI
mlflow ui
```

---

## 🎯 Model Training

### Training Configuration

Edit `params.yaml` to customize training:

```yaml
# Image preprocessing
IMAGE_SIZE: [224, 224, 3]  # Input image dimensions

# Data augmentation
AUGMENTATION: True          # Enable/disable augmentation

# Training parameters
BATCH_SIZE: 16             # Batch size for training
EPOCHS: 2                  # Number of training epochs
LEARNING_RATE: 0.02        # Learning rate for optimizer

# Model architecture
INCLUDE_TOP: False         # Use VGG16 without top layers
CLASSES: 2                 # Number of output classes
WEIGHTS: imagenet          # Pre-trained weights
```

### Training Process

```bash
# Full training pipeline
python main.py

# Monitor training progress
# Check logs/running_logs.log for detailed logs
```

### Training Tips

1. **Increase Epochs**: For better accuracy, increase epochs to 20-50
2. **Adjust Batch Size**: Reduce if running out of memory
3. **Learning Rate**: Tune for optimal convergence
4. **Data Augmentation**: Enable for better generalization

---

## 📊 Model Evaluation

### Evaluation Metrics

The model is evaluated using:
- **Loss**: Categorical cross-entropy loss
- **Accuracy**: Classification accuracy on validation set

### View Evaluation Results

**1. Scores JSON:**
```bash
cat scores.json
```

**2. MLflow UI:**
```bash
mlflow ui
# Open http://localhost:5000
```

**3. DagsHub (if configured):**
Visit your DagsHub repository to view experiments

### Model Performance

Typical performance metrics:
- Training Accuracy: ~95%+
- Validation Accuracy: ~90%+
- Loss: <0.3

---

## 🌐 API Documentation

### Endpoints

#### 1. Home Page
```
GET /
```
**Description**: Renders the web interface

**Response**: HTML page

---

#### 2. Train Model
```
POST /train
GET /train
```
**Description**: Triggers the complete training pipeline

**Response**:
```json
"Training done successfully"
```

**Example**:
```bash
curl -X POST http://localhost:8080/train
```

---

#### 3. Predict
```
POST /predict
```
**Description**: Classifies uploaded kidney CT scan image

**Request Body**:
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response**:
```json
{
  "prediction": "Normal" // or "Tumor"
}
```

**Example**:
```python
import requests
import base64

# Encode image
with open("scan.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8080/predict",
    json={"image": img_base64}
)

print(response.json())
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
# Build image
docker build -t kidney-classifier:latest .

# Verify image
docker images | grep kidney-classifier
```

### Run Docker Container

```bash
# Run container
docker run -d -p 8080:8080 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_REGION=us-east-1 \
  --name kidney-app \
  kidney-classifier:latest

# Check container status
docker ps

# View logs
docker logs kidney-app

# Stop container
docker stop kidney-app

# Remove container
docker rm kidney-app
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_REGION=${AWS_REGION}
    volumes:
      - ./model:/app/model
      - ./artifacts:/app/artifacts
```

Run with:
```bash
docker-compose up -d
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The project uses GitHub Actions for automated CI/CD with three main jobs:

#### 1. Continuous Integration
- Checkout code
- Run linting
- Execute unit tests

#### 2. Continuous Delivery
- Build Docker image
- Tag image
- Push to AWS ECR

#### 3. Continuous Deployment
- Pull latest image from ECR
- Deploy to EC2 instance
- Run container
- Clean up old images

### Setup GitHub Actions

#### 1. Configure Secrets

Add these secrets in GitHub repository settings:
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_REPOSITORY_NAME
AWS_ECR_LOGIN_URI
```

#### 2. Setup Self-Hosted Runner

On your EC2 instance:

```bash
# Navigate to repository settings → Actions → Runners → New self-hosted runner

# Download runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure runner
./config.sh --url https://github.com/Adiaparmar/Kidney-Disease-Classification \
  --token YOUR_TOKEN

# Install and start service
sudo ./svc.sh install
sudo ./svc.sh start
```

#### 3. Trigger Workflow

```bash
# Push to main branch
git add .
git commit -m "Update application"
git push origin main

# Workflow will automatically trigger
```

### Workflow File

Located at `.github/workflows/main.yaml`

Key features:
- Triggers on push to main branch
- Ignores README.md changes
- Uses AWS credentials from secrets
- Builds and pushes to ECR
- Deploys to self-hosted EC2 runner

---

## ☁️ AWS Deployment

### Architecture Overview

```
GitHub → GitHub Actions → AWS ECR → AWS EC2
```

### Step-by-Step Deployment

#### 1. Create ECR Repository

```bash
# Create repository
aws ecr create-repository \
  --repository-name kidney-classifier \
  --region us-east-1

# Note the repository URI
```

#### 2. Launch EC2 Instance

**Instance Specifications**:
- AMI: Ubuntu Server 22.04 LTS
- Instance Type: t2.medium (minimum)
- Storage: 20 GB
- Security Group: Allow ports 22, 80, 8080

**User Data Script** (optional):
```bash
#!/bin/bash
apt-get update
apt-get install -y docker.io awscli
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker
```

#### 3. Configure EC2 Instance

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@ec2-public-ip

# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker ubuntu
newgrp docker

# Install AWS CLI
sudo apt-get install -y awscli

# Configure AWS
aws configure
```

#### 4. Setup GitHub Runner

Follow the self-hosted runner setup instructions from GitHub Actions section.

#### 5. Deploy Application

**Manual Deployment**:
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin your-ecr-uri

# Pull image
docker pull your-ecr-uri/kidney-classifier:latest

# Run container
docker run -d -p 8080:8080 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_REGION=us-east-1 \
  --name kidney-app \
  your-ecr-uri/kidney-classifier:latest
```

**Automated Deployment**:
Push to main branch, GitHub Actions will handle deployment.

#### 6. Access Application

```
http://your-ec2-public-ip:8080
```

### Monitoring and Maintenance

```bash
# Check container status
docker ps

# View logs
docker logs kidney-app

# Restart container
docker restart kidney-app

# Update application
docker pull your-ecr-uri/kidney-classifier:latest
docker stop kidney-app
docker rm kidney-app
docker run -d -p 8080:8080 --name kidney-app your-ecr-uri/kidney-classifier:latest

# Clean up
docker system prune -f
```

---

## 📁 Project Structure

```
Kidney-Disease-Classification/
│
├── .github/
│   └── workflows/
│       └── main.yaml              # CI/CD pipeline configuration
│
├── artifacts/                      # Generated artifacts (gitignored)
│   ├── data_ingestion/            # Downloaded and extracted data
│   ├── prepare_base_model/        # Base model files
│   └── training/                  # Trained model files
│
├── config/
│   └── config.yaml                # Main configuration file
│
├── logs/
│   └── running_logs.log           # Application logs
│
├── mlruns/                        # MLflow experiment tracking data
│
├── model/                         # Final production model
│
├── research/                      # Jupyter notebooks for experimentation
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   └── cnnClassifier/
│       ├── __init__.py
│       ├── components/            # Core ML components
│       │   ├── data_ingestion.py
│       │   ├── prepare_base_model.py
│       │   ├── model_training.py
│       │   └── model_evaluation_mlflow.py
│       ├── config/                # Configuration management
│       │   └── configuration.py
│       ├── constants/             # Constants and paths
│       │   └── __init__.py
│       ├── entity/                # Data classes
│       │   └── config_entity.py
│       ├── pipeline/              # Training and prediction pipelines
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_prepare_base_model.py
│       │   ├── stage_03_model_training.py
│       │   ├── stage_04_model_evaluation_mlflow.py
│       │   └── prediction.py
│       └── utils/                 # Utility functions
│           └── common.py
│
├── templates/
│   └── index.html                 # Web interface
│
├── .dvcignore                     # DVC ignore file
├── .env                           # Environment variables (gitignored)
├── .gitignore                     # Git ignore file
├── app.py                         # Flask application
├── Dockerfile                     # Docker configuration
├── dvc.lock                       # DVC pipeline lock file
├── dvc.yaml                       # DVC pipeline definition
├── main.py                        # Main training script
├── params.yaml                    # Model parameters
├── requirements.txt               # Python dependencies
├── scores.json                    # Model evaluation scores
├── setup.py                       # Package setup
└── README.md                      # This file
```

---

## ⚙️ Configuration

### config.yaml

Main configuration file for pipeline stages:

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_url: https://drive.google.com/file/d/...
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.h5
  updated_base_model_path: artifacts/prepare_base_model/base_model_updated.h5

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5
```


## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the Repository**
   ```bash
   # Click 'Fork' button on GitHub
   ```

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/your-username/Kidney-Disease-Classification.git
   cd Kidney-Disease-Classification
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests if applicable

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   ```

6. **Push to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**
   - Go to GitHub and create a pull request
   - Describe your changes clearly
   - Reference any related issues

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions and classes
- Update documentation for new features
- Ensure all tests pass before submitting PR
- Keep commits atomic and well-described

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🧪 Additional tests
- 🎨 UI/UX enhancements
- ⚡ Performance optimizations

---

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 Adiaparmar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contact

### Author

**Adiaparmar**

- 📧 Email: adiaparmar@gmail.com
- 🐙 GitHub: [@Adiaparmar](https://github.com/Adiaparmar)
- 🔗 Repository: [Kidney-Disease-Classification](https://github.com/Adiaparmar/Kidney-Disease-Classification)

### Support

For questions, issues, or suggestions:

1. **Issues**: [GitHub Issues](https://github.com/Adiaparmar/Kidney-Disease-Classification/issues)
2. **Discussions**: [GitHub Discussions](https://github.com/Adiaparmar/Kidney-Disease-Classification/discussions)
3. **Email**: adiaparmar@gmail.com

---


## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [AWS Documentation](https://docs.aws.amazon.com/)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ by Adiaparmar**

[⬆ Back to Top](#-kidney-disease-classification-using-deep-learning)

</div>