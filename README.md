# 🖌️ Dig-Reg: Handwritten Digits Classification

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub--Actions-blue)

---

## **Overview**

**Dig-Reg** is a full-stack machine learning project for **handwritten digits classification**.  
It allows users to **draw digits (0–9) in the browser** and predict them in real-time using a trained **Random Forest model**.  

Key technologies include:

- **Streamlit** for interactive frontend
- **FastAPI** for backend API
- **MLflow** for experiment tracking and model registry
- **Docker** for containerized deployment
- **GitHub Actions** for CI/CD automation

---

## **Features**

- 🎨 Draw digits on a canvas in the web interface
- 🔮 Real-time prediction with a trained model
- 📊 MLflow experiment tracking: metrics, parameters, artifacts, and model registry
- 🔌 API endpoint to integrate with other applications
- 🐳 Dockerized deployment for easy scaling and portability
- 🚀 Automated CI/CD pipeline for preprocessing, training, Docker build, and deployment

---

## **Project Structure**

```
Dig-Reg/
│
├── data/processed/                   # Preprocessed train/val/test CSVs
├── src/
│   ├── app.py                       # Streamlit frontend
│   ├── main.py                      # FastAPI backend
│   ├── train.py                     # Model training & MLflow logging
│   └── process_data.py              # Dataset preprocessing
├── models/                          # Trained model artifacts
├── mlruns/                          # MLflow experiment tracking data
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker image configuration
├── .gitignore                       # Git ignore patterns
└── .github/workflows/main.yaml      # CI/CD workflow
```

---

## **Installation**

### Prerequisites
- Python 3.12+
- Docker (optional, for containerized deployment)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/munikumar229/Dig-Reg.git
cd Dig-Reg
```

2. **Create and activate virtual environment:**
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## **Usage**

### **Running Locally**

1. **Preprocess the data:**
```bash
python src/process_data.py
```

2. **Train the model and log to MLflow:**
```bash
python src/train.py
```

3. **Run FastAPI backend:**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8502
```

4. **Run Streamlit frontend (in another terminal):**
```bash
streamlit run src/app.py
```

5. **Open your browser at:** `http://localhost:8501`

---

## **Docker Deployment**

Build and run the Docker container:

```bash
# Build the Docker image
docker build -t mlops-digits:latest .

# Run the container
docker run -p 8502:8502 mlops-digits:latest
```

The app will be available at `http://localhost:8502`.

---

## **CI/CD Pipeline**

Automated pipeline using GitHub Actions includes:

1. **Data Preprocessing** (`process_data.py`)
2. **Model Training** (`train.py`)
3. **Docker Image Build & Push**
4. **Optional Deployment** to cloud platforms

Workflow is defined in `.github/workflows/main.yaml`.

---

## **MLflow Tracking**

Track experiments, metrics, parameters, and models:

```bash
mlflow ui
```

Visit `http://127.0.0.1:5000` to view:
- 📈 Experiment dashboard
- 🗂️ Model registry
- 📁 Artifacts and metrics

---

## **Technologies & Dependencies**

- **Python 3.12** - Core programming language
- **Streamlit** - Interactive web application framework
- **FastAPI** - Modern, fast web framework for building APIs
- **scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **OpenCV** - Computer vision library
- **MLflow** - ML lifecycle management
- **Docker** - Containerization platform
- **streamlit-drawable-canvas** - Canvas component for Streamlit

---

## **API Documentation**

Once the FastAPI backend is running, visit:
- **Interactive API docs:** `http://localhost:8502/docs`
- **Alternative docs:** `http://localhost:8502/redoc`

### Example API Usage:

```python
import requests
import numpy as np

# Prepare your 28x28 digit image as a flattened array
image_data = np.random.rand(784).tolist()  # Replace with actual image data

response = requests.post(
    "http://localhost:8502/predict",
    json={"image": image_data}
)

prediction = response.json()
print(f"Predicted digit: {prediction['digit']}")
```

---

## **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Contact**

**Author:** Muni Kumar  and Abhiroop

**GitHub:** [@munikumar229](https://github.com/munikumar229)   and [@shadowscythe03](https://github.com/shadowscythe03)

**Project Link:** [https://github.com/munikumar229/Dig-Reg](https://github.com/munikumar229/Dig-Reg)