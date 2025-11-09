GIT: https://github.com/Crese16/MITB_MLE

Loan Default Prediction Pipeline
📖 Overview

An end-to-end loan default prediction system built on the Medallion Architecture (Bronze → Silver → Gold) using PySpark, Airflow, and Docker.
It automates data processing and model training (Logistic Regression & Random Forest) with Out-of-Time (OOT) validation.

🚀 Quick Start
1️⃣ Start the Environment
docker-compose up --build


Airflow UI: http://localhost:8080
 (admin / admin)

JupyterLab: http://localhost:8888

2️⃣ Run Data Pipeline

Inside Jupyter or terminal:

python main.py


Generates monthly Bronze → Silver → Gold data under /datamart.

3️⃣ Run Model Training

Open the notebook:

model_train_main.ipynb


and Run All Cells — it will:

Merge Gold feature + label stores

Perform Out-of-Time split

Train & evaluate Logistic Regression and Random Forest

Save models + metrics in:

utils/model_bank/

📈 Example Output
🏆 Best Model: LogisticRegression (AUC=0.83)
📊 Metrics → utils/model_bank/oot_model_metrics_20251109.csv


✅ One-Command Startup

docker-compose up --builds