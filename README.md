💻 Developer Burnout Predictor
Predicting mental exhaustion risk through behavioral data and Machine Learning.

📌 Project Overview
This project is an end-to-end Machine Learning solution designed to identify burnout risks among software developers. Instead of relying on subjective stress reports, the model analyzes objective lifestyle habits—like sleep-to-work ratios and meeting loads—to predict risk levels.

This tool is part of my journey as a Data Science student and content creator, moving from a simple diagnostic tool to a behavioral-driven AI application.

🚀 Key Features
Behavioral Prediction: Analyzes habits (sleep, work hours, commits, meetings) rather than just self-reported stress.

Engineered Metrics: Includes custom features like work_recovery_ratio and meeting_load.

Interactive App: Deployed via Streamlit for real-time user assessment.

📊 Model Performance
The current "Behavioral Model" (Model B) uses a Random Forest Classifier and achieves:

Accuracy: 77.55%

High-Risk Precision: 81%

Status: Fully optimized to prevent "data leakage" from stress scores, ensuring realistic predictions.

🛠️ Tech Stack
Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Joblib

Deployment: Streamlit Cloud

Environment: Jupyter Notebook

📁 File Structure
app.py: The Streamlit web application script.

burnout_model.pkl: The trained and saved Random Forest model.

requirements.txt: Necessary libraries for deployment.

Burnout_Analysis.ipynb: The full data cleaning, EDA, and training pipeline.