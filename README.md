Credit Card Fraud Detection using Isolation Forest + XGBoost

Dataset-https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/code

This project focuses on detecting fraudulent credit card transactions by leveraging a hybrid Machine Learning approach. Due to the extremely imbalanced nature of fraud data in the real world, a combination of unsupervised and supervised learning techniques is used:

🔍 Isolation Forest → Identifies anomalies (potential fraud) without labels
🎯 XGBoost Classifier → Learns fraud behavior from both anomaly scores and labels to boost detection accuracy

📌 Project Objectives

Detect fraudulent credit card transactions effectively

Handle highly imbalanced dataset (<1% fraud cases)

Improve fraud recall while maintaining acceptable precision

Compare anomaly detection baseline vs supervised model performance

Provide fraud risk insights for real-world use

🧠 Techniques Used
Technique	Role
Isolation Forest	Unsupervised anomaly detection → anomaly score
Feature Scaling	Normalize input features
XGBoost	Final fraud classifier using anomaly score + features
Confusion Matrix	Analyze misclassifications
Precision, Recall	Evaluate performance for rare fraud cases
📂 Project Workflow
1️⃣ Import & explore data (EDA)
2️⃣ Handle imbalance visualization
3️⃣ Scale features using StandardScaler
4️⃣ Train Isolation Forest model → anomaly scoring
5️⃣ Add anomaly scores as feature
6️⃣ Train XGBoost for classification
7️⃣ Evaluate using precision, recall & confusion matrix
8️⃣ Fraud detection insights & documentation

📊 Evaluation Metrics

As fraud is rare, accuracy is misleading.
Focus on:

Precision: How many flagged frauds are correct?

Recall: How many real frauds were successfully caught? (Most important!)

Confusion Matrix

Classification Report

📈 Results show that XGBoost significantly improves fraud recall compared to Isolation Forest alone.

🛠️ Tech Stack

Python

pandas, numpy

scikit-learn

XGBoost

matplotlib

📦 Dataset

Dataset contains:

PCA-transformed numeric features (V1–V28)

Amount & Time columns

Class label → 0 = Legit ✓ , 1 = Fraud 🚫

Note: This dataset is commonly referenced from the popular Kaggle Credit Card Fraud Detection dataset.

🚀 Future Improvements

Hyperparameter tuning for both models

Visualizing PR and ROC curves

Cost-sensitive evaluation (reduce financial loss)

Fraud dashboard for real-time monitoring

Save and deploy final model

🧑‍💻 Author

Laksh Sandeep Oswal
B.Tech CS + AI/ML

📝 How to Run
# Clone the repo
git clone https://github.com/<your-username>/<your-repo-name>.git

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook "CreditCardFraudDetection.ipynb"
