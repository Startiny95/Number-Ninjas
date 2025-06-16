# 📊 Churn Prediction Dashboard

A **machine learning-powered dashboard** to **predict customer churn** in telecom companies. Upload customer data, generate predictions, and explore interactive visualizations to identify **high-risk customers**.

---
Video presentation link:

PPT link:

## 🌟 Why This Project?

Customer churn directly impacts company revenue. By predicting which customers are likely to leave, telecom businesses can:

- Proactively retain customers
- Improve customer satisfaction
- Reduce marketing & retention costs

This dashboard was built as part of a **hackathon challenge** focusing on churn prediction using real-world telecom data.

---

## 🚀 Features

✅ **CSV File Upload** for customer data  
✅ **Churn Prediction** using trained **Random Forest Classifier**  
✅ **Downloadable Prediction Results**  
✅ **Interactive Visualizations**:
- Churn vs Retained distribution
- Monthly & Total Charges vs Churn
- Contract type, Device Protection, Tech Support impact
✅ **Top 10 High-Risk Customers List**  
✅ **Feature Importance Graph** (What features affect churn the most)

---

## 🗂️ Project Structure

📁 churn-prediction-dashboard

 churn_prediction_model.py # Train model locally
 churn_gui.py # Streamlit dashboard app
 d-2 train dataset.csv # Training dataset (private/local)
 d-2 test dataset.csv # Example test dataset (private/local)
 Churn Prediction Dashboard final.pdf # Presentation/report of the project

_________________________________________________________________________________________________________

📤 Example CSV Format

| MonthlyCharges | TotalCharges | Contract       | TechSupport | tenure | DeviceProtection | ... |
| -------------- | ------------ | -------------- | ----------- | ------ | ---------------- | --- |
| 65.5           | 4500         | Month-to-month | No          | 22     | Yes              | ... |

## 🤖 Machine Learning Model Details

| **Property**     | **Details**                                                                     |
|------------------|-------------------------------------------------------------------------------- |
| **Algorithm**    | Random Forest Classifier                                                        |
| **Preprocessing**| Label Encoding for categorical features, Standard Scaler for numerical features |
| **Metrics**      | AUC-ROC ≈ 0.84 (Good Performance)                                               |

_________________________________________________________________________________________________________

📊 Visualizations Included

| 🔸 Visualization         | 📌 Insight                                           |
| ------------------------ | ----------------------------------------------------- |
| Churn Distribution       | % of Churned vs Retained                              |
| Monthly Charges vs Churn | Higher charges → Higher churn probability             |
| Tenure vs Churn          | Shorter tenure → More likely to churn                 |
| Contract Type Analysis   | Month-to-month contracts → Higher churn risk          |
| Tech Support Impact      | Customers without tech support → More likely to churn |
| Top 10 Risk Customers    | List of most vulnerable customers with churn %        |
| Feature Importance       | What drives churn?                                    |

_________________________________________________________________________________________________________


💻 Installation & Usage:

Here’s a step-by-step guide to set up and run this project on your local machine:

1️⃣ Clone the Repository:

First, you need to copy (clone) the repository to your computer:

git clone https://github.com/yourusername/churn-prediction-dashboard.git

cd churn-prediction-dashboard

2️⃣ Install Required Libraries:

Now, install all necessary Python libraries using:

pip install -r requirements.txt


📌 Note:
Make sure Python and pip are installed on your computer before running this command.


3️⃣ Run the Dashboard Locally:

After installing the libraries, start the dashboard using:

streamlit run churn_gui.py



______________________________________________________________________________________________________

🎯 Future Scope
✅ Add animations to the file uploader and graphs (Planned)

✅ Enhance visual realism of graphs (Planned)

🔐 Add login/authentication for secure dashboard access

🧠 Improve the model with advanced hyperparameter tuning

_______________________________________________________________________________________________________


📬 Contact: gothwalvrishty@gmail.com 

Developers: Vrishty Gothwal, Izram Sana, Kalyani Singh

GitHub: Startiny95 (Vrishty Gothwal)

Email: gothwalvrishty@gmail.com , singhkalyani720@gmail.com , rehansyed.egr@gmail.com








