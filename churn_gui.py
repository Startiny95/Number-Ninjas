import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("📊 Churn Prediction Dashboard")

@st.cache_data
def load_and_train_model():
    """Load data and train the churn prediction model"""
    df = pd.read_csv("d-2 train dataset.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop("customerID", axis=1, inplace=True)

    cat_cols = df.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop("churned", axis=1)
    y = df["churned"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y)

    return clf, label_encoders, scaler, cat_cols, X.columns.tolist()

def identify_risk_factors(data, churn_probs, top_n=10):
    """Identify top risk factors for churning customers"""
    risk_data = data.copy()
    risk_data['Churn_Probability'] = churn_probs
   
    top_risk = risk_data.nlargest(top_n, 'Churn_Probability')
    
    return top_risk

clf, label_encoders, scaler, cat_cols, feature_names = load_and_train_model()

uploaded_file = st.file_uploader("📤 Upload a CSV file with customer data", type="csv")

if uploaded_file:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.subheader("🔍 Uploaded Data Preview")
        st.dataframe(df_input.head())

        df_input["TotalCharges"] = pd.to_numeric(df_input["TotalCharges"], errors="coerce")
        df_input.dropna(inplace=True)
        
        original_input = df_input.copy()
        
        if "customerID" in df_input.columns:
            df_input.drop("customerID", axis=1, inplace=True)

        for col in cat_cols:
            if col in df_input.columns and col != 'churned':
                df_input[col] = label_encoders[col].transform(df_input[col])

        df_scaled = scaler.transform(df_input)
        
        churn_preds = clf.predict(df_scaled)
        churn_probs = clf.predict_proba(df_scaled)[:, 1]

        original_input["Predicted_Churn"] = churn_preds
        original_input["Churn_Probability"] = churn_probs
        original_input["Churn_Status"] = original_input["Predicted_Churn"].map({1: "Churned", 0: "Retained"})

        st.subheader("📊 Prediction Results")
        st.dataframe(original_input)

        csv_out = original_input.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Predictions as CSV", 
            data=csv_out, 
            file_name="churn_predictions.csv", 
            mime="text/csv"
        )

        st.subheader("📈 Churn Summary Statistics")
        churn_rate = original_input["Predicted_Churn"].mean() * 100
        total_customers = len(original_input)
        churned_customers = original_input["Predicted_Churn"].sum()
        retained_customers = total_customers - churned_customers
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Customers", total_customers)
        with col2:
            st.metric("📉 Churned Customers", churned_customers)
        with col3:
            st.metric("📈 Retained Customers", retained_customers)
        with col4:
            st.metric("📊 Churn Rate (%)", f"{churn_rate:.2f}%")

        st.divider()

        st.subheader("📊 Data Visualizations")

        plt.style.use('default')
        sns.set_palette("husl")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1️⃣ Churn Distribution")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            churn_counts = original_input["Churn_Status"].value_counts()
            colors = ['#ff6b6b', '#4ecdc4']
            ax1.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                    colors=colors, startangle=90)
            ax1.set_title("Churned vs Retained Customers", fontsize=12, fontweight='bold')
            st.pyplot(fig1, use_container_width=True)

        with col2:
            st.subheader("2️⃣ Device Protection")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            device_counts = original_input["DeviceProtection"].value_counts()
            ax2.pie(device_counts.values, labels=device_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title("Device Protection Distribution", fontsize=12, fontweight='bold')
            st.pyplot(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("3️⃣ Tech Support Analysis")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=original_input, x="TechSupport", hue="Churn_Status", ax=ax3)
            ax3.set_title("Tech Support vs Churn", fontsize=12, fontweight='bold')
            ax3.set_xlabel("Tech Support", fontsize=10)
            ax3.set_ylabel("Count", fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
            plt.legend(fontsize=8)
            st.pyplot(fig3, use_container_width=True)

        with col4:
            if "StreamingTV" in original_input.columns:
                st.subheader("4️⃣ Streaming TV")
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                sns.countplot(data=original_input, x="StreamingTV", hue="Churn_Status", ax=ax4)
                ax4.set_title("Streaming TV vs Churn", fontsize=12, fontweight='bold')
                ax4.set_xlabel("Streaming TV", fontsize=10)
                ax4.set_ylabel("Count", fontsize=10)
                plt.xticks(rotation=45, fontsize=8)
                plt.legend(fontsize=8)
                st.pyplot(fig4, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            if "tenure" in original_input.columns:
                st.subheader("5️⃣ Tenure vs Churn")
                fig5, ax5 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=original_input, x="tenure", y="Churn_Probability", 
                               hue="Churn_Status", ax=ax5, alpha=0.7, s=40)
                ax5.set_title("Tenure vs Churn Probability", fontsize=12, fontweight='bold')
                ax5.set_xlabel("Tenure (months)", fontsize=10)
                ax5.set_ylabel("Churn Probability", fontsize=10)
                plt.legend(fontsize=8)
                st.pyplot(fig5, use_container_width=True)
            else:
                st.subheader("5️⃣ Tenure vs Churn")
                st.info("Tenure column not found in the dataset. Please ensure your dataset contains a 'tenure' column.")

        with col6:
            st.subheader("6️⃣ Contract Type")
            fig6, ax6 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=original_input, x="Contract", hue="Churn_Status", ax=ax6)
            ax6.set_title("Contract Type vs Churn", fontsize=12, fontweight='bold')
            ax6.set_xlabel("Contract Type", fontsize=10)
            ax6.set_ylabel("Count", fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
            plt.legend(fontsize=8)
            st.pyplot(fig6, use_container_width=True)

        col7, col8 = st.columns(2)
        with col7:
            st.subheader("7️⃣ Monthly Charges vs Churn")
            fig7, ax7 = plt.subplots(figsize=(6, 4))
            scatter = sns.scatterplot(data=original_input, x="MonthlyCharges", y="Churn_Probability", 
                                    hue="Churn_Status", ax=ax7, alpha=0.7, s=40)
            ax7.set_title("Monthly Charges vs Churn Probability", fontsize=12, fontweight='bold')
            ax7.set_xlabel("Monthly Charges (₹)", fontsize=10)
            ax7.set_ylabel("Churn Probability", fontsize=10)
            plt.legend(fontsize=8)
            st.pyplot(fig7, use_container_width=True)

        with col8:
            st.subheader("8️⃣ Total Charges vs Churn")
            fig8, ax8 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=original_input, x="TotalCharges", y="Churn_Probability", 
                           hue="Churn_Status", ax=ax8, alpha=0.7, s=40)
            ax8.set_title("Total Charges vs Churn Probability", fontsize=12, fontweight='bold')
            ax8.set_xlabel("Total Charges (₹)", fontsize=10)
            ax8.set_ylabel("Churn Probability", fontsize=10)
            plt.legend(fontsize=8)
            st.pyplot(fig8, use_container_width=True)

        st.subheader("🚨 Top 10 High-Risk Customers")
        
        top_risk_customers = original_input.nlargest(10, 'Churn_Probability')
        
        risk_display = top_risk_customers[['Churn_Probability', 'MonthlyCharges', 'TotalCharges', 
                                         'Contract', 'TechSupport', 'DeviceProtection']].copy()
        risk_display['Churn_Probability'] = risk_display['Churn_Probability'].round(4)
        risk_display['MonthlyCharges'] = '₹' + risk_display['MonthlyCharges'].astype(str)
        risk_display['TotalCharges'] = '₹' + risk_display['TotalCharges'].astype(str)
        risk_display['Risk_Rank'] = range(1, 11)
        risk_display = risk_display[['Risk_Rank', 'Churn_Probability', 'MonthlyCharges', 
                                   'TotalCharges', 'Contract', 'TechSupport', 'DeviceProtection']]
        
        st.dataframe(risk_display, use_container_width=True)
        
        st.subheader("📊 Risk Factors Summary")
        high_risk_customers = original_input[original_input['Churn_Probability'] > 0.7]
        
        if len(high_risk_customers) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**High-Risk Customer Characteristics:**")
                avg_monthly = high_risk_customers['MonthlyCharges'].mean()
                avg_total = high_risk_customers['TotalCharges'].mean()
                st.write(f"• Average Monthly Charges: ₹{avg_monthly:.2f}")
                st.write(f"• Average Total Charges: ₹{avg_total:.2f}")
                
            with col2:
                st.write("**Most Common Risk Factors:**")
                if 'Contract' in high_risk_customers.columns:
                    common_contract = high_risk_customers['Contract'].mode().iloc[0] if not high_risk_customers['Contract'].mode().empty else "N/A"
                    st.write(f"• Most Common Contract: {common_contract}")
                if 'TechSupport' in high_risk_customers.columns:
                    common_tech = high_risk_customers['TechSupport'].mode().iloc[0] if not high_risk_customers['TechSupport'].mode().empty else "N/A"
                    st.write(f"• Most Common Tech Support: {common_tech}")
        
        else:
            st.write("No customers with high churn probability (>70%) found in this dataset.")

        st.subheader("📈 Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig9, ax9 = plt.subplots(figsize=(8, 6))
        sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature', ax=ax9)
        ax9.set_title("Top 10 Most Important Features for Churn Prediction", fontsize=14, fontweight='bold')
        ax9.set_xlabel("Feature Importance", fontsize=12)
        st.pyplot(fig9, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing the file: {str(e)}")
        st.write("Please ensure your CSV file has the correct format and column names.")

else:
    st.info("👆 Please upload a CSV file to start the churn analysis.")
    st.write("""
    **Expected CSV format:**
    - The CSV should contain customer data with features like:
    - MonthlyCharges, TotalCharges, Contract, TechSupport, DeviceProtection, tenure, etc.
    - The model will predict churn probability for each customer
    - All monetary values will be displayed in Indian Rupees (₹)
    """)