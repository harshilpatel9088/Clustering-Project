import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from data_loader import load_data
from model import train_kmeans
from predictor import predict_cluster

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.title("Mall Customer Segmentation Dashboard")

file_path = os.path.join(os.path.dirname(__file__), '..', 'mall_customers.csv')
resolved_path = os.path.abspath(file_path)

if not os.path.isfile(resolved_path):
    st.error("Dataset is missing. Please check the specified location.")
else:
    try:
        df = pd.read_csv(resolved_path)
        st.subheader("Dataset Overview")
        st.dataframe(df)

        st.subheader("Demographic & Spending Visuals")
        st.write("**Age Distribution**")
        st.bar_chart(df["Age"])

        st.write("**Income Distribution**")
        st.bar_chart(df["Annual_Income"])

        st.write("**Spending Score**")
        st.bar_chart(df["Spending_Score"])

        st.sidebar.title("Clustering Configuration")
        cluster_count = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

        model, assigned_labels, model_inertia = train_kmeans(df[["Age", "Annual_Income", "Spending_Score"]], cluster_count)
        st.success(f"K-Means trained with {cluster_count} clusters.")
        st.write(f"Model Inertia: {model_inertia:.2f}")

        df["Cluster_Group"] = assigned_labels

        st.subheader("Cluster Segmentation Plot")
        fig, ax = plt.subplots()
        scatter_plot = ax.scatter(
            df["Annual_Income"],
            df["Spending_Score"],
            c=df["Cluster_Group"],
            cmap="tab10"
        )
        ax.set_title("Customer Groups")
        ax.set_xlabel("Annual Income")
        ax.set_ylabel("Spending Score")
        st.pyplot(fig)

        st.sidebar.title("New Customer Prediction")
        user_age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
        user_income = st.sidebar.number_input("Annual Income", min_value=0, max_value=200000, value=50000)
        user_score = st.sidebar.number_input("Spending Score", min_value=0, max_value=100, value=50)

        if st.sidebar.button("Predict Cluster"):
            result_cluster = predict_cluster(model, user_age, user_income, user_score)
            st.sidebar.success(f"Predicted Cluster: {result_cluster}")

    except Exception as ex:
        st.error(f"Something went wrong while processing the data: {ex}")
