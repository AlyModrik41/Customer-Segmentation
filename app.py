import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import joblib

st.title("Customer Segmentation: KMeans & DBSCAN")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head())

    # --- Let user choose columns to include ---
    st.write("Select columns to use for clustering (exclude Gender):")
    feature_cols = st.multiselect("Features", options=[col for col in df.columns if col != 'Gender'], 
                                  default=['Age','Annual Income (k$)'])
    
    if len(feature_cols) < 2:
        st.warning("Please select at least 2 features for clustering.")
    else:
        df_features = df[feature_cols]

        # --- Scale features ---
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_features)

        # --- Algorithm selection ---
        algorithm = st.selectbox("Choose clustering algorithm", ["KMeans", "DBSCAN"])

        if algorithm == "KMeans":
            n_clusters = st.number_input("Number of clusters (K)", min_value=2, max_value=20, value=3)
            model = KMeans(n_clusters=n_clusters, random_state=42)
            model.fit(data_scaled)
            labels = model.labels_
            st.write("Cluster labels:", labels)

        elif algorithm == "DBSCAN":
            eps = st.number_input("Epsilon (neighborhood size)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
            min_samples = st.number_input("Minimum samples", min_value=1, max_value=20, value=5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            model.fit(data_scaled)
            labels = model.labels_
            st.write("Cluster labels:", labels)

        # --- Add cluster labels to dataframe ---
        df['Cluster'] = labels

        # --- Show average spending, income, age per cluster ---
        st.write("Average values per cluster:")

        cluster_summary = df.groupby('Cluster')[['Spending Score (1-100)', 'Age', 'Annual Income (k$)']].mean()
        st.dataframe(cluster_summary)

# Plot simple bar charts for each feature
        for col in cluster_summary.columns:
            st.write(f"Average {col} per cluster:")
            plt.figure(figsize=(6,4))
            plt.bar(cluster_summary.index.astype(str), cluster_summary[col], color='skyblue')
            plt.xlabel("Cluster")
            plt.ylabel(f"Average {col}")
            plt.title(f"Average {col} per Cluster")
            st.pyplot(plt)
        # --- Save model option ---
        if st.button("Save Model"):
            joblib.dump(model, f"{algorithm}_model.pkl")
            st.success(f"{algorithm} model saved!")

else:
    st.write("Please upload a CSV file to start clustering.")
