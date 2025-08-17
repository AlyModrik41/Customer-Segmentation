import streamlit as st
import pandas as pd
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
    feature_cols = st.multiselect(
        "Features",
        options=[col for col in df.columns if col != 'Gender'],
        default=['Age','Annual Income (k$)']
    )
    
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

        elif algorithm == "DBSCAN":
            eps = st.number_input("Epsilon (neighborhood size)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
            min_samples = st.number_input("Minimum samples", min_value=1, max_value=20, value=5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            model.fit(data_scaled)
            labels = model.labels_

        # --- Add cluster labels to dataframe ---
        df['Cluster'] = labels
        st.write("Cluster labels added to data.")

        # --- Average Spending Score per cluster ---
        if 'Spending Score (1-100)' in df.columns:
            avg_spending = df.groupby('Cluster')['Spending Score (1-100)'].mean()
            st.write("Average Spending Score per cluster:")
            st.dataframe(avg_spending)

            # Simple bar chart
            plt.figure(figsize=(6,4))
            plt.bar(avg_spending.index.astype(str), avg_spending.values, color='skyblue')
            plt.xlabel("Cluster")
            plt.ylabel("Average Spending Score")
            plt.title("Average Spending Score per Cluster")
            st.pyplot(plt)

        # --- 2D Scatter Plot of clusters ---
        if data_scaled.shape[1] >= 2:
            plt.figure(figsize=(8,6))
            plt.scatter(data_scaled[:,0], data_scaled[:,1], c=labels, cmap='viridis', s=50)
            plt.xlabel(feature_cols[0])
            plt.ylabel(feature_cols[1])
            plt.title(f"{algorithm} Clustering")
            plt.colorbar(label='Cluster')
            st.pyplot(plt)

        # --- Save model option ---
        if st.button("Save Model"):
            joblib.dump(model, f"{algorithm}_model.pkl")
            st.success(f"{algorithm} model saved!")

else:
    st.write("Please upload a CSV file to start clustering.")
