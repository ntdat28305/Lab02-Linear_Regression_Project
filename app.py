import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="California Housing Analysis - Nhóm 01", page_icon="🏠", layout="wide")

# LOAD Data
@st.cache_resource
def load_assets():
    model = joblib.load('model_6.pkl')
    kmeans = joblib.load('spatial_cluster_model.pkl')
    df = pd.read_csv(os.path.join('data', '1553768847-housing.csv'))
    
    performance_data = {
        'Model': ['M6: Spatial Cluster', 'M4: Interaction', 'M1: Baseline', 'M5: Polynomial', 'M2: VIF Optimized', 'M7: Outlier Cleaned', 'M3: Log-Log'],
        'MAE': [49924.74, 50787.82, 50888.66, 51565.22, 53818.61, 52449.49, 60254.90],
        'RMSE': [71084.90, 72565.18, 72668.53, 74201.56, 77307.84, 81337.26, 685760.86],
        'R2': [0.614, 0.598, 0.597, 0.579, 0.543, 0.495, -34.88]
    }
    return model, kmeans, df, pd.DataFrame(performance_data)

try:
    model, kmeans, df, perf_df = load_assets()
except Exception as e:
    st.error(f"Missing model files or dataset: {e}")
    st.stop()

st.title("California Housing Analysis")
st.markdown("### **Group 01** - Linear Regression Project (Lab 02)")
st.divider()

tab1, tab2, tab3 = st.tabs(["Real-time Prediction", "EDA Analysis", "Comparison & Evaluation"])

# --- TAB 1: Real-time Prediction --- ---
with tab1:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        with st.form("input_form"):
            st.subheader("Input Parameters for Prediction")
            lon = st.number_input("Longitude:", value=-122.23)
            lat = st.number_input("Latitude:", value=37.88)
            income = st.slider("Income ($10k):", 0.5, 15.0, 3.8)
            age = st.slider("Housing Median Age:", 1, 52, 28)
            rooms = st.number_input("Total Rooms:", value=800)
            beds = st.number_input("Total Bedrooms:", value=150)
            pop = st.number_input("Population:", value=300)
            hh = st.number_input("Households:", value=120)
            submit = st.form_submit_button("Predict House Value", use_container_width=True)
            
            if submit:
                inp = pd.DataFrame([[lon, lat, age, rooms, beds, pop, hh, income]], 
                                   columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income'])
                inp['rooms_per_household'] = inp['total_rooms'] / inp['households']
                inp['bedrooms_per_room'] = inp['total_bedrooms'] / inp['total_rooms']
                inp['population_per_household'] = inp['population'] / inp['households']
                cluster = kmeans.predict(inp[['longitude', 'latitude']])[0]
                for i in range(5): inp[f'cluster_{i}'] = 1 if cluster == i else 0
                for col in ['ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']: inp[col] = 0
                cols_order = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household', 'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN', 'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4']
                prediction = model.predict(inp[cols_order])
                st.balloons()
                st.markdown(f"<div style='background-color:#d4edda; padding:20px; border-radius:10px; border: 1px solid #c3e6cb;'><h2 style='color:#155724; margin:0;'>Prediction value : ${prediction[0]:,.2f}</h2></div>", unsafe_allow_html=True)

with c2:
        st.subheader("California Map Visualization")

        fig_loc, ax_loc = plt.subplots(figsize=(8, 7))

        sc = ax_loc.scatter(df['longitude'], df['latitude'], alpha=0.1, 
                            c=df['median_house_value'], cmap='jet', s=df['population']/200)

        ax_loc.scatter(lon, lat, color='black', marker='X', s=250, 
                       edgecolors='white', linewidths=1.5, zorder=10, label='Prediction Location')
        
        ax_loc.legend(loc='upper right') 
        plt.colorbar(sc, ax=ax_loc, label='Median House Value (USD)') 
        
        ax_loc.set_xlabel("Longitude")
        ax_loc.set_ylabel("Latitude")
        ax_loc.set_title("Prediction Location on California Map")
        
        st.pyplot(fig_loc)
# --- TAB 2: EDA ---
with tab2:
    st.header("Exploratory Data Analysis (EDA)")
    
    col_r1_1, col_r1_2 = st.columns([1.2, 1])
    with col_r1_1:
        st.subheader("Geographical Heatmap (Interactive)")
        fig_m = px.scatter(df.sample(5000), x="longitude", y="latitude", color="median_house_value",
                           size="population", color_continuous_scale='Jet', height=450)
        st.plotly_chart(fig_m, use_container_width=True)
    with col_r1_2:
        st.subheader("Target Variable Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4.5))
        sns.histplot(df['median_house_value'], kde=True, color='teal', ax=ax_dist)
        st.pyplot(fig_dist)

    st.divider()

    col_r2_1, col_r2_2 = st.columns([1, 1.2])
    with col_r2_1:
        st.subheader("Feature Correlation Matrix")
        fig_c, ax_c = plt.subplots(figsize=(6, 5))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 6}, ax=ax_c)
        st.pyplot(fig_c)
    with col_r2_2:
        st.subheader(" Income vs. House Value")
        fig_reg, ax_reg = plt.subplots(figsize=(7, 5.5))
        sns.regplot(data=df.sample(1000), x='median_income', y='median_house_value', 
                    scatter_kws={'alpha':0.2, 'color':'teal'}, line_kws={'color':'red'}, ax=ax_reg)
        st.pyplot(fig_reg)

    st.divider()

    st.subheader("Check Outliers ")
    fig_b, ax_b = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df[['median_income', 'housing_median_age']], palette='Set2', ax=ax_b)
    st.pyplot(fig_b)

# --- TAB 3: Comparison & Evaluation ---
with tab3:
    st.header("Model Performance Evaluation")
    st.dataframe(perf_df.style.highlight_max(subset=['R2'], color='#90ee90'), use_container_width=True)
    st.info("**Note:** Model 6 (Spatial Cluster) is the best-performing model.")
    fig_bar = px.bar(perf_df[perf_df['R2']>0], x='Model', y='R2', color='R2', text_auto='.3f', color_continuous_scale='Viridis')
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.caption("© 2026 Group 01 | Lab 02: Linear Regression Project")