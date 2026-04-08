# California Housing Price Prediction - Group 01

This project is part of **Lab 02 (Linear Regression)**. We have developed a comprehensive machine learning pipeline to analyze and predict house values in California using 7 different model variants. The final product includes a trained model integrated into a **Streamlit** web application.

## Project Structure
* `data/`: Contains the original California Housing dataset.
* `lab2-ml.ipynb`: The main Jupyter Notebook containing EDA, Feature Engineering, and Model Training (7 variants).
* `app.py`: Streamlit application script for deployment.
* `model_6.pkl`: The best-performing trained regression model.
* `spatial_cluster_model.pkl`: The K-Means clustering model for geographical features.
* `requirements.txt`: List of necessary Python libraries.

## Execution Guide

Follow these steps to set up and run the project on your local machine:

### 1. Install Dependencies
First, ensure you have Python installed. Then, install all required libraries using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt

---
```

### 2. Generate the Models
Open the Jupyter Notebook lab2-ml.ipynb and Run All Cells 
* This process will perform Exploratory Data Analysis (EDA) and train 7 model variants.
* Important: Running the notebook will automatically export the two serialized files: model_6.pkl and spatial_cluster_model.pkl. These are required for the web application to function.

### 3. Launch the Application 
Once the .pkl files are generated in the root directory, you can deploy the web interface by running:
```bash
streamlit run app.py
---
```