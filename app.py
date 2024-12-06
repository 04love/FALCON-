import streamlit as st
import pandas as pd
import os

# Check if the file exists before loading
def load_data():
    try:
        if not os.path.exists('merged_cleaned.csv'):
            st.error("Error: 'merged_cleaned.csv' file not found. Please check the file path or upload it.")
            return None
        
        df = pd.read_csv('df.csv')  # Load the raw dataset
        merged_cleaned = pd.read_csv('merged_cleaned.csv')  # Load the cleaned dataset
        return df, merged_cleaned
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Load the data
data = load_data()
if data:
    df, merged_cleaned = data

# If you're using file uploader
def upload_data():
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())  # Display the first few rows of the uploaded data
        return df
    return None

# In your main app logic, check if the data is loaded
if 'merged_cleaned' not in locals():
    uploaded_data = upload_data()
    if uploaded_data:
        merged_cleaned = uploaded_data

# After loading, you can inspect the first few rows
st.write(df.head())
st.write(merged_cleaned.head())

