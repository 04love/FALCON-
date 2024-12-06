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

# Page 1: Dashboard
def dashboard():
    st.subheader("💡 Abstract:")
    inspiration = '''
Food Security: Ensuring everyone has access to sufficient, nutritious food is a key challenge. This project seeks to predict food hamper demand and help organizations optimize resource allocation.
Feature Selection: By analyzing factors such as income levels, family size, and location, we identified key variables influencing food hamper requests.
Model Evaluation: We used various metrics to evaluate our prediction model, ensuring it generalizes well to unseen data and provides reliable forecasts for future donations.
Deployment Challenges: We focused on ensuring scalability and accuracy, integrating the predictive models with local food bank systems to streamline food distribution.
The project sheds light on food security issues and demonstrates how data science can enhance community outreach and food distribution efforts.
    '''
    st.write(inspiration)
    st.subheader("👨🏻‍💻 What our Project Does?")
    what_it_does = '''
  The goal of this research is to predict food hamper demand in communities. By utilizing machine learning techniques, we aim to forecast the number of food hampers needed in various regions based on demographic factors, historical donation data, and community needs.
  The project involves three primary stages: data cleaning and exploration, predictive modeling, and deployment. In the EDA phase, we clean and preprocess the data, identify trends and patterns, and investigate correlations between various features like income, family size, and location.
  During the machine learning phase, we build models to predict the number of food hampers required in each community. We may also categorize areas based on the level of need (e.g., low, medium, high demand).
  Finally, in the deployment phase, the predictive model will be accessible to local food banks through a web application, enabling real-time forecasting and more efficient food distribution.
    '''
    st.write(what_it_does)

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")

    # Histograms for numerical columns
    client_clean['age'].hist(bins=10)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()
    average_prices_bathrooms = data.groupby('Bathrooms')['Price'].mean().reset_index()
    fig = px.bar(average_prices_bathrooms, x='Bathrooms', y='Price', title='Average Price by Bathrooms')
    st.plotly_chart(fig)


    average_prices = data.groupby('Bedrooms')['Price'].mean().reset_index()
    fig = px.bar(average_prices, x='Bedrooms', y='Price', title='Average Price by Bedrooms')
    st.plotly_chart(fig)

    fig = px.box(data, x='Type', y='Price', title='Price Distribution by Property Type')
    st.plotly_chart(fig)
    
