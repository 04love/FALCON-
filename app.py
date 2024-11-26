import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
def load_data():
    # If your data is stored in Google Drive in Colab, download and use the file path here
    # For this example, we're assuming the data is in the current directory
    df = pd.read_csv('merged_cleaned.csv')# Adjust path to where your file is located
    return df

# Page 1: Dashboard
def dashboard():
    st.title("Data Science Streamlit App")
    st.subheader("Overview of the Data")

    # Load data
    df = load_data()
    
    # Show the first few rows of the dataframe
    st.write(df.head())

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Plot Histogram for a sample column
    st.subheader("Visualizations")
    fig = px.histogram(df, x="column_name", title="Histogram of Column")
    st.plotly_chart(fig)

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    df = load_data()

    # Example: Distribution of a specific column (e.g., 'Age')
    st.subheader("Distribution of Age")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], kde=True, ax=ax)
    st.pyplot(fig)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr_matrix = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Page 3: Machine Learning Model
def machine_learning_modeling():
    st.title("Machine Learning Model")

    # Load data
    df = load_data()
    X = df[['feature1', 'feature2', 'feature3']]  # Replace with your actual feature columns
    y = df['target']  # Replace with your target column

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model evaluation (R^2 score)
    st.write("Model Accuracy (R^2): ", model.score(X_test, y_test))

    # Save the model (optional)
    joblib.dump(model, 'model.pkl')

    # Make predictions
    predictions = model.predict(X_test)
    st.write("Predictions:", predictions[:10])

# Page 4: Community Mapping (if relevant)
def community_mapping():
    st.title("Community Mapping")

    # Example for creating a map using Plotly
    geo_data = pd.read_csv("community_data.csv")  # Replace with actual file name or data source

    # Plot with Plotly (adjust the columns as necessary)
    fig = px.scatter_mapbox(
        geo_data,
        lat='Latitude',
        lon='Longitude',
        color='IncomeLevel',  # Adjust as per your data
        size='HamperRequests',  # Adjust as per your data
        color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=15,
        zoom=10,
        hover_name='Location',  # Customize for your dataset
        title="Community Mapping for Food Hamper Needs"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

# Main function that connects all the pages
def main():
    st.sidebar.title("Streamlit App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "Machine Learning", "Community Mapping"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "Machine Learning":
        machine_learning_modeling()
    elif app_page == "Community Mapping":
        community_mapping()

if __name__ == "__main__":
    main()

