import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load the dataset with a specified encoding

df = pd.read_csv('df.csv')  # Load the raw dataset
merged_cleaned = pd.read_csv('merged_cleaned.csv')

# Page 1: Dashboard
def dashboard():
    st.image('Logo.PNG', use_column_width=True)  # Add a logo for your app
    st.subheader("💡 Abstract:")
    inspiration = '''
Food Security: Ensuring everyone has access to sufficient, nutritious food is a key challenge. This project seeks to predict food hamper demand and help organizations optimize resource allocation.
Feature Selection: By analyzing factors such as income levels, family size, and location, we identified key variables influencing food hamper requests.
Model Evaluation: We used various metrics to evaluate our prediction model, ensuring it generalizes well to unseen data and provides reliable forecasts for future donations.
Deployment Obstacles: Scalability, security, and system integration are just a few of the challenges that come with deploying machine learning models to commercial settings. Addressing these issues required cross-team collaboration and technical expertise.
The project sheds light on food security issues and demonstrates how data science can improve community outreach and food distribution efforts.
    '''
    st.write(inspiration)
    st.subheader("👨🏻‍💻 What our Project Does?")
    what_it_does = '''
The goal of this research is to predict food hamper demand in communities. Using machine learning techniques, we forecast the number of food hampers needed in various regions based on demographic factors, historical donation data, and community needs.
The project involves three main stages: data cleaning and exploration, predictive modeling, and deployment. During the exploratory data analysis (EDA) phase, we preprocess the data, identify trends, and investigate correlations between variables like income, family size, and location.
The machine learning phase involves constructing models to predict food hamper demand, potentially categorizing areas by their level of need. In the deployment phase, the prediction model will be made available to food banks for real-time forecasting and more efficient food distribution.
    '''
    st.write(what_it_does)

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")

    # Food hamper demand vs. family size
    fig = px.scatter(data, x='FamilySize', y='HamperRequests', trendline="ols", title='Family Size vs. Hamper Requests')
    st.plotly_chart(fig)

    # Average hamper requests by income level
    average_requests_income = data.groupby('IncomeLevel')['HamperRequests'].mean().reset_index()
    fig = px.bar(average_requests_income, x='IncomeLevel', y='HamperRequests', title='Average Hamper Requests by Income Level')
    st.plotly_chart(fig)

    # Hamper requests by community location
    fig = px.box(data, x='Location', y='HamperRequests', title='Hamper Requests Distribution by Location')
    st.plotly_chart(fig)

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Food Hamper Demand Prediction")
    st.write("Enter the details of the community to predict food hamper demand:")

    # Input fields for user to enter data
    income_level = st.selectbox("Income Level", ['Low', 'Medium', 'High'])
    family_size = st.slider("Family Size", 1, 10, 3)
    location = st.selectbox("Location", data['Location'].unique())

    if st.button("Predict"):
        # Load the trained model including preprocessing
        model = joblib.load('food_hamper_model.pkl')  # Assuming your trained model is saved as 'food_hamper_model.pkl'

        # Prepare input data as a DataFrame to match the training data structure
        input_df = pd.DataFrame({
            'IncomeLevel': [income_level],
            'FamilySize': [family_size],
            'Location': [location]
        })

        # Make prediction
        prediction = model.predict(input_df)

        # Map the predicted classes to labels
        demand_bins = [0, 100, 500, float('inf')]
        demand_labels = ['Low', 'Medium', 'High']
        demand_category = pd.cut(prediction, bins=demand_bins, labels=demand_labels)

        # Display the predictions
        st.success(f"Predicted Hamper Demand: {prediction[0]:,.0f} hampers")
        st.success(f"Predicted Demand Category: {demand_category[0]}")

# Page 4: Community Mapping
def community_mapping():
    st.title("Community Mapping: Areas in Need of Food Hampers")
    geodata = pd.read_csv("community_data.csv")  # Assuming 'community_data.csv' contains geographic data

    # Create the map using Plotly Express
    fig = px.scatter_mapbox(geodata,
                            lat='Latitude',
                            lon='Longitude',
                            color='IncomeLevel',  # Color points by income level
                            size='HamperRequests',  # Size points by number of hamper requests
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            size_max=15,
                            zoom=10,
                            hover_name='Location',  # Display location when hovering over points
                            hover_data={'HamperRequests': True, 'IncomeLevel': True, 'FamilySize': True},
                            title='Community Map for Food Hamper Needs')

    fig.update_layout(mapbox_style="open-street-map")  # Use OpenStreetMap style
    st.plotly_chart(fig)

# Main App Logic
def main():
    st.sidebar.title("Food Hamper Prediction App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "Community Mapping"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Community Mapping":
        community_mapping()

if __name__ == "__main__":
    main()
