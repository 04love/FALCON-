import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load the dataset with a specified encoding
data = pd.read_csv('kijiji_cleaned.csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
    st.image('Logo.PNG', use_column_width=True)
    st.subheader("💡 Abstract:")
    inspiration = '''
Data Quality: It is impossible to exaggerate the significance of data quality. An essential first step in guaranteeing the precision and dependability of our analysis and models was cleaning and preparing the dataset.
Feature Selection: The effectiveness of machine learning models is greatly impacted by the identification of pertinent features. We identified the key variables influencing Ontario rental pricing through iterative experimentation.
Model Evaluation: To appropriately determine a machine learning model's performance and capacity for generalization, a thorough evaluation of the model is necessary. We assessed and improved our models using a range of metrics and methods.
Deployment Obstacles: Scalability, security, and system integration are just a few of the difficulties that come with deploying machine learning models to commercial settings. Working together across several teams and areas of expertise was necessary to address these problems.
Overall, this study offered insightful information about the rental market in Ontario and the practical uses of machine learning methods. It emphasized how crucial it is for data science projects to have interdisciplinary collaboration and ongoing learning.
    '''
    st.write(inspiration)
    st.subheader("👨🏻‍💻 What our Project Does?")
    what_it_does = '''
  The purpose of this research is to use machine learning techniques to perform an extensive examination of the rental market in Ontario, Canada. The project will be broken down into three primary stages: the creation of machine learning (ML) models, deployment, and exploratory data analysis (EDA) and visualization.In order to obtain insights into the trends, patterns, and factors impacting rental pricing in the rental market, a range of statistical approaches and visualization tools will be utilized throughout the EDA phase. In this stage, the rental data will be cleaned and preprocessed, outliers and missing values will be found, and correlations between various factors will be investigated.Using supervised learning methods like regression and classification, predictive models will be constructed throughout the machine learning model building phase in order to forecast rental prices and examine the variables influencing price fluctuations.
  Furthermore, rental market segmentation based on various attributes may be achieved through the use of unsupervised learning techniques such as clustering.In the Deployment phase, the built machine learning models will be made available to customers via a web platform or application. This will enable them to interactively explore insights about the rental market and receive rental price projections based on predetermined criteria.
     '''



# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")

    # Price Distribution
    fig = px.scatter(data, x='Size', y='Price', trendline="ols", title='Relationship between Size and Price')
    st.plotly_chart(fig)

    average_prices_bathrooms = data.groupby('Bathrooms')['Price'].mean().reset_index()
    fig = px.bar(average_prices_bathrooms, x='Bathrooms', y='Price', title='Average Price by Bathrooms')
    st.plotly_chart(fig)


    average_prices = data.groupby('Bedrooms')['Price'].mean().reset_index()
    fig = px.bar(average_prices, x='Bedrooms', y='Price', title='Average Price by Bedrooms')
    st.plotly_chart(fig)

    fig = px.box(data, x='Type', y='Price', title='Price Distribution by Property Type')
    st.plotly_chart(fig)

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Kijiji Rental Price Prediction")
    st.write("Enter the details of the property to predict its rental price:")

    # Input fields for user to enter data
    property_type = st.selectbox("Type of Property", ['Apartment', 'House', 'Condo', 'Townhouse'])
    bedrooms = st.slider("Number of Bedrooms", 1, 5, 2)
    bathrooms = st.slider("Number of Bathrooms", 1, 3, 1)
    size = st.slider("Size (sqft)", 300, 5000, 1000)
    unique_locations = data['CSDNAME'].unique()
    location = st.selectbox("Location", unique_locations)

    if st.button("Predict"):
        # Load the trained model including preprocessing
        model = joblib.load('random_forest_regressor_model.pkl')

        # Assuming the model_with_preprocessing is a pipeline that ends with your estimator
        # Prepare input data as a DataFrame to match the training data structure
        input_df = pd.DataFrame({
            'Type': [property_type],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Size': [size],
            'CSDNAME': [location]
        })

        # Make prediction
        prediction = model.predict(input_df)
        prediction = model.predict(input_df)

    # Map the predicted classes to labels
    price_bins = [0, 1700, 2300, float('inf')]
    price_labels = ['low', 'medium', 'high']
    price_category = pd.cut(prediction, bins=price_bins, labels=price_labels)

    # Display the predictions
    st.success(f"Predicted Rental Price: ${prediction[0]:,.2f}")
    st.success(f"Predicted Price Category: {price_category[0]}")



# Page 4: Community Mapping
def community_mapping():
    st.title("Small Communities Map: Population <10000")
    geodata = pd.read_csv("small_communities.csv")

    # Optional: Set your Mapbox token (if you want to use Mapbox styles)
    # px.set_mapbox_access_token('YOUR_MAPBOX_TOKEN_HERE')

    # Create the map using Plotly Express
    fig = px.scatter_mapbox(geodata,
                            lat='Latitude',
                            lon='Longitude',
                            color='Population',  # Color points by population, or choose another column
                            size='Price',  # Size points by price, or choose another column
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            size_max=15,
                            zoom=10,
                            hover_name='Type',  # Display property type when hovering over points
                            hover_data={'Price': True, 'Population': True, 'Bathrooms': True, 'Bedrooms': True, 'Size': True, 'Latitude': False, 'Longitude': False},
                            title='Small Communities Map')

    fig.update_layout(mapbox_style="open-street-map")  # Use OpenStreetMap style
    st.plotly_chart(fig)


# Main App Logic
def main():
    st.sidebar.title("Kijiji Community App")
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
