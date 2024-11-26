import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

# Load your datasets
df = pd.read_csv('df.csv')  # Load the raw dataset
merged_cleaned = pd.read_csv('merged_cleaned.csv')  # Load the cleaned dataset

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

    # Plot 2: Total Demand by Month for Each Year
    month_demand = merged_cleaned.groupby(['Year', 'Month']).agg({'quantity': 'sum'}).reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Month', y='quantity', hue='Year', data=month_demand, palette='viridis')
    plt.title('Total Demand by Month for Each Year', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Quantity', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Year', loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)

    # Plot 3: Scatter Plot for Age Distribution (using Plotly)
    fig = px.scatter(merged_cleaned, x='Age', y='Frequency', trendline="ols", title='Age Distribution')
    st.plotly_chart(fig)

# Page 3: Machine Learning Modeling (with ARIMA Simulation)
def simulate_future_pickups(future_days):
    # Load the trained ARIMA model
    loaded_arima_model = joblib.load('arima_model.pkl')

    # Get the last 'actual_pickup' value as the starting point for simulation
    last_actual_pickup = merged_cleaned['actual_pickup'].iloc[-1]

    # Simulate future values using the ARIMA model
    forecast = loaded_arima_model.get_forecast(steps=int(future_days))

    # Get the predicted values and confidence intervals
    predicted_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    # Create a date range for the future predictions
    last_date = merged_cleaned['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=int(future_days))

    # Create a DataFrame for the predicted values
    future_predictions = pd.DataFrame({
        'date': future_dates,
        'predicted_pickup': predicted_values
    })

    # Convert the DataFrame to an HTML table for display
    table_html = future_predictions.to_html(index=False)

    return table_html

def machine_learning_modeling():
    st.title("Future Pickup Count Simulation")
    st.write("Enter the number of days to simulate future pickup counts:")

    # Input for number of days to simulate
    future_days = st.number_input("Number of Days to Simulate", min_value=1, max_value=365, value=30)

    if st.button("Simulate"):
        # Get the future predictions using ARIMA
        table_html = simulate_future_pickups(future_days)

        # Display the table with predictions
        st.markdown(table_html, unsafe_allow_html=True)

# Page 4: Community Mapping
def community_mapping():
    st.title("Community Mapping: Areas in Need of Food Hampers")
    geodata = pd.read_csv("community_data.csv")

    # Optional: Set your Mapbox token (if you want to use Mapbox styles)
    px.set_mapbox_access_token('YOUR_MAPBOX_TOKEN_HERE')

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
                            hover_data={'HamperRequests': True, 'FamilySize': True, 'IncomeLevel': True, 'Latitude': False, 'Longitude': False},
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
