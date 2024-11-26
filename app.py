import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA


# Load your datasets (replace with your actual file paths)
try:
    merged_cleaned = pd.read_csv('merged_cleaned.csv', parse_dates=['date'])  # Ensure 'date' is datetime
    geodata = pd.read_csv("Edmonton_postal_code.csv")
    # ... load other datasets as needed
except FileNotFoundError as e:
    st.error(f"Error: {e.filename} not found. Please check the file paths.")
    st.stop()


# Page 1: Dashboard
def dashboard():
    # ... your dashboard content (plots, text, etc.) ...
    st.subheader("💡 Abstract:")
    # ... your abstract

    # Plot 2: Total Demand by Month for Each Year (example)
    if 'Year' in merged_cleaned.columns and 'Month' in merged_cleaned.columns and 'quantity' in merged_cleaned.columns:
        month_demand = merged_cleaned.groupby(['Year', 'Month']).agg({'quantity': 'sum'}).reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes objects
        sns.barplot(x='Month', y='quantity', hue='Year', data=month_demand, palette='viridis', ax=ax)
        ax.set_title('Total Demand by Month for Each Year', fontsize=16)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Total Quantity', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.error("Required columns 'Year', 'Month', or 'quantity' are missing in the dataset.")
    # ... other parts of your dashboard

# Page 3: Machine Learning Modeling (with ARIMA Simulation)
def simulate_future_pickups(future_days):
    # Load the trained ARIMA model (ensure it's saved properly)
    try:
        loaded_arima_model = joblib.load('arima_model.pkl')
    except FileNotFoundError:
        st.error("Error: ARIMA model file 'arima_model.pkl' not found.")
        return None

    # Ensure 'actual_pickup' exists and get the last value for prediction start
    if 'actual_pickup' not in merged_cleaned.columns:
        st.error("'actual_pickup' column is missing in the dataset.")
        return None
    last_actual_pickup = merged_cleaned['actual_pickup'].iloc[-1]

    # Create the index for forecasting (important!)
    last_date = merged_cleaned['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    # Use the model to make predictions with the correct index
    forecast = loaded_arima_model.get_forecast(steps=future_days)
    predicted_values = forecast.predicted_mean
    future_predictions = pd.DataFrame({'date': future_dates, 'predicted_pickup': predicted_values})

    return future_predictions  # Return the DataFrame

def machine_learning_modeling():
    # ... your ML modeling page content ...
    future_days = st.number_input("Number of Days to Simulate", min_value=1, max_value=365, value=30)
    if st.button("Simulate"):
        future_predictions = simulate_future_pickups(int(future_days))  # Convert to integer

        if future_predictions is not None:
            st.write(future_predictions) # Display DataFrame directly

            fig, ax = plt.subplots() # Create a figure and an axes
            ax.plot(future_predictions['date'], future_predictions['predicted_pickup']) # Plot date vs prediction

            st.pyplot(fig) # Display the plot

# Page 4: Community Mapping (using Plotly Express)
def community_mapping():
    # ... your community mapping content ...
    if 'Latitude' not in geodata.columns or 'Longitude' not in geodata.columns:
        st.error("Missing required columns 'Latitude' or 'Longitude' in the community data.")
        return
    # ... (your mapping code using Plotly Express) ...
# ... other functions ...



# Main App Logic
def main():
    # ... your main function (sidebar, page selection, etc.) ...
  if __name__ == "__main__":
    main()
