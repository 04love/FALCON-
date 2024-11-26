import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your datasets
merged_cleaned = pd.read_csv('merged_cleaned.csv')  # Load the cleaned dataset

# Check the columns in merged_cleaned to confirm 'DayOfWeek' exists
st.write(merged_cleaned.columns)  # Display columns for debugging

# If 'DayOfWeek' is missing, derive it from a date column (if it exists)
if 'date' in merged_cleaned.columns:
    merged_cleaned['date'] = pd.to_datetime(merged_cleaned['date'])
    merged_cleaned['DayOfWeek'] = merged_cleaned['date'].dt.day_name()

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")

    # Plot 1: Total Demand by Day of the Week
    if 'DayOfWeek' not in merged_cleaned.columns:
        st.error("'DayOfWeek' column is missing in the dataset.")
        return

    day_demand = merged_cleaned.groupby('DayOfWeek').agg({'quantity': 'sum'}).reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DayOfWeek', y='quantity', data=day_demand, palette='viridis')
    plt.title('Total Demand by Day of the Week', fontsize=16)
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Total Quantity', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    # Additional plots (if needed)

# Main App Logic
def main():
    st.sidebar.title("Food Hamper Prediction App")
    app_page = st.sidebar.radio("Select a Page", ["EDA"])

    if app_page == "EDA":
        exploratory_data_analysis()

if __name__ == "__main__":
    main()
   

