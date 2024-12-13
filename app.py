#pip install streamlit joblib
import plotly.express as px
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
#from PyPDF2 import PdfReader
import google.generativeai as genai
from snowflake.snowpark import Session

# Establish Snowflake session
@st.cache_resource
def create_session():
    return Session.builder.configs(st.secrets.snowflake).create()

session = create_session()
st.success("Connected to Snowflake!")

# Function to load datasets
def load_data():
    try:
        # Load datasets
        df = pd.read_csv('df.csv')
        merged_cleaned = pd.read_csv('merged_cleaned.csv', low_memory=False)
        food_hampers_fact = pd.read_csv('Food Hampers Fact.csv', low_memory=False)
        clients_data = pd.read_csv('Clients Data Dimension1.csv', low_memory=False)
        return df, merged_cleaned, food_hampers_fact, clients_data
    except FileNotFoundError as e:
        st.error(f"Error: {e.filename} not found. Please upload the file.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Load the data
data = load_data()
if data:
    df, merged_cleaned, food_hampers_fact, clients_data = data


# Set up the API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', st.secrets.get("GOOGLE_API_KEY"))
genai.configure(api_key=GOOGLE_API_KEY)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     try:
#         reader = PdfReader(pdf_file)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text()
#         return text.strip()
#     except Exception as e:
#         st.error(f"Error reading PDF: {e}")
#         return ""
# Function to generate response from the model
def generate_response(prompt, context):
    try:
        model = genai.GenerativeModel('gemini-pro')
        # Include context from uploaded data in the prompt
        response = model.generate_content(f"{prompt}\n\nContext:\n{context}")
        return response.text  # Use 'text' attribute
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request."
        
# Page 1: Dashboard
def dashboard():
    st.image('Logo.png', use_column_width=True)
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
    merged_cleaned['age'].hist(bins=10)
    #fig = plt.figure(figsize = (8,8))
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()
    st.pyplot(plt.gcf())
    
    merged_cleaned['pickup_date'] = pd.to_datetime(merged_cleaned['pickup_date'])
#checking the average pickups on day basis
    merged_cleaned['DayOfWeek'] = merged_cleaned['pickup_date'].dt.day_name()

#  Group by 'DayOfWeek' and calculate total demand (quantity) for each day
    day_demand = merged_cleaned.groupby('DayOfWeek')['quantity'].sum().reset_index()

#  Sort days in the correct order (Monday to Sunday)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_demand['DayOfWeek'] = pd.Categorical(day_demand['DayOfWeek'], categories=day_order, ordered=True)
    day_demand = day_demand.sort_values('DayOfWeek')

#  Create a bar plot to visualize the demand by day
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DayOfWeek', y='quantity', data=day_demand, palette='viridis')
    plt.title('Total Demand by Day of the Week', fontsize=16)
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Total Quantity', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

# Show the plot
    plt.show()
    st.pyplot(plt.gcf())
    
# Page 3: Exploratory Data Analysis (EDA)
def Visualizations():
    st.title('Streamlit App with Embedded My Google Map')
    st.write('Here is an example of embedded LDS Londonderry Map:')

    # Embedding Google Map using HTML iframe
    st.markdown("""
   <iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/20395848-bdf5-407a-a1f9-38981b686102/page/VVqIE" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>""", unsafe_allow_html=True)
   

# Page 4: Machine Learning Modeling (ARIMA for Food Hamper Prediction)
def machine_learning_modeling():
    st.title("Food Hamper Demand Prediction")
    st.write("Enter the details to predict the number of food hampers needed in the future:")

    # Input field to select the number of days to predict
    future_days = st.number_input("Number of Days to Forecast", min_value=1, max_value=365, value=15)

    if st.button("Predict"):
        # Load the trained ARIMA model
        try:
            model = joblib.load('arima_model.pkl')  # Make sure the ARIMA model is saved with this name
        except FileNotFoundError:
            st.error("Error: ARIMA model file 'arima_model.pkl' not found.")
            return

        # Assuming we are using historical data from merged_cleaned DataFrame
        if 'pickup_date' not in merged_cleaned.columns or 'record_count' not in df.columns:
            st.error("Required columns 'date' or 'actual_pickup' are missing in the dataset.")
            return
        
        # Convert the 'date' column to datetime if it's not already
        merged_cleaned['pickup_date'] = pd.to_datetime(merged_cleaned['pickup_date'])
        merged_cleaned.set_index('pickup_date', inplace=True)

        # Use ARIMA model to forecast future demand (number of hampers)
        forecast = model.get_forecast(steps=future_days)

        # Get the predicted values and confidence intervals
        predicted_values = forecast.predicted_mean.round().astype(int)
        confidence_intervals = forecast.conf_int()

        # Create a DataFrame to show the results
        future_dates = pd.date_range(start=merged_cleaned.index[-1] + pd.Timedelta(days=1), periods=future_days)
        future_predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted Hamper Demand': predicted_values
        })

        # Display the predictions
        st.write("Predicted Food Hamper Demand for the Next {0} Days".format(future_days))
        st.write(future_predictions)

        st.line_chart(future_predictions.set_index('Date')['Predicted Hamper Demand'])
        
# Page 5: XAI
def Explainable_AI():
    st.image('XAI.png', use_column_width=True)
    st.image('XAI1.png', use_column_width=True)  

# Page 6: chatbot

# Streamlit app
def Chat_With_Data():
    st.title("Food Hamper Demand Forecasting")
    st.write("Upload Food Hamper project related files and ask questions based on the data.")
    # File upload
    uploaded_files = st.file_uploader("Upload your project files (CSV/Excel)", type=["csv", "xlsx"], accept_multiple_files=True)

    # Prepare data context
    dataframes = {}
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.name.endswith('.csv'):
                    dataframes[file.name] = pd.read_csv(file)
                elif file.name.endswith('.xlsx'):
                    dataframes[file.name] = pd.read_excel(file)
                st.success(f"Successfully loaded {file.name}")
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")

    # Create context from data
    context = ""
    for file_name, df in dataframes.items():
        context += f"\nData from {file_name}:\n"
        context += df.head(5).to_string()  # Include a preview of the data (first 5 rows)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about your project:", key="input")
    if st.button("Send"):
        if user_input and context:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = generate_response(user_input, context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        elif not context:
            st.error("Please upload relevant files to ask project-specific questions.")

    for message in st.session_state.chat_history:
        st.write(f"{message['role'].capitalize()}: {message['content']}")
# Main App Logic
def main():
    st.sidebar.title("Food Hamper Prediction App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard","Visualizations", "Looker vis", "ML Modeling", "Explainable AI","Chat With Data"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Visualizations":
        exploratory_data_analysis()
    elif app_page == "Looker vis":
        Visualizations()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Explainable AI":
        Explainable_AI()
    elif app_page == "Chat With Data":
        Chat_With_Data()                                              
                                                  

if __name__ == "__main__":
    main()
    
