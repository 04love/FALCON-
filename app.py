# Import required libraries
import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col
import altair as alt

# Set up the Streamlit app
st.title("Food Hamper Distribution Visualizer")
st.write(
    """
    Explore how different parameters impact the distribution of food hamper items, quantities, and prices!
    """
)

# Get the current Snowflake session
session = get_active_session()

# Add sliders to interact with mean and standard deviation for quantities and prices
st.markdown("# Adjust the sliders and watch the results update 👇")
col1, col2 = st.columns(2)
with col1:
    quantity_mean = st.slider("Mean of Item Quantity", 1, 20, 5)
with col2:
    price_stdev = st.slider("Standard Deviation of Item Price", 1, 20, 5)

# Generate data dynamically in Snowflake for the food hamper project
try:
    session.sql(f"""
        CREATE OR REPLACE TABLE FOOD_HAMPER_CATALOG AS
        SELECT CONCAT('ITEM-', UNIFORM(1000, 9999, RANDOM())) AS ITEM_ID,
               ABS(NORMAL({quantity_mean}, {price_stdev}, RANDOM())) AS ITEM_QUANTITY,
               ABS(NORMAL(30, 10::FLOAT, RANDOM())) AS ITEM_PRICE
        FROM TABLE(GENERATOR(ROWCOUNT => 100));
    """).collect()

    st.success("Food hamper data has been successfully generated in Snowflake!")
except Exception as e:
    st.error(f"Error generating data: {e}")

# Read data from Snowflake table and visualize the histogram for item quantity distribution
try:
    df = session.table("FOOD_HAMPER_CATALOG").to_pandas()
    st.write("Data preview:")
    st.dataframe(df.head(10))

    # Create the histogram using Altair for item quantity distribution
    chart = alt.Chart(df).mark_bar().encode(
        alt.X("ITEM_QUANTITY", bin=alt.Bin(step=1), title="Item Quantity Distribution"),
        y='count()',
    ).properties(
        title="Histogram of Item Quantities in Food Hampers"
    )

    # Display the histogram
    st.altair_chart(chart, use_container_width=True)
except Exception as e:
    st.error(f"Error fetching or visualizing data: {e}")

# Footer
st.write("---")
st.write("Powered by Snowflake and Streamlit")
