import os
import streamlit as st
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from langchain_community.utilities import SQLDatabase
import google.generativeai as genai
import pymysql
import pandas as pd  # For handling table format
import matplotlib.pyplot as plt  # For plotting
from decimal import Decimal  # For handling decimal values

# Configure GenAI Key
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
api_key = os.getenv("GOOGLE_API_KEY")

# Database connection parameters
db_user = os.getenv("MYSQL_USER")
db_password = os.getenv("MYSQL_PASSWORD")
db_host = os.getenv("MYSQL_HOST")
db_name = os.getenv("MYSQL_DATABASE")

# Create SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# Initialize LLM with Gemini AI model
LLM = GoogleGenerativeAI(model="gemini-pro", api_key=api_key)

# Initialize SQLDatabase
db = SQLDatabase(engine, sample_rows_in_table_info=3)

# Create the SQL query chain
chain = create_sql_query_chain(LLM, db)

# Few-shot examples for LLM
few_shot_examples = """
Q: Count the number of transactions made each month.
SQL: SELECT DATE_FORMAT(transaction_date, '%Y-%m') AS Month, COUNT(*) AS TransactionCount FROM transactions GROUP BY Month ORDER BY Month;

Q: Identify the top 10 spending customers based on their total amount spent.
SQL: SELECT customer_name, SUM(total_spent) AS TotalSpent FROM transactions GROUP BY customer_name ORDER BY TotalSpent DESC LIMIT 10;

Q: Calculate the average age of customers grouped by gender.
SQL: SELECT gender, AVG(age) AS AverageAge FROM customers GROUP BY gender;

Q: Calculate the total sales amount per product category.
SQL: SELECT product_category, SUM(sales_amount) AS TotalSales FROM sales GROUP BY product_category;

Q: How many unique customers are there for each product category?
SQL: SELECT product_category, COUNT(DISTINCT customer_id) AS UniqueCustomers FROM sales GROUP BY product_category;

Q: Count the total number of records in the transactions table.
SQL: SELECT COUNT(*) AS TotalRecords FROM transactions;
"""

# Function to clean the generated SQL query
def clean_sql_query(response):
    cleaned_query = response.replace("SQLQuery:", "").replace("sql", "").replace("```", "").strip()
    if "COUNT()" in cleaned_query:
        cleaned_query = cleaned_query.replace("COUNT()", "COUNT(*)")  # Change COUNT() to COUNT(*)
    cleaned_query = cleaned_query.replace("FROM", " FROM ").replace("GROUP BY", " GROUP BY ").replace("ORDER BY", " ORDER BY ")
    cleaned_query = cleaned_query.replace("strftime", "DATE_FORMAT")
    return cleaned_query

# Function to convert decimal.Decimal objects in the DataFrame to float
def convert_decimal_to_float(df):
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
    return df

# Function to generate and execute the SQL query based on user input
def execute_query(question):
    try:
        prompt = f"""
        You are an expert SQL query generator. Given a natural language question, translate it into a SQL query.
        Here are a few examples:
        
        {few_shot_examples}
        
        Now, translate the following question into SQL:
        Q: {question}
        SQL:
        """
        
        # Generate SQL query from the prompt using LLM
        response = chain.invoke({"question": prompt})

        # Clean the SQL query
        cleaned_query = clean_sql_query(response)

        # Execute the SQL query on the database
        with engine.connect() as connection:
            result = connection.execute(text(cleaned_query))
            rows = result.fetchall()
            column_names = result.keys()

            # Convert the result into a Pandas DataFrame
            if rows:
                result_df = pd.DataFrame(rows, columns=column_names)
                # Convert decimal.Decimal to float
                result_df = convert_decimal_to_float(result_df)
                return cleaned_query, result_df
            else:
                return cleaned_query, None
    
    except ProgrammingError as e:
        st.error(f"SQL Syntax Error: {e}")
        return None, None
    except SQLAlchemyError as e:
        st.error(f"Database Error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        return None, None

# Streamlit interface
st.title("MySQL Chatbot Using Gemini LLM")

# User input in the main section
question = st.text_input("Enter your question:")

# Sidebar for buttons and download
with st.sidebar:
    # Button to execute the query
    execute_button = st.button("Execute")
    
    
    
    # Button to show the graph
    show_graph_button = st.button("Show Graph")

# CSS for uniform button size and bold text
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Execute query and show result in the main area
if execute_button:
    if question:
        # Call the function to generate the SQL query and execute it
        cleaned_query, query_result = execute_query(question)

        if cleaned_query and query_result is not None:
            st.write("Generated SQL Query:")
            st.code(cleaned_query, language="sql")

            # Display query result as a table
            st.write("Query Result:")
            st.table(query_result)  # Display the DataFrame as a table

            # Store the query result in session state for later use
            st.session_state['query_result'] = query_result
           # Add a download button (after Execute)
            if 'query_result' in st.session_state:
                csv = st.session_state['query_result'].to_csv(index=False)
                st.download_button(
                    label="Download Report",
                    data=csv,
                    file_name='report.csv',
                    mime='text/csv',
                )


        else:
            st.write("No result returned due to an error.")
    else:
        st.write("Please enter a question.")

# Show the graph in the main area when "Show Graph" is clicked
if show_graph_button and 'query_result' in st.session_state:
    query_result = st.session_state['query_result']

    # Check if DataFrame contains numeric columns for plotting
    numeric_columns = query_result.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 0:
        st.write("Visualization:")
        fig, ax = plt.subplots()

        # Plot using the first numeric column (x-axis) and the others (y-axis)
        if len(numeric_columns) > 1:
            query_result.plot(kind='bar', x=numeric_columns[0], y=numeric_columns[1:], ax=ax)
        else:
            query_result.plot(kind='bar', y=numeric_columns[0], ax=ax)

        st.pyplot(fig)
