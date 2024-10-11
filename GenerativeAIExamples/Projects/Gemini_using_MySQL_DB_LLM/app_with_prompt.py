import os
import streamlit as st
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from langchain_community.utilities import SQLDatabase
import google.generativeai as genai
import pymysql

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
    # Clean the response by removing 'sql', 'SQLQuery:', and unwanted formatting
    cleaned_query = response.replace("SQLQuery:", "").replace("sql", "").replace("```", "").strip()

    # Ensure COUNT() has a column name
    if "COUNT()" in cleaned_query:
        cleaned_query = cleaned_query.replace("COUNT()", "COUNT(*)")  # Change COUNT() to COUNT(*)

    # Add a space between SQL keywords and table/column names if missing
    cleaned_query = cleaned_query.replace("FROM", " FROM ").replace("GROUP BY", " GROUP BY ").replace("ORDER BY", " ORDER BY ")

    # Post-process the query to ensure MySQL compatibility
    cleaned_query = cleaned_query.replace("strftime", "DATE_FORMAT")

    return cleaned_query

# Function to generate and execute the SQL query based on user input
def execute_query(question):
    try:
        # Few-shot prompt with user question appended
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

        # Clean the SQL query using the updated cleaning function
        cleaned_query = clean_sql_query(response)

        print("Generated SQL Query:", cleaned_query)

        # Execute the SQL query on the database
        result = db.run(cleaned_query)
        
        # Return the generated query and the result
        return cleaned_query, result
    
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

# User input
question = st.text_input("Enter your question:")

# Button to execute the query
if st.button("Execute"):
    if question:
        # Call the function to generate the SQL query and execute it
        cleaned_query, query_result = execute_query(question)
        
        if cleaned_query and query_result is not None:
            st.write("Generated SQL Query:")
            st.write(cleaned_query)
            st.write("Query Result:")
            st.write(query_result)
        else:
            st.write("No result returned due to an error.")
    else:
        st.write("Please enter a question.")
