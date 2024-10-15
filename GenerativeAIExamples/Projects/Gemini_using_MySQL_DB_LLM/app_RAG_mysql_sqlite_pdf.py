import os 
import streamlit as st
from langchain.chains import create_sql_query_chain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from langchain_community.utilities import SQLDatabase
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate  # Added this import
import google.generativeai as genai
import pymysql
import sqlite3
import pandas as pd  # For handling table format
import matplotlib.pyplot as plt  # For plotting
from decimal import Decimal  # For handling decimal values
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Configure GenAI Key
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
api_key = os.getenv("GOOGLE_API_KEY")

# Database connection parameters for MySQL
db_user = os.getenv("MYSQL_USER")
db_password = os.getenv("MYSQL_PASSWORD")
db_host = os.getenv("MYSQL_HOST")
db_name = os.getenv("MYSQL_DATABASE")

# Create SQLAlchemy engine for MySQL
mysql_engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# Create SQLite engine
sqlite_db_path = "student.db"  # Path to your SQLite database
sqlite_engine = create_engine(f"sqlite:///{sqlite_db_path}")

# Initialize LLM with Gemini AI model
LLM = GoogleGenerativeAI(model="gemini-pro", api_key=api_key)

# Initialize SQLDatabase for both MySQL and SQLite
mysql_db = SQLDatabase(mysql_engine, sample_rows_in_table_info=3)
sqlite_db = SQLDatabase(sqlite_engine, sample_rows_in_table_info=3)

# Create the SQL query chain for both databases
mysql_chain = create_sql_query_chain(LLM, mysql_db)
sqlite_chain = create_sql_query_chain(LLM, sqlite_db)

# Few-shot examples for MySQL
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

# SQLite few-shot example
sqlite_prompt = """
You are an expert in converting English questions to SQL queries!
The SQL database has the name STUDENT and has the following columns: ID, NAME, CLASS, SECTION, 
MARKS, AGE, GENDER, ADDRESS, CONTACT_NUMBER, EMAIL, DATE_OF_BIRTH, ADMISSION_DATE, GUARDIAN_NAME, 
GUARDIAN_CONTACT_NUMBER, TOTAL_FEES, FEES_PAID, PERCENTAGE, REMARKS.
\n\n
For example:
Example 1 - How many entries of records are present?
The SQL command will be: SELECT COUNT(*) FROM STUDENT;
Example 2 - Tell me all the students studying in Data Science class?
The SQL command will be: SELECT * FROM STUDENT WHERE CLASS='Data Science';
The SQL code should not have ``` or the word SQL in it.
"""

# Function to clean the generated SQL query
def clean_sql_query(response):
    cleaned_query = response.replace("SQLQuery:", "").replace("sql", "").replace("```", "").strip()
    if "COUNT()" in cleaned_query:
        cleaned_query = cleaned_query.replace("COUNT()", "COUNT(*)")  # Change COUNT() to COUNT(*)
    return cleaned_query

# Function to convert decimal.Decimal objects in the DataFrame to float
def convert_decimal_to_float(df):
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
    return df

# Function to generate and execute the SQL query based on user input
def execute_query(question, db_choice):
    try:
        # MySQL prompt
        mysql_prompt = f"""
        You are an expert SQL query generator. Given a natural language question, translate it into a SQL query.
        Here are a few examples:
        
        {few_shot_examples}
        
        Now, translate the following question into SQL:
        Q: {question}
        SQL:
        """

        # Use the appropriate chain depending on the selected database
        if db_choice == "MySQL":
            response = mysql_chain.invoke({"question": mysql_prompt})
            engine = mysql_engine
        elif db_choice == "SQLite":
            response = sqlite_chain.invoke({"question": sqlite_prompt + f"\n\nQ: {question}"})
            engine = sqlite_engine

        # Clean the SQL query
        cleaned_query = clean_sql_query(response)

        # Execute the SQL query on the selected database
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

# Function to handle PDF embedding and vector search
def vecotr_embading():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("D:/GEN AI/project/Gemini_using_MySQL_DB_LLM/sample_pdf")  # Updated PDF path
        st.session_state.docs = st.session_state.loader.load()  # Load documents from PDFs
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Split documents
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Chunk the docs
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Apply FAISS

# Function to query the PDF content
def query_pdf(prompt1):
    if prompt1 and "vectors" in st.session_state:
        prompt_template = PromptTemplate(
            input_variables=["context", "input"],
            template="""
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            <context>
            Questions: {input}
            """
        )

        document_chain = create_stuff_documents_chain(LLM, prompt_template)  # Use the prompt object
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start} seconds")

        if response and 'answer' in response:
            st.write(response['answer'])

        # Display the document similarity context
        with st.expander("Document Similarity Search"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")

# Streamlit interface
st.title("SQL and PDF Query Chatbot Using Gemini LLM")

# Sidebar for database selection
with st.sidebar:
    db_choice = st.selectbox("Select Data Source", ["MySQL", "SQLite", "PDF"])
    execute_button = st.button("Execute")
    show_graph_button = st.button("Show Graph")
    #embed_button = st.button("Click for PDF Embedding")


# CSS for uniform button size and bold text
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# User input in the main section with a single prompt
if db_choice == "PDF":
    vecotr_embading()
    st.success("PDF Embedding Completed!")
    question = st.text_input("Enter your question for PDF:")
else:
    question = st.text_input("Enter your SQL question:")

# Initialize query result
query_result = None

# Execute SQL or PDF query based on user input
if execute_button:
    if db_choice in ["MySQL", "SQLite"]:
        query_result = execute_query(question, db_choice)
        if query_result[1] is not None:
            query_result_df = query_result[1]
            st.write("Generated SQL Query:", query_result[0])
            st.dataframe(query_result_df)
            # Store the query result in session state for later use
            st.session_state['query_result'] = query_result

    elif db_choice == "PDF":
        vecotr_embading()  # Call to embed the PDFs
        query_pdf(question)

# Show the graph in the main area when "Show Graph" is clicked
if show_graph_button and 'query_result' in st.session_state:
    query_result = st.session_state['query_result']

    if query_result is not None:
        # query_result is a tuple, where the first element is the query and the second is the DataFrame
        result_df = query_result[1]  # Extract the DataFrame from the tuple
        
        if result_df is not None:
            # Check if DataFrame contains numeric columns for plotting
            numeric_columns = result_df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_columns) > 0:
                st.write("Visualization:")
                fig, ax = plt.subplots()

                # Plot using the first numeric column (x-axis) and the others (y-axis)
                if len(numeric_columns) > 1:
                    result_df.plot(kind='bar', x=numeric_columns[0], y=numeric_columns[1:], ax=ax)
                else:
                    result_df.plot(kind='bar', y=numeric_columns[0], ax=ax)

                st.pyplot(fig)
            else:
                st.warning("No numeric columns available in the query result to plot.")
        else:
            st.warning("No data available for plotting.")
    else:
        st.error("No query results to display.")
