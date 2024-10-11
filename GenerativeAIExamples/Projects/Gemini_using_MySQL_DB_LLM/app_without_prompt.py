import os
import streamlit as st
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from langchain_community.utilities import SQLDatabase
import google.generativeai as genai
import pymysql
## Configure Genai Key
from dotenv import load_dotenv
load_dotenv() 
## load all the environemnt variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

api_key=os.getenv("GOOGLE_API_KEY")


#Database connection parameters 
db_user = os.getenv("MYSQL_USER")
db_password = os.getenv("MYSQL_PASSWORD")
db_host = os.getenv("MYSQL_HOST")
db_name = os.getenv("MYSQL_DATABASE")

# Create SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

#initialised LLM 
LLM = GoogleGenerativeAI(model="gemini-pro",api_key=api_key)
#LLM = genai.GenerativeModel('gemini-pro')

# Initialize SQLDatabase
db = SQLDatabase(engine, sample_rows_in_table_info=3)

#create sql query chain 
chain = create_sql_query_chain(LLM,db)


def execute_query(question):
    try:
        #generate sql query from queation 
        response = chain.invoke({"question": question})
        #cleaned_query = response.strip('```sql\n').strip('\n```')
        #print("cleaned_query",cleaned_query)

        # Clean the response by removing any unwanted characters
        cleaned_query = response.replace("SQLQuery:", "").replace("```", "").strip()
        print("cleaned_query", cleaned_query)

        # Execute the query
        result = db.run(cleaned_query)
                
        # Return the query and the result
        return cleaned_query, result
    except ProgrammingError as e:
        st.error(f"An error occurred: {e}")
        return None, None

# Streamlit interface
st.title("MySql CHATBOT Using Gemini LLM")

# Input from user
question = st.text_input("Enter your question:")

if st.button("Execute"):
    if question:
        cleaned_query, query_result = execute_query(question)
        
        if cleaned_query and query_result is not None:
            print("cleaned_query.......",cleaned_query)
            st.write("Generated SQL Query Convert Text To SQL:")
            st.write(cleaned_query)
            st.write("Query Result:")
            st.write(query_result)
        else:
            st.write("No result returned due to an error.")
    else:
        st.write("Please enter a question.")
        