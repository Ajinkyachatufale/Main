import streamlit as st
import os
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown

from dotenv import load_dotenv

#read env from document
load_dotenv()

#remove special charactor from response text 
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(question):
  model = genai.GenerativeModel('gemini-pro')
  response = model.generate_content(question)
  return response.text

##initialize streamlit app

#st.set_page_config(page_tital="Q and A Gemini demo")
st.header("Gemini Application")
input = st.text_input("input: ",key ="input")
submit = st.button("Ask Question")

#if submit clicked 
if submit:
  response=get_gemini_response(input)
  st.subheader("The Response is")
  st.write(response)