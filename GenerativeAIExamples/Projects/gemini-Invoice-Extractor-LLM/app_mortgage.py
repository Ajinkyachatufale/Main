import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  
# Take environment variables from .env.

# Load API key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API key not found. Please check your .env file.")

# Function to get response from Gemini model
def get_gemini_response(input_text, image_data, prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input_text, image_data[0], prompt])
        return response.text
    except Exception as e:
        st.error(f"Error getting response from Gemini model: {e}")
        return ""

# Function to set up image for processing
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize Streamlit app
st.header("Mortgage Document Analysis with Gemini AI")

# User input for question
input_text = st.text_input("Enter your question about the mortgage document:", key="input")
uploaded_file = st.file_uploader("Upload a mortgage document image (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Document", use_column_width=True)

# Prompt for AI model
mortgage_prompt = """
    You are an AI specialized in reading and analyzing mortgage documents.
    You will receive input images of mortgage documents and should answer questions
    based on the content, terms, conditions, and specific details within the document.
    Please interpret and provide accurate information from the document.
"""

# Variable to store the response text
response_text = ""

# Analyze document
if st.button("Analyze Document"):
    if uploaded_file is not None:
        image_data = input_image_setup(uploaded_file)
        response_text = get_gemini_response(input_text, image_data, mortgage_prompt)
        st.subheader("Response:")
        st.write(response_text)
    else:
        st.error("Please upload a mortgage document to analyze.")

# Provide download button for the response text
if response_text:
    st.download_button(
        label="Download Analysis Result",
        data=response_text,
        file_name="mortgage_document_analysis.txt",
        mime="text/plain"
    )
else:
    st.info("No response available to download. Please analyze the document first.")
