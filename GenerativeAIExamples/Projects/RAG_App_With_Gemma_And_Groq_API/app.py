import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
#split charactor in document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
#vector store db
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#word to vector conversion
from dotenv import load_dotenv
import time


#load Groq api key and google api key
#need to create .env file and assign below two API key
gorq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is None:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")
os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("Gamma model document Q&A")

# model declaration 
#llm = ChatGroq(gorq_api_key=gorq_api_key, model_name="Llama3-8b-8192")
llm=ChatGroq(groq_api_key=gorq_api_key,model_name="Llama3-8b-8192")

#chat prompt template it required to input to model
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)


def vecotr_embading(): 

    if "vectors" not in st.session_state: 
        
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./sample_pdf") # Data ingestion
        st.session_state.docs=st.session_state.loader.load() #document loader 
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)#check creation based on document split document into small chunk
        #spalitter 
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) 
        # apply FAISS DB and google embadding techinique
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)




prompt1=st.text_input("Enter Question")

if st.button("Click for Document Embading"):
    vecotr_embading()
    st.write("FAISS vector Database is ready to use ")


if prompt1: 

    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    print("start",start)
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])
    #print("response['answer']",response['answer'])

     # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")