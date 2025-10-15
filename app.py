import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    os.getenv("OPENAI_API_KEY")
    st.set_page_config(page_title="AskPDF-RAG", page_icon=":tada:", layout="wide")
    st.header("AskPDF-RAG")
    st.subheader("Ask questions about your PDF documents")
    st.file_uploader("Upload your PDF files", type=["pdf"])


if __name__=='__main__':
    main()