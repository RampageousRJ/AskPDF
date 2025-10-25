import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def main():
    st.set_page_config(page_title="AskPDF-RAG", page_icon=":tada:", layout="wide")
    st.header("AskPDF-RAG")
    st.subheader("Ask questions about your PDF documents")

    # Load OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found in environment variables.")
        return

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF files", type=["pdf"])
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Split text
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings + vector store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        knowledge_base = Chroma.from_texts(chunks, embeddings)

        # User query
        user_input = st.text_input("Ask your question about the PDF:")
        if user_input:
            # Show similar documents for transparency
            docs = knowledge_base.similarity_search(user_input)
            with st.expander("View similar documents"):
                st.write(docs)

            # Create LLM
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=openai_api_key
            )
            
            # Create RAG prompt
            template = """Answer the question based only on the following context: {context}
                    Question: {question}
                    Answer:"""
            prompt = ChatPromptTemplate.from_template(template)
            
            # Create retriever
            retriever = knowledge_base.as_retriever()
            
            # Create RAG chain using LCEL (LangChain Expression Language)
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Get response
            response = rag_chain.invoke(user_input)
            st.write("**Answer:**")
            st.write(response)

if __name__ == "__main__":
    main()