import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = st.secrets["api_keys"]["openai"]

st.header("Chatbot exercise")

with st.sidebar:
    st.title("Your document")
    file = st.file_uploader("Upload a PDF file and start asking questions", type=["pdf"])

# Extract text from the PDF file
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
        #st.write(text)

    # Break the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    st.write(text_chunks)

    # Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Creating vector store - FAISS storage
    vector_store = FAISS.from_texts(text_chunks, embeddings)

    # Get user question
    user_question = st.text_input()

    # Do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)
        
        # Define LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0, # Set temperature to 0 for deterministic output
            openai_api_key=OPENAI_API_KEY,
            max_tokens=1000
        )
        
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write("Response: " + response)