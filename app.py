import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    '''
        This function is responsible for extracting all the text from the pdf files

        ARGS : pdf_docs (pdf files)

        return -> text from the pdf
    '''
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)          # CREATING A PDF READER FOR EACH PDF
        for page in pdf_reader.pages:        # LOOPING THROUGH EACH PAGE IN THE PDF
            text += page.extract_text()      # EXTRACTING THE CONTENT OF EACH PAGE AND APPENDING IT TO THE TEXT VARIABLE
    return text

def get_text_chunks(raw_text):
    '''
        Function is responsible of converting the raw_text into chunks of text

        ARGS : raw_text

        return -> chunks from the raw_text
    '''
    text_splitter = CharacterTextSplitter(
            separator = '\n',
            chunk_size = 200,
            chunk_overlap = 20,
            length_function = len            # THE LEN FUNCTION WILL BE USED TO CALCULATE THE LENGTH OF THE CHUNK
    )
    chunk = text_splitter.split_text(raw_text)
    return chunk

def get_vectorstore(text_chunks):
    '''
        Function is responsible for converting the chunks into vector using embedding models 

        ARGS (text_chinks) - Chunks 

        return - A vectorstore for the converted chunks
    '''
    embedding = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embedding)       # CREATES A SEARCHABLE INDEX OF TEXT CHUNKS USING THEIR 
                                                                                    # CORRESPONDING EMBEDDINGS
    return vectorstore

def main():
    load_dotenv()     # LOADING ALL THE ENVIRONMENT VARIABLES FROM THE .env FILE
    st.set_page_config(page_title = 'Chat with multiple documents ', page_icon = ':books:')

    st.header('Chat with multiple PDFs :books:')
    st.text_input('Ask a question about your documents')

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader('Upload your PDFs here and click on process ', accept_multiple_files = True)

        if st.button('Process'):
            with st.spinner('Processing'):

                # GETTING ALL THE TEXT FROM THE PDF
                raw_text = get_pdf_text(pdf_docs)

                # GETTING THE TEXT CHUNKS
                text_chunks = get_text_chunks(raw_text)

                # CONVERTING THE CHUNKS INTO VECTOR STORE
                vectoestore = get_vectorstore(text_chunks)


if __name__ == "__main__":
    main()


