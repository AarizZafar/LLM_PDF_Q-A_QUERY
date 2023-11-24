import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import huggingface_hub
from htmlTemplates import css,bot_template, user_template

# ConversationalRetrievalChain -> FOCUSES IN RETRIEVING INFORMATION FROM A KNOWLEDGE BASE, PROVIDES THE
# FRAME WORK FOR CONSTRUCTING THE SEARCH QUERY, ACCESSING THE KNOWLEDGE BASE

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
            separator        = '\n',
            chunk_size       = 200,
            chunk_overlap    = 20,
            length_function  = len            # THE LEN FUNCTION WILL BE USED TO CALCULATE THE LENGTH OF THE CHUNK
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
    print('1')
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embedding)       # CREATES A SEARCHABLE INDEX OF TEXT CHUNKS USING THEIR 
                                                                                    # CORRESPONDING EMBEDDINGS
    print('2')
    return vectorstore

def get_conversation_chain(vectorstore): 
    # max_length SPECIFIES THE MAX NUMBER OF TOKENS THAT THE LANGUAGE MODLE WILL GENERATE IN ITS RESPONSE
    llm = huggingface_hub(repo_id = 'google/flan-t5-xxl', model_kwargs = {'temperature' : 0.5, 'max_length' : 512})
    # memory_key = 'chat_history' THE MEMORY WHERE THE CONVERSATION WILL BE STORED THE MEMORY WILL BE LABELED WITH A NAME

    # return_message = True DETERMINES WHETHER OR NOT THE CONVERSATION HISTORY SHOULD BE INCLUDED WHEN THE MEMORY IS 
    # ACCESSED.
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question' : user_question})      # THIS CONTAINES ALL THE CONFIGURATION 
                                                                                # FROM THE VECTOR STORE AND FROM OUR MEMORY    
    # 
    st.write(response)
    # 

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace('{{MSG}}', message.content), unsafe_allow_html = True)
        else:
            st.write(bot_template.replace('{{MSG}}', message.content), unsafe_allow_html = True)

def main():
    load_dotenv()     # LOADING ALL THE ENVIRONMENT VARIABLES FROM THE .env FILE
    st.set_page_config(page_title = 'Chat with multiple documents ', page_icon = ':books:')

    st.write(css, unsafe_allow_html=True)

    print('**************************')
    print(st.session_state)
    print('**************************')

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with multiple PDFs :books:')
    user_question = st.text_input('Ask a question about your documents')

    if user_question:
        handle_userinput(user_question)

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
                vectorstore = get_vectorstore(text_chunks)

                # CREATE CONVERSATION CHAIN 
                # (STREAMLIT HAS THE TENDANCY TO RELOAD ITS COMPLETE CODE TO PREVENT IT FROM HAPPENING WE USE THE 
                # SESSION_STATE)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()