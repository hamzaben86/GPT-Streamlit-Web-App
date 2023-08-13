import os
import logging
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

load_dotenv()


def get_doc_text(docs):
    """Get raw text from doc repo

    Args:
        docs: uploaded documents

    Returns:
        text(str): raw text data extract from documents
    """

    text = ""
    try:
        for doc in docs:
            if ".pdf" in doc.name:
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    text += page.extract_text()

            if ".docx" in doc.name:
                text += docx2txt.process(doc)
    except:
        logging.error(f"Unable to extract text from {doc}:", exc_info=True)

    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # characters count
        chunk_overlap=200,  # overlap goal is preserve context
        length_function=len,  # len function from Python
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks, embedding="text-embedding-ada-002"):
    st.write("Embedding model: ", embedding)
    if embedding == "text-embedding-ada-002":
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    elif embedding == "instructor-base":
        embedding_model = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"}
        )
    else:
        logging.error("Invalid embedding model provided")

    st.write("Creating local FAISS vector store..")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )  # buffer for storing conversation memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.write(response)


def main():
    # Stremlit App
    st.set_page_config(page_title="Chat with Documents using GPT-4")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with Documents using GPT-4")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    st.write(user_template, unsafe_allow_html=True)
    st.write(bot_template, unsafe_allow_html=True)

    with st.sidebar:
        # sidebar context
        st.subheader("Your Documents")

        docs = st.file_uploader(
            "Upload your Documents and click on PROCESS", accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing"):
                # get the raw contents of the document
                raw_text = get_doc_text(docs)
                # st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vector store (knowledge base)
                vectorstore = get_vector_store(text_chunks=text_chunks)

                # create conversation chain (make conversation variable peristent)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
