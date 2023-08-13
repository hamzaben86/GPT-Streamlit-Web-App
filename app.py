import os
import logging
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
import qdrant_client
import requests

load_dotenv()


def get_qdrant_client():
    client = qdrant_client.QdrantClient(
        url=os.environ["QDRANT_HOST"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    return client


def create_qdrant_collection(collection_name: str):
    client = get_qdrant_client()
    st.write("Create new qdrant collection: ", collection_name)
    client.recreate_collection(
        collection_name="{collection_name}",
        vectors_config=qdrant_client.http.models.VectorParams(
            size=1536,  # 1536 for OpenAI embeddings, 768 for instructor-xl
            distance=qdrant_client.http.models.Distance.COSINE,  # distance metric for similarity search
        ),
    )


def create_qdrant_vector_store():
    client = get_qdrant_client()
    embeddings = get_embedding_model()
    doc_store = Qdrant(
        client=client,
        collection_name="texts",
        embeddings=embeddings,
    )


def get_qdrant_collection(collection_name: str = "my-collection"):
    response = requests.get(
        url=os.environ["QDRANT_HOST"] + "/collections",
        headers={"api-key": f'{os.environ["QDRANT_API_KEY"]}'},
    )
    collections = response["result"]["collections"]

    if not any(
        collection["name"] == collection_name
        for collection in collections["result"]["collections"]
    ):
        create_qdrant_collection(collection_name=collection_name)


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
        chunk_overlap=200,  # overlap goal to preserve context
        length_function=len,  # len function from Python
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_embedding_model(embedding_model_name: str = "text-embedding-ada-002"):
    if embedding_model_name == "text-embedding-ada-002":
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    elif embedding_model_name == "instructor-base":
        embedding_model = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"}
        )
    else:
        logging.error("Invalid embedding model name:", embedding_model_name)

    st.write("Embedding model: ", embedding_model)
    return embedding_model


def get_vector_store(text_chunks, embedding="text-embedding-ada-002"):
    # get_embedding_model
    embedding_model = get_embedding_model()
    # FAISS vector DB (local)
    st.write("Creating local FAISS vector store..")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
    return vectorstore


def get_llm_model(llm_model: str = "OpenAI"):
    if llm_model == "OpenAI":
        llm = ChatOpenAI()

    elif llm_model == "HuggingFace":
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.5, "max_length": 512},
        )
    else:
        logging.error("Invalid LLM name:", llm_model)

    return llm


def get_conversation_chain(vectorstore, llm_model="OpenAI"):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = get_llm_model(llm_model)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )


def main():
    # Stremlit App
    st.set_page_config(page_title="Chat with Documents using GPT-4")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Documents using GPT-4")
    user_question = st.text_input("Ask a question about your documents:")

    try:
        if user_question:
            handle_userinput(user_question)
    except:
        pass

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
                st.session_state.conversation = get_conversation_chain(
                    vectorstore=vectorstore, llm_model="HuggingFace"
                )


if __name__ == "__main__":
    main()
