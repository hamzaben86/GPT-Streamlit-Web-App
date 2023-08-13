import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt
import logging

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
        print(f"Unable to extract text from {doc}:", exc_info=True)

    return text


def main():
    # Stremlit App
    st.set_page_config(page_title="Chat with Documents using GPT-4")

    st.header("Chat with Documents using GPT-4")
    st.text_input("Ask a question about your documents:")

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
                st.write(raw_text)
                # get the text chunks

                # create vector store (knowledge base)


if __name__ == "__main__":
    main()
