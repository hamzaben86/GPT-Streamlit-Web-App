import streamlit as st


def main():
    st.set_page_config(page_title="Chat with Documents using GPT-4")

    st.header("Chat with Documents using GPT-4")
    st.text_input("Ask a question about your documents:")

    with st.sidebar:
        # sidebar context
        st.subheader("Your Documents")
        st.file_uploader("Upload your Documents and click on PROCESS")
        st.button("Process")


if __name__ == "__main__":
    main()
