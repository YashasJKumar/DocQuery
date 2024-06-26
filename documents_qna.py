# ------------------------------------- IMPORT STATEMENTS --------------------------------------------------------------

import os
import time
import tempfile
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ------------------------------------- STREAMLIT UI -------------------------------------------------------------------
st.logo(image="./groq-logo.png")
st.title("DocQuery")
st.sidebar.markdown(":orange[Disclaimer:]")
st.sidebar.write("")
st.sidebar.write("Upload your Document & unleash the power of Large Language Models to answer your queries.")
chosen_model = st.sidebar.selectbox(label="Select LLM here: ", label_visibility="visible",
                            options=["Llama3-8b-8192", "Llama3-70b-8192", "Mixtral-8x7b-32768",
                                     "Gemma-7b-It"])
st.sidebar.write("")

# ------------------------------------- LOADING THE DOCUMENT-----------------------------------------------------------

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
uploaded_file = st.sidebar.file_uploader("Upload your Document here: ", type=["pdf", "txt", "doc", "docx", "csv"],
                                         accept_multiple_files=False)

temp_file_path = None
if uploaded_file is not None:
    # Create a temporary file and write the uploaded file's content to it
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name


# --------------------------------------- VECTOR CONVERSION ------------------------------------------------------------


def file_loader():
    if "vectors" not in st.session_state:
        st.sidebar.write("You have uploaded: " + str(uploaded_file.type) + " file.")
        # Splitting into smaller chunks.
        if uploaded_file.type == "application/pdf":
            st.session_state.loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.type == "text/plain":
            st.session_state.loader = TextLoader(file_path=temp_file_path)
        elif uploaded_file.type == "text/csv":
            st.session_state.loader = CSVLoader(file_path=temp_file_path)
        st.session_state.file = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_documents = st.session_state.text_splitter.split_documents(st.session_state.file)

        # Unlink the temporary file after use.
        os.unlink(temp_file_path)

        try:
            # Converting into embeddings.
            st.session_state.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Initializing the FAISS DB & storing the embeddings in it
            st.session_state.vectors = FAISS.from_documents(documents=splitted_documents,
                                                            embedding=st.session_state.embedding_model)
        except Exception:
            st.warning("Please Start your Ollama Server.")


# ------------------------------------- DEFINE PROMPT TEMPLATE ---------------------------------------------------------

prompt_template = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context.Think step by step before providing the 
    detailed answer. 
    <context> 
    {context} 
    </context> 
    Question: {input}
    """
)

# ------------------------------------- LOAD THE LLM -------------------------------------------------------------------

try:
    llm = ChatGroq(groq_api_key=st.secrets["GROQ_API_KEY"],
                   model_name=chosen_model)
except Exception as e:
    st.warning("There was a problem with the Groq API. Please try again.")

if uploaded_file is not None:
    if "vectors" not in st.session_state:
        with st.spinner("Loading.."):
            file_loader()

        if "vectors" in st.session_state:
            success_msg = st.success("Embedding are ready..")

            # Create a chain for LLM & Prompt Template to inject to LLM for inferencing
            document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
            retriever = st.session_state.vectors.as_retriever()

            # Create a retrieval chain which links the retriever & document chain
            st.session_state.retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Remove the success_msg
            time.sleep(1)
            success_msg.empty()
else:
    st.warning("Please upload any file before asking questions..")

# ------------------------------------- USER QUERY --------------------------------------------------------------------

if uploaded_file is not None and "vectors" in st.session_state:
    # Ask prompt from user
    if user_prompt := st.chat_input("Enter your query: "):
        message_container = st.container(height=600, border=False)
        message_container.markdown(":red[User Prompt: ]")
        message_container.write(user_prompt)
        with st.spinner("Generating response.."):
            start_time = time.time()
            response = st.session_state.retrieval_chain.invoke({"input": user_prompt})
            st.sidebar.write("")
            st.sidebar.markdown("\n\n\n:green[Response Time : ]" + " " +
                                str(round((time.time() - start_time), 4)) + " sec.")
        message_container.markdown(":blue[Response:]")
        message_container.write(response['answer'])
