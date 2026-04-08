import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder  # <-- added MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage  # <-- added
import os
import unicodedata

def clean_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")

upload_dir = 'raw_pdf'

def save_pdf(document):
    os.makedirs(name=upload_dir, exist_ok=True)
    if document is not None:
        file_path = os.path.join(upload_dir, document.name)
        with open(file=file_path, mode='wb') as file:
            file.write(document.getbuffer())
    return file_path

def delete_pdf(path):
    if os.path.exists(path=path):
        os.remove(path=path)
        st.success('Files Deleted Successfully')
    else:
        st.error('Error in deletion, path doesnot exits')

def create_embeddings(file_path, model, chunk_size, chunk_overlap):
    loader = PyMuPDFLoader(file_path=file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.split_documents(documents=docs)
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=model
    )
    return vector_store


# --- CHANGED: prompt now includes MessagesPlaceholder for chat history ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),  # history slots in here
    ("human", "{question}")
])

parser = StrOutputParser()

def chat_with_user(model):
    chain = prompt | model | parser
    return chain


# --- ADDED: initialize chat history in session state once ---
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    chat_model = None
    embedding_model = None
    st.header('Settings')
    st.text('Configure Models and Upload PDFs from here.')
    chat_model = st.selectbox(label="Select Chat Model", placeholder='Select Your Model', options=['qwen/qwen3-32b', 'groq/compound', 'llama-3.1-8b-instant', 'openai/gpt-oss-120b'])
    chat_api_key = st.text_input(label='Enter ChatModels API Key', type='password', placeholder='Enter API Key')

    embedding_model = st.selectbox(
        label="Select Embedding Model",
        placeholder="Select Your Model",
        options=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "microsoft/harrier-oss-v1-0.6b"
        ]
    )
    embedding_api_key = st.text_input(label='Enter Embedding Models API Key', type='password', placeholder='Enter API Key')

    chunk_size = st.slider(label='Select Chunk Size', min_value=100, max_value=1500, step=100, value=800)
    chunk_overlap = st.slider(label='Select Chunk Overlap', min_value=10, max_value=150, step=10, value=80)
    document = st.file_uploader(label='Upload User Documents(PDF Format)', type=['pdf'], accept_multiple_files=False)

    file_path = None
    vector_store = None

    if chat_api_key:
        chat_api_key = chat_api_key.encode("ascii", "ignore").decode("ascii").strip()  # sanitize
        chat_model = ChatGroq(
            model=chat_model,
            api_key=chat_api_key
        )

    if embedding_api_key:
        embedding_api_key = embedding_api_key.encode("ascii", "ignore").decode("ascii").strip()  # sanitize
        embedding_model = HuggingFaceEndpointEmbeddings(
            repo_id=embedding_model,
            huggingfacehub_api_token=embedding_api_key
        )

    if document is not None:
        file_path = save_pdf(document=document)
        st.success(f"Saved: {document.name}")
        vector_store = create_embeddings(file_path=file_path, model=embedding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.success('Embeddings Creation Successful')

    reset = st.button(label='Delete Your Docs')
    if reset:
        if file_path:
            delete_pdf(file_path)
            vector_store = None
        else:
            st.warning("No file to delete")

    # --- ADDED: button to clear chat history ---
    if st.button(label='Clear Chat History'):
        st.session_state.messages = []
        st.success('Chat history cleared')


st.title(body='RAG Chatbot')
st.text('Update your keys, Models and Upload the PDF You want to chat with..')

if chat_api_key:
    if embedding_api_key:
        if document:

            # --- ADDED: display all past messages before the input box ---
            for message in st.session_state.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(message.content)

            query = st.chat_input("Enter Your Query")

            if query:
                retriever = vector_store.as_retriever(search_type="mmr")
                docs = retriever.invoke(query)
                context = "\n\n".join([clean_text(doc.page_content) for doc in docs])
                context = context[:4000]
                query = clean_text(query)

                chain = chat_with_user(model=chat_model)

                # --- CHANGED: pass chat_history, and trim to last 10 messages to avoid context overflow ---
                response = chain.invoke({
                    'question': query,
                    'context': context,
                    'chat_history': st.session_state.messages[-10:]  # only last 10 messages
                })

                # --- ADDED: save the new exchange to session state ---
                st.session_state.messages.append(HumanMessage(content=query))
                st.session_state.messages.append(AIMessage(content=response))

                # display the latest exchange
                with st.chat_message("user"):
                    st.write(query)
                with st.chat_message("assistant"):
                    st.write(response)

        else:
            st.error('Error With Document: Please add Your Document')
    else:
        st.error('Error With API Key: Please Enter Your Embedding API key')
else:
    st.error('Error With API Key: Please Enter Your Chat API key')