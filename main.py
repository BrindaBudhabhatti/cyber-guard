import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import os
from langchain_groq import ChatGroq

api_key = os.getenv("GROQ_API_KEY")

@st.cache_resource(show_spinner=False)
def load_chain():
    loader = TextLoader("cyber_data.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key) 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

qa_chain = load_chain()


st.set_page_config(page_title="CyberGuard AI", page_icon="🛡️", layout="wide")


st.markdown("""
<div class="header">
    <h2 style='text-align:center;'>🛡️ CyberGuard AI</h2>
    <p style='text-align:center;'>Your AI-powered Cyber Crime Awareness Assistant by <b>Brinda Budhabhatti</b></p>
    <hr style="border-color: #30363D;">
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("⚙️ About CyberGuard AI")
    st.write("Developed by **Brinda Budhabhatti**")
    st.write("Built using **LangChain + Groq + Streamlit**")
    st.markdown("[🔗 Report Cyber Crime (India)](https://cybercrime.gov.in/)")



# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    with st.chat_message("assistant"):
            st.markdown("**Welcome to the CyberCrime Support Bot!**  "
            "***I'm here to assist with cybercrime-related inquiries.***")

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "You":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)

# with st.container(vertical_alignment='bottom'):
# Input box
user_query = st.chat_input("Type your question about cyber safety...")

if user_query:
    st.session_state.chat_history.append(("You", user_query))
    with st.spinner("Analyzing your question..."):
        result = qa_chain.invoke({"question": user_query})
        answer = result["answer"]
    st.session_state.chat_history.append(("CyberGuard AI", answer))
    st.rerun()


