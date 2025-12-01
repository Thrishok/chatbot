import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_groq import ChatGroq



# Load API Keys
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# -------------------------------------------------------------
# CACHED FUNCTIONS
# -------------------------------------------------------------

@st.cache_resource
def load_pdf():
    loader = PyPDFLoader("/workspaces/chatbot/Price-Action-Trading-Guide.pdf")
    return loader.load()

@st.cache_resource
def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

@st.cache_resource
def build_vectorstore():
    docs = load_pdf()
    chunks = split_text(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db

@st.cache_resource
def build_chain():
    db = build_vectorstore()
    retriever = db.as_retriever()
    model = ChatGroq(model_name="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.
    Use ONLY the below context if it contains the answer.
    If not found, use your general knowledge.

    <context>
    {context}
    </context>

    Question:
    {input}

    Answer:
    """)

    doc_chain = create_stuff_documents_chain(model, prompt)
    return create_retrieval_chain(retriever, doc_chain)


# -------------------------------------------------------------
# UI CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="🤖", layout="wide")

# -------------------------------------------------------------
# CHATGPT-LIKE CSS STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
/* Hide Streamlit elements */
.stToolbar {display: none !important;}
.stBlockContainer {border: none !important; padding: 0 !important;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background image */
.stApp {
    background-image: url('https://images.unsplash.com/photo-1557683316-973673baf926?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.9);
    z-index: -1;
}

/* Message styling */
.user-message {
    background: #007bff;
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px auto;
    max-width: 80%;
    width: fit-content;
    margin-left: auto;
    display: block;
    box-shadow: 0 2px 8px rgba(0,123,255,0.3);
    font-size: 14px;
    line-height: 1.4;
}

.bot-message {
    background: rgba(241, 243, 245, 0.95);
    color: #333;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px auto 8px 0;
    max-width: 80%;
    width: fit-content;
    display: block;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    font-size: 14px;
    line-height: 1.4;
    border: 1px solid #e9ecef;
}

/* Simple title styling */
.chat-title {
    text-align: center;
    font-size: 32px;
    font-weight: 600;
    color: #333;
    margin: 20px 0 10px 0;
}

.chat-subtitle {
    text-align: center;
    font-size: 16px;
    color: #666;
    margin: 0 0 30px 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.markdown('<h1 class="chat-title">🤖 PDF Q&A Assistant</h1>', unsafe_allow_html=True)
#st.markdown('<p class="chat-subtitle">Ask me anything about your PDF document</p>', unsafe_allow_html=True)


# -------------------------------------------------------------
# SESSION STATE TO STORE CHAT MESSAGES
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]


# -------------------------------------------------------------
# CHAT DISPLAY AREA
# -------------------------------------------------------------
# Show welcome message if no messages
if not st.session_state.messages:
    st.markdown("""
    <div class='bot-message'>
        👋 Hello! I'm your PDF assistant. I can help you find information from your uploaded PDF document. 
        Just ask me a question and I'll do my best to provide you with accurate answers!
    </div>
    """, unsafe_allow_html=True)

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-message'>{msg['content']}</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# CHAT INPUT
# -------------------------------------------------------------
user_input = st.chat_input("💬 Type your question here...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show typing indicator
    with st.spinner("🤔 Thinking..."):
        try:
            chain = build_chain()
            result = chain.invoke({"input": user_input})
            answer = result["answer"]
            
            # Add bot message
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Sorry, I encountered an error: {str(e)}. Please try again."
            })
    
    st.rerun()
