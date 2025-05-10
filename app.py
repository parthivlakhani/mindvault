import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
import bcrypt
from pymongo import MongoClient
import json
import requests
from datetime import datetime, UTC
import re

# Load environment variables
load_dotenv()

# Set CUDA visibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# MongoDB setup
client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
db = client['pdf_chatbot']
users_collection = db['users']
subjects_collection = db['subjects']
chat_history_collection = db['chat_history']

# OpenRouter API setup
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def init_session_state():
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'current_subject' not in st.session_state:
        st.session_state.current_subject = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def login_user(username, password):
    user = users_collection.find_one({'username': username})
    if user and verify_password(password, user['password']):
        st.session_state.user = {'username': username, 'is_guest': False}
        return True
    return False

def register_user(username, password, confirm_password):
    if password != confirm_password:
        return False, "Passwords do not match"
    if users_collection.find_one({'username': username}):
        return False, "Username already exists"
    hashed_password = hash_password(password)
    users_collection.insert_one({
        'username': username,
        'password': hashed_password,
        'created_at': datetime.now(UTC)
    })
    return True, "Registration successful"

def guest_login():
    st.session_state.user = {'username': f'guest_{datetime.now().timestamp()}', 'is_guest': True}

def get_user_subjects():
    if st.session_state.user['is_guest']:
        return list(subjects_collection.find({'username': st.session_state.user['username']}))
    return list(subjects_collection.find({'username': st.session_state.user['username']}))

def add_subject(subject_name):
    if not subject_name.strip():
        return False, "Subject name cannot be empty"
    subjects_collection.insert_one({
        'username': st.session_state.user['username'],
        'name': subject_name,
        'created_at': datetime.now(UTC)
    })
    return True, "Subject added successfully"

def delete_subject(subject_id):
    subjects_collection.delete_one({'_id': subject_id})

def save_chat_message(subject_id, message, response):
    chat_history_collection.insert_one({
        'username': st.session_state.user['username'],
        'subject_id': subject_id,
        'message': message,
        'response': response,
        'timestamp': datetime.now(UTC)
    })

def get_chat_history(subject_id):
    return list(chat_history_collection.find({
        'username': st.session_state.user['username'],
        'subject_id': subject_id
    }).sort('timestamp', 1))

def get_openrouter_response(prompt, subject_name):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "nousresearch/deephermes-3-mistral-24b-preview:free",
        "messages": [
            {"role": "system", "content": f"You are a helpful assistant specialized in {subject_name}."},
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
    try:
        result = response.json()
        if 'choices' in result and result['choices']:
            return result['choices'][0]['message']['content']
        else:
            st.error(f"OpenRouter API error: {result}")
            return f"OpenRouter API error: {result}"
    except Exception as e:
        st.error(f"Failed to parse OpenRouter response: {e}")
        return f"Failed to parse OpenRouter response: {e}"

def display_chat_message(message, is_user=True):
    if is_user:
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-end; margin: 8px 0;'>
            <div style='background: #3a86ff; color: #fff; padding: 12px 18px; border-radius: 18px 18px 4px 18px; max-width: 70%; box-shadow: 0 2px 8px rgba(58,134,255,0.08); font-size: 1.05rem;'>
                <span style='font-weight: 500;'>You</span><br>{message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-start; margin: 8px 0;'>
            <div style='background: #23272f; color: #e0e0e0; padding: 12px 18px; border-radius: 18px 18px 18px 4px; max-width: 70%; box-shadow: 0 2px 8px rgba(35,39,47,0.08); font-size: 1.05rem;'>
                <span style='font-weight: 500;'>Assistant</span><br>{message}
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    print("App started.")
    init_session_state()
    
    if not st.session_state.user:
        st.title("Welcome to MindVault")
        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Guest Login"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                if submit:
                    print(f"Login attempt for user: {username}")
                    if login_user(username, password):
                        print("Login successful.")
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        print("Login failed.")
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                signup = st.form_submit_button("Sign Up")
                if signup:
                    print(f"Signup attempt for user: {new_username}")
                    success, message = register_user(new_username, new_password, confirm_password)
                    if success:
                        print("Registration successful.")
                        st.success(message)
                    else:
                        print(f"Registration failed: {message}")
                        st.error(message)
        
        with tab3:
            if st.button("Continue as Guest"):
                print("Guest login.")
                guest_login()
                st.rerun()
    
    else:
        print(f"User session: {st.session_state.user['username']}")
        st.sidebar.title(f"Welcome, {st.session_state.user['username']}")
        
        # Subject Management
        st.sidebar.header("Subjects")
        with st.sidebar.form("add_subject"):
            new_subject = st.text_input("Add New Subject")
            if st.form_submit_button("Add"):
                print(f"Adding subject: {new_subject}")
                success, message = add_subject(new_subject)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        # Display existing subjects
        subjects = get_user_subjects()
        for subject in subjects:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(subject['name'], key=f"subject_{subject['_id']}"):
                    print(f"Selected subject: {subject['name']}")
                    st.session_state.current_subject = subject
                    st.session_state.chat_history = get_chat_history(subject['_id'])
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{subject['_id']}"):
                    print(f"Deleting subject: {subject['name']}")
                    delete_subject(subject['_id'])
                    st.rerun()
        
        # Main content area
        if st.session_state.current_subject:
            print(f"Current subject: {st.session_state.current_subject['name']}")
            st.header(f"Chat about {st.session_state.current_subject['name']}")
            
            # Display chat history
            for chat in st.session_state.chat_history:
                display_chat_message(chat['message'], True)
                display_chat_message(chat['response'], False)
            
            # PDF Upload
            pdfs = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
            if pdfs:
                print(f"PDFs uploaded: {[pdf.name for pdf in pdfs]}")
                texts = []
                for pdf in pdfs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        texts.append(page.extract_text())
                print("Extracted text from all PDFs.")
                
                # Store in Redis using HuggingFace embeddings
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                print("Created embeddings.")
                VectorStore = Redis.from_texts(texts, embeddings, redis_url="redis://localhost:6379")
                print("Stored texts in Redis.")
            
            # Chat interface
            query = st.text_input("Ask questions about your PDFs")
            if query:
                print(f"User query: {query}")
                display_chat_message(query, True)
                
                results = VectorStore.similarity_search(query=query, k=3)
                if results:
                    # Concatenate top-k relevant chunks as context
                    context_chunks = [r.page_content for r in results if r.page_content]
                    context = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
                    print("Found relevant content in PDFs. Constructing RAG prompt.")
                    prompt = (
                        "You are a helpful assistant. Use the following context from documents to answer the question.\n\n"
                        f"Context from docs:\n{context}\n\nQuestion: {query}"
                    )
                    print("Calling OpenRouter LLM with RAG prompt.")
                    response = get_openrouter_response(prompt, st.session_state.current_subject['name'])
                else:
                    print("No relevant results in PDFs, calling LLM directly.")
                    response = get_openrouter_response(query, st.session_state.current_subject['name'])
                
                display_chat_message(response, False)
                save_chat_message(st.session_state.current_subject['_id'], query, response)
                st.session_state.chat_history = get_chat_history(st.session_state.current_subject['_id'])
        
        if st.sidebar.button("Logout"):
            print("User logged out.")
            st.session_state.user = None
            st.session_state.current_subject = None
            st.session_state.chat_history = []
            st.rerun()

if __name__ == '__main__':
    main()
