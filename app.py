import os
import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import openai
import hmac
import hashlib
from datetime import datetime
from streamlit_pdf_viewer import pdf_viewer

# File paths for user data and activity logs
USER_DATA_FILE = "user_data.json"
ACTIVITY_LOG_FILE = "activity_log.txt"
SECRET_KEY = "supersecretkey"  # Replace this with a proper secret key

# Set up your OpenAI API key securely
openai_api_key = st.secrets["OpenAI_key"]
openai.api_key = openai_api_key

# Define the base directory for your PDF files
base_dir = "docs"

# Load user data from a JSON file
def load_users():
    """Load existing users from a JSON file or create a new file if not exists."""
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'w') as f:
            json.dump({}, f)
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

# Save user data to a JSON file
def save_users(users):
    """Save users to a JSON file."""
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# Log user activity with a timestamp
def log_activity(message):
    """Log activity messages with a timestamp to a log file."""
    with open(ACTIVITY_LOG_FILE, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")

# Hash a password using SHA-256
def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(f"{password}{SECRET_KEY}".encode()).hexdigest()

# Check user credentials and provide login or registration options
def check_password():
    """Checks user credentials and provides login or registration options."""
    users = load_users()

    def login_and_register_form():
        """Form with widgets to collect user login information."""
        st.subheader("Login")
        with st.form("Login Form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
            if submitted:
                hashed_password = hash_password(password)
                if username in users and hmac.compare_digest(hashed_password, users[username]['password']):
                    st.session_state["authenticated"] = True
                    st.session_state["current_user"] = username
                    log_activity(f"User {username} logged in")
                    st.experimental_rerun()
                else:
                    st.error("Incorrect username or password.")

        st.subheader("New User?")
        st.write("Register to create a new account.")
        if st.button("Register"):
            st.session_state["register"] = True

    login_and_register_form()

# Displays a registration form for new users
def registration_form(users):
    """Displays a registration form for new users."""
    st.subheader("User Registration")
    with st.form("Register"):
        name = st.text_input("Name")
        username = st.text_input("Username")
        email = st.text_input("Email")
        contact_no = st.text_input("Contact No.")
        password = st.text_input("Password", type="password")
        organization = st.text_input("Organization")
        purpose = st.text_area("Purpose for using this tool")
        submit_button = st.form_submit_button("Register")

        if submit_button:
            if username in users:
                st.error("Username already exists. Please choose a different username.")
            else:
                users[username] = {
                    "name": name,
                    "password": hash_password(password),
                    "organization": organization,
                    "purpose": purpose
                }
                save_users(users)
                log_activity(f"User {username} registered")
                st.success("Registration successful! Redirecting to login page...")
                st.session_state["register"] = False
                st.experimental_rerun()

# Load document embeddings, texts, and sources from a JSON file
def load_embeddings_and_docs(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    embeddings = np.array([np.array(doc['embedding']) for doc in data])
    documents = [doc['document'] for doc in data]
    sources = [doc['document'] for doc in data]  # Use PDF names as sources
    return embeddings, documents, sources

# Set up a FAISS index for efficient document similarity search
def setup_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retrieve documents based on the query using FAISS
def retrieve_documents_by_query(query, index, model, embeddings, documents, sources, k=5):
    query_embedding = model.encode([query])[0]
    _, indices = index.search(np.array([query_embedding]), k)
    retrieved_docs = [documents[idx] for idx in indices.flatten()]
    retrieved_sources = [sources[idx] for idx in indices.flatten()]
    return retrieved_docs, retrieved_sources

# Initialize the Sentence Transformer model for query encoding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load embeddings, document data, and sources
embeddings, documents, sources = load_embeddings_and_docs('docs/merged_embeddings.json')
faiss_index = setup_faiss_index(embeddings)

# Check authentication or registration status
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "register" not in st.session_state:
    st.session_state["register"] = False

if not st.session_state["authenticated"]:
    if st.session_state["register"]:
        registration_form(load_users())
    else:
        check_password()
else:
    # Streamlit user interface for the application
    st.title("ComplAi Genie")
    st.markdown("""
    If you like ComplAi Genie, please show us some love by sharing your feedback ❤️: 
    [Feedback Form](https://forms.gle/8WmZqafnWGivfV4p9)
    """)
    query = st.text_input("Ask any question about RBI circulars.")

    if st.button("Submit"):
        try:
            retrieved_docs, retrieved_sources = retrieve_documents_by_query(query, faiss_index, model, embeddings, documents, sources, k=5)
            context = " ".join(retrieved_docs)  # Context from documents
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a trained and experienced compliance RBI Official. Provide an answer to the query related to compliance from all the RBI regulations issued till now. Once answered, provide a list of all circular and regulation links referred to at the end along with crisp and very pointed action items for the regulated entities that their staff should be able to implement."},
                    {"role": "user", "content": query}
                ]
            )
            answer = response.choices[0].message['content']
            st.write("Answer:")
            st.write(answer)

            # Display the sources with clickable links that open in the PDF viewer
            st.write("Sources:")
            for source in set(retrieved_sources):
                file_path = os.path.join(base_dir, source)
                if os.path.exists(file_path):
                    st.write(f"{source}:")
                    pdf_viewer(input=file_path, height=500)  # Adjust the height as needed
                else:
                    st.write(f"File {source} not found")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Application content (logout button)
    if st.button("Logout"):
        log_activity(f"User {st.session_state['current_user']} logged out")
        st.session_state["authenticated"] = False
        st.session_state["register"] = False
        del st.session_state["current_user"]
        st.experimental_rerun()
