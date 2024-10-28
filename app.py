import os
import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import openai
import fitz  # Correct import for PyMuPDF
from datetime import datetime
import hmac
import hashlib
from streamlit_pdf_viewer import pdf_viewer
import re
import Levenshtein as lev




# File paths for user data and activity logs
USER_DATA_FILE = "user_data.json"
ACTIVITY_LOG_FILE = "activity_log.txt"
RLHF_DATA_FILE = "rlhf_data.txt"
SECRET_KEY = "supersecretkey"  # Replace this with a proper secret key

# Set up your OpenAI API key securely
openai_api_key = st.secrets["OpenAI_key"]
openai.api_key = openai_api_key

# Define the base directory for your PDF files
base_dir = "ALL_PDFs"

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


#Clean Circular Numbers
def clean_code(code):
    if isinstance(code, float):  # Handle float values which might be NaN
        code = ""
    # Ensure the code is a string and remove newlines and leading/trailing whitespace
    code = str(code).replace('\n', '').strip()
    # Optional: remove all non-alphanumeric characters except slashes and periods
    code = re.sub(r'[^a-zA-Z0-9/.]', '', code)
    return code


#Find Best Match for Code
def find_best_match(input_code, all_codes):
    # Clean the input code
    input_code = clean_code(input_code)
    # Calculate Levenshtein distance to each code, return the code with the smallest distance
    best_match, min_distance = None, float('inf')
    for code in all_codes:
        current_distance = lev.distance(input_code, clean_code(code))
        if current_distance < min_distance:
            min_distance = current_distance
            best_match = code
    return best_match

# Log user activity with a timestamp
def log_activity(message, query=None, response=None, answer=None):
    """Log activity messages with a timestamp to a log file, including query, response, and answer."""
    with open(ACTIVITY_LOG_FILE, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
        if query:
            f.write(f"Query: {query}\n")
        if response:
            f.write(f"Response: {response}\n")
        if answer:
            f.write(f"Answer: {answer}\n")
        f.write("\n")


# Log RLHF data to a file
def log_rlhf_data(question, response, feedback):
    """Log question, response, and feedback data for RLHF."""
    with open(RLHF_DATA_FILE, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Question: {question}\n")
        f.write(f"[{timestamp}] Response: {response}\n")
        f.write(f"[{timestamp}] Feedback: {feedback}\n\n")

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
                    st.rerun()  # Use experimental_rerun here
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
                st.rerun()
# Cache document embeddings, texts, and sources
@st.cache_data
def load_embeddings_and_docs(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Filter out withdrawn circulars
    data = [doc for doc in data if doc.get('status', '').lower() != 'withdrawn']
    
    embeddings = np.array([np.array(doc['embedding']) for doc in data])
    documents = [doc.get('text', '') for doc in data]  # Use full text or substantial content as document content
    sources = [doc['pdf_filename'] for doc in data]  # Use PDF names as sources
    metadata = [
        {
            "title": doc.get("title", ""),
            "code": doc.get("code", ""),
            "department": doc.get("department", ""),
            "pdf_filename": doc['pdf_filename'],
            "link": doc.get("link", ""),
            "date_of_issue": doc.get("date_of_issue", ""),
            "status": doc.get("status", "")
        }
        for doc in data
    ]  # Extract metadata including circular number and department
    return embeddings, documents, sources, metadata



# Cache FAISS index setup
@st.cache_resource
def setup_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retrieve documents based on the query using FAISS
def retrieve_documents_by_query(query, index, model, embeddings, documents, sources, metadata, k=15):
    query_embedding = model.encode([query])[0]
    _, indices = index.search(np.array([query_embedding]), k)
    retrieved_docs = [documents[idx] for idx in indices.flatten()]
    retrieved_sources = [sources[idx] for idx in indices.flatten()]
    retrieved_metadata = [metadata[idx] for idx in indices.flatten()]
    return retrieved_docs, retrieved_sources, retrieved_metadata

# Highlight text in a PDF file
def highlight_text_in_pdf(pdf_path, passages):
    """Highlights passages in a PDF file."""
    doc = fitz.open(pdf_path)
    for page in doc:
        for passage in passages:
            text_instances = page.search_for(passage)
            for inst in text_instances:
                page.add_highlight_annot(inst)
    highlighted_pdf_path = pdf_path.replace(".pdf", "_highlighted.pdf")
    doc.save(highlighted_pdf_path, garbage=4, deflate=True)
    doc.close()
    return highlighted_pdf_path

# Create an iframe for PDF viewing
def create_pdf_iframe(pdf_path, height=500):
    """Creates an iframe to view a PDF document."""
    encoded_pdf = pdf_path.replace("docs/", "")
    pdf_url = f"https://raw.githubusercontent.com/your-github-username/your-repo/main/docs/{encoded_pdf}"
    iframe_code = f"""
        <iframe src="{pdf_url}" width="100%" height="{height}px" style="border:none;"></iframe>
    """
    return iframe_code

# Initialize the Sentence Transformer model for query encoding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load embeddings, document data, and sources
embeddings, documents, sources, metadata = load_embeddings_and_docs('ALL_PDFs/combined_embeddings_with_metadata.json')
faiss_index = setup_faiss_index(embeddings)
# Check authentication or registration status
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "register" not in st.session_state:
    st.session_state["register"] = False

# Initialize feedback status in session state
if "feedback_status" not in st.session_state:
    st.session_state["feedback_status"] = None

if "query_response" not in st.session_state:
    st.session_state["query_response"] = {"query": "", "response": "", "sources": [], "metadata": []}

if 'search_by_code' not in st.session_state:
    st.session_state['search_by_code'] = ''

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
            retrieved_docs, retrieved_sources, retrieved_metadata = retrieve_documents_by_query(query, faiss_index, model, embeddings, documents, sources, metadata, k=5)
            context = " ".join(retrieved_docs)  # Context from documents
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a trained and experienced compliance RBI Official. Provide detailed and comprehensive answer to the query about compliance basis all the RBI regulations issued till now. Once answered, provide crisp and very pointed action items for the regulated entities."},
                    {"role": "user", "content": f"Query: {query}\n\nContext: {context}"}
                ]
            )
            answer = response.choices[0].message['content']
            st.session_state["query_response"] = {"query": query, "response": answer, "sources": retrieved_sources, "metadata": retrieved_metadata}
            
            # Log the query, response, and answer
            log_activity(f"Query processed for user {st.session_state['current_user']}", query=query, response=context, answer=answer)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Button for searching by code with near match
    # Add text input for searching by circular code
    st.session_state['search_by_code'] = st.text_input("Search by Circular Code (e.g., RBI/2024-25/32 A. P. (DIR Series) Circular No. 04)")

    # Button for searching by code with near match
    if st.button("Search by Code"):
        try:
            input_code = st.session_state['search_by_code']  # Get the user input from session state
            all_codes = [meta.get('code', '') for meta in metadata]  # Ensure all codes are treated as strings
            best_match = find_best_match(input_code, all_codes)
            
            if best_match:
                result = next((meta for meta in metadata if clean_code(meta.get('code', '')) == clean_code(best_match)), None)
                if result:
                    st.write("**Title**: ", result.get("title", "N/A"))
                    st.write("**Code**: ", result.get("code", "N/A"))
                    st.write("**Department**: ", result.get("department", "N/A"))
                    st.write("**Date of Issue**: ", result.get("date_of_issue", "N/A"))

                    # Display the PDF if it exists
                    file_path = os.path.join(base_dir, result['pdf_filename'])
                    if os.path.exists(file_path):
                        pdf_viewer(input=file_path, height=500)
                    else:
                        st.write("File not found")
                else:
                    st.write("No close matches found.")
            else:
                st.write("No matches found.")

        except Exception as e:
            st.error(f"An error occurred: {e}")




    query_response = st.session_state["query_response"]


    if query_response["response"]:
        st.write("Answer:")
        st.write(query_response["response"])

        # Feedback buttons with icons side by side
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍", key="upvote"):
                log_rlhf_data(query_response["query"], query_response["response"], "upvote")
                st.session_state["feedback_status"] = "upvote"
        with col2:
            if st.button("👎", key="downvote"):
                log_rlhf_data(query_response["query"], query_response["response"], "downvote")
                st.session_state["feedback_status"] = "downvote"

        # Display a thank-you message based on the vote status
        if st.session_state["feedback_status"] == "upvote" or st.session_state["feedback_status"] == "downvote":
            st.success("Thank you for your feedback!")

        # Display the sources with clickable links that open in the PDF viewer
        st.write("Sources:")
        for i, (source, metadata) in enumerate(zip(query_response["sources"], query_response["metadata"])):
            file_path = os.path.join(base_dir, source)
            if os.path.exists(file_path):
                st.write(f"**Title**: {metadata.get('title', 'N/A')} \n")
                st.write(f"**Circular Number**: {metadata.get('code', 'N/A')} \n")
                st.write(f"**Addressed To**: {metadata.get('department', 'N/A')} \n")
                st.write(f"**Date of Issue**: {metadata.get('date_of_issue', 'N/A')}")
                pdf_viewer(input=file_path, height=500, key=f"pdf_viewer_{i}")  # Ensure unique key for each viewer
            else:
                st.write(f"File {source} not found")
        

    # Application content (logout button)
    if st.button("Logout"):
        log_activity(f"User {st.session_state['current_user']} logged out")
        st.session_state["authenticated"] = False
        st.session_state["register"] = False
        st.session_state["feedback_status"] = None
        st.session_state["query_response"] = {"query": "", "response": "", "sources": [], "metadata": []}
        del st.session_state["current_user"]
        st.rerun()


