
import os
import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import openai
import hmac

# Set up your OpenAI API key securely
openai_api_key = st.secrets["OpenAI_key"]

# Set the API key for OpenAI
openai.api_key = openai_api_key

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()


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
                {"role": "system", "content": "Provide answer to the query related to RBI circulars for the following user question. Double check the info to include info from the latest circular. Start with regulations in the ongoing year, if nothing has been issued, then try documents from previous years and so on. Once answered, provide a list of paragraphs and all circular links referred in the end."},
                {"role": "user", "content": query}
            ]
        )
        # Display the answer
        answer = response.choices[0].message['content']
        st.write("Answer:")
        st.write(answer)
        
        # Display the sources with clickable links
        st.write("Sources:")
        for source in set(retrieved_sources):  # Use set to avoid duplicate sources
            # Generate the URL path based on the source filenames
            file_path = os.path.join('docs', source)
            url_path = f"./{file_path}"
            link = f"[{source}]({url_path})"
            st.markdown(link)
    except Exception as e:
        st.error(f"An error occurred: {e}")

