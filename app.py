import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader

# Step 1: Frontend Styling
# Define Streamlit markdown CSS for UI customization
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #000000; }
    .stChatInput input { background-color: #F0F0F0 !important; color: #000000 !important; border: 1px solid #B0B0B0 !important; margin-top: 20px; padding: 10px; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background-color: #F0F0F0 !important; border: 1px solid #B0B0B0 !important; color: #000000 !important; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) { background-color: #E0E0E0 !important; border: 1px solid #B0B0B0 !important; color: #000000 !important; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stFileUploader { background-color: #F0F0F0; border: 1px solid #B0B0B0; border-radius: 5px; padding: 15px; }
    h1, h2, h3 { color: #000000 !important; }
    </style>
""", unsafe_allow_html=True)

# Step 2: Define Constants and Models
# Define prompt template for generating responses
PROMPT_TEMPLATE = """
You are a skilled assistant. Refer to the provided context to respond to the query.
If uncertain, acknowledge your lack of knowledge. 
Keep your answers brief and factual, using no more than fifty words.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Define embedding model, vector store, and language model
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")
PDF_STORAGE_PATH = 'pdf_doc/'

# Step 3: Utility Functions
# Function to save uploaded PDF file locally
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# Function to load PDF documents using PDFPlumberLoader
def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

# Function to split documents into chunks for processing
def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

# Function to index document chunks into the vector database
def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

# Function to search for relevant documents based on query
def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

# Function to generate AI response based on context documents
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# Step 4: Layout Configuration
# Define Streamlit page title and layout
st.title("RAG App with DeepSeek-R1")
st.markdown("---")

# Step 5: Sidebar for PDF Upload
with st.sidebar:
    st.markdown("### Upload Document (PDF)")
    uploaded_pdf = st.file_uploader(
        "Select a PDF document", type="pdf", accept_multiple_files=False
    )

# Step 6: Document Processing
if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    st.success("Document processed successfully!")
    
    # Step 7: Query Processing
    user_input = st.text_input("Ask your question ...")
    
    if user_input:
        # Display user query
        with st.markdown(f"""
            <div class="query-container">
                <div class="query-box"><strong>You:</strong> {user_input}</div>
            </div>
        """, unsafe_allow_html=True):
            pass
        
        # Process query and find related documents
        with st.spinner("Reviewing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
        
        # Display AI response
        with st.markdown(f"""
            <div class="response-container">
                <div class="response-box"><strong>Bot:</strong> {ai_response}</div>
            </div>
        """, unsafe_allow_html=True):
            pass
