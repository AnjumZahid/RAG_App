# Step 1: Set Up the User Interface with Streamlit
# Installation: Use "pip install streamlit" to install Streamlit if you haven't already.
import streamlit as st

# Set page configuration and title
st.set_page_config(page_title="PDF QA System")
st.title("PDF QA System")

# Add a reset button to clear session state
if st.sidebar.button("Reset Chat"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# Manage session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type='pdf')

# Step 2: Create a Chat Interface with streamlit_chat
# Installation: Use "pip install streamlit-chat" to install this library.
from streamlit_chat import message  # Import only at this step
from deep_translator import GoogleTranslator

# Function to handle user input and display response
def handle_userinput(user_question):
    # Translate user question from Urdu to English for processing
    translator = GoogleTranslator(source='auto', target='en')
    user_question_in_english = translator.translate(user_question)
    
    with st.spinner('Generating response...'):
        # Invoke the conversation chain and get the result in English
        result = st.session_state.conversation.invoke({"question": user_question_in_english})
        
        # Ensure result has content
        response_in_english = result.content if hasattr(result, 'content') else "Sorry, I couldn't retrieve a proper response."
        
        # Translate the response back to Urdu
        translator_urdu = GoogleTranslator(source='en', target='ur')
        response_in_urdu = translator_urdu.translate(response_in_english)

        # Append user question and translated Urdu response to chat history
        st.session_state.chat_history.append(f"You: {user_question}")
        st.session_state.chat_history.append(f"Bot: {response_in_urdu}")

    # Display each message in the chat history in the response container
    response_container = st.container()
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages, is_user=True, key=str(i))
            else:
                message(messages, key=str(i))


# Step 3: Process PDF Files
# Installation: Use "pip install pypdf langchain-core langchain-community" to install these libraries for PDF processing.
import os  # Import only at this step
from langchain_core.documents import Document  # Import only at this step
from langchain_community.document_loaders import PyPDFLoader  # Import only at this step
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import only at this step


# Function to process the uploaded PDF file
def process_pdf(pdf_file):
    loaders = PyPDFLoader(pdf_file)
    pages = loaders.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    doc_list = []
    translator = GoogleTranslator(source='auto', target='en')
    
    for page in pages:
        # Translate each page content to English for processing in the QA system
        translated_page_content = translator.translate(page.page_content)
        pg_split = text_splitter.split_text(translated_page_content)
        for pg_sub_split in pg_split:
            metadata = {"source": "Uploaded PDF"}
            doc_string = Document(page_content=pg_sub_split, metadata=metadata)
            doc_list.append(doc_string)
    
    return doc_list

# Step 4: Convert Text into Embeddings for Vector Search
# Installation: Use "pip install langchain-community sentence-transformers" to get HuggingFaceEmbeddings.
from langchain_community.embeddings import HuggingFaceEmbeddings  # Import only at this step


# Initialize embedding model with BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')


# Step 5: Store Document Embeddings in Qdrant
# Installation: Use "pip install qdrant-client langchain-qdrant" for Qdrant support.
from qdrant_client import QdrantClient  # Import only at this step
from langchain_qdrant import QdrantVectorStore  # Import only at this step

# Initialize Qdrant vector store
if uploaded_file:
    # Save the uploaded file temporarily
    with open("uploaded_temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Process the PDF
    doc_list = process_pdf("uploaded_temp.pdf")
    
    vectorstore = QdrantVectorStore.from_documents(
        doc_list,
        embed_model,
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
        collection_name="Urdu_doc",
        prefer_grpc=True,
        force_recreate=True
    )
    st.session_state.processComplete = True


# Step 6: Generate Answers with Google Gemini Model
# Installation: Use "pip install langchain-google-genai" for Google Gemini Model and "pip install langchain-core" for LangChain components.
from langchain_google_genai import ChatGoogleGenerativeAI  # Import only at this step
from langchain_core.prompts import ChatPromptTemplate  # Import only at this step
from langchain_core.runnables import RunnableSequence, RunnableMap  # Import only at this step
from operator import itemgetter  # Import only at this step

# Function to define the retrieval and answer generation pipeline
def get_qa_chain(vectorstore, num_chunks):
    # Define the retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
    
    # Define the prompt template
    prompt_str = """
    Answer the user question based only on the following context:
    {context}

    Question: {question}
    """
    _prompt = ChatPromptTemplate.from_template(prompt_str)
    
    # Initialize the Gemini Chat model
    chat_llm = ChatGoogleGenerativeAI(
        api_key=st.secrets["GEMINI_API_KEY"],
        model="gemini-1.5-flash",
        temperature=0
    )
    
    # Separate steps for RunnableSequence
    query_fetcher = itemgetter("question")
    retrieval_pipeline = query_fetcher | retriever
    
    # Create a RunnableMap to split context and question handling
    setup_pipeline = RunnableMap({"question": query_fetcher, "context": retrieval_pipeline})
    
    # Chain together the steps in a sequence without using a list
    qa_chain = setup_pipeline | _prompt | chat_llm
    
    return qa_chain

# Step 7: Handle User Input and Display Responses
if st.session_state.processComplete:
    num_chunks = 3
    st.session_state.conversation = get_qa_chain(vectorstore, num_chunks)
    
    user_question = st.chat_input("Ask a question about the PDF:")
    if user_question:
        handle_userinput(user_question)

# Step 8: Build the Main App
def main():
    # All setup is done here, so we simply run the app
    st.sidebar.button("Upload PDF and ask questions!")

if __name__ == "__main__":
    main()
