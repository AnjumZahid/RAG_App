# Step 1: Set Up the User Interface with Streamlit
# -----------------------------------------------
# Install Streamlit: pip install streamlit
import streamlit as st

# Configure the page settings and set up the title
st.set_page_config(page_title="PDF QA System with LangGraph")
st.title("PDF QA System with LangGraph")

# Add a reset button in the sidebar to clear session state
if st.sidebar.button("Reset Chat"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# Initialize session state variable to track if the processing is complete
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

# Sidebar file uploader for PDF input
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type='pdf')


# Step 2: Process the PDF File
# -----------------------------------------------
# Install LangChain community packages: pip install langchain_community
from langchain_community.document_loaders import PyPDFLoader
# Install text_splitter: pip install langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Install langchain_core for Document type: pip install langchain_core
from langchain_core.documents import Document

# Function to process the uploaded PDF file
def process_pdf(pdf_file):
    loaders = PyPDFLoader(pdf_file)  # Initialize the loader
    pages = loaders.load()  # Load the content of the PDF

    # Define a text splitter to break down content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    doc_list = []
    
    # Split each page and add to document list
    for page in pages:
        pg_split = text_splitter.split_text(page.page_content)
        for pg_sub_split in pg_split:
            metadata = {"source": "Uploaded PDF"}
            doc_string = Document(page_content=pg_sub_split, metadata=metadata)
            doc_list.append(doc_string)
    
    return doc_list  # Return the processed document list


# Step 3: Setup Document Embeddings and Vector Store
# -----------------------------------------------
# Install the required embeddings package: pip install langchain_community
from langchain_community.embeddings import HuggingFaceEmbeddings
# Install Qdrant vector store integration: pip install langchain_qdrant
from langchain_qdrant import QdrantVectorStore

# Initialize the embedding model
embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

if uploaded_file:
    # Temporarily save the uploaded file
    with open("uploaded_temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Process the PDF
    doc_list = process_pdf("uploaded_temp.pdf")
    
    # Initialize Qdrant vector store for storing document embeddings
    vectorstore = QdrantVectorStore.from_documents(
        doc_list,
        embed_model,
        url=st.secrets["QDRANT_URL"],         # Qdrant URL from secrets
        api_key=st.secrets["QDRANT_API_KEY"], # API key from secrets
        collection_name="Urdu_doc",
        prefer_grpc=True,
        force_recreate=True
    )
    st.session_state.processComplete = True  # Update session state


# Step 4: Initialize LangGraph
# -----------------------------------------------
# Install LangGraph and typing extensions: pip install langgraph typing_extensions
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

# Define a custom TypedDict for managing the graph's state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the state graph
graph_builder = StateGraph(State)


# Step 5: Initialize Language Model (Google Gemini)
# -----------------------------------------------
# Install LangChain's Google Gemini module: pip install langchain_google_genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the language model with Google Gemini
llm = ChatGoogleGenerativeAI(
    api_key=st.secrets["GEMINI_API_KEY"],  # Google Gemini API key from secrets
    model="gemini-1.5-flash",
)


# Step 6: Define Chatbot Functionality
# -----------------------------------------------
# This function manages the chatbot's conversation flow and retrieves relevant context from the vector store
def chatbot(state: State):
    # Get the latest user message content
    user_question = state["messages"][-1].content

    # Retrieve relevant documents from the vector store based on the question
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_chunks = retriever.get_relevant_documents(user_question)

    # Build context by combining content from retrieved documents
    context = "\n".join([doc.page_content for doc in retrieved_chunks])

    # Formulate the prompt with the relevant context for the language model
    prompt_str = f"""
    Answer the user question based only on the following context:
    {context}

    Question: {user_question}
    """
    
    # Invoke the language model and return its response
    response = llm.invoke(prompt_str)
    return {"messages": [response]}


# Step 7: Configure Graph Workflow
# -----------------------------------------------
# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Define the start and end of the chatbot workflow
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph for execution
graph = graph_builder.compile()




# Step 8: Handle User Input and Display Responses

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------------------------
# Accept user input and interact with the chatbot if PDF processing is complete
if st.session_state.processComplete:
    user_question = st.chat_input("Ask a question about the PDF:")

    if user_question:
        # Append the user question to the chat history
        st.session_state.chat_history.append(("user", user_question))

        # Process the input with the graph and retrieve responses
        for event in graph.stream({"messages": [("user", user_question)]}):
            for value in event.values():
                response = value["messages"][-1].content  # Get the assistant's response
                # Append the assistant response to the chat history
                st.session_state.chat_history.append(("assistant", response))

    # Layout for displaying chat history in the app
    response_container = st.container()
    with response_container:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for sender, message in st.session_state.chat_history:
            if sender == "user":
                # Display user messages with formatting
                st.markdown(f"<div class='user-bubble' style='background-color: #d1e7dd; color: #0f5132; padding: 10px; border-radius: 10px; margin: 5px; text-align: right;'><strong>You:</strong> {message}</div>", unsafe_allow_html=True)
            else:
                # Display assistant messages with formatting
                st.markdown(f"<div class='bot-bubble' style='background-color: #f8d7da; color: #842029; padding: 10px; border-radius: 10px; margin: 5px; text-align: left;'><strong>Bot:</strong> {message}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# Step 9: Build the Main App Interface
# -----------------------------------------------
# Final function to initialize the interface and sidebar instructions
def main():
    st.sidebar.button("Upload PDF and ask questions!")

if __name__ == "__main__":
    main()

