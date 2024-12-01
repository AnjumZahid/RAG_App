import streamlit as st
import os
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Class Definition
class SimpleQAChain:
    def __init__(self, model):
        self.model = model

    def QAchain(self, query, template, stream_callback=None):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.model | StrOutputParser()

            if stream_callback:
                for partial_response in chain.stream({"question": query}):
                    stream_callback(partial_response)

            return chain.invoke({"question": query})
        except Exception as e:
            return f"Error executing chain: {str(e)}"

    def conversational_chain(self, query, history, template, stream_callback=None):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.model | StrOutputParser()
            input_dict = {"HISTORY": history, "QUESTION": query}

            if stream_callback:
                for partial_response in chain.stream(input_dict):
                    stream_callback(partial_response)

            return chain.invoke(input_dict)
        except Exception as e:
            return f"Error executing conversational chain: {str(e)}"

    def QA_Retrieval(self, query, template, vector_store, k, stream_callback=None):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
            _filter = LLMChainFilter.from_llm(self.model)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=retriever
            )

            setup_and_retrieval = RunnableParallel(
                {
                    "CONTEXT": compression_retriever,
                    "question": RunnablePassthrough(),
                }
            )
            rag_chain = setup_and_retrieval | prompt | self.model | StrOutputParser()

            if stream_callback:
                for partial_response in rag_chain.stream(query):
                    stream_callback(partial_response)

            return rag_chain.invoke(query)
        except Exception as e:
            return f"Error executing retrieval chain: {str(e)}"
        
    # Conversational Retrieval method
    def Conversational_Retrieval(self, query, history, template, vector_store, k, stream_callback=None):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
            _filter = LLMChainFilter.from_llm(self.model)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=retriever
            )

            setup_and_retrieval = RunnableParallel(
                {
                    "CONTEXT": compression_retriever,
                    "QUESTION": RunnablePassthrough(),
                    "HISTORY": RunnablePassthrough(),
                }
            )
            output_parser = StrOutputParser()
            rag_chain = (
                setup_and_retrieval
                | prompt
                | self.model
                | output_parser
            )

            input_dict = {"QUESTION": query, "HISTORY": history}
            
            # If streaming is enabled, use the callback function to stream output
            if stream_callback:
                for partial_response in rag_chain.stream(str(input_dict)):
                    stream_callback(partial_response)
            else:
                return rag_chain.invoke(str(input_dict))
        except Exception as e:
            return f"Error executing conversational retrieval chain: {str(e)}"
        


# Streamlit Interface with updated layout

st.set_page_config(page_title="QA System with Retrieval", layout="wide")

# Display the title with different sizes and move it up
st.markdown(
    """
    <h1 style="text-align: center; margin-top: -50px;">LCEL (LangChain Expression Language)</h1>
    <h3 style="text-align: center; margin-top: -20px;">QA System with Simple, Conversational, and Retrieval Modes</h3>
    """,
    unsafe_allow_html=True
)

# Add CSS for styling
st.markdown(
    """
    <style>
    .query-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
    }
    .response-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 10px;
    }
    .query-box {
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 5px;
        background-color: #f0f8ff; /* Light blue */
        max-width: 70%;
        text-align: right;
    }
    .response-box {
        padding: 10px;
        border-radius: 8px;
        margin-top: 5px;
        background-color: #f5f5f5; /* Light gray */
        max-width: 70%;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize embeddings, model, and QA Chain
@st.cache_resource
def initialize_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=url,
        api_key=api_key,
        collection_name="Urdu_doc"
    )

@st.cache_resource
def initialize_model():
    return ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-1.5-flash",
        temperature=0.5,
        streaming=True
    )

@st.cache_resource
def initialize_qa_chain():
    llm = initialize_model()
    return SimpleQAChain(model=llm)

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mode" not in st.session_state:
    st.session_state.mode = "Simple QA"

# Sidebar for mode selection and additional parameters
with st.sidebar:
    st.subheader("Select Mode")
    st.session_state.mode = st.radio("Choose your mode:", ["Simple QA", "Conversational", "Retrieval", "Conversational_Retrieval"])

    # Check if the mode has changed, if so, reset the chat history
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = st.session_state.mode  # Initialize last_mode

    # If the mode has changed, reset the chat history
    if st.session_state.mode != st.session_state.last_mode:
        st.session_state.chat_history = []  # Clear the chat history when switching modes
        st.session_state.last_mode = st.session_state.mode  # Update last_mode to the current mode

    # Number of documents to retrieve (k)
    if st.session_state.mode in ["Retrieval", "Conversational_Retrieval"]:
        st.subheader("Select Number of Documents (k)")
        k_value = st.slider("Number of Documents (k)", min_value=1, max_value=5, value=2)
    else:
        k_value = None

qa_chain = initialize_qa_chain()
vector_store = initialize_vectorstore()

# Templates
simple_qa_template = """You are a helpful assistant. Please answer the user's question: "{question}"."""
conversational_template = """
You are a helpful assistant. Answer based on the conversation history.
Previous conversation:
{HISTORY}
Current question: {QUESTION}
"""
retrieval_template = """You are an assistant. Use the following pieces of {CONTEXT} to generate an answer to the provided question.

question: {question}.

Helpful Answer:"""


def display_chat_history():
    """Display the current chat history with latest query and response first."""
    if st.session_state.chat_history:
        # Reverse the chat history to show latest first
        for user_query, bot_response in reversed(st.session_state.chat_history):
            # Query box (right-aligned)
            st.markdown(
                f"""
                <div class="query-container">
                    <div class="query-box"><strong>You:</strong> {user_query}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Response box (left-aligned)
            st.markdown(
                f"""
                <div class="response-container">
                    <div class="response-box"><strong>Bot:</strong> {bot_response}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )



def handle_simple_qa(user_query):
    """Handle simple QA."""
    # Add the user query to the chat history with a placeholder for the bot response
    st.session_state.chat_history.append((user_query, ""))  # Add query with empty response initially

    # Create a container for dynamically updating the bot's response
    response_container = st.empty()
    streamed_response = ""

    # Function to update the response dynamically as it is streamed
    def update_streamed_output(partial_response):
        nonlocal streamed_response
        streamed_response += partial_response

        # Update the chat history with the latest response (not duplicating)
        st.session_state.chat_history[-1] = (user_query, streamed_response)  # Update last response

        # Dynamically update the response container with the latest streamed response
        response_container.markdown(
            f"""
            <div class="response-container">
                <div class="response-box"><strong>Bot:</strong> {streamed_response}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Stream the response from the QA chain
    with st.spinner("Generating response..."):
        try:
            qa_chain.QAchain(
                query=user_query,
                template=simple_qa_template,
                stream_callback=update_streamed_output
            )
        except Exception as e:
            st.error(f"Error during response streaming: {str(e)}")

    response_container.empty()  # Clear the container once streaming is done
    display_chat_history()  # Refresh the chat display

   
def handle_conversational(user_query, history):
    """Handle conversational QA."""
    # Add the user query to the chat history with a placeholder for the bot response
    st.session_state.chat_history.append((user_query, ""))

    # Create a container for dynamically updating the bot's response
    response_container = st.empty()
    streamed_response = ""

    # Function to update the response dynamically as it is streamed
    def update_streamed_output(partial_response):
        nonlocal streamed_response
        streamed_response += partial_response

        # Update the chat history with the latest response (not duplicating)
        st.session_state.chat_history[-1] = (user_query, streamed_response)  # Update last response

        # Dynamically update the response container with the latest streamed response
        response_container.markdown(
            f"""
            <div class="response-container">
                <div class="response-box"><strong>Bot:</strong> {streamed_response}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Stream the response from the conversational QA chain
    with st.spinner("Generating response..."):
        try:
            qa_chain.conversational_chain(
                query=user_query,
                history=history,
                template=conversational_template,
                stream_callback=update_streamed_output
            )
        except Exception as e:
            st.error(f"Error during response streaming: {str(e)}")

    response_container.empty()  # Clear the container once streaming is done
    display_chat_history()  # Refresh the chat display

# Modify the handle_retrieval function to take the vector_store and k value
def handle_retrieval(user_query, vector_store, k):
    """Handle QA with retrieval."""
    # Add the user query to the chat history with a placeholder for the bot response
    st.session_state.chat_history.append((user_query, ""))

    # Create a container for dynamically updating the bot's response
    response_container = st.empty()
    streamed_response = ""

    # Function to update the response dynamically as it is streamed
    def update_streamed_output(partial_response):
        nonlocal streamed_response
        streamed_response += partial_response

        # Update the chat history with the latest response (not duplicating)
        st.session_state.chat_history[-1] = (user_query, streamed_response)  # Update last response

        # Dynamically update the response container with the latest streamed response
        response_container.markdown(
            f"""
            <div class="response-container">
                <div class="response-box"><strong>Bot:</strong> {streamed_response}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Stream the response from the QA retrieval chain
    with st.spinner("Generating response..."):
        try:
            qa_chain.QA_Retrieval(
                query=user_query,
                template=retrieval_template,
                vector_store=vector_store,
                k=k,
                stream_callback=update_streamed_output
            )
        except Exception as e:
            st.error(f"Error during response streaming: {str(e)}")

    response_container.empty()  # Clear the container once streaming is done
    display_chat_history()  # Refresh the chat display

def handle_conversational_retrieval(user_query, history, vector_store, k):
    """Handle conversational QA with retrieval."""
    # Add the user query to the chat history with a placeholder for the bot response
    st.session_state.chat_history.append((user_query, ""))

    # Create a container for dynamically updating the bot's response
    response_container = st.empty()
    streamed_response = ""

    # Function to update the response dynamically as it is streamed
    def update_streamed_output(partial_response):
        nonlocal streamed_response
        streamed_response += partial_response

        # Update the chat history with the latest response (not duplicating)
        st.session_state.chat_history[-1] = (user_query, streamed_response)  # Update last response

        # Dynamically update the response container with the latest streamed response
        response_container.markdown(
            f"""
            <div class="response-container">
                <div class="response-box"><strong>Bot:</strong> {streamed_response}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Stream the response from the conversational QA with retrieval chain
    with st.spinner("Generating response..."):
        try:
            qa_chain.Conversational_Retrieval(
                query=user_query,
                history=history,
                template=conversational_template,
                vector_store=vector_store,
                k=k,
                stream_callback=update_streamed_output
            )
        except Exception as e:
            st.error(f"Error during response streaming: {str(e)}")

    response_container.empty()  # Clear the container once streaming is done
    display_chat_history()  # Refresh the chat display


# Main content: Full-width query and response area
# st.subheader("Ask your question:")
user_input = st.text_input("Your Question", placeholder="Type your question here...")

if st.button("Submit"):
    if user_input:
        if st.session_state.mode == "Simple QA":
            handle_simple_qa(user_input)
        elif st.session_state.mode == "Conversational":
            handle_conversational(user_input, st.session_state.chat_history)
        elif st.session_state.mode == "Retrieval":
            handle_retrieval(user_input, vector_store, k_value)
        elif st.session_state.mode == "Conversational_Retrieval":
            handle_conversational_retrieval(user_input, st.session_state.chat_history, vector_store, k_value)
    else:
        st.warning("Please enter a question before submitting!")

