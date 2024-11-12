# Step 1: 
# Install Streamlit if not already installed:
# pip install streamlit

import streamlit as st


# Step 2: Define a TypedDict for the chatbot's state
# Install typing-extensions for TypedDict if not already installed:
# pip install typing-extensions

from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, "add_messages"]  # List of messages to store conversation

# Step 3: Initialize the StateGraph for managing conversation flow
# Install langgraph if not already installed:
# pip install langgraph

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Define the graph with the state schema
graph_builder = StateGraph(State)

# Step 4: Set up the Google Gemini language model for the chatbot
# Install langchain-google-genai if not already installed:
# pip install langchain-google-genai

from langchain_google_genai import ChatGoogleGenerativeAI

# Load the API key from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]  # Store your API key in `.streamlit/secrets.toml` with the format {"GEMINI_API_KEY": "YOUR_API_KEY"}

# Initialize the language model with your API key
llm = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-1.5-flash"
)

# Step 5: Define the chatbot function
# This function will handle user input, call the language model, and return the response

def chatbot(state: State):
    # Invoke the language model with the user's messages and return the response in updated state
    return {"messages": [llm.invoke(state["messages"])]}

# Step 6: Add the chatbot function as a node in the graph
# The chatbot node represents a single step where user input is processed by the model

graph_builder.add_node("chatbot", chatbot)

# Step 7: Define the flow of conversation by adding edges to the graph
# The flow starts at the START node, goes to chatbot node, and ends at the END node

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Step 8: Compile the graph for use
# Compilation of the graph makes it ready to handle conversational inputs

graph = graph_builder.compile()

# Step 9: Set up Streamlit page configuration and display the title
# This step sets the page title and heading that the user will see

st.set_page_config(page_title="QA System")
st.title("QA System")

# Step 10: Initialize chat history in Streamlit’s session state
# Chat history is stored to maintain context across user interactions

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Step 11: Define a function to handle user input and interact with the graph
# This function sends the user input to the model and displays the responses in real time

def handle_user_input(user_question):
    with st.spinner('Generating response...'):
        # Stream updates from the graph as it processes the input
        for event in graph.stream({"messages": [("user", user_question)]}):
            for value in event.values():
                # Get the last message from the response and update chat history
                response = value["messages"][-1].content
                st.session_state.chat_history.append(f"You: {user_question}")
                st.session_state.chat_history.append(f"Bot: {response}")

# Step 12: Capture user input through a text input field
# This input field lets the user type a question, which triggers the handle_user_input function

user_input = st.text_input("You:", "")
if user_input:
    handle_user_input(user_input)

# Step 13: Display the chat history in a formatted layout
# The conversation history is displayed in a color-coded style, distinguishing user and bot messages

response_container = st.container()
with response_container:
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # Style user messages in blue
            st.markdown(f"<div style='color:blue;'><b>{message}</b></div>", unsafe_allow_html=True)
        else:
            # Style bot responses in green
            st.markdown(f"<div style='color:green;'><b>{message}</b></div>", unsafe_allow_html=True)
