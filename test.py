import streamlit as st
import os
# from dotenv import load_dotenv

# load_dotenv()

# # Load API keys from .env file
# groq_api_key = os.getenv('GROQ_API_KEY')
# tavily_api_key = os.getenv('TAVILY_API_KEY')
# if not groq_api_key:
#     raise ValueError("GROQ API key not found in .env file")

groq_api_key = st.secrets['GROQ_API_KEY']  
tavily_api_key = st.secrets['TAVILY_API_KEY']

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun

# Set up the page
st.set_page_config(layout="wide")
st.title("Chat with Web Search (Powered by Groq)")

# Initialize session state for visible messages and full conversation context
if "messages" not in st.session_state:
    st.session_state.messages = []

if "context" not in st.session_state:
    system_message = "You are a helpful assistant. Provide informative and concise responses based on your knowledge."
    st.session_state.context = [
        {"role": "system", "content": system_message}
    ]

# Get chatbot tools
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama3-8b-8192",  
        temperature=0.7, 
        streaming=True,
        api_key=groq_api_key
    )

@st.cache_resource
def get_search_tool():
    return DuckDuckGoSearchRun()

@st.cache_resource
def get_query_generator():
    return ChatGroq(
        model="llama3-70b-8192", 
        temperature=0.1,
        api_key=groq_api_key
    )

llm = get_llm()
search_tool = get_search_tool()
query_generator = get_query_generator()

# Create sidebar with model selection
with st.sidebar:
    st.header("Settings")
    
    # Add model selection dropdown
    model_option = st.selectbox(
        "Select Groq Model",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    
    # Update model if changed
    if model_option != getattr(llm, "model", "llama3-8b-8192"):
        llm = ChatGroq(
            model=model_option,
            temperature=0.7,
            streaming=True,
            api_key=groq_api_key
        )
        st.success(f"Model changed to {model_option}")
    
    use_search = st.checkbox("Enable web search", value=True)
    
    if st.button("Clear Chat"):
        # Reset both visible messages and context
        st.session_state.messages = []
        
        system_message = "You are a helpful assistant. Provide informative and concise responses based on your knowledge."
        st.session_state.context = [
            {"role": "system", "content": system_message}
        ]
        st.rerun()

# Display visible chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
user_input = st.chat_input("Ask anything...")

def generate_search_query(user_input):
    # If we have conversation history, use it to generate a better search query
    if len(st.session_state.messages) > 0:
        # Prepare the prompt for query generation
        conversation_history = ""
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_history += f"{role}: {msg['content']}\n\n"
            
        prompt = [
            SystemMessage(content="""You are a search query generator. 
            If any part if the user input is in inverted commas, you must search it as it is.
            Your task is to generate an effective search query based on the user's question and the conversation history.
            Generate a concise, specific query that will return the most relevant information.
            Return ONLY the search query text, nothing else.
            You MUST NOT include any urls or links like site:sap.com in the search query."""),
            HumanMessage(content=f"""Conversation history:
            {conversation_history}

            User's current question: {user_input}
            You MUST NOT include any urls or links like site:sap.com in the search query.
            Generate an optimal search query based on this context:""")
        ]
        
        # Get the generated query
        response = query_generator.invoke(prompt)
        generated_query = response.content.strip()
        
        # Store the generated query in session state
        st.session_state.generated_query = generated_query
        return generated_query
    else:
        # If no history, just use the user's input
        return user_input

def prepare_messages_for_llm():
    messages = []
    
    # Add system message
    messages.append(SystemMessage(content=st.session_state.context[0]["content"]))
    
    # Add the rest of the context (which contains both visible messages and any augmented queries)
    for message in st.session_state.context[1:]:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
            
    return messages

if user_input:
    # Add user message to visible chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Search for information if enabled
    search_results = None
    if use_search:
        with st.spinner("Generating search query and searching the web..."):
            # Generate a search query based on conversation history
            search_query = generate_search_query(user_input)
            
            # Perform the search with the generated query
            search_results = search_tool.invoke(search_query)
            
            # Show search query and results in expander
            with st.expander("Web Search Details", expanded=False):
                if hasattr(st.session_state, 'generated_query'):
                    st.write(f"**Generated search query:** {st.session_state.generated_query}")
                
                # Display the search results as text
                if search_results:
                    st.text_area("Search Results", search_results, height=200)
    
    # Update system message if we have search results
    if search_results:
        system_content = """You are a helpful assistant with access to real-time web search results. 
        When answering questions, you should:
        1. Use the web search results provided to give current, accurate information
        2. Refer to specific details from the search results when relevant
        3. Mention that your information comes from web searches when applicable
        4. Answer confidently based on the search results without disclaimers about not having real-time information

        Remember: You DO have access to current information through web searches that were just performed."""
        
        # Update system message in context
        st.session_state.context[0] = {"role": "system", "content": system_content}
    
    # Add user message to context (either original or augmented with search results)
    if search_results:
        search_query_info = ""
        if hasattr(st.session_state, 'generated_query'):
            search_query_info = f"Search query used: \"{st.session_state.generated_query}\"\n\n"
            
        augmented_query = f"""Web search results:
        {search_query_info}{search_results}

        Using the information from these search results, please answer the following question:
        {user_input}"""
        
        # Add the augmented query to the context (not visible chat)
        st.session_state.context.append({"role": "user", "content": augmented_query})
    else:
        # Just add the regular user input to context
        st.session_state.context.append({"role": "user", "content": user_input})
    
    # Prepare messages for LLM from full context
    messages_for_llm = prepare_messages_for_llm()
    
    # Display info about the model being used
    with st.expander("Model Information", expanded=False):
        st.write(f"Using Groq model: **{getattr(llm, 'model', 'llama3-70b-8192')}**")
    
    # Get response from LLM
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stream the response
            for chunk in llm.stream(messages_for_llm):
                if chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response)
        except Exception as e:
            # Handle any API errors gracefully
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            full_response = f"I'm sorry, I encountered an error while generating a response. Please try again or select a different model. Error details: {str(e)}"
            response_placeholder.markdown(full_response)
    
    if full_response:
        # Add assistant response to both visible chat and context
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.context.append({"role": "assistant", "content": full_response})
