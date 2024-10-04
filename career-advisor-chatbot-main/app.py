from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper

import streamlit as st
from dotenv import load_dotenv
import emoji

import os
from itertools import zip_longest

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.title(f"Career Advisor Chatbot {emoji.emojize(':robot:')}")

# Define your directory containing PDF files here
pdf_dir = 'Career couselling book.pdf'

# Check if the file exists before trying to open it
if os.path.exists(pdf_dir):
    with open(pdf_dir, 'rb') as file:
        loader = PyPDFLoader(file)
        documents = loader.load()
        pdf_texts = " ".join([doc.page_content for doc in documents])
else:
    st.error(f"The file '{pdf_dir}' does not exist in the current directory.")

if "pdf_texts" not in st.session_state:
    st.session_state["pdf_texts"] = pdf_texts if 'pdf_texts' in locals() else ""
    
if "vectors" not in st.session_state:
    with st.spinner("Creating a Database..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(st.session_state["pdf_texts"])
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state["vectors"] = FAISS.from_texts(chunks, embeddings)
    st.success("Database creation completed!")

def get_response(history, user_message, temperature=0):
    DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and a Career Advisor. The Advisor guides the user regarding jobs, interests, and other domain selection decisions.
    It follows the previous conversation to do so.

    Relevant pieces of previous conversation:
    {context},

    Useful information from career guidance books:
    {text}, 

    Useful information about career guidance from Web:
    {web_knowledge},

    Current conversation:
    Human: {input}
    Career Expert:"""

    PROMPT = PromptTemplate(
        input_variables=['context', 'input', 'text', 'web_knowledge'], template=DEFAULT_TEMPLATE
    )
    
    docs = st.session_state["vectors"].similarity_search(user_message)

    params = {
        "engine": "bing",
        "gl": "us",
        "hl": "en",
    }

    search = SerpAPIWrapper(params=params)
    web_knowledge = search.run(user_message)

    gemini_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)

    conversation_with_summary = LLMChain(
        llm=gemini_model,
        prompt=PROMPT,
        verbose=False
    )
    
    response = conversation_with_summary.predict(context=history, input=user_message, web_knowledge=web_knowledge, text=docs)
    
    return response

# Function to get conversation history
def get_history(history_list):
    history = ''
    for message in history_list:
        if message['role'] == 'user':
            history += f'input {message["content"]}\n'
        elif message['role'] == 'assistant':
            history += f'output {message["content"]}\n'
    
    return history

# Streamlit UI
def get_text():
    input_text = st.sidebar.text_input("You: ", "Hello, how are you?", key="input")
    if st.sidebar.button('Send'):
        return input_text
    return None

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

user_input = get_text()

if user_input:
    user_history = list(st.session_state["past"])
    bot_history = list(st.session_state["generated"])

    combined_history = []
    
    for user_msg, bot_msg in zip_longest(user_history, bot_history):
        if user_msg is not None:
            combined_history.append({'role': 'user', 'content': user_msg})
        if bot_msg is not None:
            combined_history.append({'role': 'assistant', 'content': bot_msg})

    formatted_history = get_history(combined_history)

    output = get_response(formatted_history, user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Chat History", expanded=True):
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            st.markdown(emoji.emojize(f":speech_balloon: **User {str(i)}**: {st.session_state['past'][i]}"))
            st.markdown(emoji.emojize(f":robot: **Assistant {str(i)}**: {st.session_state['generated'][i]}"))

# Example questions for users to consider
st.write("### Example Questions:")
st.write("- What factors should I keep in mind before deciding a career?")
st.write("- What are the growing sectors of the global economy?")
st.write("- If I decide to be a software engineer, what would my salary be?")
