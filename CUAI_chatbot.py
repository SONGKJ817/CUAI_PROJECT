import os
import openai
import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler

import os
os.environ['OPENAI_API_KEY'] = 'sk-proj-pkJIhc0fM3iuKGlCvvxAT3BlbkFJABN9ZkEh9WrIVZw0oXBH'

openai.api_key = 'sk-proj-pkJIhc0fM3iuKGlCvvxAT3BlbkFJABN9ZkEh9WrIVZw0oXBH'

# load data
loader = TextLoader(file_path='./data/all_data.txt', encoding = "UTF-8")
data = loader.load()

# chunk documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    length_function = len)
texts = text_splitter.split_documents(data)

# create an index
store = LocalFileStore("./cachce/")

# create an embedder
core_embeddings_model = OpenAIEmbeddings()

embedder = CacheBackedEmbeddings.from_bytes_store(
    core_embeddings_model,
    store,
    namespace = core_embeddings_model.model
)

# retriever ensemble
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 2

faiss_vector = FAISS.from_documents(texts, embedder)
faiss_retriever = faiss_vector.as_retriever(search_kwargs={'k':2})

ensemble_retriever = EnsembleRetriever(
                    retrievers = [bm25_retriever, faiss_retriever]
                    , weight = {0.5,0.5})

# 앱 제목과 설명
st.set_page_config(page_title="CUAI Chatbot", page_icon=":robot_face:")
st.header("CUAI CHATBOT :robot_face:")
st.markdown("""
    중앙대학교 CUAI에 대한 정보를 알려주는 chatbot입니다.
    """)
st.markdown("""
    - 예시) 2024년 NLP 관련 프로젝트를 알려줘.
    """)

# 사이드바에 설명 추가
st.sidebar.title("About")
st.sidebar.info("""
    This is a demo chatbot built using Streamlit and OpenAI's GPT-3.5-turbo model.
    """)

def generate_response(prompt):
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0)
    handler = StdOutCallbackHandler()
    qa_with_sources_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=ensemble_retriever,
        callbacks=[handler],
        return_source_documents=True
    )
    response = qa_with_sources_chain({"query" : f"{prompt}"})
    return response["result"]

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    
if 'past' not in st.session_state:
    st.session_state['past'] = []
    
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('You: ', '', key='input')
    submitted = st.form_submit_button('Send')
    
if submitted and user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))

# CSS 스타일 추가
st.markdown("""
    <style>
    .stMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .stMessageUser {
        background-color: #DCF8C6;
        text-align: right;
    }
    .stMessageBot {
        background-color: #E1E1E1;
    }
    </style>
    """, unsafe_allow_html=True)