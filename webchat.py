from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
import streamlit as st  # type: ignore

load_dotenv()

st.set_page_config(page_title="Web Chat", page_icon=":rocket:")
st.title("WebChat")

with st.sidebar:
    st.header("Description")
    st.write("**Webchat** is a **RAG/AI** application that allows you to ask questions and get answers from a website.")


base_llm = ChatAnthropic(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-sonnet-20240229",
    temperature=0.2,
    max_tokens=1024,
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the the provided context:
     <context>
         {context}
     </context>

     Question: {input}
    """
)

document_chain = create_stuff_documents_chain(base_llm, prompt)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
)

text_splitter = RecursiveCharacterTextSplitter()


def load_website_data(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    return retriever


# retriever = load_website_data('https://www.anthropic.com/claude')
# retriever = load_website_data("https://en.wikipedia.org/wiki/Perplexity.ai")
# loader = WebBaseLoader("https://www.anthropic.com/claude")

try:
    input_url = st.text_input("URL: ")
    # if st.button('Get site data'):
    with st.spinner("Getting site data"):
        retriever = load_website_data(input_url)
        st.success(f"Successfully retrieved data from {input_url}")

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    
    query = st.text_input(label="Ask question")

    if st.button("Ask question"):
        with st.spinner("Generating Answers"):
            result = retrieval_chain.invoke({"input": query})
       
            message = st.chat_message('assistant')
            message.write(result["answer"])

except Exception as e:
    st.error(f"{e} Please enter a valid URL")
