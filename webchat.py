from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

anthropic_llm = ChatAnthropic(anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))

st.set_page_config(page_title="Web Chat")
