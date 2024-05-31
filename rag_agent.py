import os 
import streamlit as st 

# Models
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# LangChain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# HuggingFace
# from langchain.embeddings import HuggingFaceEmbeddings        # OLD
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector databases
from langchain_pinecone import PineconeVectorStore


index_name = "datatrust"


# LLM
# llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0)
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY_ZOT"],
    model_name=st.secrets["GROQ_MODEL"], temperature=0)

# Embeddings model
EMBEDDINGS_MODEL = 'BAAI/bge-base-en-v1.5'                      # Ranking 8, 768
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

# Vector store
docsearch = PineconeVectorStore.from_existing_index(embedding=embeddings, index_name=index_name)
retriever = docsearch.as_retriever(search_type="mmr")

system_prompt = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Keep the answer concise and to the point."
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

def run_query(query):
    return chain.invoke({"input": query})


print("rag_agent initialized!")


if __name__ == "__main__":
    print("Running tests")
    print(run_query("What is data governance?"))
