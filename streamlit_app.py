import os
# Warning about parallel tokenizers
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st 
from rag_agent import run_query

st.title("Zotero chatbot")

st.sidebar.write('Groq configuration')
st.sidebar.selectbox('LLM Model', [os.getenv("GROQ_MODEL"), 'llama3-8b-8192'])


# Initialize message history in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Retrieve the chat history from the session_state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def log_response(response):
    print("INPUT  : ", response['input'])
    print("CONTEXT: ", response['context'])     # A list of Document(page_content,metadata)
    print("ANSWER : ", response['answer'])

def xform_doc(doc):
    return {
        'title': doc.metadata['title'],
        'page': doc.metadata['page'],
        'url': f"<a href=\"{doc.metadata['url']}\">Link</A>",
    }

def docs_with_df(docs):
    list_dict = list(map(xform_doc, docs))
    # st.dataframe(pd.DataFrame.from_dict(list_dict))
    st.table(pd.DataFrame.from_dict(list_dict))

def docs_with_cols(docs):
    st.caption("Source references")
    cols = st.columns(4)
    for col,doc in zip(cols,docs):
        with col:
            with st.container(border=True):
                st.write(doc.metadata['title'])
                st.write(f"Page: {doc.metadata['page']}")
                st.link_button("Zot", doc.metadata['zot_url'])
                # print("***URL*** : ", doc.metadata['url'])
                if 'url' in doc.metadata and doc.metadata['url']!='':
                    st.link_button("Url", doc.metadata['url'])


# Await for user input
if prompt := st.chat_input("Enter query"):

    # Add user input to the history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({ "role":"user", "content":prompt })

    response = run_query(prompt)
    # log_response(response)
    st.divider()
    # docs_with_df(response['context'])
    docs_with_cols(response['context'])

    # Add response
    with st.chat_message("assistant"):
        st.markdown(response['answer'])
    st.session_state.messages.append({ "role": "assistant", "content":response['answer'] })


