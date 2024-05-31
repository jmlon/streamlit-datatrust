import os
# Warning about parallel tokenizers
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st 
from rag_agent import run_query


