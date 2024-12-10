# UNICEF studio for chatbot

from IPython.display import clear_output
import subprocess
import os
import sys
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["LITERAL_API_KEY"] = "lsk_Zr7hzZASa388Pl2uYUUNzJdYHNJXznumk8tOZrvLkLE"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
URI = "http://ec2-63-176-97-200.eu-central-1.compute.amazonaws.com:19530"

print("Installing packages for UNICEF Studio ... Please wait 2 minutes ...")

install = [
    "pip", "install",
    "gradio==4.44.1",
    "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git",
    "llama-index",
    "pymilvus>=2.4.2",
    "llama-index-vector-stores-milvus",
    "llama-index-embeddings-huggingface",
    "literalai",
    "llama-index-callbacks-literalai",
    "llama-index-llms-huggingface-api",
    "huggingface_hub[inference]"
]


install = subprocess.Popen(install)
install.wait()

# from utils import query_store
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import load_indices_from_storage, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore

from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)
import warnings

warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "subprocess")

##############uitls.py################
# @title Imports
#data loading imports
import re


# construct vector store query
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.core.retrievers import VectorIndexRetriever

#logging
from llama_index.core import set_global_handler
set_global_handler("literalai")

# from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer


#miscellaneous
from typing import List, Optional


# @title Query engine
def query_store(query_str, index, nodes, embed_model, vector_store, filters, llm, callback_manager, stats = False, viz = False) -> None:
    """Processes user query, calls stats, vector store and LLM. Returns LLM response

    Args:
        query_str (str): Query being asked.
        index (VectorStoreIndex(BaseIndex[IndexDict])): index of vector data.
        nodes (): all nodes from vector store for viz purpose. 
        embed_model (): embedding model used from HF
        vector_store (): milvus vector client 
        filters (list[dict]): list of filters for metadata.
        llm (): llm used from HF
        callback_manager (): For logging llm and vector retrieval calls.
        stats (bool): Whether to return retrival stats or not.
        viz (bool): Whether to return visualization or not.

    """
    emb_pants_str = "Represent this sentence for searching relevant passages: "
    query_embedding = embed_model.get_query_embedding(f"{emb_pants_str}{query_str}")

    metadata_filters = MetadataFilters(
    filters = [MetadataFilter(
        key=filter['metadata_key'], 
        value = filter['metadata_value'], 
        operator = FilterOperator.EQ) 
        for filter in filters if filter['metadata_key']]
    )

    ###################### LLM based ###########

    #llm based on index (in-memory and not from vector store)
    retriever = VectorIndexRetriever(
        index=index, similarity_top_k=32,
        filters = metadata_filters
    ) #llm context len=8192, so 4 chunk = 512x4 = 2048 (+prompt) < 8192, reponse = 8k-2k = 6k. Use Tree_summarizer for individual LLM calls.
    synthesizer = get_response_synthesizer(response_mode="simple_summarize",
                                           callback_manager = callback_manager,
                                           use_async = True
                                           ) #use this to inject custom prompt.
    query_engine = RetrieverQueryEngine( retriever=retriever,
                                        response_synthesizer=synthesizer
                                        )
   
    response = query_engine.query(query_str)

    return str(response)




######################################

print("Loading model ... Please wait a few more seconds! ...")

embed_model = HuggingFaceEmbedding(
model_name="BAAI/bge-small-en-v1.5"
)

clear_output()

# set embed model for index and quering
Settings.embed_model = embed_model

llm = HuggingFaceInferenceAPI(
    model_name=MODEL_NAME,
)
Settings.llm = llm

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

vector_store = MilvusVectorStore(
    uri=URI,
    collection_name="coar_collection",
    similarity_metric = "IP", #default for sparse. else use COSINE,L2. Used while index creation
    batch_size = 100, #default
    dim=384,
    overwrite=False, # put False while reading from milvus.
    search_config = {"metric_type":"IP"} # used while search. should be same as similarity_matric during index creation of milvus.
)

# loading storage context that contains our vector store
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./ooi-meelr-km/coar_storage_context"),
    vector_store=vector_store,
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="./ooi-meelr-km/coar_storage_context"),
)

# csv+json was inserted into 1 index so below returns only 1 index
indices = load_indices_from_storage(storage_context)

# fetching all nodes takes time, switching it off.
nodes = None

clear_output()

import gradio
gradio.strings.en["SHARE_LINK_DISPLAY"] = ""

def process_chatbot(message, history):
    
    filters = [{'metadata_key' : 'country_name', 'metadata_value' : 'Malawi'},
           {'metadata_key' : 'year', 'metadata_value' : 2019}]
    
    return query_store(message, indices[0], nodes, embed_model, vector_store, filters, llm, callback_manager, stats = False, viz = False)

#BUG:Data incompatible with messages format. Each message should be a dictionary with 'role' and 'content' keys or a ChatMessage object."
studio_theme = gradio.themes.Soft(
    primary_hue = "teal",
)


scene = gradio.ChatInterface(
    process_chatbot,
    type="messages",
    # chatbot = gradio.Chatbot(
    #     height = 325,
    #     label = "UNICEF Studio Chat",
    # ),
    # textbox = gradio.Textbox(
    #     placeholder = "Message UNICEF Chat",
    #     container = False,
    # ),
    title = None,
    theme = studio_theme,
    examples = None,
    cache_examples = False,
    retry_btn = None,
    undo_btn = "Remove Previous Message",
    clear_btn = "Restart Entire Chat",
)

scene.launch(quiet = True)#(debug=True)