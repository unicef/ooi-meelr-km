# @title Imports
#data loading imports
import re


# construct vector store query
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.core.retrievers import VectorIndexRetriever




#llms and query engines

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

    metadata_filters = None
    if filters:
        metadata_filters = MetadataFilters(
            filters = [MetadataFilter(
                key=filter['metadata_key'],
                value = filter['metadata_value'],
                operator = FilterOperator.EQ)
                for filter in filters if filter['metadata_key']]
        )
    else: 
        metadata_filters = None

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
