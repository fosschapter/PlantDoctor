import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def setup_chroma():
    """
    Set up ChromaDB with data for RAG
    
    Returns:
        Tuple of (ChromaDB client, collection)
    """
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.Client()
        
        # Use SentenceTransformer for embeddings
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        collection = chroma_client.get_or_create_collection(
            name="plant_disease_data",
            embedding_function=sentence_transformer_ef
        )
        
        # Check if collection is empty, if so, populate it
        if collection.count() == 0:
            logger.info("Loading data into ChromaDB collection")
            
            # Load RAG data
            if os.path.exists('data_for_rag.json'):
                with open('data_for_rag.json', 'r') as f:
                    rag_data = json.load(f)
                
                # Prepare data for ChromaDB
                documents = []
                metadatas = []
                ids = []
                
                for i, item in enumerate(rag_data):
                    documents.append(item['content'])
                    metadatas.append({"source": item['source']})
                    ids.append(f"doc_{i}")
                
                # Add data to collection
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(documents)} documents to ChromaDB collection")
            else:
                logger.warning("No data_for_rag.json file found. ChromaDB collection will be empty.")
        else:
            logger.info(f"ChromaDB collection already contains {collection.count()} documents")
        
        return chroma_client, collection
    except Exception as e:
        logger.error(f"Error setting up ChromaDB: {str(e)}")
        raise

def load_llm():
    """
    Load the lightweight LLM for generating recommendations
    
    Returns:
        Loaded LLM pipeline
    """
    try:
        logger.info("Loading LLM from Hugging Face")
        
        # Load model with optimizations
        model_name = "arnir0/Tiny-LLM"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            low_cpu_mem_usage=True
        )
        
        # Create pipeline
        llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,  # Limit token generation for efficiency
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        logger.info("LLM loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM: {str(e)}")
        raise

# Cache the LLM to avoid reloading
_llm_cache = None

def get_llm():
    """Get the LLM, loading it if necessary"""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = load_llm()
    return _llm_cache

def query_rag(collection, query, n_results=3):
    """
    Query the RAG system with user input
    
    Args:
        collection: ChromaDB collection
        query: User query string
        n_results: Number of results to retrieve
        
    Returns:
        Generated response
    """
    try:
        logger.info(f"RAG query: {query}")
        
        # Query ChromaDB for relevant documents
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Extract the documents
        documents = results['documents'][0]
        
        # Create context from retrieved documents
        context = "\n\n".join(documents)
        
        # Prepare prompt for the LLM
        prompt = f"""Based on the following information about plant diseases and treatments,
provide a helpful response to the query: "{query}"

Reference information:
{context}

Provide a concise, informative answer focusing on accurate treatment methods and disease management.
"""
        
        # Get the LLM
        llm = get_llm()
        
        # Generate response
        response = llm(prompt)[0]['generated_text']
        
        # Extract just the generated part (after the prompt)
        response = response[len(prompt):].strip()
        
        # If the response is empty or too short, provide a fallback
        if len(response) < 50:
            response = f"I don't have enough information about {query}. Consider consulting a local agricultural extension service or plant pathologist for specific advice."
        
        return response
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        return f"I encountered an error while processing your query. Please try again or rephrase your question."
