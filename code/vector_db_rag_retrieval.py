import os
import logging
from dotenv import load_dotenv
from utils import load_yaml_config
from prompt_builder import build_prompt_from_config
from langchain_groq import ChatGroq
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from collections import deque

logger = logging.getLogger()


def setup_logging():

    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "rag_assistant.log"))
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


load_dotenv()

# To avoid tokenizer parallelism warning from huggingface
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VECTOR_DB_DIR = "../vector_db"
COLLECTION_NAME = "ethiopian_history"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Set up ChromaDB persistent client and collection
client = chromadb.PersistentClient(path=VECTOR_DB_DIR, settings=Settings(allow_reset=True))
collection = client.get_or_create_collection(COLLECTION_NAME)

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Load config for memory strategy
config = load_yaml_config(APP_CONFIG_FPATH)
window_size = config.get("memory_strategies", {}).get("trimming_window_size", 6)

# Conversation memory: keep only the last N messages (N = window_size)
conversation_history = deque(maxlen=window_size)


def add_to_memory(user, assistant):
    conversation_history.append({"user": user, "assistant": assistant})


def get_memory_context():
    if not conversation_history:
        return ""
    return "\n".join(
        [
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in conversation_history
        ]
    )


def retrieve_relevant_documents(
    query: str,
    n_results: int = 5,
    threshold: float = 15,  # Raise threshold for debugging
) -> list[str]:
    """
    Query the ChromaDB database with a string query.

    Args:
        query (str): The search query string
        n_results (int): Number of results to return (default: 5)
        threshold (float): Threshold for the cosine similarity score (default: 0.3)

    Returns:
        dict: Query results containing ids, documents, distances, and metadata
    """
    logging.info(f"Retrieving relevant documents for query: {query}")
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }
    # Embed the query using the same model used for documents
    logging.info("Embedding query...")
    query_embedding = embedding_model.encode([query])[0].tolist()

    logging.info("Querying collection...")
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )
    logger.info(f"Raw query results: {results}")

    logging.info("Filtering results...")
    keep_item = [False] * len(results["ids"][0])
    for i, distance in enumerate(results["distances"][0]):
        if distance < threshold:
            keep_item[i] = True

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    return relevant_results["documents"]


def respond_to_query(
    prompt_config: dict,
    query: str,
    llm: str,
    n_results: int = 5,
    threshold: float = 0.3,
) -> str:
    """
    Respond to a query using the ChromaDB database.
    """

    relevant_documents = retrieve_relevant_documents(
        query, n_results=n_results, threshold=threshold
    )

    logging.info("-" * 100)
    logging.info("Relevant documents: \n")
    for doc in relevant_documents:
        logging.info(doc)
        logging.info("-" * 100)
    logging.info("")

    logging.info("User's question:")
    logging.info(query)
    logging.info("")
    logging.info("-" * 100)
    logging.info("")
    input_data = (
        f"Relevant documents:\n\n{relevant_documents}\n\nUser's question:\n\n{query}"
    )

    rag_assistant_prompt = build_prompt_from_config(
        prompt_config, input_data=input_data
    )

    logging.info(f"RAG assistant prompt: {rag_assistant_prompt}")
    logging.info("")

    llm = ChatGroq(model=llm)

    response = llm.invoke(rag_assistant_prompt)
    return response.content


if __name__ == "__main__":
    setup_logging()
    app_config = load_yaml_config(APP_CONFIG_FPATH)
    prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)

    rag_assistant_prompt = prompt_config["rag_assistant_prompt"]

    vectordb_params = app_config["vectordb"]
    llm = app_config["llm"]

    exit_app = False
    rude_words = ["stupid", "idiot", "hate", "dumb", "kill", "shut up", "ugly", "fool", "suck", "darn", "hate you", "hate this", "dumbest", "worst"]
    while not exit_app:
        query = input(
            "Enter a question, 'config' to change the parameters, or 'exit' to quit: "
        )
        if query == "exit":
            exit_app = True
            exit()

        elif query == "config":
            threshold = float(input("Enter the retrieval threshold: "))
            n_results = int(input("Enter the Top K value: "))
            vectordb_params = {
                "threshold": threshold,
                "n_results": n_results,
            }
            continue

        # Kid-friendly input enforcement
        if any(word in query.lower() for word in rude_words):
            print("Please ask your question in a kind and kid-friendly way! 😊 Try rephrasing your question politely.")
            continue

        response = respond_to_query(
            prompt_config=rag_assistant_prompt,
            query=query,
            llm=llm,
            **vectordb_params,
        )
        logging.info("-" * 100)
        logging.info("LLM response:")
        logging.info(response + "\n\n")
