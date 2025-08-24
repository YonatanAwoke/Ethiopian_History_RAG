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


    def print_all_ids_and_metadata():
        print("\nAll stored chunk IDs and metadata:")
        results = collection.get(include=["metadatas"])
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            print(f"ID: {doc_id}, Title: {meta.get('title')}, URL: {meta.get('fullurl')}")

    def evaluate_retrieval(test_queries, k=5, threshold=0.3, match_on="id"):
        """
        Evaluate retrieval using recall@k and precision@k.
        test_queries: list of dicts with 'query' and 'relevant_values' fields
        match_on: 'id', 'title', or 'fullurl'
        """
        total_recall = 0
        total_precision = 0
        for test in test_queries:
            query = test['query']
            relevant_values = set(test['relevant_values'])
            if match_on == "id":
                results = collection.query(
                    query_embeddings=[embedding_model.encode([query])[0].tolist()],
                    n_results=k
                )
            else:
                results = collection.query(
                    query_embeddings=[embedding_model.encode([query])[0].tolist()],
                    n_results=k,
                    include=["metadatas"]
                )
            if match_on == "id":
                retrieved = set(results["ids"][0])
            else:
                retrieved = set(meta.get(match_on) for meta in results["metadatas"][0])
            true_positives = len(retrieved & relevant_values)
            recall = true_positives / len(relevant_values) if relevant_values else 0
            precision = true_positives / k if k else 0
            print(f"Query: {query}")
            print(f"Recall@{k}: {recall:.2f}, Precision@{k}: {precision:.2f}")
            total_recall += recall
            total_precision += precision
        n = len(test_queries)
        print(f"\nAverage Recall@{k}: {total_recall/n:.2f}")
        print(f"Average Precision@{k}: {total_precision/n:.2f}")

    # Example test set (replace with real data and correct match_on):
    test_queries = [
        {"query": "Battle of Adwa", "relevant_values": ["Battle of Adwa", "https://en.wikipedia.org/wiki/Battle_of_Adwa"]},
        {"query": "Haile Selassie", "relevant_values": ["Haile Selassie", "https://en.wikipedia.org/wiki/Haile_Selassie"]},
        # Add more test cases as needed
    ]

    exit_app = False
    rude_words = ["stupid", "idiot", "hate", "dumb", "kill", "shut up", "ugly", "fool", "suck", "darn", "hate you", "hate this", "dumbest", "worst"]
    while not exit_app:

        query = input(
            "Enter a question, 'config' to change the parameters, 'eval' to run retrieval evaluation, 'printids' to list all stored IDs/metadata, or 'exit' to quit: "
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

        elif query == "printids":
            print_all_ids_and_metadata()
            continue

        elif query == "eval":
            k = int(input("Enter k for recall@k and precision@k: "))
            match_on = input("Match on which field? (id/title/fullurl): ").strip().lower()
            if match_on not in ("id", "title", "fullurl"):
                print("Invalid match_on. Using 'id'.")
                match_on = "id"
            evaluate_retrieval(test_queries, k=k, threshold=vectordb_params.get("threshold", 0.3), match_on=match_on)
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
