
Ethiopian History RAG System
===========================

This repository implements a Retrieval-Augmented Generation (RAG) system for exploring Ethiopian history, designed for educational and interactive use. The system leverages Wikipedia data, chunking, embedding, and a vector database to provide accurate, context-aware answers to user queries, with a Streamlit-based chat interface.

## Project Description

This project enables users (especially kids) to ask questions about Ethiopian history and receive well-formatted, document-grounded answers. The backend uses pre-processed Wikipedia data, chunked and embedded, and stored in a ChromaDB vector database. The frontend is a modern, interactive Streamlit app with persona-based responses.

## Project Structure

The directory structure is as follows:

```
├── code/
│   ├── data_acquisition_wikipedia.py         # Wikipedia data acquisition
│   ├── data_processing_chunking.py           # Cleans and chunks Wikipedia data
│   ├── embedding_and_storage.py              # Embeds and stores chunks in ChromaDB
│   ├── ethiopian_history_streamlit_chat.py   # Streamlit chat app
│   ├── paths.py                              # Centralized file paths
│   ├── prompt_builder.py                     # Prompt construction utilities
│   ├── save_persona_image.py                 # Persona image utility
│   ├── vector_db_rag_retrieval.py            # CLI RAG retrieval and LLM interface
│   └── vector_db/                            # Vector DB files and chunked data
├── config/
│   ├── config.yaml                           # App configuration
│   └── prompt_config.yaml                    # Prompt configuration
├── data/
│   ├── ethiopian_history_am.json             # Amharic Wikipedia data
│   └── ethiopian_history_en.json             # English Wikipedia data
├── vector_db/
│   ├── chroma.sqlite3                        # ChromaDB database
│   ├── ethiopian_history_am_chunked.json     # Chunked Amharic data
│   └── ethiopian_history_en_chunked.json     # Chunked English data
├── requirements.txt                          # Python dependencies
├── LICENSE                                   # License
├── README.md                                 # This documentation
```


## System Scope and Limitations

**Coverage:**
- The assistant covers Ethiopian history from prehistory (e.g., Lucy/Australopithecus) through the 20th century, including major events, figures, empires, and cultural developments.
- Sources include English and Amharic Wikipedia articles, curated for educational use.
- Focus is on factual, well-documented history suitable for students and the general public.

**Limitations:**
- The system does not cover modern political events after the 20th century, current news, or non-historical folklore.
- It does not provide legal, medical, or political advice.
- Some topics may be simplified for clarity and age-appropriateness.
- Answers are grounded in the available Wikipedia data and may not reflect the latest academic research.

**Excluded Topics:**
- Modern Ethiopian politics (post-2000)
- Unverified legends or folklore not covered in Wikipedia
- Non-historical or speculative content

If you have suggestions for expanding the scope or notice missing topics, please open an issue or contribute!

## Usage

### Preparing Your Data

1. Place your Wikipedia JSON data in the `data/` directory (see `ethiopian_history_en.json` and `ethiopian_history_am.json` for format).
2. Run `code/data_processing_chunking.py` to clean and chunk the data. This will output chunked JSON files in `vector_db/`.
3. Run `code/embedding_and_storage.py` to embed the chunks and store them in ChromaDB.

### Running the System Locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Start the Streamlit chat app:
   ```
   streamlit run code/ethiopian_history_streamlit_chat.py
   ```
   - The app will be available at `http://localhost:8501`.
3. (Optional) Use the CLI RAG retrieval tool:
   ```
   python code/vector_db_rag_retrieval.py
   ```

### Project Components

- **code/data_acquisition_wikipedia.py**: Script for acquiring Wikipedia data (customize as needed).
- **code/data_processing_chunking.py**: Cleans and chunks Wikipedia articles for embedding.
- **code/embedding_and_storage.py**: Embeds text chunks and stores them in ChromaDB.
- **code/ethiopian_history_streamlit_chat.py**: Streamlit app for interactive Q&A with persona selection.
- **code/vector_db_rag_retrieval.py**: Command-line tool for RAG retrieval and LLM-based answering.
- **code/paths.py**: Centralizes all file and directory paths.
- **code/prompt_builder.py**: Utilities for building prompts for the LLM.
- **config/config.yaml**: Main application configuration.
- **config/prompt_config.yaml**: Prompt and persona configuration.
- **data/**: Contains raw Wikipedia data in JSON format.
- **vector_db/**: Contains the ChromaDB database and chunked data files.

### Model Inputs and Outputs

- **Inputs**: User questions (via Streamlit or CLI), Wikipedia JSON data for preprocessing.
- **Outputs**: Answers to user questions, chat history, and logs (see `outputs/` if configured).

## Requirements

Install all dependencies with:

```
pip install -r requirements.txt
```

## License

See the LICENSE file for details.

## Contact

Repository created by Yonatan Awoke. For questions, please open an issue on GitHub.
