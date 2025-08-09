import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_FPATH = os.path.join(ROOT_DIR, ".env")

CODE_DIR = os.path.join(ROOT_DIR, "code")

APP_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "config.yaml")
PROMPT_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "prompt_config.yaml")


OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")


DATA_DIR = os.path.join(ROOT_DIR, "data")

VECTOR_DB_DIR = os.path.join(OUTPUTS_DIR, "vector_db")

CHAT_HISTORY_DB_FPATH = os.path.join(OUTPUTS_DIR, "chat_history.db")
