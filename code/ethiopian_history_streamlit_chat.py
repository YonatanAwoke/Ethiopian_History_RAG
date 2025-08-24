import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv
import re
from PIL import Image

# Load environment variables
load_dotenv()

# ChromaDB and embedding model setup
VECTOR_DB_DIR = "../vector_db"
COLLECTION_NAME = "ethiopian_history"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

client = chromadb.PersistentClient(path=VECTOR_DB_DIR, settings=Settings(allow_reset=True))
collection = client.get_or_create_collection(
    COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Function to strictly format Wikipedia-style sections, references, and paragraphs
def format_wiki_sections(text):
    # Section headers: == Section == to <h4>Section</h4>
    def repl_section(match):
        title = match.group(1).strip()
        return f"<h4>{title}</h4>"
    text = re.sub(r"==+\s*(.*?)\s*==+", repl_section, text)
    # References: [1], [2], etc. to superscript
    text = re.sub(r"\[(\d+)\]", r"<sup>[\1]</sup>", text)
    # External links: [http(s)://...] to clickable links
    text = re.sub(r"\[(https?://[^\s\]]+)\]", r'<a href="\1" target="_blank">[external link]</a>', text)
    # Paragraphs: split by double newlines, wrap in <p>
    paragraphs = [f"<p>{p.strip()}</p>" for p in re.split(r"\n{2,}", text) if p.strip()]
    return "\n".join(paragraphs)

# Function to strictly format LLM output (markdown/HTML)
def format_llm_output(text):
    # Ensure sentences start with capital, end with period if missing
    sentences = re.split(r'(?<=[.!?])\s+', text)
    formatted = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not s[0].isupper():
            s = s[0].upper() + s[1:]
        if s[-1] not in '.!?':
            s += '.'
        formatted.append(s)
    return ' '.join(formatted)

# Retrieval function
def retrieve_relevant_documents(query, n_results=5, threshold=0.7):
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )
    docs = results["documents"][0] if results["documents"] else []
    # Format section headers in each doc
    docs = [format_wiki_sections(d) for d in docs]
    return docs



# --- Modern Navigation Bar ---
st.set_page_config(page_title="Ethiopian History for Kids", page_icon="🦁", layout="centered")

def nav_bar():
    st.markdown("""
    <style>
    .nav-bar {
        display: flex;
        justify-content: center;
        align-items: center;
        background: #fcfbee;
        border-bottom: 2px dotted #e0e0c0;
        border-top: 2px dotted #e0e0c0;
        padding: 0.7em 0 0.4em 0;
        margin-bottom: 1.2em;
        font-family: 'Fira Mono', 'Consolas', monospace;
    }
    .nav-link {
        font-size: 1.1em;
        font-weight: 500;
        color: #3a5d2c;
        margin: 0 1.5em;
        text-decoration: none;
        padding: 0.1em 0.5em;
        border-radius: 4px;
        background: none;
        border: none;
        outline: none;
        cursor: pointer;
        position: relative;
        transition: color 0.2s, font-weight 0.2s;
        letter-spacing: 0.01em;
    }
    .nav-link.selected {
        font-weight: bold;
        color: #2d4d1c;
    }
    .nav-link:after {
        content: ' ▼';
        font-size: 0.8em;
        color: #b2bfa3;
        margin-left: 0.2em;
        vertical-align: middle;
        opacity: 0.7;
    }
    .nav-link:last-child:after {
        content: '';
    }
    .nav-link:hover {
        color: #1b2e13;
        background: #f5f7e6;
    }
    </style>
    <script>
    const links = window.parent.document.querySelectorAll('.nav-link');
    links.forEach(link => link.onclick = function() {
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: link.id}, '*');
    });
    </script>
    """, unsafe_allow_html=True)



# Ensure nav_page is initialized
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "nav-home"
# Fallback for navigation (simulate click)
nav_choice = st.selectbox("Go to page:", ["🏠 Home", "👑 Mode"], index=0 if st.session_state.nav_page=="nav-home" else 1, key="nav_select", label_visibility="collapsed")
if nav_choice == "🏠 Home":
    st.session_state.nav_page = "nav-home"
else:
    st.session_state.nav_page = "nav-mode"


# --- Persona List ---
personas = [
    {"name": "Emperor Menelik II", "desc": "Unifier and modernizer of Ethiopia.", "tone": "Speak with authority and vision, referencing unity and progress."},
    {"name": "Empress Taytu Betul", "desc": "Strategic leader and co-founder of Addis Ababa.", "tone": "Speak wisely and diplomatically, with strategic insight."},
    {"name": "Emperor Haile Selassie", "desc": "Reformer and symbol of African unity.", "tone": "Speak with inspiration and dignity, referencing reform and unity."},
    {"name": "Queen of Sheba", "desc": "Legendary monarch of ancient Ethiopia.", "tone": "Speak with mystery and regal poise, referencing ancient wisdom."},
    {"name": "King Ezana", "desc": "First Christian king of Axum.", "tone": "Speak thoughtfully and as a pioneer, referencing faith and new beginnings."},
]

def get_persona_image():
    try:
        img = Image.open("code/ethiopian_leader.png")
    except Exception:
        img = None
    return img


# --- Home Page (Chatbot) ---
if st.session_state.nav_page == "nav-home":
    st.title("🦁 Ethiopian History Explorer")
    persona_idx = st.session_state.get("selected_persona", 0)
    persona = personas[persona_idx]
    st.markdown(f"<div style='background:#fffbe7;padding:0.7em;border-radius:10px;margin-bottom:0.7em; color:black;'><b>Current Leader:</b> <span style='color:#e65100;font-weight:bold;'>{persona['name']}</span> <span style='font-size:0.95em;color:#888;'>(Default)</span></div>", unsafe_allow_html=True)
    st.markdown("""
    Welcome! Ask me anything about Ethiopian history.\nLet's learn together in a fun way! 🎉
    """)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("Type your question here:", key="user_input")
    ask_button = st.button("Ask Lion Guide 🦁")
    # Persona prompt injection
    persona_tone = persona["tone"]
    persona_name = persona["name"]
    if ask_button and user_input.strip():
        # Check for kid-friendly input
        rude_words = ["stupid", "idiot", "hate", "dumb", "kill", "shut up", "ugly", "fool", "suck", "darn", "hate you", "hate this", "dumbest", "worst"]
        if any(word in user_input.lower() for word in rude_words):
            st.warning("Please ask your question in a kind and kid-friendly way! 😊 Try rephrasing your question politely.")
        else:
            # --- Input Normalization ---
            import re
            stopwords = set([
                'the', 'is', 'in', 'at', 'of', 'a', 'an', 'and', 'to', 'for', 'on', 'with', 'as', 'by', 'from', 'about', 'that', 'this', 'it', 'be', 'or', 'are', 'was', 'were', 'which', 'who', 'whom', 'whose', 'has', 'have', 'had', 'but', 'not', 'so', 'if', 'then', 'than', 'too', 'very', 'can', 'will', 'just', 'do', 'does', 'did', 'into', 'out', 'up', 'down', 'over', 'under', 'again', 'further', 'once'
            ])
            synonyms = {
                'ethiopia': ['abyssinia'],
                'king': ['emperor', 'negus'],
                'queen': ['empress'],
                'battle': ['war', 'conflict'],
                'capital': ['addis ababa'],
                'leader': ['ruler'],
                'church': ['cathedral'],
                'ancient': ['old'],
                'modern': ['recent'],
            }
            def normalize_query(q):
                q = q.lower()
                q = re.sub(r'[^a-z0-9\s]', '', q)
                tokens = [w for w in q.split() if w not in stopwords]
                # Expand synonyms
                expanded = []
                for w in tokens:
                    expanded.append(w)
                    for k, syns in synonyms.items():
                        if w == k or w in syns:
                            expanded.extend([k] + syns)
                return ' '.join(sorted(set(expanded)))
            normalized_query = normalize_query(user_input)
            # --- Query Classification ---
            def classify_query(q):
                q = q.lower()
                if any(x in q for x in ["when", "where", "who", "how many", "date", "year", "time", "list", "name all", "give me", "show me"]):
                    return "fact"
                if any(x in q for x in ["why", "explain", "reason", "cause", "meaning", "significance", "describe", "tell me about"]):
                    return "explanation"
                if any(x in q for x in ["compare", "difference", "similarity", "versus", "vs", "better", "worse"]):
                    return "comparison"
                return "fact"
            query_type = classify_query(user_input)
            with st.spinner(f"{persona_name} is thinking..."):
                docs = retrieve_relevant_documents(normalized_query)
                # Compose persona-aware system prompt with strict formatting instructions and query type
                system_prompt = (
                    f"You are {persona_name}, a historical leader of Ethiopia. {persona_tone} "
                    f"The user has asked a {query_type} question. "
                    "Answer as if you are this leader, using the provided historical documents.\n"
                    "Strictly follow these formatting rules:\n"
                    "- Use clear, complete sentences and paragraphs.\n"
                    "- Properly format references, citations, and external links.\n"
                    "- Use HTML or markdown for section headers, paragraphs, and lists.\n"
                    "- Do not leave sentences unfinished.\n"
                    "- If you use content from the document, preserve its structure and references.\n"
                    "- If you summarize, ensure the summary is well-structured and readable.\n"
                    "- Always answer in a positive, encouraging, and kid-friendly tone.\n"
                )
                if docs:
                    formatted_docs = [format_wiki_sections(d) for d in docs[:3]]
                    answer = (
                        f"<div style='font-size:1.2em;'>"
                        f"<span style='font-size:2em;'>👑</span> <b>{persona_name} says:</b>"
                        "<ul>" + "".join([f'<li>📚 {d}</li>' for d in formatted_docs]) + "</ul>"
                        "<span style='font-size:1.1em;'>✨ Keep exploring and ask more questions! ✨</span>"
                        "</div>"
                    )
                    answer = format_llm_output(answer)
                else:
                    answer = f"<div style='font-size:1.2em;'><span style='font-size:2em;'>👑</span> <b>{persona_name}:</b> I couldn't find anything about that in my history books. Try another question! 🦋</div>"
                st.session_state.chat_history.insert(0, (persona_name, answer))
                st.session_state.chat_history.insert(0, ("You", user_input))
    feedback_given = st.session_state.get('feedback_given', {})
    for idx, (speaker, msg) in enumerate(st.session_state.chat_history):
        if speaker == "You":
            st.markdown(f"<div style='background:#e0f7fa;padding:0.7em;border-radius:10px;margin-bottom:0.3em; color:black;'><b>🧒 {speaker}:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#fff3e0;padding:0.7em;border-radius:10px;margin-bottom:0.3em;animation:fadein 1s; color:black;'><b>{msg}</b></div>", unsafe_allow_html=True)
            # Feedback buttons (only for answers, not user input)
            if idx not in feedback_given:
                col1, col2 = st.columns([1, 3])
                with col1:
                    unclear = st.button("❓ Unclear", key=f"unclear_{idx}")
                with col2:
                    unsat = st.button("👎 Not Helpful", key=f"unsat_{idx}")
                if unclear or unsat:
                    feedback_given[idx] = True
                    st.session_state['feedback_given'] = feedback_given
                    if unclear:
                        st.info("Thank you for your feedback! We'll work to make answers clearer.")
                    if unsat:
                        st.info("Thank you for your feedback! We'll work to improve answer helpfulness.")

    st.markdown("---")
    with st.expander("ℹ️ System Scope, Limitations, and Excluded Topics", expanded=False):
        st.markdown("""
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

<span style='font-size:0.95em;'>If you have suggestions for expanding the scope or notice missing topics, please open an issue or contribute!</span>
        """, unsafe_allow_html=True)
    st.markdown("Made with ❤️ for curious kids.")


# --- Mode Page (Persona Selection) ---
elif st.session_state.nav_page == "nav-mode":
    st.title("👑 Choose a Historical Leader")
    st.markdown("Select a leader to chat with. They will answer in their unique tone!")
    img = get_persona_image()
    if img:
        st.image(img, use_column_width=True, caption="Ethiopian Leader")
    st.markdown("""
    <style>
    .persona-card {
        background: #fffbe7;
        border-radius: 16px;
        box-shadow: 0 2px 8px #00000022;
        padding: 1.2em 1em;
        margin-bottom: 1.2em;
        transition: box-shadow 0.2s, transform 0.2s;
        cursor: pointer;
        animation: fadein 0.8s;
    }
    .persona-card.selected {
        border: 2px solid #e65100;
        background: #ffe0b2;
        transform: scale(1.03);
    }
    .persona-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #e65100;
    }
    .persona-desc {
        font-size: 1em;
        color: #333;
        margin-bottom: 0.3em;
    }
    </style>
    """, unsafe_allow_html=True)
    # Show default persona at the top
    default_idx = 0
    default_persona = personas[default_idx]
    st.markdown(f"<div style='background:#fffbe7;padding:0.7em;border-radius:10px;margin-bottom:0.7em;'><b>Default Leader:</b> <span style='color:#e65100;font-weight:bold;'>{default_persona['name']}</span></div>", unsafe_allow_html=True)
    for idx, p in enumerate(personas):
        selected = idx == st.session_state.selected_persona
        card_class = "persona-card selected" if selected else "persona-card"
        if st.button(f"{p['name']}", key=f"persona_{idx}"):
            st.session_state.selected_persona = idx
            st.rerun()
        st.markdown(f"""
        <div class='{card_class}'>
            <div class='persona-title'>{p['name']}</div>
            <div class='persona-desc'>{p['desc']}</div>
            <div style='font-size:0.95em;color:#888;'>Tone: <i>{p['tone']}</i></div>
        </div>
        """, unsafe_allow_html=True)
    st.success(f"You have selected: {personas[st.session_state.selected_persona]['name']}")
    st.info("Go back to Home to chat with your selected leader!")

# Add simple CSS animation
st.markdown("""
<style>
@keyframes fadein {
  from { opacity: 0; }
  to   { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = 0 
