import streamlit as st
import os
import json
import chromadb
import networkx as nx
import PyPDF2
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from streamlit_agraph import agraph, Node, Edge, Config

# --- ENGINE CONFIGURATION ---
# Switched to 8b-instant: 500k Daily Token Limit (Solves Error 429)
MODEL_NAME = "llama-3.1-8b-instant" 
LIME_VIBE = "#39FF14" 
DANGER_RED = "#FF3131" 

# --- PAGE SETUP ---
st.set_page_config(page_title="ACE: Neon Command", layout="wide", page_icon="🧠")

# --- VIBRANT LIVE BACKGROUND ---
video_url = "https://cdn.pixabay.com/video/2020/07/03/43336-435160910_large.mp4"
st.markdown(f"""
    <style>
    #background-video {{ position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%; z-index: -1; filter: brightness(0.4) contrast(1.2); object-fit: cover; }}
    .stApp {{ background: transparent; color: white; }}
    [data-testid="stSidebar"] {{ background-color: rgba(0, 0, 0, 0.9) !important; backdrop-filter: blur(20px); border-right: 2px solid {LIME_VIBE}; }}
    .stChatMessage, .context-card, .stTabs, div[data-baseweb="tab-panel"] {{ background: rgba(0, 0, 0, 0.6) !important; backdrop-filter: blur(25px); border: 1px solid rgba(57, 255, 20, 0.2); border-radius: 20px; padding: 25px; margin-bottom: 20px; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {LIME_VIBE}; }}
    h1, h2, h3 {{ color: {LIME_VIBE} !important; text-transform: uppercase; letter-spacing: 4px; font-weight: 900; }}
    .main .block-container {{ max-width: 98% !important; padding: 1rem 2rem !important; }}
    .stButton>button:contains('NUKE') {{ background-color: {DANGER_RED}; color: white; }}
    </style>
    <video autoplay muted loop id="background-video"><source src="{video_url}" type="video/mp4"></video>
    """, unsafe_allow_html=True)

# --- RESOURCE KERNEL ---
@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path="./memex_db")
    collection = client.get_or_create_collection(name="ace_memex_v4")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    graph_path = "./memex_db/knowledge_graph.json"
    if os.path.exists(graph_path):
        with open(graph_path, "r") as f:
            kg = nx.node_link_graph(json.load(f))
    else:
        kg = nx.DiGraph()
    return client, collection, model, kg

client, collection, model, kg = load_resources()

# --- ENGINE UTILITIES ---
def save_graph():
    with open("./memex_db/knowledge_graph.json", "w") as f:
        json.dump(nx.node_link_data(kg), f)

def nuke_system():
    """Wipes all vectors and the entire graph (Delete All)."""
    global kg
    all_docs = collection.get()
    if all_docs['ids']: collection.delete(ids=all_docs['ids'])
    kg.clear()
    save_graph()
    st.toast("SYSTEM PURGED. All files deleted.", icon="☢️")

def delete_from_memex(source_name):
    """Prunes a single specific file from Vectors and Graph (Selective Delete)."""
    collection.delete(where={"source": source_name})
    to_remove = [n for n, d in kg.nodes(data=True) if d.get('source') == source_name]
    kg.remove_nodes_from(to_remove)
    # Aggressively remove orphaned concepts
    orphans = [n for n, d in kg.nodes(data=True) if kg.degree(n) == 0 and d.get('type') == 'CONCEPT']
    kg.remove_nodes_from(orphans)
    save_graph()
    st.toast(f"Deleted: {source_name}", icon="🗑️")

def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def extract_entities(groq_client, text):
    """Bulletproof JSON Parser for 8b Model (Prevents Dict/String crash)."""
    try:
        prompt = f"Extract 3 technical concepts. Return ONLY a JSON object with one key 'entities' containing a list of strings. Example: {{\"entities\": [\"Concept 1\", \"Concept 2\"]}}\nText: {text[:800]}"
        res = groq_client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        raw_entities = json.loads(res.choices[0].message.content).get("entities", [])
        
        clean_entities = []
        for item in raw_entities:
            # Force everything into a string to prevent NetworkX crashing
            if isinstance(item, dict):
                clean_entities.append(str(list(item.values())[0]))
            else:
                clean_entities.append(str(item))
        return clean_entities
    except: 
        return []

def add_to_memex(doc_id, text, source, groq_client=None):
    chunks = chunk_text(text)
    with st.container():
        st.write(f"🧬 **Indexing:** {source}")
        bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            c_id = f"{source}_{i}"
            collection.add(ids=[c_id], embeddings=[model.encode(chunk).tolist()], documents=[chunk], metadatas=[{'source': source}])
            kg.add_node(c_id, type="CHUNK", source=source)
            if groq_client:
                for ent in extract_entities(groq_client, chunk):
                    kg.add_node(ent, type="CONCEPT")
                    kg.add_edge(ent, c_id)
            bar.progress((i + 1) / len(chunks))
        save_graph()
        st.success(f"✅ {source} Online.")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ ACE CORE")
    api_key = st.text_input("Groq API Key:", type="password")
    groq_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1") if api_key else None

    st.markdown("### 💬 Sessions")
    if st.button("➕ NEW SESSION", use_container_width=True):
        nid = f"Session {len(st.session_state.get('chats', {})) + 1}"
        st.session_state.chats[nid] = []
        st.session_state.current_chat = nid
        st.rerun()

    if "chats" not in st.session_state: st.session_state.chats = {"Default Session": []}
    if "current_chat" not in st.session_state: st.session_state.current_chat = "Default Session"
    
    chats = list(st.session_state.chats.keys())
    sel = st.radio("History", chats, index=chats.index(st.session_state.current_chat), label_visibility="collapsed")
    if sel != st.session_state.current_chat:
        st.session_state.current_chat = sel
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Telemetry")
    c1, c2 = st.columns(2)
    c1.metric("Nodes", f"{kg.number_of_nodes()}")
    c2.metric("Vectors", f"{collection.count()}")

    st.markdown("---")
    st.subheader("📚 Memex Inventory")
    all_items = collection.get()
    if all_items['ids']:
        unique_sources = sorted(set([meta['source'] for meta in all_items['metadatas']]))
        for s in unique_sources:
            col_x, col_name = st.columns([0.2, 0.8])
            # SELECTIVE DELETE OPTION
            if col_x.button("❌", key=f"del_{s}"):
                delete_from_memex(s)
                st.rerun()
            col_name.caption(s)
    else:
        st.caption("Memex Empty.")

    st.markdown("---")
    # NUKE ALL OPTION
    if st.button("☢️ NUKE SYSTEM (DELETE ALL)", use_container_width=True):
        nuke_system()
        st.rerun()

# --- MAIN COMMAND CENTER ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📝 Paste", "📄 Upload", "🌌 Associative Galaxy"])

with tab1:
    st.subheader(f"📟 {st.session_state.current_chat}")
    history = st.session_state.chats[st.session_state.current_chat]
    for msg in history:
        with st.chat_message(msg["role"]):
            if msg.get("ctx"):
                with st.expander("🔗 Evidence"):
                    for d in msg["ctx"]: st.markdown(f'<div class="context-card">{d}</div>', unsafe_allow_html=True)
            st.write(msg["content"])

    query = st.chat_input("Query ACE...")
    if query and groq_client:
        history.append({"role": "user", "content": query})
        with st.spinner("Bridging..."):
            res = collection.query(query_embeddings=[model.encode(query).tolist()], n_results=10)
            docs = res['documents'][0] if res['documents'] else []
            
            # Strict RAG logic to prevent Hallucination
            prompt = f"""
            You are ACE, a strict Retrieval-Augmented Generation engine.
            CRITICAL: You MUST ONLY answer using the provided Context. 
            If the Context is empty, reply EXACTLY with: 'Memex is empty. I cannot answer.' Do not use pre-trained knowledge.
            
            Context: {' '.join(docs)}
            Query: {query}
            """
            ans = groq_client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
            history.append({"role": "assistant", "content": ans.choices[0].message.content, "ctx": docs})
            st.rerun()

with tab2:
    st.subheader("📝 Manual Data Entry (Paste)")
    s_title = st.text_input("Entry Label:")
    s_text = st.text_area("Source Text:", height=300)
    if st.button("Index Snippet") and s_title and s_text and groq_client:
        add_to_memex(s_title, s_text, f"Snippet: {s_title}", groq_client)

with tab3:
    st.subheader("📄 Document Ingestion (Upload)")
    up = st.file_uploader("Upload PDF", type="pdf")
    if st.button("Process Document") and up and groq_client:
        reader = PyPDF2.PdfReader(up)
        full = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
        add_to_memex(up.name, full, up.name, groq_client)

with tab4:
    st.subheader("🌌 Associative Galaxy Map")
    if kg.number_of_nodes() > 0:
        nodes = [Node(id=n, label=n, color=LIME_VIBE, size=35) for n in list(kg.nodes())]
        edges = [Edge(source=u, target=v) for u, v in list(kg.edges())]
        config = Config(width=2000, height=1200, directed=True, nodeHighlightBehavior=True, highlightColor=DANGER_RED, staticGraph=False)
        agraph(nodes=nodes, edges=edges, config=config)
    else: st.info("Galaxy offline. Ingest data to initialize.")