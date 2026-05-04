ACE: Associative Context Engine

ACE is a hybrid retrieval framework designed to bridge the structural gap in traditional RAG (Retrieval-Augmented Generation) pipelines. Inspired by Vannevar Bush’s 1945 vision of the Memex, ACE moves beyond simple semantic similarity by coupling a high-density vector database with a directed knowledge graph. This allows the system to surface "associative" connections between disparate data silos that standard search would otherwise miss.


🚀 Key Features

- Monolithic Architecture: The entire engine logic, ingestion pipeline, and interactive HUD are contained within a single app.py for streamlined deployment and execution.

- Dual-Pathway Retrieval: Synchronized lookup across ChromaDB (for semantic intuition) and NetworkX (for structural logic).

- Granular Sliding-Window Chunking: Utilizes 1500-character blocks with a 200-character "contextual glue" overlap to ensure technical details and equations remain coherent during vectorization.

- Associative Galaxy: A real-time, interactive 2D visualization of the knowledge graph rendered via streamlit-agraph, allowing users to explore the "connective tissue" of their data.

- Universal Ingestion (Upload & Paste): Supports batch PDF/TXT uploads alongside a "Paste" portal for immediate indexing of unstructured data like emails, Slack messages, or lecture discussion posts.


🛠️ Setup Instructions

1. Clone the Repository:


2. Install Dependencies:
Ensure you have Python installed, then run "pip install -r requirements.txt"


3. Run the Application:
streamlit run app.py


4. API Configuration:
Once the application launches in your browser, enter your Groq API Key in the sidebar. This key is required for the Llama-3.1-8b entity extraction and synthesis loops.


📂 File Structure

app.py: The core monolithic script handling backend logic, graph traversal, and the Streamlit UI.

requirements.txt: List of necessary Python libraries for the engine.

README.md: Project documentation and setup guide.