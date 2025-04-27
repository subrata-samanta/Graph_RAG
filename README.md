# Graph RAG with Neo4j and LangChain 🚀

This project implements a Graph-based Retrieval Augmented Generation (RAG) system using Neo4j, LangChain, and various LLM techniques to analyze and query Apple Inc.'s 2024 annual report data. 📊

## Project Overview 🎯

This project combines traditional vector-based retrieval with graph-based knowledge extraction to provide more accurate and contextual answers to queries about Apple's financial and business information. It uses a hybrid approach that leverages both structured (graph) and unstructured (vector) data for comprehensive information retrieval.

## Features ✨

- **Document Processing**: 📄
  - PDF document ingestion and chunking
  - Recursive text splitting with configurable chunk sizes
  - Graph document transformation using LLMGraphTransformer

- **Knowledge Graph Construction**: 🕸️
  - Neo4j graph database integration
  - Entity extraction and relationship mapping
  - Graph document storage with source tracking

- **Vector Store Integration**: 🔍
  - HuggingFace embeddings integration
  - Hybrid search capabilities
  - Vector indexing for efficient retrieval

- **Query Processing**: 💡
  - Natural language query understanding
  - Entity extraction from queries
  - Context-aware response generation
  - Chat history management

## Technical Architecture 🏗️

### Core Components 🔧

1. **Document Processing Pipeline**
   - PyPDFLoader for PDF ingestion
   - RecursiveCharacterTextSplitter for text chunking
   - LLMGraphTransformer for graph document creation

2. **Storage Layer**
   - Neo4j Graph Database for structured data
   - Vector store for embeddings
   - Pickle-based document caching

3. **Retrieval System**
   - Hybrid retrieval combining graph and vector search
   - Entity-aware structured queries
   - Context-based response generation

### Key Technologies 💻

- **Databases**: Neo4j 🗄️
- **Embeddings**: HuggingFace (sentence-transformers/all-mpnet-base-v2) 🤗
- **LLM Integration**: Groq 🧠
- **Framework**: LangChain ⚡
- **Development**: Python, Jupyter Notebook 🐍

## Setup Requirements 🛠️

1. **Environment Variables**:
   ```
   AURA_INSTANCENAME=<your-neo4j-instance>
   NEO4J_URI=<your-neo4j-uri>
   NEO4J_USERNAME=<your-username>
   NEO4J_PASSWORD=<your-password>
   OPENAI_API_KEY=<your-api-key>
   ```

2. **Python Dependencies**: 📦
   - langchain
   - neo4j
   - groq
   - huggingface_hub
   - python-dotenv
   - pypdf
   - pydantic

## Usage 🚀

1. **Initialize Environment**:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

2. **Process Documents**:
   ```python
   raw_documents = PyPDFLoader("Annual Report/NASDAQ_AAPL_2024.pdf").load()
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
   documents = text_splitter.split_documents(raw_documents)
   ```

3. **Create Graph Documents**:
   ```python
   llm_transformer = LLMGraphTransformer(llm=chat)
   graph_documents = process_documents(documents)
   ```

4. **Query the System**:
   ```python
   response = chain.invoke({
       "question": "Your question here",
       "chat_history": []
   })
   ```

## Example Queries 💭

The system can handle complex queries about:
- 📈 Capital expenditure evolution
- ⚠️ Emerging technology risks
- 🌱 Sustainability reporting
- 🌍 Geographical revenue diversification
- 💰 Debt profile analysis

## Project Structure 📁

```
Graph_RAG/
├── graph_documents.pkl       # Cached graph documents
├── graph_rag.ipynb          # Main notebook
├── README.md               # This file
├── Annual Report/          # Source documents
│   └── NASDAQ_AAPL_2024.pdf
└── __pycache__/           # Python cache
```

## Best Practices 🌟

1. **Document Processing**: 📝
   - Use appropriate chunk sizes based on document content
   - Maintain overlap between chunks for context preservation

2. **Query Formation**: 🔍
   - Be specific with questions
   - Provide context when needed
   - Utilize chat history for follow-up questions

3. **System Maintenance**: ⚙️
   - Regularly update the knowledge graph
   - Monitor embedding quality
   - Validate response accuracy

## Contributing 🤝

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Submitting a pull request

## License 📜

This project is licensed under MIT License.

## Acknowledgments 🙏

- Neo4j team for graph database capabilities
- LangChain community for the framework
- HuggingFace for embeddings model
