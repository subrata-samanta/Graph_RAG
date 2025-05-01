
"""
### Environment Setup and Imports

This section initializes the environment and imports the necessary modules and classes for the notebook. It includes loading environment variables, importing libraries for working with Neo4j, LangChain, and other utilities for processing and querying data.

The following key imports and initializations are performed:
- `load_dotenv`: Loads environment variables from a `.env` file.
- `Neo4jGraph` and `Neo4jVector`: For interacting with Neo4j databases and vector stores.
- LangChain components such as `RunnableBranch`, `RunnableLambda`, `ChatPromptTemplate`, and `LLMGraphTransformer`.
- `ChatGroq` and `ChatOpenAI`: For using Groq and OpenAI models.
- Other utilities like `WikipediaLoader`, `TokenTextSplitter`, and `PyPDFLoader` for document processing.

Additionally, the `ChatGroq` instance is initialized with specific parameters for temperature and model name.
"""

from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_neo4j import Neo4jVector

from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

chat = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

"""### Environment Variables Setup for Neo4j

This section defines and initializes the necessary environment variables for the notebook. These variables are used to configure connections to external services such as Neo4j and OpenAI.

The following environment variables are set:
- `AURA_INSTANCENAME`: The name of the Neo4j Aura instance.
- `NEO4J_URI`: The URI for connecting to the Neo4j database.
- `NEO4J_USERNAME`: The username for authenticating with the Neo4j database.
- `NEO4J_PASSWORD`: The password for authenticating with the Neo4j database.
- `AUTH`: A tuple containing the Neo4j username and password.
- `OPENAI_API_KEY`: The API key for accessing OpenAI services.

"""

AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
) #database=NEO4J_DATABASE,

"""### Data Processing and Document Loading

This section processes the Annual Report data from a PDF file. It performs the following steps:

1. Loads the NASDAQ AAPL 2024 Annual Report using PyPDFLoader
2. Splits the document into manageable chunks using RecursiveCharacterTextSplitter with:
    - Chunk size: 1200 characters
    - Overlap: 200 characters
3. Prints the total number of document chunks created

The processed documents will be used for further analysis and querying.

"""

raw_documents = PyPDFLoader(r"Annual Report\NASDAQ_AAPL_2024.pdf").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)
print(len(documents))

"""## Below cell performs document processing and graph transformation using LangChain's LLMGraphTransformer. Here's what the code does:

1. Imports necessary libraries:
    - ThreadPoolExecutor for parallel processing
    - tqdm for progress tracking
    - pickle for serialization

2. Initializes LLMGraphTransformer with the chat model

3. Defines a helper function `process_document` that converts a single document to graph format

4. Processes documents in parallel batches of 100 using ThreadPoolExecutor:
    - Submits each document in the batch for processing
    - Collects results and extends the graph_documents list
    - Shows progress with tqdm

5. Saves the processed graph documents to a pickle file for later use

The processing leverages multithreading to speed up the graph transformation of the documents while providing visual progress feedback.

"""

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle

llm_transformer = LLMGraphTransformer(llm=chat)

# Function to process a single document
def process_document(doc):
    return llm_transformer.convert_to_graph_documents([doc])


graph_documents = []
with ThreadPoolExecutor() as executor:
    with tqdm(total=len(documents), desc="Processing Documents") as pbar:
        for i in range(0, len(documents), 100):  # Batch size of 100
            batch = documents[i:i+100]
            futures = {executor.submit(process_document, doc): doc for doc in batch}
            for future in futures:
                result = future.result()
                graph_documents.extend(result)
                pbar.update(1)

# Save graph_documents to a pickle file
with open("graph_documents.pkl", "wb") as f:
    pickle.dump(graph_documents, f)

"""### Load Graph Documents

This cell loads the previously processed graph documents from a pickle file and displays the total number of loaded documents. These documents contain the structured graph representation of the annual report data.

"""

import pickle
with open("graph_documents.pkl", "rb") as f:
    graph_documents = pickle.load(f)

print(f"Loaded {len(graph_documents)} graph documents.")

"""### Neo4j Graph Storage

This cell stores the processed graph documents to the Neo4j database. The following operations are performed:

1. Uses the Neo4jGraph instance (`kg`) to add the graph documents
2. Includes source information for traceability
3. Uses base entity labels for node classification

The documents contain structured information extracted from the annual report that will be used for graph-based querying and analysis.
"""

# # store to neo4j
res = kg.add_graph_documents(
    graph_documents,
    include_source=True,
    baseEntityLabel=True,
)

"""### Vector Store Creation

This cell creates a vector store in Neo4j using HuggingFace embeddings:

1. Initializes HuggingFaceEmbeddings with:
    - Model: sentence-transformers/all-mpnet-base-v2
    - Device: CPU
    - Normalize embeddings: False

2. Creates Neo4jVector index from existing graph with:
    - Search type: hybrid
    - Node label: Document
    - Text properties: text
    - Embedding property: embedding

"""

from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# create vector index
vector_index = Neo4jVector.from_existing_graph(
    hf,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)

"""### Entity Extraction from Text

This cell defines a class `Entities` to extract organization and person entities from a given text. It uses a `ChatPromptTemplate` to structure the prompt for entity extraction and invokes the `entity_chain` to process the input question. The extracted entities are printed as the output.

The following steps are performed:
1. Define the `Entities` class to structure the extracted entity information.
2. Create a `ChatPromptTemplate` to guide the entity extraction process.
3. Use the `entity_chain` to extract entities from the input question.
4. Print the extracted entities.

"""

class Entities(BaseModel):
    """Entity information extracted from text."""
    names: List[str] = Field(
        description="List of person, organization, or business entities appearing in the text"
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract organization and person etc entities from the text."),
    ("human", "Extract all the entities from the following input: {question}")
])

entity_chain = prompt | chat.with_structured_output(Entities)

# Test entity extraction
entities = entity_chain.invoke({"question": "How has the company's capital expenditure (CapEx) evolved over the past five years, and what strategic initiatives have driven significant changes in CapEx allocations"})
print(entities.names)

"""The code at cell index 17 contains a function `generate_full_text_query` that processes input strings for full-text search in Neo4j. Here's an analysis:

The function:
1. Takes an input string
2. Cleans it using `remove_lucene_chars` (already imported)
3. Splits it into words
4. Adds fuzzy matching (~2) to allow for minor spelling variations
5. Combines words with AND operator

The function is used to help match entities from user questions to database values, with some tolerance for misspellings. The fuzzy matching allows for up to 2 character differences.

Sample input/output:
- Input: "artificial intelligence"
- Output: "artificial~2 AND intelligence~2"

This is particularly useful in the context of the notebook which handles queries about Apple's annual report, allowing for more flexible text matching in the Neo4j database.

"""

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    words = [w for w in remove_lucene_chars(input).split() if w]
    if not words:
        return ""
    return " AND ".join(f"{word}~2" for word in words)

"""### Structured Data Retrieval and Query Matching

This cell defines the `structured_retriever` function that handles querying the Neo4j knowledge graph. The function:

1. Takes a question string as input
2. Uses entity extraction to identify relevant entities
3. Generates Cypher queries using case-insensitive pattern matching
4. Returns formatted query results showing entity relationships

The function is a key component of the RAG (Retrieval Augmented Generation) pipeline, working alongside vector search to provide comprehensive answers.

"""

def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        print(f" Getting Entity: {entity}")
        # Using standard pattern matching instead of fulltext search
        response = kg.query(
            """
            MATCH (node)
            WHERE node.name =~ $query
            OR node.id =~ $query
            WITH node
            MATCH (node)-[r]->(neighbor)
            RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
            UNION ALL
            MATCH (node)<-[r]-(neighbor)
            WHERE node.name =~ $query
            OR node.id =~ $query
            RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            LIMIT 50
            """,
            {"query": f"(?i).*{entity}.*"}  # Case-insensitive pattern matching
        )
        result += "\n".join([el["output"] for el in response])
    return result

"""### Retrieval and Chain Processing

This section processes retrieval and chain operations for the Jupyter notebook. The retriever function performs the following steps:

1. Takes a question string as input
2. Performs structured retrieval using the knowledge graph
3. Executes vector similarity search for unstructured data
4. Combines both structured and unstructured data into a final response

The results are used by the RAG chain to provide comprehensive answers based on both graph relationships and document content.

"""

# Final retrieval step
def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [
        el.page_content for el in vector_index.similarity_search(question)
    ]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    print(f"\nFinal Data::: ==>{final_data}")
    return final_data

"""### RAG Pipeline Components and Configuration

This section describes the setup of the Retrieval-Augmented Generation (RAG) pipeline. The pipeline integrates the following key components:

1. Question Template:
    - Condenses chat history and follow-up questions into standalone queries
    - Uses a template format to maintain context across conversations

2. Chat History Formatter:
    - Converts chat history into structured message format
    - Maintains conversation flow for context-aware responses

3. Search Query Processing:
    - Handles queries with and without chat history
    - Implements conditional branching based on query context

4. Response Generation:
    - Combines retrieved context with user questions
    - Uses natural language processing for concise answers
    - Incorporates both structured and unstructured data sources


"""

# Define the RAG chain
# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

"""### Chat History Management and Query Processing

This section defines the core functionality for managing chat history and processing queries. It consists of:

1. Chat History Formatting:
    - Converts chat history tuples into message objects
    - Creates structured HumanMessage and AIMessage instances
    - Maintains conversation context

2. Search Query Processing Logic:
    - Handles queries with and without chat history
    - Implements conditional branching for different query types
    - Uses RunnableBranch for execution path selection
    - Condenses follow-up questions with chat history into standalone queries


"""

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | chat
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x: x["question"]),
)

"""### Response Generation Template and Chain Configuration

This section configures the response generation template and chain for processing queries. It includes:

1. Question-Answer Template:
    - Takes context and question as inputs
    - Structures responses for natural language output
    - Emphasizes concise answers based on provided context

2. Chain Configuration:
    - Combines context retrieval with question processing
    - Uses parallel processing for efficient response generation
    - Integrates chat model for natural language generation


"""

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | chat
    | StrOutputParser()
)

"""### Testing RAG for Complex Multi-Hop Question Answering

The following questions explore various aspects of Apple Inc.'s financial, technological, and operational performance based on their 2024 annual report:

1. Capital Expenditure Evolution and Strategic Initiatives
2. Emerging Technology Risks and Mitigation Strategies
3. Sustainability Reporting and ESG Outcomes
4. Geographical Revenue Diversification
5. Debt Profile Changes and Financial Implications

The responses demonstrate our RAG system's ability to extract and synthesize complex information from the annual report, providing detailed insights into each area of inquiry.
"""

res_hist = chain.invoke(
    {
        "question": "How has the company's capital expenditure (CapEx) evolved over the past five years, and what strategic initiatives have driven significant changes in CapEx allocations?",
        "chat_history": [

        ],
    }
)

print(f"\n === {res_hist}\n\n")

res_hist = chain.invoke(
    {
        "question": "What are the key risks identified by the company related to emerging technologies, such as artificial intelligence, and how has the company's risk mitigation strategy evolved in response to these technologies??",
        "chat_history": [

        ],
    }
)

print(f"\n === {res_hist}\n\n")

res_hist = chain.invoke(
    {
        "question": "How does the company's approach to sustainability reporting align with industry best practices, and what measurable outcomes have been achieved in environmental, social, and governance (ESG) areas over the past three years?",
        "chat_history": [

        ],
    }
)

print(f"\n === {res_hist}\n\n")

res_hist = chain.invoke(
    {
        "question": "What are the trends in the company's revenue diversification across different geographical regions, and how have geopolitical factors influenced these trends over the past five years?",
        "chat_history": [

        ],
    }
)

print(f"\n === {res_hist}\n\n")

res_hist = chain.invoke(
    {
        "question": "How has the company's debt profile changed over the last five years, and what implications does this have for its financial stability and investment capacity?",
        "chat_history": [

        ],
    }
)

print(f"\n === {res_hist}\n\n")
