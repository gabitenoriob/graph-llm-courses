import os

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

DOCS_PATH = "llm-knowledge-graph\\data\\dados"

llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",  # URL do servidor API do LM Studio
    model_name="multi-qa-MiniLM-L6-cos-v1",  # Nome do modelo configurado no LM Studio
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0.0,  
)

# Fazer uma consulta ao modelo
response = llm.predict("Explique a importância dos embeddings no NLP.")
print(response)

embedding_provider = OpenAIEmbeddings(
    openai_api_base="http://localhost:1234/v1",  # URL base do LM Studio
    model="multi-qa-MiniLM-L6-cos-v1",  # Substitua pelo modelo configurado no LM Studio
    openai_api_key=os.getenv('OPENAI_API_KEY'),  # Não usado, mas necessário
)

# embedding_provider = HuggingFaceEmbeddings(
#     model_name="multi-qa-MiniLM-L6-cos-v1",
#     model_kwargs={"device": "cpu"}  # ou "cuda" se você estiver usando GPU
# )


# Exemplo: Gerar embeddings para uma query
query = "Como os embeddings ajudam na recuperação de informações?"
vector = embedding_provider.embed_query(query)
print(vector)


graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

doc_transformer = LLMGraphTransformer(
    llm=llm,
    )

# Load and split the documents
loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

docs = loader.load()
chunks = text_splitter.split_documents(docs)

for chunk in chunks:

    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f"{filename}.{chunk.metadata['page']}"
    print("Processing -", chunk_id)

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }
    
    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, 
        properties
    )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        for node in graph_doc.nodes:

            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node, 
                    type="HAS_ENTITY"
                    )
                )

    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)

# Create the vector index
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
    }};""")
