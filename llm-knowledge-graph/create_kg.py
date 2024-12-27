from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
#from langchain.llms import HuggingFacePipeline
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship


from dotenv import load_dotenv
load_dotenv()

# Entrada
DOCS_PATH = "llm-knowledge-graph\\data\\dados"

# # Configurando o modelo LLM do LM Studio
# model_name = "multi-qa-MiniLM-L6-cos-v1"  

# # Carregar tokenizer e modelo localmente
# #tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name, proxies={'https': ''}, verify=False)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Caminho local
embedding_model_path = Path(r'C:\\Users\\gabrielabtn\\.cache\\huggingface\\hub\\multi-qa-MiniLM-L6-cos-v1')

# Verificar se o diretório local existe e contém arquivos
if embedding_model_path.exists() and any(embedding_model_path.iterdir()):
    embedding_tokenizer = AutoTokenizer.from_pretrained(str(embedding_model_path))
    embedding_model = AutoModelForSequenceClassification.from_pretrained(str(embedding_model_path))
else:
    # Caso não exista ou esteja vazio, use o repo_id para baixar
    repo_id = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    embedding_tokenizer = AutoTokenizer.from_pretrained(repo_id)
    embedding_model = AutoModelForSequenceClassification.from_pretrained(repo_id)


# Configurando provedor de embeddings
embedding_provider = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs={"device": "cpu"}  # Altere para "cuda" se estiver usando GPU
)

# Gerar embeddings para uma query
query = "Como os embeddings ajudam na recuperação de informações?"
vector = embedding_provider.embed_query(query)
print("Vector gerado para a query:", vector)

# Caminho para o modelo de geração de texto 
generation_model_path = Path(r'C:\\Users\\gabrielabtn\\.cache\\huggingface\\hub\\Phi-3.5-mini-instruct')

# Verificar se o diretório local existe e contém arquivos
if generation_model_path.exists() and any(generation_model_path.iterdir()):
    generation_tokenizer = AutoTokenizer.from_pretrained(str(generation_model_path))
    generation_model = AutoModelForCausalLM.from_pretrained(str(generation_model_path))
else:
    # Caso não exista ou esteja vazio, use o repo_id para baixar
    # Load model directly
    generation_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
    generation_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)

# Criar pipeline para geração de texto
hf_pipeline = pipeline("text-generation", model=generation_model, tokenizer=generation_tokenizer, device=-1)  # CPU

# Configurando o LLM para LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Fazer uma consulta ao modelo como teste
response = llm.invoke("Explique a importância dos embeddings no NLP.")
print("Resposta do modelo:", response)

#Conexão com o BD
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

#Cria um transformador de grafo que usará o LLM para extrair entidades e relacionamentos do texto.
doc_transformer = LLMGraphTransformer(
    llm=llm,
    )

#Carrega arquivos PDF da pasta data usando o DirectoryLoader.
loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
     
#Divide o texto dos PDFs em chunks (blocos) menores usando o CharacterTextSplitter.
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

docs = loader.load()
chunks = text_splitter.split_documents(docs)


#Para cada chunk gera identificadores,cria embeddings,armazena o chunk no Neo4j
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

    #Extrai entidades e relacionamentos usando o LLM:
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    #Relaciona entidades ao chunk
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

#Cria um índice vetorial chamado chunkVector no Neo4j para nós Chunk
#Configura a similaridade vetorial como cosine com dimensões 384 (compatível com os embeddings gerados pelo modelo multi-qa-MiniLM-L6-cos-v1).
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
    }};""")
