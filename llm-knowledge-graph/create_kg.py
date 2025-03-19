import os
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import login

# Carregar variáveis de ambiente
load_dotenv()

# Autenticação na Hugging Face
login(token=os.getenv('HUGGINGFACE_TOKEN'))

# Carregar CSV
df = pd.read_csv("llm-knowledge-graph/data/base-problemas - enunciados e resoluções.csv")

# Carregar modelo de embeddings
embedding_provider = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Carregar modelo de geração de texto
generation_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
generation_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", attn_implementation='eager')

# Criar pipeline
txt_pipeline = pipeline(
    "text-generation",
    model=generation_model,
    tokenizer=generation_tokenizer,
    device=0,  # Use -1 para CPU
    max_new_tokens=200
)

# Configurar LLM para LangChain
llm = HuggingFacePipeline(pipeline=txt_pipeline)

# Conectar ao Neo4j
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

# Criar transformador de grafo
doc_transformer = LLMGraphTransformer(llm=llm)

# Criar entidades e relacionamentos
for index, row in df.iterrows():
    enunciado = row["enunciado"]
    resolucao = row["resolucao"]

    enunciado_id = f"Enunciado_{index}"
    resolucao_id = f"Resolucao_{index}"

    # Criar embeddings
    enunciado_embedding = embedding_provider.embed_query(enunciado)
    resolucao_embedding = embedding_provider.embed_query(resolucao)

    # Criar nós no Neo4j
    graph.query(
        """
        MERGE (e:Enunciado {id: $enunciado_id})
        SET e.text = $enunciado, e.embedding = $enunciado_embedding
        MERGE (r:Resolucao {id: $resolucao_id})
        SET r.text = $resolucao, r.embedding = $resolucao_embedding
        MERGE (e)-[:TEM_RESOLUCAO]->(r)
        """,
        {
            "enunciado_id": enunciado_id,
            "enunciado": enunciado,
            "enunciado_embedding": enunciado_embedding,
            "resolucao_id": resolucao_id,
            "resolucao": resolucao,
            "resolucao_embedding": resolucao_embedding
        }
    )

# Criar índice vetorial para busca
graph.query(
    """
    CREATE VECTOR INDEX `enunciadoVector`
    IF NOT EXISTS
    FOR (e:Enunciado) ON (e.embedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
    }};
    """
)

graph.query(
    """
    CREATE VECTOR INDEX `resolucaoVector`
    IF NOT EXISTS
    FOR (r:Resolucao) ON (r.embedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
    }};
    """
)

print("Knowledge Graph criado com sucesso!")
