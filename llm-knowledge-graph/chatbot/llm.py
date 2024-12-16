import os
from dotenv import load_dotenv
load_dotenv()

# tag::llm[]
# Create the LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",  # URL do servidor API do LM Studio
    model_name="multi-qa-MiniLM-L6-cos-v1",  # Nome do modelo configurado no LM Studio
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0.0,  
)
# end::llm[]

# tag::embedding[]
# Create the Embedding model
from langchain_openai import OpenAIEmbeddings

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

# end::embedding[]
