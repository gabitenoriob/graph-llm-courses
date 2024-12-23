import os
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

#Inicializa LLM
llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",  # URL do servidor API do LM Studio
    model_name="multi-qa-MiniLM-L6-cos-v1",  # Nome do modelo configurado no LM Studio
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0.0,  
)

#Conexão com BD
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

#Configuração do prompt para geração de Cypher
CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Only include the generated Cypher statement in your response.

Always use case insensitive search when matching strings.

Schema:
{schema}

The question is:
{question}"""

#O PromptTemplate organiza as variáveis schema (estrutura do banco de dados) e question (pergunta em linguagem natural) para gerar o prompt que será usado pelo modelo.
cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

#Inicializa o GraphCypherQAChain
cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,#Ativa logs detalhados para depuração.
    enhanced_schema=True, #Mais precisão
)

# A pergunta é passada para o modelo de linguagem.
# O modelo gera um comando Cypher
# O comando é executado no Neo4j.
# Os resultados são retornados e exibidos no terminal.
def run_cypher(q):
    return cypher_chain.invoke({"query": q})

#Loop de interação com o usuário
while True:
    q = input("> ")
    print(run_cypher(q))