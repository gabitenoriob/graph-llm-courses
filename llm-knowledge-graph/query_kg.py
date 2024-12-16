import os
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",  # URL do servidor API do LM Studio
    model_name="multi-qa-MiniLM-L6-cos-v1",  # Nome do modelo configurado no LM Studio
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0.0,  
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

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

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    enhanced_schema=True, #Mais precisão
)

def run_cypher(q):
    return cypher_chain.invoke({"query": q})

while True:
    q = input("> ")
    print(run_cypher(q))