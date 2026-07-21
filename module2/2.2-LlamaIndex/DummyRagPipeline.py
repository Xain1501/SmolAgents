# general imports
import os
from dotenv import load_dotenv

# loader imports
from llama_index.core import SimpleDirectoryReader

# chunking imports
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# vector store and chroma db imports
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# vector store and embeddings imports
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# query imports
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# evaluation imports
from llama_index.core.evaluation import FaithfulnessEvaluator





# Load the .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Retrieve HF_TOKEN from the environment variables
hf_token = os.getenv("HF_TOKEN")

# trying out the SimpleDirectoryReader to read dummy data
reader = SimpleDirectoryReader(input_dir=r"C:\DESKTOP STUFF\Programming and Uni Shit here\Programming and Projects\Ai Agent\SmolAgent-HuggingFaceCourse\module2\2.2-LlamaIndex\DummyDirectory")

documents = reader.load_data()

# creating persistentchroma db vector database  for storing embeddings and vector space

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# ingestion pipeline turning the data into chunks/nodes
# using tools like text splitter and hugging face embeddinggs

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)

nodes = pipeline.run(documents=documents)

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto",
)

# embedding the vector space
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

# query engine

query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)


# evaluation <checking if model hallucinates out of provided document context>

evaluator = FaithfulnessEvaluator(llm=llm)
response = query_engine.query(
    "What is the meaning of being a hero?"
)

print(response)
eval_result = evaluator.evaluate_response(response=response)

print(eval_result.passing)
print(eval_result.score)
print(eval_result.feedback)

#observability

