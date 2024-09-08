from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from load_llama2 import OurLLM

documents = SimpleDirectoryReader(
    input_files=["../data/Henry.txt"]
).load_data()

local_model_path = "E:/pythonProject_RAG/Retrieval-Augmented-Generation-Intro-Project2/bge-base-zh-v1.5"
Settings.embed_model = HuggingFaceEmbedding(model_name=local_model_path)

# 创建 VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

Settings.llm = OurLLM()

query_engine = index.as_query_engine()

response = query_engine.query(
    "Who is the pretty boy in Hong Kong?"
)
print(str(response))

#scp -r E:\pythonProject_RAG guoyimo@192.168.2.203:/data-extend/guoyimo