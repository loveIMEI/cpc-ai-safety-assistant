from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

client_openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Embedding模型
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# ChromaDB
client_db = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client_db.get_collection(
    name="safety_knowledge"
)

question = input("請輸入問題：")

# 轉向量
query_embedding = model.encode(
    question
).tolist()

# 搜尋
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

context = "\n\n".join(
    results["documents"][0]
)

prompt = f"""
你是台灣中油工安知識庫助理。

請根據以下文件內容回答問題。

【文件內容】

{context}

【問題】

{question}

回答規則：
1. 使用繁體中文
2. 只根據文件內容回答
3. 若文件沒有答案，請明確說明
4. 先給結論，再給說明
"""

response = client_openai.responses.create(
    model="gpt-5-mini",
    input=prompt
)

print("\n")
print("=" * 60)
print("AI回答")
print("=" * 60)
print("\n")

print(response.output_text)