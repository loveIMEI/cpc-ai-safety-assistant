from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(name="safety_knowledge")

question = input("請輸入問題：")

query_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\n找到的相關內容：\n")

for i, doc in enumerate(results["documents"][0]):
    source = results["metadatas"][0][i]["source"]
    print(f"【來源：{source}】")
    print(doc)
    print("-" * 50)