from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os

# 載入Embedding模型
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# ChromaDB
client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="safety_knowledge"
)

# PDF資料夾
pdf_folder = "knowledge_base"

doc_count = 0

for filename in os.listdir(pdf_folder):

    if filename.endswith(".pdf"):

        filepath = os.path.join(pdf_folder, filename)

        print(f"讀取中：{filename}")

        reader = PdfReader(filepath)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        # 每500字切一段
        chunks = []

        chunk_size = 500

        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])

        for idx, chunk in enumerate(chunks):

            embedding = model.encode(chunk).tolist()

            collection.add(
                ids=[f"{filename}_{idx}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": filename}]
            )

            doc_count += 1

print(f"完成，共建立 {doc_count} 筆知識")