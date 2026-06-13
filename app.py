import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document

# =========================
# 基本設定
# =========================
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="中油工安 AI 助理 V10-3",
    page_icon="🏭",
    layout="wide"
)

# =========================
# RAG 載入
# =========================
@st.cache_resource
def load_rag():
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="safety_knowledge")
    return embedding_model, collection


embedding_model, collection = load_rag()


# =========================
# 文件讀取
# =========================
def read_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def read_word(file):
    doc = Document(file)
    text = ""

    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    return text


# =========================
# 知識庫寫入
# =========================
def add_text_to_knowledge_base(text, source_name):
    chunks = []

    chunk_size = 800
    overlap = 100

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    if not chunks:
        return 0

    embeddings = embedding_model.encode(chunks).tolist()

    ids = [
        f"{source_name}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {"source": source_name}
        for _ in chunks
    ]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    return len(chunks)


# =========================
# 知識庫查詢
# =========================
def search_knowledge_base(query, n_results=5):
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_text = ""
    sources = []

    for i, doc in enumerate(documents):
        source = metadatas[i].get("source", "未知來源") if i < len(metadatas) else "未知來源"
        context_text += f"\n【來源：{source}】\n{doc}\n"
        sources.append(source)

    return context_text, list(set(sources))


# =========================
# Agent 設定
# =========================
AGENTS = {
    "工安專家 Agent": {
        "role": "你是工業安全衛生專家，熟悉職業安全衛生法規、製程安全、現場風險辨識、承攬商管理與事故預防。",
        "goal": "協助使用者進行工安風險分析、改善建議、稽核準備與安全管理文件整理。",
        "style": "正式、專業、條列清楚，適合公司內部使用。"
    },
    "承攬商管理 Agent": {
        "role": "你是承攬商安全管理專家，熟悉承攬商入廠管理、教育訓練、保險、危害告知與工作許可流程。",
        "goal": "協助檢視承攬商文件、找出管理缺失，並提出改善建議。",
        "style": "務實、明確、可直接用於稽核與公文回覆。"
    },
    "工作許可 Agent": {
        "role": "你是工作許可審查專家，熟悉動火、高架、局限空間、吊掛、開挖及一般作業許可。",
        "goal": "協助檢查工作許可內容是否完整，辨識作業風險與管制措施。",
        "style": "以檢核表與條列方式回答，重視現場可執行性。"
    },
    "PSM Agent": {
        "role": "你是製程安全管理 PSM 專家，熟悉 PSI、PHA、MOC、MI、事故調查與教育訓練。",
        "goal": "協助整理 PSM 查核重點、缺失回覆與改善追蹤內容。",
        "style": "邏輯清楚、偏稽核與制度管理角度。"
    },
    "儲槽開放檢查 Agent": {
        "role": "你是儲槽開放檢查與非破壞檢測專家，熟悉儲槽內外部檢查、腐蝕、厚度量測、PT、MT、UT、真空試漏。",
        "goal": "協助整理儲槽檢查計畫、檢查結果、異常分析與改善建議。",
        "style": "技術性、條列式、可用於檢查報告與公文。"
    },
    "危險性機械設備 Agent": {
        "role": "你是危險性機械及設備管理專家，熟悉鍋爐、壓力容器、起重機、升降機與定期檢查制度。",
        "goal": "協助整理設備清冊、檢查狀態、法定檢查與異常追蹤。",
        "style": "精準、表格化、適合設備管理。"
    },
    "公文撰寫 Agent": {
        "role": "你是公司內部公文與簽辦文件撰寫專家，熟悉主旨、說明、辦法、簽核語氣與正式用語。",
        "goal": "協助將口語或草稿改寫成正式公文、通知、簽辦或電子郵件。",
        "style": "正式、精簡、符合公司行政文件語氣。"
    },
    "稽核缺失回覆 Agent": {
        "role": "你是稽核缺失回覆與矯正預防措施撰寫專家，熟悉缺失原因分析、改善措施、佐證資料與追蹤管理。",
        "goal": "協助撰寫稽核回覆、改善說明、原因分析與預防再發措施。",
        "style": "客觀、穩健、避免過度承認責任，適合稽核回覆。"
    },
    "公司內部知識庫 Agent": {
        "role": "你是台灣中油公司內部知識庫專家，熟悉公司內部規章、SOP、工安制度、稽核資料、會議紀錄、設備檢查文件與教育訓練資料。",
        "goal": "根據公司內部知識庫內容，協助使用者快速查詢、摘要、比對與整理內部文件重點。",
        "style": "正式、清楚、條列化。若知識庫沒有資料，必須明確說明「目前知識庫未找到相關依據」，不可自行編造。"
    }
}


# =========================
# Task 設定
# =========================
TASKS = {
    "一般問答": "請依照 Agent 專業身份回答使用者問題。",
    "風險分析": "請分析可能危害、風險等級、原因與改善建議。",
    "文件摘要": "請摘要文件重點，整理成條列式重點。",
    "稽核缺失回覆": "請協助撰寫稽核缺失原因分析、改善措施與預防再發措施。",
    "公文撰寫": "請協助撰寫正式公文主旨、說明與辦法。",
    "檢查表產生": "請協助產生可執行的檢查表或查核表。",
    "查詢公司內部知識庫": "請根據公司內部知識庫內容回答使用者問題，並整理成條列重點。",
    "摘要內部文件重點": "請根據公司內部知識庫內容，摘要相關文件的重點、規定、注意事項與可執行建議。",
    "比對內部規定差異": "請根據公司內部知識庫內容，比對不同文件、規定或版本之間的差異。",
    "整理內部作業依據": "請根據公司內部知識庫內容，整理可作為公文、稽核回覆或工作說明的依據。"
}


# =========================
# GPT 回答
# =========================
def ask_gpt(selected_agent, selected_task, user_question, context_text):
    agent_info = AGENTS[selected_agent]
    task_instruction = TASKS[selected_task]

    system_prompt = f"""
你現在扮演的 Agent 是：{selected_agent}

Agent 身份：
{agent_info["role"]}

Agent 目標：
{agent_info["goal"]}

回答風格：
{agent_info["style"]}

目前執行的 Task：
{selected_task}

Task 說明：
{task_instruction}

重要規則：
1. 必須優先依據知識庫內容回答。
2. 不可編造公司內部規定、法規、SOP 或不存在的文件內容。
3. 如果知識庫沒有找到明確依據，請回答：「目前知識庫未找到相關依據」。
4. 回答要適合公司內部使用，可用於工安、稽核、公文、設備管理或作業說明。
5. 回答請盡量使用條列式、表格化、正式語氣。
"""

    user_prompt = f"""
以下是知識庫查詢到的內容：

{context_text}

使用者問題：
{user_question}

請依據以上知識庫內容回答。
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# =========================
# Streamlit 介面
# =========================
st.title("🏭 中油工安 AI 助理 V10-3")
st.caption("Agent = 專家身份｜Task = 工作模式｜新增：公司內部知識庫 Agent")

tab1, tab2 = st.tabs(["🤖 AI Agent 問答", "📚 上傳內部文件到知識庫"])


# =========================
# Tab 1：AI Agent 問答
# =========================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        selected_agent = st.selectbox(
            "選擇 Agent 專家身份",
            list(AGENTS.keys())
        )

    with col2:
        selected_task = st.selectbox(
            "選擇 Task 工作模式",
            list(TASKS.keys())
        )

    user_question = st.text_area(
        "請輸入你的問題",
        height=160,
        placeholder="例如：請查詢公司內部知識庫，整理承攬商入廠安全衛生講習的重點。"
    )

    if st.button("開始分析", type="primary"):
        if not user_question.strip():
            st.warning("請先輸入問題。")
        else:
            with st.spinner("正在查詢知識庫並產生回答..."):
                context_text, sources = search_knowledge_base(user_question)

                if not context_text.strip():
                    st.error("目前知識庫未找到相關依據。")
                else:
                    answer = ask_gpt(
                        selected_agent,
                        selected_task,
                        user_question,
                        context_text
                    )

                    st.subheader("✅ AI 回答")
                    st.write(answer)

                    st.subheader("📌 引用來源")
                    for source in sources:
                        st.write(f"- {source}")


# =========================
# Tab 2：上傳內部文件
# =========================
with tab2:
    st.subheader("📚 上傳公司內部文件到知識庫")
    st.caption("支援 PDF、Word。適合放入 SOP、會議資料、稽核資料、工安制度、設備檢查文件。")

    uploaded_files = st.file_uploader(
        "請上傳 PDF 或 Word 文件",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if st.button("寫入知識庫"):
        if not uploaded_files:
            st.warning("請先上傳文件。")
        else:
            total_chunks = 0

            with st.spinner("正在解析文件並寫入 ChromaDB 知識庫..."):
                for file in uploaded_files:
                    file_name = file.name

                    if file_name.lower().endswith(".pdf"):
                        text = read_pdf(file)
                    elif file_name.lower().endswith(".docx"):
                        text = read_word(file)
                    else:
                        text = ""

                    if text.strip():
                        chunk_count = add_text_to_knowledge_base(text, file_name)
                        total_chunks += chunk_count
                        st.success(f"{file_name} 已寫入知識庫，共 {chunk_count} 筆片段。")
                    else:
                        st.warning(f"{file_name} 無法讀取文字內容。")

            st.info(f"本次共寫入 {total_chunks} 筆知識片段。")