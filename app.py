import os
import time
import uuid
import json
import chromadb
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document


# =========================
# 基本設定
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

MEMORY_FILE = "memory_db.json"

st.set_page_config(
    page_title="中油工安 AI 助理 V12",
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
# 長期記憶系統
# =========================
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        data = {
            "cases": [],
            "equipments": [],
            "contractors": [],
            "audit_findings": [],
            "qa_history": []
        }
        save_memory(data)
        return data

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_memory(memory_type, title, content, tags=""):
    data = load_memory()

    item = {
        "id": str(uuid.uuid4())[:8],
        "title": title,
        "content": content,
        "tags": tags,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    data[memory_type].append(item)
    save_memory(data)
    return item


def search_memory(keyword):
    data = load_memory()
    results = []

    for memory_type, items in data.items():
        for item in items:
            text = f"{item.get('title', '')} {item.get('content', '')} {item.get('tags', '')}"
            if keyword.lower() in text.lower():
                results.append({
                    "type": memory_type,
                    **item
                })

    return results


def delete_memory(memory_type, memory_id):
    data = load_memory()
    data[memory_type] = [
        item for item in data[memory_type]
        if item["id"] != memory_id
    ]
    save_memory(data)


# =========================
# Session State
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "multi_agent_history" not in st.session_state:
    st.session_state.multi_agent_history = []


# =========================
# 文件讀取
# =========================
def read_pdf(file):
    text = ""
    reader = PdfReader(file)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def read_word(file):
    text = ""
    doc = Document(file)

    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    return text


def read_txt(file):
    return file.read().decode("utf-8", errors="ignore")


# =========================
# 文字切片
# =========================
def split_text(text, chunk_size=800, overlap=120):
    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


# =========================
# 知識庫寫入
# =========================
def add_text_to_knowledge_base(text, source_name, doc_type="內部文件"):
    chunks = split_text(text)

    if not chunks:
        return 0

    embeddings = embedding_model.encode(chunks).tolist()

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    batch_id = str(uuid.uuid4())[:8]

    ids = [
        f"{source_name}_{batch_id}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": source_name,
            "doc_type": doc_type,
            "created_at": now
        }
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

    if not documents:
        return "", []

    context_text = ""
    sources = []

    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        source = metadata.get("source", "未知來源")
        doc_type = metadata.get("doc_type", "未知類型")

        context_text += f"\n【來源：{source}｜類型：{doc_type}】\n{doc}\n"
        sources.append(source)

    return context_text, sorted(list(set(sources)))


# =========================
# Agent 設定
# =========================
AGENTS = {
    "🏭 工安專家 Agent": {
        "role": "你是工業安全衛生專家，熟悉職業安全衛生法規、製程風險、現場危害辨識、事故預防與安全管理制度。",
        "goal": "協助進行工安風險分析、改善建議、稽核準備、安全管理文件整理。",
        "style": "正式、專業、條列清楚，適合公司內部使用。"
    },
    "📋 工作許可 Agent": {
        "role": "你是工作許可審查專家，熟悉動火、高架、局限空間、吊掛、開挖、電氣及一般作業許可。",
        "goal": "協助檢查工作許可完整性，辨識風險與管制措施。",
        "style": "以檢核表、風險點、改善建議呈現，重視現場可執行性。"
    },
    "👷 承攬商管理 Agent": {
        "role": "你是承攬商安全管理專家，熟悉入廠管理、教育訓練、保險、危害告知、資格查驗與工作許可流程。",
        "goal": "協助檢視承攬商文件、找出管理缺失，提出改善建議。",
        "style": "務實、明確，可直接用於稽核與公文回覆。"
    },
    "⚙️ PSM Agent": {
        "role": "你是製程安全管理 PSM 專家，熟悉 PSI、PHA、MOC、MI、事故調查、教育訓練、承攬商管理與稽核追蹤。",
        "goal": "協助整理 PSM 查核重點、缺失回覆與改善追蹤內容。",
        "style": "邏輯清楚，偏制度管理與稽核角度。"
    },
    "🛢️ 儲槽開放檢查 Agent": {
        "role": "你是儲槽開放檢查與非破壞檢測專家，熟悉儲槽內外部檢查、腐蝕、厚度量測、PT、MT、UT、真空試漏與修補追蹤。",
        "goal": "協助整理儲槽檢查計畫、檢查結果、異常分析與改善建議。",
        "style": "技術性、條列式，可用於檢查報告與公文。"
    },
    "🏗️ 危險性機械設備 Agent": {
        "role": "你是危險性機械及設備管理專家，熟悉鍋爐、壓力容器、起重機、升降機與定期檢查制度。",
        "goal": "協助整理設備清冊、檢查狀態、法定檢查與異常追蹤。",
        "style": "精準、表格化，適合設備管理。"
    },
    "🔍 工安稽核 Agent": {
        "role": "你是工安稽核與缺失改善專家，熟悉稽核查證、原因分析、矯正措施、預防再發與佐證資料整理。",
        "goal": "協助撰寫稽核缺失回覆、改善說明與追蹤表。",
        "style": "客觀、穩健，避免過度承認責任，適合正式稽核回覆。"
    },
    "✍️ 公文撰寫 Agent": {
        "role": "你是公司內部公文、簽辦、通知與電子郵件撰寫專家。",
        "goal": "協助將口語內容改寫成正式主旨、說明、辦法、簽辦或通知。",
        "style": "正式、精簡、符合公司行政文件語氣。"
    },
    "📚 公司內部知識庫 Agent": {
        "role": "你是公司內部知識庫查詢專家，熟悉將 SOP、規章、會議資料、稽核資料與設備檢查文件整理成可引用依據。",
        "goal": "根據知識庫內容快速查詢、摘要、比對與整理內部文件重點。",
        "style": "正式、清楚、條列化。若知識庫沒有資料，必須明確說明。"
    },
    "🧠 長期記憶 Agent": {
        "role": "你是長期記憶管理專家，負責協助整理案件、設備、廠商、稽核缺失與歷史問答資料。",
        "goal": "協助將零散資訊整理成可追蹤、可查詢、可延續管理的記憶資料。",
        "style": "清楚、分類明確、適合後續追蹤。"
    }
}


# =========================
# Task 設定
# =========================
TASKS = {
    "一般問答": "依照 Agent 專業身份回答使用者問題。",
    "風險分析": "分析可能危害、風險等級、發生原因、現有管制與改善建議。",
    "文件摘要": "摘要文件重點，整理成條列式重點。",
    "稽核缺失回覆": "撰寫缺失原因分析、改善措施、預防再發與佐證資料建議。",
    "公文撰寫": "撰寫正式公文主旨、說明與辦法。",
    "檢查表產生": "產生可執行的檢查表或查核表。",
    "查詢公司內部知識庫": "根據公司內部知識庫回答，並列出依據與重點。",
    "摘要內部文件重點": "根據知識庫摘要相關文件重點、規定、注意事項與可執行建議。",
    "比對內部規定差異": "比對不同文件、規定或版本之間的差異。",
    "整理內部作業依據": "整理可作為公文、稽核回覆或工作說明的依據。",
    "主管決策建議": "整理重點、風險、優先順序、建議決策與後續追蹤事項。",
    "長期記憶整理": "將案件、設備、廠商、稽核缺失或歷史問答整理成可追蹤記憶。"
}


QUICK_PROMPTS = {
    "風險分析": "請針對下列作業進行工安風險分析，包含危害、可能後果、風險等級、現有管制及改善建議：",
    "工作許可檢核": "請協助檢查下列工作許可內容是否完整，並列出缺漏、風險與改善建議：",
    "承攬商缺失回覆": "請協助撰寫下列承攬商管理缺失之原因分析、改善措施與預防再發措施：",
    "PSM 查核重點": "請整理下列議題在 PSM 查核時應注意的重點、可能缺失與佐證資料：",
    "儲槽檢查摘要": "請針對下列儲槽開放檢查資料，整理檢查結果、異常、改善建議與後續追蹤事項：",
    "公文草稿": "請將下列內容改寫成正式公文格式，包含主旨、說明、辦法："
}


# =========================
# 單一 Agent 回答
# =========================
def ask_gpt(selected_agent, selected_task, user_question, context_text, memory_context=""):
    agent_info = AGENTS[selected_agent]
    task_instruction = TASKS[selected_task]

    system_prompt = f"""
你現在是「中油工安 AI 助理 V12」中的專業 Agent。

目前 Agent：
{selected_agent}

Agent 身份：
{agent_info["role"]}

Agent 目標：
{agent_info["goal"]}

回答風格：
{agent_info["style"]}

目前 Task：
{selected_task}

Task 說明：
{task_instruction}

重要規則：
1. 若有知識庫內容，必須優先依據知識庫內容回答。
2. 若有長期記憶內容，請納入案件、設備、廠商或稽核脈絡。
3. 不可編造公司內部規定、SOP、法規條文或不存在的文件內容。
4. 若知識庫沒有明確依據，請明確寫：「目前知識庫未找到明確依據，以下為一般工安管理建議。」
5. 回答需適合公司內部使用。
6. 優先使用條列、表格、標題分段。
7. 回答結尾請列出「後續建議」。
"""

    user_prompt = f"""
以下是長期記憶內容：

{memory_context}

以下是知識庫查詢到的內容：

{context_text}

使用者問題：
{user_question}

請依據上述內容與 Agent 專業身份回答。
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
# V11 / V12 多 Agent 協作
# =========================
def ask_multi_agents(user_question, context_text, selected_multi_agents, memory_context=""):
    agent_outputs = []

    for agent_name in selected_multi_agents:
        agent_info = AGENTS[agent_name]

        system_prompt = f"""
你是「中油工安 AI 助理 V12 多 Agent 協作系統」中的一位專家。

目前 Agent：
{agent_name}

Agent 身份：
{agent_info["role"]}

Agent 目標：
{agent_info["goal"]}

回答風格：
{agent_info["style"]}

請只從你的專業角度回答，不要代替其他 Agent。

回答格式：
一、你看到的重點
二、主要風險或問題
三、改善建議
四、需追蹤事項
"""

        user_prompt = f"""
長期記憶內容：
{memory_context}

知識庫內容：
{context_text}

使用者問題：
{user_question}

請從你的專業角度分析。
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        agent_outputs.append({
            "agent": agent_name,
            "answer": response.choices[0].message.content
        })

    summary_prompt = "以下是多位 Agent 的分析結果，請整合成主管可閱讀的結論：\n\n"

    for item in agent_outputs:
        summary_prompt += f"\n\n【{item['agent']}】\n{item['answer']}"

    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
你是「中油工安 AI 助理 V12」的主控 Agent。

請整合多位專家 Agent 的意見，產出主管可直接閱讀的整合結論。

請輸出：
一、綜合判斷
二、主要風險
三、優先處理事項
四、改善建議
五、後續追蹤事項
六、主管決策建議
"""
            },
            {"role": "user", "content": summary_prompt}
        ],
        temperature=0.2
    )

    return agent_outputs, summary_response.choices[0].message.content


def build_memory_context(keyword):
    if not keyword.strip():
        return ""

    results = search_memory(keyword)

    if not results:
        return ""

    text = ""
    for item in results[:10]:
        text += f"""
【記憶類型：{item['type']}】
標題：{item['title']}
標籤：{item.get('tags', '')}
內容：{item['content']}
建立時間：{item['created_at']}
"""
    return text


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.title("🏭 V12 控制台")
    st.caption("中油工安 AI Agent 平台")

    selected_agent = st.selectbox(
        "選擇 Agent",
        list(AGENTS.keys())
    )

    selected_task = st.selectbox(
        "選擇 Task",
        list(TASKS.keys())
    )

    rag_results = st.slider(
        "知識庫引用筆數",
        min_value=3,
        max_value=10,
        value=5
    )

    use_memory_sidebar = st.checkbox("啟用長期記憶", value=True)

    st.divider()

    st.subheader("目前版本路線")
    st.write("✅ V10 Agent 架構優化")
    st.write("✅ V11 多 Agent 協作")
    st.write("✅ V12 長期記憶系統")
    st.write("➡️ V13 工安文件中心")
    st.write("➡️ V14 工作許可 Agent")
    st.write("➡️ V15 承攬商管理 Agent")
    st.write("➡️ V16 PSM Agent")
    st.write("➡️ V17 儲槽開放檢查 Agent")
    st.write("➡️ V18 工安稽核 Agent")
    st.write("➡️ V19 主管決策中心")
    st.write("➡️ V20 AI 工安總管平台")

    st.divider()

    if st.button("清除本次對話紀錄"):
        st.session_state.chat_history = []
        st.session_state.multi_agent_history = []
        st.success("已清除。")


# =========================
# 主畫面
# =========================
st.title("🏭 中油工安 AI 助理 V12")
st.caption("V12 長期記憶系統｜案件記憶｜設備記憶｜廠商記憶｜稽核缺失追蹤｜多 Agent 協作")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🤖 AI Agent 問答",
    "🤝 多 Agent 協作",
    "🧠 長期記憶",
    "📚 文件寫入知識庫",
    "📄 單次文件分析",
    "📊 平台狀態"
])


# =========================
# Tab 1：AI Agent 問答
# =========================
with tab1:
    st.subheader("🤖 AI Agent 問答")

    st.info(f"目前 Agent：{selected_agent}｜目前 Task：{selected_task}")

    quick = st.selectbox(
        "快捷任務提示",
        ["不使用"] + list(QUICK_PROMPTS.keys())
    )

    default_text = QUICK_PROMPTS[quick] if quick != "不使用" else ""

    user_question = st.text_area(
        "請輸入問題",
        value=default_text,
        height=180
    )

    col_a, col_b, col_c = st.columns([1, 2, 2])

    with col_a:
        start_btn = st.button("開始分析", type="primary")

    with col_b:
        use_rag = st.checkbox("使用 RAG 知識庫", value=True)

    with col_c:
        memory_keyword = st.text_input("記憶查詢關鍵字", placeholder="例如：A-14、動火、承攬商")

    if start_btn:
        if not user_question.strip():
            st.warning("請先輸入問題。")
        elif not OPENAI_API_KEY:
            st.error("找不到 OPENAI_API_KEY，請確認 .env 是否設定完成。")
        else:
            with st.spinner("正在分析中..."):
                context_text, sources = search_knowledge_base(user_question, rag_results) if use_rag else ("", [])
                memory_context = build_memory_context(memory_keyword) if use_memory_sidebar else ""

                answer = ask_gpt(
                    selected_agent,
                    selected_task,
                    user_question,
                    context_text,
                    memory_context
                )

                add_memory(
                    "qa_history",
                    title=user_question[:40],
                    content=f"問題：{user_question}\n\n回答：{answer}",
                    tags=f"{selected_agent},{selected_task}"
                )

                st.session_state.chat_history.append({
                    "agent": selected_agent,
                    "task": selected_task,
                    "question": user_question,
                    "answer": answer,
                    "sources": sources
                })

    if st.session_state.chat_history:
        st.subheader("✅ 回答紀錄")
        for i, item in enumerate(reversed(st.session_state.chat_history), start=1):
            with st.expander(f"{i}. {item['agent']}｜{item['task']}", expanded=(i == 1)):
                st.markdown("### 使用者問題")
                st.write(item["question"])
                st.markdown("### AI 回答")
                st.write(item["answer"])

                if item["sources"]:
                    st.markdown("### 引用來源")
                    for source in item["sources"]:
                        st.write(f"- {source}")


# =========================
# Tab 2：多 Agent 協作
# =========================
with tab2:
    st.subheader("🤝 V12 多 Agent 協作")
    st.caption("多位 Agent 分別分析，再由主控 Agent 整合成主管版結論。")

    default_multi_agents = [
        "🏭 工安專家 Agent",
        "📋 工作許可 Agent",
        "👷 承攬商管理 Agent",
        "⚙️ PSM Agent",
        "🔍 工安稽核 Agent"
    ]

    selected_multi_agents = st.multiselect(
        "選擇參與協作的 Agent",
        list(AGENTS.keys()),
        default=default_multi_agents
    )

    multi_question = st.text_area(
        "請輸入多 Agent 協作問題",
        height=180
    )

    col_m1, col_m2, col_m3 = st.columns([1, 2, 2])

    with col_m1:
        multi_start = st.button("啟動多 Agent 協作", type="primary")

    with col_m2:
        use_multi_rag = st.checkbox("使用 RAG 知識庫", value=True, key="multi_rag")

    with col_m3:
        multi_memory_keyword = st.text_input("記憶查詢關鍵字", key="multi_memory_keyword")

    if multi_start:
        if not multi_question.strip():
            st.warning("請先輸入問題。")
        elif not selected_multi_agents:
            st.warning("請至少選擇一位 Agent。")
        else:
            with st.spinner("多 Agent 正在分工分析中..."):
                context_text, sources = search_knowledge_base(multi_question, rag_results) if use_multi_rag else ("", [])
                memory_context = build_memory_context(multi_memory_keyword) if use_memory_sidebar else ""

                agent_outputs, final_summary = ask_multi_agents(
                    multi_question,
                    context_text,
                    selected_multi_agents,
                    memory_context
                )

                add_memory(
                    "qa_history",
                    title=f"多Agent：{multi_question[:30]}",
                    content=f"問題：{multi_question}\n\n整合結論：{final_summary}",
                    tags="多Agent,主管結論"
                )

                st.session_state.multi_agent_history.append({
                    "question": multi_question,
                    "agents": selected_multi_agents,
                    "agent_outputs": agent_outputs,
                    "final_summary": final_summary,
                    "sources": sources
                })

    if st.session_state.multi_agent_history:
        st.subheader("✅ 多 Agent 協作紀錄")
        for i, item in enumerate(reversed(st.session_state.multi_agent_history), start=1):
            with st.expander(f"{i}. 多 Agent 協作結果", expanded=(i == 1)):
                st.markdown("### 使用者問題")
                st.write(item["question"])

                st.markdown("### ✅ 主控 Agent 整合結論")
                st.write(item["final_summary"])

                st.markdown("### 🤖 各 Agent 分析內容")
                for output in item["agent_outputs"]:
                    with st.expander(output["agent"]):
                        st.write(output["answer"])

                if item["sources"]:
                    st.markdown("### 📌 引用來源")
                    for source in item["sources"]:
                        st.write(f"- {source}")


# =========================
# Tab 3：長期記憶
# =========================
with tab3:
    st.subheader("🧠 V12 長期記憶系統")

    memory_map = {
        "案件記憶": "cases",
        "設備記憶": "equipments",
        "廠商記憶": "contractors",
        "稽核缺失記憶": "audit_findings",
        "歷史問答紀錄": "qa_history"
    }

    mode = st.radio(
        "選擇功能",
        ["新增記憶", "查詢記憶", "管理記憶"],
        horizontal=True
    )

    if mode == "新增記憶":
        memory_label = st.selectbox("記憶類型", list(memory_map.keys()))
        memory_title = st.text_input("記憶標題", placeholder="例如：A-14 儲槽 115年開放檢查")
        memory_tags = st.text_input("標籤", placeholder="例如：A-14, 儲槽, NDT, 115年")
        memory_content = st.text_area("記憶內容", height=220)

        if st.button("新增到長期記憶", type="primary"):
            if not memory_title.strip() or not memory_content.strip():
                st.warning("請輸入標題與內容。")
            else:
                item = add_memory(
                    memory_map[memory_label],
                    memory_title,
                    memory_content,
                    memory_tags
                )
                st.success(f"已新增記憶：{item['id']}")

    elif mode == "查詢記憶":
        keyword = st.text_input("輸入查詢關鍵字", placeholder="例如：A-14、承攬商、動火、PSM")
        if st.button("查詢"):
            results = search_memory(keyword)

            if not results:
                st.info("未找到相關記憶。")
            else:
                st.success(f"找到 {len(results)} 筆記憶")
                for item in results:
                    with st.expander(f"{item['type']}｜{item['title']}｜{item['id']}"):
                        st.write(f"建立時間：{item['created_at']}")
                        st.write(f"標籤：{item.get('tags', '')}")
                        st.write(item["content"])

    elif mode == "管理記憶":
        data = load_memory()

        for memory_type, items in data.items():
            st.markdown(f"### {memory_type}（{len(items)} 筆）")

            for item in items:
                with st.expander(f"{item['title']}｜{item['id']}"):
                    st.write(f"建立時間：{item['created_at']}")
                    st.write(f"標籤：{item.get('tags', '')}")
                    st.write(item["content"])

                    if st.button(f"刪除此記憶 {item['id']}", key=f"delete_{memory_type}_{item['id']}"):
                        delete_memory(memory_type, item["id"])
                        st.warning("已刪除，請重新整理頁面。")


# =========================
# Tab 4：文件寫入知識庫
# =========================
with tab4:
    st.subheader("📚 文件寫入知識庫")

    doc_type = st.selectbox(
        "文件類型",
        [
            "SOP / 作業標準",
            "工安制度",
            "稽核資料",
            "承攬商資料",
            "工作許可資料",
            "PSM 資料",
            "儲槽檢查資料",
            "危險性機械設備資料",
            "會議紀錄",
            "教育訓練資料",
            "其他內部文件"
        ]
    )

    uploaded_files = st.file_uploader(
        "請上傳文件",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("寫入 ChromaDB 知識庫", type="primary"):
        if not uploaded_files:
            st.warning("請先上傳文件。")
        else:
            total_chunks = 0

            with st.spinner("正在解析並寫入知識庫..."):
                for file in uploaded_files:
                    file_name = file.name

                    if file_name.lower().endswith(".pdf"):
                        text = read_pdf(file)
                    elif file_name.lower().endswith(".docx"):
                        text = read_word(file)
                    elif file_name.lower().endswith(".txt"):
                        text = read_txt(file)
                    else:
                        text = ""

                    if text.strip():
                        chunk_count = add_text_to_knowledge_base(text, file_name, doc_type)
                        total_chunks += chunk_count
                        st.success(f"{file_name} 已寫入知識庫，共 {chunk_count} 筆片段。")
                    else:
                        st.warning(f"{file_name} 無法讀取文字內容。")

            st.info(f"本次共寫入 {total_chunks} 筆知識片段。")


# =========================
# Tab 5：單次文件分析
# =========================
with tab5:
    st.subheader("📄 單次文件分析")

    analysis_file = st.file_uploader(
        "請上傳要分析的文件",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
        key="analysis_file"
    )

    analysis_question = st.text_area(
        "請輸入分析需求",
        height=120
    )

    if st.button("分析文件", type="primary"):
        if not analysis_file:
            st.warning("請先上傳文件。")
        elif not analysis_question.strip():
            st.warning("請輸入分析需求。")
        else:
            with st.spinner("正在讀取文件並分析..."):
                file_name = analysis_file.name

                if file_name.lower().endswith(".pdf"):
                    file_text = read_pdf(analysis_file)
                elif file_name.lower().endswith(".docx"):
                    file_text = read_word(analysis_file)
                elif file_name.lower().endswith(".txt"):
                    file_text = read_txt(analysis_file)
                else:
                    file_text = ""

                context_text = f"【本次上傳文件：{file_name}】\n{file_text[:12000]}"

                answer = ask_gpt(
                    selected_agent,
                    selected_task,
                    analysis_question,
                    context_text
                )

                st.subheader("✅ 文件分析結果")
                st.write(answer)


# =========================
# Tab 6：平台狀態
# =========================
with tab6:
    st.subheader("📊 平台狀態")

    memory_data = load_memory()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Agent 數量", len(AGENTS))

    with col2:
        st.metric("Task 數量", len(TASKS))

    with col3:
        st.metric("單 Agent 對話數", len(st.session_state.chat_history))

    with col4:
        st.metric("多 Agent 協作數", len(st.session_state.multi_agent_history))

    st.divider()

    st.markdown("### 長期記憶統計")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("案件", len(memory_data["cases"]))

    with c2:
        st.metric("設備", len(memory_data["equipments"]))

    with c3:
        st.metric("廠商", len(memory_data["contractors"]))

    with c4:
        st.metric("稽核缺失", len(memory_data["audit_findings"]))

    with c5:
        st.metric("歷史問答", len(memory_data["qa_history"]))

    st.divider()

    st.markdown("### V12 已完成項目")
    st.write("✅ V10 Agent 架構優化")
    st.write("✅ V11 多 Agent 協作")
    st.write("✅ V12 長期記憶系統")
    st.write("✅ 案件記憶")
    st.write("✅ 設備記憶")
    st.write("✅ 廠商記憶")
    st.write("✅ 稽核缺失記憶")
    st.write("✅ 歷史問答紀錄")
    st.write("✅ memory_db.json 本地儲存")

    st.markdown("### 下一版 V13：工安文件中心")
    st.write("下一步會加入：文件分類、文件清單、文件搜尋、文件摘要、文件比對。")