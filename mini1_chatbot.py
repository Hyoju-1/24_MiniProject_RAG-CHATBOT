import os
import torch
import pandas as pd
import pinecone  # 공식 Pinecone 클라이언트
from pinecone import ServerlessSpec
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LC_Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from tiktoken import encoding_for_model, get_encoding
from functools import lru_cache
from dotenv import load_dotenv
import re
import streamlit as st
import random

# .env 파일에서 환경 변수 로드
load_dotenv()

# ----------------------------------------------------------------------------------
# 1) 환경 변수 로드
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_region = os.getenv("PINECONE_ENVIRONMENT")  # 예: "us-east-1"
index_name = os.getenv("INDEX_NAME")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not pinecone_api_key or not pinecone_region:
    raise ValueError("Pinecone API 키 또는 환경 변수가 설정되지 않았습니다.")
if not index_name:
    raise ValueError("INDEX_NAME 환경 변수를 설정되지 않았습니다.")
if not gemini_api_key:
    raise ValueError("Gemini API 키가 설정되지 않았습니다.")

# ----------------------------------------------------------------------------------
# 2) Pinecone 클라이언트 인스턴스 생성 & 인덱스 준비
pc = pinecone.Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# ----------------------------------------------------------------------------------
# 3) 장치 설정 (GPU/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 4) LangChain 임베딩 & VectorStore
embed_model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

vectorstore = LC_Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    text_key="text"  
)

# 5) Google Generative AI (Gemini-1.5-flash) LLM 생성
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    top_p=0.7,
    top_k=50,
    api_key=gemini_api_key
)

# 맨 위쪽(전역)에 추가
if "already_greeted" not in st.session_state:
    st.session_state.already_greeted = False

def remove_repeated_greeting(text: str) -> str:
    """
    이미 인사했으면, 답변에서 '안녕하세요' 문구를 제거한다.
    아직 인사 전이면, 인사 발견 시점부터 '이미 인사한 상태'로 바꾼다.
    """
    if not st.session_state.already_greeted:
        # 만약 이번 답변에 '안녕하세요'가 있으면, 인사 상태 True로 전환
        if "안녕하세요" in text:
            st.session_state.already_greeted = True
        return text
    else:
        # 이미 인사를 했다면, 반복 '안녕하세요'를 제거
        # 필요하다면 정규식으로 더 꼼꼼히 처리 가능
        new_text = text.replace("안녕하세요! ", "").replace("안녕하세요!", "")
        return new_text




# 6) 시스템 프롬프트 
system_prompt = """
너는 일본 애니메이션 및 공포영화 추천 및 정보 제공 및 추천 전문가야.

1. 사용자가 질문하면 해당 분야의 작품 정보를 바탕으로 추천과 상세 정보를 제공해.
2. 추천 정보는 작품 제목, 장르, 줄거리, 평점, 개봉 날짜 등 주요 정보를 포함해야 해.
3. 이전에 추천한 작품은 중복되지 않도록 새로운 작품을 추천해줘.
4. 만약에 사용자가 제공하는 정보가 부족하다면, 추가 질문을 하도록 해.
5. 사용자의 이전 질문을 기억해서 맞춤화된 답변을 제공해.
6. 한 번 인사를 한 이후에는 같은 대화에서 '안녕하세요!'라고 인사하지마.
6. 답변은 친근하고 따듯한 어조로 작성해줘.

"""

try:
    global_encoding = encoding_for_model("gemini-1.5-flash")
except KeyError:
    global_encoding = get_encoding("cl100k_base")

@lru_cache(maxsize=10000)
def cached_token_count(text):
    return len(global_encoding.encode(text))

def count_tokens_text(text: str) -> int:
    return len(global_encoding.encode(text))

# 대화 이력 관련 변수 (메모리 내 저장)
response_cache = {}
conversation_history = {}
user_llm_calls = {}

def initialize_conversation(user_id: str, system_prompt: str):
    prev = conversation_history.get(user_id, {"messages": [], "filters": {}, "recommended_ids": set()})
    if not prev["messages"]:
        prev["messages"].append(SystemMessage(content=system_prompt))
    conversation_history[user_id] = prev
    return prev

def update_conversation_history(user_id, new_message, filters):
    prev = conversation_history.get(user_id, {"messages": [], "filters": {}, "recommended_ids": set()})
    prev["messages"].append(new_message)
    prev["filters"] = filters
    conversation_history[user_id] = prev

def summarize_conversation(user_id: str, max_messages: int = 10):
    hist = conversation_history.get(user_id)
    if not hist: 
        return
    msgs = hist["messages"]
    if len(msgs) <= max_messages:
        return
    system_msg = msgs[0]
    last_msg = msgs[-1]
    middle = msgs[1:-1]
    if not middle:
        return

    user_llm_calls[user_id] = user_llm_calls.get(user_id, 0)
    if user_llm_calls[user_id] >= 3:
        return

    conversation_text = "\n".join([f"{m.__class__.__name__}: {m.content}" for m in middle])
    prompt = f"다음 대화를 간단히 요약해줘:\n{conversation_text}"
    try:
        summary_result = llm.invoke([HumanMessage(content=prompt)])
        user_llm_calls[user_id] += 1
    except Exception as e:
        print("요약 오류:", e)
        return

    summary_msg = SystemMessage(content=f"이전 대화 요약: {summary_result.content}")
    new_msgs = [system_msg, summary_msg, last_msg]
    hist["messages"] = new_msgs
    conversation_history[user_id] = hist

def create_search_filter(query: str):
    filters = {}
    query_lower = query.lower()

    category_map = {
        "공포": "공포영화",
        "호러": "공포영화",
        "애니": "애니메이션",
        "만화": "애니메이션",
        "일본": "애니메이션",
        "애니메이션": "애니메이션"
    }

    matched_categories = set()
    for kw, cat in category_map.items():
        if kw in query_lower:
            matched_categories.add(cat)

    if len(matched_categories) == 1:
        filters["카테고리"] = {"$eq": matched_categories.pop()}
    elif len(matched_categories) > 1:
        filters["카테고리"] = {"$in": list(matched_categories)}

    year_pattern = re.findall(r"(\d{4})년", query_lower)
    if year_pattern:
        year_int = int(year_pattern[0])
        filters["개봉 날짜"] = {"$eq": year_int}

    if "최근" in query_lower:
        now_year = 2023
        lower_year_bound = now_year - 2
        filters["개봉 날짜"] = {"$gte": lower_year_bound}

    decade_pattern = re.search(r"(\d{4})년대", query_lower)
    if decade_pattern:
        decade_start = int(decade_pattern.group(1))
        filters["개봉 날짜"] = {"$gte": decade_start, "$lte": decade_start + 9}

    rating_match = re.findall(r"(전체|13세|15세|18세|19금)", query_lower)
    if rating_match:
        rating = rating_match[0]
        filters["시청 등급"] = {"$eq": rating}

    rating_pattern = re.search(r"(평점|점수)\s*(\d+(\.\d+)?)(점|점대| )?\s*(이상|이하|초과|미만)?", query_lower)
    if rating_pattern:
        rating_str = rating_pattern.group(2)
        compare_keyword = rating_pattern.group(5)
        rating_val = float(rating_str)
        if compare_keyword == "이상" or compare_keyword == "초과":
            filters["점수"] = {"$gte": rating_val}
        elif compare_keyword == "이하" or compare_keyword == "미만":
            filters["점수"] = {"$lte": rating_val}
        else:
            filters["점수"] = {"$eq": rating_val}

    return filters

def rank_documents_by_score(docs, weight_score=0.7, weight_votes=0.3):
    def compute_rank(doc):
        score = float(doc.metadata.get("점수", 0))
        votes = float(doc.metadata.get("투표수", 0))
        return weight_score * score + weight_votes * votes

    sorted_docs = sorted(docs, key=lambda doc: compute_rank(doc), reverse=True)
    return sorted_docs

def retrieve_documents(query: str, filters: dict, k=5, exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = set()
    try:
        docs = vectorstore.similarity_search(query, k=k*2, filter=filters)
    except Exception as e:
        print("Pinecone 검색 오류:", e)
        return []

    filtered_docs = []
    for doc in docs:
        doc_id = doc.metadata.get("id")
        if doc_id not in exclude_ids:
            filtered_docs.append(doc)
        if len(filtered_docs) >= k:
            break

    query_lower = query.lower()
    if "평점" in query_lower or "점수" in query_lower:
        filtered_docs = rank_documents_by_score(filtered_docs, weight_score=0.7, weight_votes=0.3)
        filtered_docs = filtered_docs[:k]

    return filtered_docs

def create_prompt(system_prompt, prev_text, question, context):
    return f"""
{system_prompt}

이전 대화 내용:
{prev_text}

사용자가 질문한 내용:
{question}

검색된 작품 정보:
{context}

위 정보를 바탕으로 사용자 질문에 대해 답변해줘.
"""

def reduce_context(docs, prev_text, system_prompt, question, max_token_limit=5000):
    while len(docs) > 0:
        docs = docs[:-1]
        context = "\n\n".join([
            "\n".join([f"{k}: {v}" for k,v in doc.metadata.items()])
            for doc in docs
        ])
        prompt = create_prompt(system_prompt, prev_text, question, context)
        total_tokens = count_tokens_text(prompt)
        if total_tokens <= max_token_limit:
            return prompt, docs
    raise ValueError("토큰 제한 초과")






def call_llm(llm, prompt_with_context):
    try:
        resp = llm.invoke([HumanMessage(content=prompt_with_context)])
        return resp
    except Exception as e:
        print("LLM 호출 오류:", e)
        return AIMessage(content="답변 생성 중 오류가 발생했습니다.")


def query_llm(user_id: str, question: str, forced_category: str):

    # 1) 대화 초기화
    initialize_conversation(user_id, system_prompt)
    
    # 2) 검색 필터 생성
    filters = create_search_filter(question)


    # (중요) 사용자가 라디오 버튼으로 고른 장르를 '카테고리' 필드에 강제 주입
    filters["카테고리"] = {"$eq": forced_category}

    # 3) 이전 대화 요약
    summarize_conversation(user_id, max_messages=10)
    hist = conversation_history[user_id]

    # 사용자가 보낸 메시지를 대화 기록에 추가
    hist["messages"].append(HumanMessage(content=question))
    recommended_ids = hist.get("recommended_ids", set())

    # 4) Pinecone 등 VectorStore에서 문서 검색
    docs = retrieve_documents(question, filters, k=5, exclude_ids=recommended_ids)

    # 5) 검색 결과가 없으면 안내
    if not docs:
        no_result = f"'{forced_category}' 카테고리에서 작품을 찾을 수 없네요!"
        update_conversation_history(user_id, AIMessage(content=no_result), filters)
        response_cache[user_id] = no_result
        return no_result

    # 6) 검색된 문서를 바탕으로 컨텍스트 정리
    context = "\n\n".join([
        "\n".join([
            f"원제목: {doc.metadata.get('원제목','없음')}",
            f"한글제목: {doc.metadata.get('한글제목','없음')}",
            f"타입: {doc.metadata.get('타입','없음')}",
            f"장르: {doc.metadata.get('장르','없음')}",
            f"줄거리: {doc.metadata.get('줄거리','없음')[:100]}...",
            f"점수: {doc.metadata.get('점수','없음')}",
            f"개봉 날짜: {doc.metadata.get('개봉 날짜','없음')}",
            f"id: {doc.metadata.get('id','')}",
        ])
        for doc in docs
    ])

    prev_msgs_text = "\n\n".join([f"{m.__class__.__name__}: {m.content}" for m in hist["messages"]])
    prompt_with_context = create_prompt(system_prompt, prev_msgs_text, question, context)

    # 7) 토큰 제한 체크 후 필요 시 컨텍스트 축소
    total_tokens = count_tokens_text(prompt_with_context)
    if total_tokens > 5000:
        prompt_with_context, docs = reduce_context(docs, prev_msgs_text, system_prompt, question, 5000)

    # 8) LLM 호출
    resp = call_llm(llm, prompt_with_context)
    final_answer = resp.content if isinstance(resp, AIMessage) else resp.content

    # --- 여기서 "안녕하세요" 중복 방지 후처리 ---
    final_answer = remove_repeated_greeting(final_answer)



    # 9) 추천된 문서 ID를 기록 (중복 추천 방지용)
    for d in docs:
        rid = d.metadata.get("id")
        if rid:
            recommended_ids.add(rid)
    hist["recommended_ids"] = recommended_ids

    # 10) 대화 이력과 캐시에 최종 답변 업데이트
    update_conversation_history(user_id, AIMessage(content=final_answer), filters)
    response_cache[user_id] = final_answer
    
    return final_answer

# ----------------------------------------------------------------------------------
# Streamlit 앱 설정
st.set_page_config(page_title="일본 애니메이션 & 공포 영화 추천 챗봇", layout="wide")
st.title("🎭 Anime & Horror 같이 보아~ 즈! 👻🌟")

# 상태 저장 (대화 기록, 마지막 장르, 테마, 사용자 ID)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_genre" not in st.session_state:
    st.session_state.last_genre = "공포영화"  # 기본값
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "user_id" not in st.session_state:
    st.session_state.user_id = "streamlit_user"

# 사용자에게 장르 선택 요청
choice = st.radio("장르 선택", ["공포영화", "애니메이션"], key="genre_choice")

# 사용자가 장르를 고르면 세션에 반영
st.session_state.last_genre = choice

# 테마 변경 함수 (배경 & 글씨색)
def set_theme(theme, sidebar_bg_color):
    bg_color = "#df3dff" if theme == "dark" else "#fffdba"
    text_color = "#FFFFFF" if theme == "dark" else "#000000"
    button_bg_color = "#FFFFFF"
    button_text_color = "#000000"
    
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {bg_color} !important;
                color: {text_color} !important;
            }}
            .stRadio {{
                background-color: transparent !important;
            }}
            .stRadio label {{
                color: {button_text_color} !important;
                font-weight: bold;
                padding: 10px;
                display: inline-block;
            }}
            .stRadio div {{
                color: {text_color} !important;
            }}
            .stButton button {{
                background-color: {button_bg_color} !important;
                color: {button_text_color} !important;
                font-weight: bold;
                border-radius: 5px;
            }}
            .sidebar {{
                background-color: {sidebar_bg_color} !important;
                padding: 15px;
                border-radius: 10px;
            }}
            .stChatMessage .stMarkdown p {{
                color: {text_color} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    return text_color

if choice == "공포영화":
    text_color = set_theme("dark", "#D1B3FF")
    st.session_state.last_genre = "공포영화"
elif choice == "애니메이션":
    text_color = set_theme("light", "#FFEB99")
    st.session_state.last_genre = "애니메이션"




# 왼쪽 사이드바에 챗봇 소개 추가
with st.sidebar:
    if st.session_state.get("last_genre") == "애니메이션":
        st.markdown("""<div class='sidebar'>
            <h3>🎬 Anime Movie Bot</h3>
            <p>애니메이션 추천 챗봇입니다! 나만의 취향저격 애니메이션을 찾아보세요! 다양한 장르를 입력해보세요!</p>
            <img src='https://cdn.pixabay.com/animation/2023/11/01/09/27/09-27-12-411_512.gif' width='150'>
        </div>""", unsafe_allow_html=True)
    elif st.session_state.get("last_genre") == "공포영화":
        st.markdown("""<div class='sidebar'>
            <h3>🎬 Horror Movie Bot</h3>
            <p>공포영화 추천 챗봇입니다! 공포영화가 필요한 순간에 적합한 영화를 추천해드려요!</p>
            <img src='https://cdn.pixabay.com/animation/2022/10/13/14/20/14-20-47-878_512.gif' width='150'>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='sidebar'>
            <h3>🎬 Choose a Genre!</h3>
            <p>영화 장르를 선택해주세요.</p>
        </div>""", unsafe_allow_html=True)

# 채팅 초기화 버튼
if st.button("🔄 채팅 초기화"):
    st.session_state.chat_history = []
    st.session_state.last_genre = "공포영화"
    st.session_state.theme = "light"
    st.session_state.user_id = "streamlit_user"
    st.session_state.clear()
    st.rerun()

# 이전 대화 표시
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for content in st.session_state.chat_history:
    with st.chat_message(content["role"]):
        if content["role"] == "ai" and st.session_state.last_genre == "공포영화":
            st.markdown(f'<p style="color: #FFFFFF;">{content["message"]}</p>', unsafe_allow_html=True)
        else:
            st.markdown(content["message"])


# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "message": prompt})

    # LLM 호출 (예시)
    response = query_llm(st.session_state.get("user_id", "test_user"), prompt, st.session_state.last_genre)

    # 챗봇 답변 (공포영화 -> 흰글씨, 애니 -> 기본값)
    with st.chat_message("ai"):
        if st.session_state.last_genre == "공포영화":  # 공포영화일 경우
            st.markdown(f'<p style="color: #FFFFFF;">{response}</p>', unsafe_allow_html=True)
        else:  # 애니메이션일 경우
            st.markdown(f'<p style="color: #000000;">{response}</p>', unsafe_allow_html=True)

    st.session_state.chat_history.append({"role": "ai", "message": response})

    


# 오른쪽 상단에 이미지 추가
col1, col2 = st.columns([3, 1])
with col2:
    if choice == "공포영화":
        st.markdown("""<div style="position: fixed; top: 100px; right: 10px;">
            <img src="https://cdn.pixabay.com/animation/2025/01/27/23/38/23-38-11-320_512.gif" width="250" alt="공포영화의 세계">
        </div>""", unsafe_allow_html=True)
    elif choice == "애니메이션":
        anime_images = [
            "https://mblogthumb-phinf.pstatic.net/MjAyMTA0MjZfMjcx/MDAxNjE5MzY1Mjg1MzUz.bqKdXoiQfDvluBmKLvnRnHpC7c9evgcimqI2nvB4CoYg.czF81H9WfvdUHJDOrEdkBg_ZG6xo6yMjToRGnssqr58g.GIF.badtutle124/%EC%A7%80%EB%A6%BC2.gif?type=w800",
            "https://img.onnada.com/201703/1794831217_11697e17_22.gif",
            "https://blog.kakaocdn.net/dn/A4Jys/btrsSvkfSir/JdfPZxZ5LA9kGz3gsg3LcK/img.gif",
            "https://i.pinimg.com/originals/2f/e4/32/2fe432a6fb2eda99bee011c12027a500.gif",
            "https://mblogthumb-phinf.pstatic.net/MjAyMDA3MjZfMTk2/MDAxNTk1NzM4MTE0MTg0.7G2PZFTiVbr9r-0VRX5OTLAVZroxPRC_A1u1FdL-S9Ug.RRUT4rNhLDCprwkn4PZRwDLY2LrhSAZ2JXdgy0ZQ9KMg.GIF.wogus2003/1595738107665.gif?type=w800",
            "https://i.imgur.com/lYAuac9.gif",
            "https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F2304A04654190CFB16",
            "https://blog.kakaocdn.net/dn/cw6Klq/btqNqBFUmC9/WsumPEAcns5SDSgEL503W1/img.gif",
            "https://i1.ruliweb.com/img/5/7/1/C/571CC988433B9F0024"
        ]
        selected_image = random.choice(anime_images)
        st.markdown(f"""
            <div style="position: fixed; top: 100px; right: 10px;">
                <img src="{selected_image}" width="350" alt="애니메이션 이미지">
            </div>
        """, unsafe_allow_html=True)
