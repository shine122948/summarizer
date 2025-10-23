import textwrap
import streamlit as st
from openai import OpenAI

# ---------------------------
# 설정 & 상수
# ---------------------------
st.set_page_config(page_title="학생 프로젝트 보고서 요약기+", page_icon="📝", layout="wide")
st.title("📝 학생 프로젝트 보고서 요약기+")
st.caption("보고서를 50/100/300/500자, 세특형태(500자 내외)로 요약하고, AI 추천 질문 기반 관점 요약도 생성합니다.")

# OpenAI 클라이언트
client = OpenAI(api_key=st.secrets["openai_api_key"])

SAMPLE_REPORT = (
    "우리 팀은 기후 변화로 인한 이상기온과 자연재해 발생을 예측하기 위해 인공지능 기술을 활용한 프로젝트를 진행하였다. "
    "먼저 지난 20년간의 국내외 기상 데이터를 수집하여 평균 기온, 강수량, 이산화탄소 농도 등의 주요 변수를 정리하였다. "
    "이후 데이터를 학습시키기 위해 Python과 TensorFlow를 활용하여 기온 예측 모델을 설계하였다. 초기에는 단순 선형회귀를 적용했지만 예측 오차가 컸기 때문에, "
    "다층 퍼셉트론(MLP) 모델로 구조를 바꾸고 학습률과 은닉층 수를 조정하면서 정확도를 높였다. 또한 기상청 오픈데이터 API를 통해 실시간 데이터를 추가로 받아 "
    "모델이 새로운 입력에도 대응할 수 있도록 했다. 모델 학습 결과, 평균 제곱 오차(MSE)가 0.15로 줄어들며 성능이 향상되었고, 시각화를 통해 특정 지역의 온도 상승 추세를 "
    "확인할 수 있었다. 예를 들어, 서울과 강릉 지역은 지난 10년간 여름철 평균기온이 꾸준히 상승하는 경향을 보였고, 우리 모델은 향후 5년간 평균기온이 약 1.2도 상승할 것으로 "
    "예측했다. 프로젝트 후반부에는 단순한 예측을 넘어 ‘기후 행동’으로의 연결을 고민하였다. 우리는 예측 결과를 바탕으로 지역별 온실가스 감축 시나리오를 제안하고, 이를 시각화 "
    "대시보드로 구현하였다. Streamlit을 이용해 누구나 접근 가능한 웹 형태로 배포했으며, 이를 통해 학급 친구들이 자신의 지역 데이터를 직접 탐색하고 기후 변화의 심각성을 "
    "체감할 수 있도록 했다. 이번 활동을 통해 우리는 인공지능이 단순한 기술이 아니라 사회 문제 해결의 강력한 도구가 될 수 있음을 배웠다. 또한 데이터의 품질과 전처리 과정의 "
    "중요성을 실감했으며, 앞으로는 더 다양한 기후 변수와 지역 데이터를 반영하여 예측의 정확도를 높이고 싶다. 무엇보다 협업 과정에서 각자의 역할을 책임감 있게 수행하는 것이 "
    "프로젝트 성공의 핵심이라는 점을 깨달았다."
)

# ---------------------------
# 세션 상태 초기화
# ---------------------------
if "report_input" not in st.session_state:
    st.session_state.report_input = ""
if "reco_questions" not in st.session_state:
    st.session_state.reco_questions = []
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# ---------------------------
# 사이드바 설정
# ---------------------------
with st.sidebar:
    st.header("⚙️ 옵션")
    model = st.selectbox("모델 선택", ["gpt-4o-mini", "gpt-4o"], index=0)
    temperature = st.slider("창의성(temperature)", 0.0, 1.0, 0.2, 0.05)
    st.caption("※ 낮을수록 간결·정확, 높을수록 다양·창의적")

# ---------------------------
# 유틸 함수
# ---------------------------
def trim_to_chars(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text.strip()
    cut = text[:limit].rstrip()
    endings = ["다.", ".", "!", "?", "요.", "임.", "습니다.", "했다."]
    last_end = -1
    for end in endings:
        pos = cut.rfind(end)
        if pos > last_end:
            last_end = pos + len(end)
    if last_end >= max(10, int(limit * 0.4)):
        return cut[:last_end].strip()
    return cut.strip()

def summarize_with_limit(report: str, limit: int, teacher_hint: str | None = None) -> str:
    base_rules = (
        "규칙:\n"
        "1) 한국어 3문장 이내로 요약 (문장 수 3개 이하)\n"
        "2) 새로운 사실 추가 금지, 원문 핵심만\n"
        "3) 목적→주요 수행→성과/한계 흐름 유지\n"
        f"4) 공백 포함 {limit}자 이내 목표\n"
    )
    perspective = ""
    if teacher_hint:
        perspective = f"\n교사 질문 관점 지시: '{teacher_hint}' 관점에서 핵심적으로 요약.\n"

    prompt = (
        f"다음은 고등학생의 프로젝트 보고서다.\n{base_rules}{perspective}\n"
        "[보고서]\n" + report
    )
    resp = client.responses.create(model=model, input=prompt, temperature=float(temperature))
    return trim_to_chars(resp.output_text, limit)

def summarize_as_student_record(report: str) -> str:
    """세특형태 500자 내외 요약"""
    prompt = (
        "다음은 고등학생의 프로젝트 활동 보고서이다. "
        "이 내용을 바탕으로 학생부 세부능력 및 특기사항(세특) 형태로 500자 내외로 요약하라.\n"
        "- 문체: '~함', '~함을 보임' 등 교사 기록형\n"
        "- 항목 없음, 한 단락으로 자연스럽게 작성\n"
        "- 핵심 포함: 주제, 수행 내용, 역량(탐구·문제해결·협업), 태도, 성과\n\n"
        f"[보고서]\n{report}\n\n"
        "출력: 500자 내외 한 단락의 세특 문장"
    )
    resp = client.responses.create(model=model, input=prompt, temperature=0.3)
    return trim_to_chars(resp.output_text, 520)

def generate_recommended_questions(report: str, k: int = 7) -> list:
    prompt = (
        "다음 학생 프로젝트 보고서를 읽고, 교사가 관점 요약에 활용할 수 있는 질문을 한국어로 7개 제안하라.\n"
        "- 각 질문은 한 줄, 40자 이내\n"
        "- 관점 예: 문제 정의, 데이터 수집, 분석, 협업, 성과, 한계, 개선 등\n"
        f"[보고서]\n{report}\n\n"
        "출력은 번호 없이 줄바꿈으로만 구분된 7개 질문."
    )
    resp = client.responses.create(model=model, input=prompt, temperature=0.3)
    lines = [ln.strip("-• ").strip() for ln in resp.output_text.split("\n") if ln.strip()]
    cleaned = []
    for q in lines:
        if len(q) <= 40 and q not in cleaned:
            cleaned.append(q)
        if len(cleaned) == k:
            break
    return cleaned[:k]

# ---------------------------
# 입력창
# ---------------------------
st.subheader("1) 1000자 보고서 입력")
col_top = st.columns([1, 2, 1])
with col_top[0]:
    use_sample = st.checkbox("샘플 입력 사용", value=False)
with col_top[2]:
    clear_btn = st.button("입력 초기화")

if clear_btn:
    st.session_state.report_input = ""
    st.session_state.reco_questions = []
    st.session_state.selected_question = None

if use_sample and not st.session_state.report_input.strip():
    st.session_state.report_input = SAMPLE_REPORT

report = st.text_area("학생 보고서", key="report_input", height=280, placeholder="학생이 작성한 프로젝트 활동 보고서를 붙여넣어 주세요.")

# ---------------------------
# 버튼 및 기능 실행
# ---------------------------
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    gen_default = st.button("기본 요약 (50/100/300/500자)", type="primary", use_container_width=True)
with colB:
    gen_sect = st.button("세특형태 500자 요약", use_container_width=True)
with colC:
    gen_questions = st.button("AI 추천 질문 생성", use_container_width=True)

# 기본 요약
if gen_default and report.strip():
    tabs = st.tabs(["50자", "100자", "300자", "500자"])
    for tab, limit in zip(tabs, [50, 100, 300, 500]):
        with tab:
            with st.spinner(f"{limit}자 요약 생성 중..."):
                summary = summarize_with_limit(report, limit)
                st.write(summary)
                st.caption(f"문자 수: {len(summary)}")

# 세특 요약
if gen_sect and report.strip():
    with st.spinner("세특형태 요약 생성 중..."):
        summary = summarize_as_student_record(report)
        st.subheader("🧾 세특형태 요약 (500자 내외)")
        st.write(summary)
        st.caption(f"문자 수: {len(summary)}")

# 추천 질문 생성
if gen_questions and report.strip():
    with st.spinner("AI가 추천 질문을 생성 중입니다..."):
        st.session_state.reco_questions = generate_recommended_questions(report)
        st.success("추천 질문이 생성되었습니다.")

if st.session_state.reco_questions:
    st.markdown("**추천 질문 선택:**")
    st.session_state.selected_question = st.radio("질문 선택", st.session_state.reco_questions)
    gen_q_summary = st.button("선택한 질문으로 관점 요약 생성")
    if gen_q_summary and st.session_state.selected_question:
        q = st.session_state.selected_question
        with st.spinner(f"'{q}' 관점 요약 중..."):
            s = summarize_with_limit(report, 500, teacher_hint=q)
            st.write(s)
            st.caption(f"문자 수: {len(s)}")

st.divider()
st.markdown("**💡 사용 팁**  \n- 세특형태 요약은 학생부 기록 문체로 자동 변환됩니다.")
