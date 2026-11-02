"""🎧 Churn Care — 음악 스트리밍 이탈 위험 고객 관리 서비스 (Streamlit)

실행 (저장소 폴더에서):  streamlit run service/app.py
저장된 머신러닝·신경망 모델을 연결합니다. 앱 시작 시 학습하지 않습니다.
고객 자료를 추가·갱신하면 service_data/ 에 저장됩니다.
"""
from __future__ import annotations

import io
import hashlib
import sys
from dataclasses import astuple, replace
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import core as C  # noqa: E402
import ui  # noqa: E402
import operations as O
from operating_views import response_results
from sequence_runtime import SequenceInputError, register_inputs

st.set_page_config(page_title="Churn Care · 이탈 위험 고객 관리", page_icon="🎧", layout="wide")
ui.inject_css()

PAGES = ["🏠 홈", "👤 고객 상세", "💰 캠페인", "🗂️ 고객 자료", "⚙️ 설정"]


@st.cache_resource(show_spinner=False)
def get_service(signature) -> C.Service:
    bar = st.progress(0.0, text="저장된 모델을 연결하는 중...")
    svc = C.load_service(progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
    bar.empty()
    return svc


@st.cache_data(show_spinner="고객별 위험 점수와 이유를 계산하는 중...")
def analyze(_svc: C.Service, setting: tuple, data_key: str, model_signature: tuple, _df: pd.DataFrame) -> pd.DataFrame:
    out = _svc.analyze(C.Setting(*setting), _df)
    return out.join(_df[C.FEATURES + ["churn"]]).sort_values("score", ascending=False)


def go_to(page: str):
    st.session_state.nav = page


def open_customer(uid):
    st.session_state.detail_uid = uid
    st.session_state.nav = PAGES[1]


def wide_button(label: str, **kw):
    """Streamlit 버전에 따라 너비 옵션 이름이 달라서 둘 다 지원."""
    try:
        new = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 46)
    except ValueError:
        new = False
    return st.button(label, width="stretch", **kw) if new else st.button(label, use_container_width=True, **kw)


MODEL_SIGNATURE = C.service_signature()
svc = get_service(MODEL_SIGNATURE)
if "nav" not in st.session_state:
    st.session_state.nav = PAGES[0]
if st.session_state.nav not in PAGES:
    st.session_state.nav = PAGES[0]
PROFILE = O.optimal_profile()
S: C.Setting = C.load_setting(svc)  # 운영 설정은 파일에 저장되어 모든 사용자에게 같게 적용
REG = svc.reg["models"]
THR = svc.threshold(S)

customers = C.load_customers(svc.team)
is_default_data = not (C.STORE / C.CURRENT).exists()
active_date=date.fromisoformat(C.load_data_date()) if not is_default_data else date(2018,11,1)
customers.attrs['as_of']=active_date.isoformat()
input_issue=None
try:
    svc.validate_population(S,customers)
except SequenceInputError as error:
    input_issue=str(error)
data_key = hashlib.sha256(pd.util.hash_pandas_object(customers,index=True).values.tobytes()).hexdigest()
A = analyze(svc, astuple(S), data_key, MODEL_SIGNATURE, customers) if input_issue is None else pd.DataFrame(columns=['score','churn'])
N = len(customers)
flag = A['score'] >= THR
has_truth = A['churn'].notna().any()
data_desc = "검증 예시 고객" if is_default_data else "저장된 고객 자료"

# ---------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">🎧</div><div class="name">Churn Care</div></div>'
                '<div style="color:#9A9AB0;font-size:.8rem;margin-bottom:6px">이탈 위험 고객 관리</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="lib">메뉴</div>', unsafe_allow_html=True)
    st.radio("메뉴", PAGES, key="nav", label_visibility="collapsed")
    st.markdown('<div class="lib">운영 현황</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-card">📁 <b>{N:,}명</b> · {data_desc}<br>🎚️ <b>{S.label}</b><br>'
        f'🤖 {REG[S.ml]["name"].split(" · ")[0]} + {REG[S.nn]["name"].split(" · ")[0]}<br>'
        f'🎯 기준 <b>{THR:.1f}점</b> 이상 = 이탈 예상<br><br>'
        f'관찰 {active_date-timedelta(days=28):%Y-%m-%d} ~ {active_date:%m-%d} (28일)<br>예측 {active_date:%Y-%m-%d} ~ {active_date+timedelta(days=14):%m-%d} (14일)</div>', unsafe_allow_html=True)

page = st.session_state.nav

if input_issue and page not in [PAGES[3],PAGES[4]]:
    st.error(input_issue)
    st.button('고객 자료에서 입력 등록',on_click=go_to,args=(PAGES[3],))
    st.button('설정 확인',on_click=go_to,args=(PAGES[4],))
    st.stop()
if input_issue:
    st.warning(input_issue)


def player():
    if input_issue:return
    ui.player_bar(f"지금 운영 중 · {S.label}",
                  f"{REG[S.ml]['name']} + {REG[S.nn]['name']} · 기준 {THR:.1f}점",
                  float(flag.mean()), f"이탈 예상 {int(flag.sum()):,}명", f"{N:,}명",
                  f"분석 자료 {N:,}명 · {'검증 예시 고객' if is_default_data else '저장된 자료'}")


def show_prediction_comparison():
    ui.playlist_header("🔥", "현재 자료", "두 모델의 예측 비교",
                       "현재 운영 자료에서 두 모델의 판단이 일치하는지 확인합니다.",
                       [f"<b>Churn Care</b>", f"{N:,}명", f"이탈 예상 <b>{int(flag.sum()):,}명</b>",
                        f"운영 기준 {S.label}"])
    k = st.columns(4)
    with k[0]:
        ui.kpi("분석 고객", f"{N:,}명", data_desc)
    with k[1]:
        ui.kpi("운영 기준의 이탈 예상", f"{int(flag.sum()):,}명", f"전체의 {flag.mean():.1%} · {THR:.1f}점 이상", hot=True)
    with k[2]:
        ui.kpi("두 모델의 판정이 다른 고객", f"{int((A['agree'] == '다름').sum()):,}명", "머신러닝과 신경망이 엇갈린 고객")
    with k[3]:
        ui.kpi("평균 위험 점수", f"{A['score'].mean():.1f}점", "고객의 상대적인 위험 순위")

    st.write("")
    c1, c2 = st.columns([2.1, 1])
    with c1:
        with st.container(border=True):
            ui.section("두 모델이 바라본 고객", "점 하나가 고객 한 명 · 점선 = 각 모델의 판정 기준")
            t_ml, t_nn = svc.model_threshold(S.ml, "ml",S), svc.model_threshold(S.nn, "nn",S)
            fig = go.Figure()
            for name, m, color in [("둘 다 이탈 예상", (A["ml_score"] >= t_ml) & (A["nn_score"] >= t_nn), ui.RED),
                                   ("판정 다름", A["agree"] == "다름", ui.AMBER),
                                   ("둘 다 유지 예상", (A["ml_score"] < t_ml) & (A["nn_score"] < t_nn), ui.VIOLET)]:
                d = A[m]
                fig.add_scatter(x=d["ml_score"], y=d["nn_score"], mode="markers", name=f"{name} ({len(d):,})",
                                marker=dict(size=6, color=color, opacity=.75, line=dict(width=0)),
                                customdata=d[["user_id", "score"]],
                                hovertemplate="고객 %{customdata[0]}<br>머신러닝 %{x:.1f} · 신경망 %{y:.1f}"
                                              "<br>운영 점수 %{customdata[1]:.1f}<extra></extra>")
            fig.add_vline(x=t_ml, line_dash="dot", line_color="#6B6B85")
            fig.add_hline(y=t_nn, line_dash="dot", line_color="#6B6B85")
            fig.update_layout(xaxis_title=f"머신러닝 점수 · {REG[S.ml]['name'].split(' · ')[0]}",
                              yaxis_title=f"신경망 점수 · {REG[S.nn]['name'].split(' · ')[0]}",
                              xaxis_range=[0, 101], yaxis_range=[0, 101])
            ui.plot(ui.style(fig, 400))
    with c2:
        with st.container(border=True):
            ui.section("현재 운영 설정")
            ui.setting_card(REG[S.ml]["name"], REG[S.nn]["name"])
            st.markdown(f"**운영 기준: {S.label}**")
            st.caption(("검증에서 선택한 기준" if S.use_saved else "직접 정한 기준") + f" · {THR:.2f}점 이상")


# ---------------------------------------------------------------
# 🏠 홈
# ---------------------------------------------------------------
if page == PAGES[0]:
    ui.topline([f"{data_desc} {N:,}명", f"예측 대상 기간 {active_date:%m-%d} ~ {active_date+timedelta(days=14):%m-%d}"], live="운영 중")
    ui.playlist_header("🎯", "오늘의 대응", "관리 목록",
                       "위험 고객을 확인하고, 맞춤 대응을 제공한 뒤 이탈 감소를 확인하세요.",
                       [f"이탈 예상 <b>{int(flag.sum()):,}명</b>", "고객 선정 → 대응 → 결과 확인"])
    st.caption('확정된 최적 조합 적용 중 · 신규 기준일·고객은 정형 자료와 신경망 입력을 함께 등록하세요.' if S.mode=='optimal' else '상세 옵션에서 변경한 운영 설정을 사용 중입니다.')
    target_tab, response_tab = st.tabs(["대응 고객 목록", "대응 결과 확인"])
    with target_tab:
        with st.container(border=True):
            a, b, d = st.columns([1.3, 1.2, 1])
            method = a.radio("선정 방식", ["점수 구간별 관리", "판정 기준", "상위 N명", "예산"], horizontal=True)
            if method == "점수 구간별 관리":
                st.caption("메모장에 적은 위험 점수 구간으로 고객을 나누고, 각 구간의 혜택 안내 푸시를 시뮬레이션합니다.")
                threshold_cols = st.columns(5)
                defaults = [30, 50, 70, 80, 90]
                threshold_labels = ["관심군 시작", "중위험군 시작", "고위험군 시작", "초고위험군 시작", "최고 위험군 시작"]
                cutoffs = [int(col.number_input(label, min_value=1, max_value=99, value=default, step=1,
                                                key=f"risk_cutoff_{i}"))
                           for i, (col, label, default) in enumerate(zip(threshold_cols, threshold_labels, defaults))]
                if cutoffs != sorted(set(cutoffs)):
                    st.error("점수 구간 기준은 낮은 점수부터 서로 다른 값으로 입력해 주세요.")
                    st.stop()
                levels = d.multiselect("요금제", ["free", "paid"], default=["free", "paid"],
                                       format_func={"free":"무료", "paid":"유료"}.get)
                mask = np.ones(N, dtype=bool)
            elif method == "판정 기준":
                thr = b.slider("위험 점수 기준", 0.0, 100.0, float(THR), 0.5, format="%.1f")
                mask = A["score"].to_numpy() >= thr
                levels = d.multiselect("요금제", ["free", "paid"], default=["free", "paid"], format_func={"free":"무료", "paid":"유료"}.get)
            else:
                if method == "상위 N명":
                    topn = b.number_input("관리 인원", 1, N, max(1, N // 10), 10)
                else:
                    budget = b.number_input("예산 (원)", 0, 1_000_000_000, 3_000_000, 100_000)
                    unit = b.number_input("1명당 비용 (원)", 1, 1_000_000, 5_000, 1_000)
                    topn = int(budget // unit)
                mask = np.zeros(N, dtype=bool)
                mask[: min(int(topn), N)] = True
                levels = d.multiselect("요금제", ["free", "paid"], default=["free", "paid"], format_func={"free":"무료", "paid":"유료"}.get)
        m2 = mask & A["level_last"].isin(levels).to_numpy()
        sel = A[m2]
        if method == "점수 구간별 관리":
            score = A["score"]
            bands = [
                ("최고 위험군", cutoffs[4], 100.000001, "1개월 무료 또는 3개월간 월 990원", "원클릭 복귀 링크를 포함한 SMS/알림톡"),
                ("초고위험군", cutoffs[3], cutoffs[4], "3개월 연속 50% 할인 또는 1개월 무료 연장", "해지 단계에서 즉시 노출하는 방어 혜택"),
                ("고위험군", cutoffs[2], cutoffs[3], "2–3개월간 30–40% 할인 + 맞춤 플레이리스트", "개인화 큐레이션 앱 푸시/이메일"),
                ("중위험군", cutoffs[1], cutoffs[2], "1–2개월간 15–20% 할인 쿠폰 또는 포인트", "신규 기능·앨범 안내 푸시"),
                ("관심군", cutoffs[0], cutoffs[1], "음원 다운로드 쿠폰·고음질 업그레이드·제휴 콘텐츠 이용권", "가격 할인 없는 이용 혜택 안내"),
                ("저위험군", 0, cutoffs[0], "추가 혜택 없음 · 일반 멤버십 유지", "마케팅 예산 대상에서 제외"),
            ]
            st.subheader("점수 구간별 고객과 혜택")
            st.caption("위험 점수는 모델 예측을 0~100점으로 표시한 값입니다. 버튼은 화면 시뮬레이션이며 실제 알림을 보내지 않습니다.")
            st.info("운영 가이드: 70점 이상 고객을 우선 관리합니다. 할인 혜택은 종료 후에도 이용이 이어지도록 기간을 나눠 제안하세요.")
            for i, (name, low, high, benefit, action) in enumerate(bands):
                band_mask = (score >= low) & (score < high) & A["level_last"].isin(levels)
                band = A[band_mask]
                with st.expander(f"{name} · {low}점 이상 {('100점 이하' if high > 100 else f'{high}점 미만')} · {len(band):,}명", expanded=i < 2):
                    st.markdown(f"**추천 혜택:** {benefit}  \n**대응 방식:** {action}")
                    push_col, _ = st.columns([1, 3])
                    if push_col.button(f"📣 {name} 푸시 알림 시뮬레이션", key=f"push_sim_{i}",
                                       disabled=band.empty or name == "저위험군"):
                        st.session_state[f"push_sim_result_{i}"] = f"{name} {len(band):,}명에게 혜택 안내를 표시했습니다. (시뮬레이션 · 실제 발송 없음)"
                    if f"push_sim_result_{i}" in st.session_state:
                        st.success(st.session_state[f"push_sim_result_{i}"])
                    if band.empty:
                        st.caption("현재 선택한 요금제에 해당 구간 고객이 없습니다.")
                    elif name == "저위험군":
                        st.caption("메모장 운영 가이드에 따라 푸시 발송 대상에서 제외합니다.")
                    else:
                        st.dataframe(band[["user_id", "score", "level_last", "action"]].rename(columns={
                            "user_id":"고객 번호", "score":"위험 점수", "level_last":"요금제", "action":"기존 추천 대응"
                        }), hide_index=True, height=min(240, 38 * min(len(band), 6) + 42))
        k = st.columns(4)
        with k[0]:
            ui.kpi("선택된 고객", f"{len(sel):,}명", f"전체의 {len(sel) / max(N, 1):.1%}", hot=True)
        with k[1]:
            ui.kpi("평균 위험 점수", f"{sel['score'].mean():.1f}점" if len(sel) else "-", "0~100점")
        with k[2]:
            ui.kpi("7일 이상 미접속", f"{int((sel['days_since_last_activity'] >= 7).sum()):,}명", "재방문 대응 우선 확인")
        with k[3]:
            ui.kpi("맞춤 대응 제안", f"{len(sel):,}건", "고객별 확인 신호와 추천 대응")
        st.write("")
        cols = ["user_id", "score", "pred", "level_last", "reason_1", "action", "reason_2",
                "days_since_last_activity", "recent_7d_song_count"]
        names = {"user_id": "고객 번호", "score": "위험 점수", "pred": "예측 결과",
                 "level_last": "요금제", "reason_1": "확인할 신호", "reason_2": "추가 신호",
                 "action": "추천 대응", "days_since_last_activity": "마지막 활동(일 전)",
                 "recent_7d_song_count": "최근 7일 재생"}
        show = sel[cols].rename(columns=names).copy()
        show["요금제"] = show["요금제"].map({"free":"무료", "paid":"유료"})
        st.dataframe(show, hide_index=True, height=470, column_config={
            "위험 점수": st.column_config.ProgressColumn("위험 점수", min_value=0.0, max_value=100.0, format="%.1f"),
            "마지막 활동(일 전)": st.column_config.NumberColumn(format="%.1f")})
        st.download_button("⬇ 대응 고객 목록 내려받기", show.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"관리목록_{date.today():%Y%m%d}.csv", mime="text/csv", type="primary")
        if len(sel):
            a, b = st.columns([3, 1])
            chosen = a.selectbox("대응할 고객 상세 확인", sel['user_id'].tolist(),
                                 format_func=lambda uid: f"{uid} · {float(sel.loc[sel.user_id==uid,'score'].iloc[0]):.1f}점")
            b.write("")
            b.button("고객 상세 열기", on_click=open_customer, args=(chosen,), type="primary")
        st.caption("추천 대응은 고객의 실제 불편을 확인한 뒤 제공하세요.")
    with response_tab:
        response_results(active_date)
    player()

# ---------------------------------------------------------------
# 👤 고객 상세
# ---------------------------------------------------------------
elif page == PAGES[1]:
    a, b = st.columns([1, 1])
    top = A.head(300)
    score_of = dict(zip(top["user_id"], top["score"]))
    typed = b.text_input("고객 번호로 찾기", placeholder="예: 1000035", key="detail_uid")
    uid = typed.strip() or a.selectbox("위험 높은 순 상위 300명", list(score_of),
                                       format_func=lambda u: f"{u} · {score_of[u]:.1f}점")
    hit = A[A["user_id"] == str(uid)]
    if hit.empty:
        st.warning(f"고객 {uid}를 현재 분석 자료에서 찾을 수 없습니다.")
        player()
        st.stop()
    r = hit.iloc[0]
    row_df = customers[customers["user_id"].astype(str) == str(uid)].head(1)
    contrib = svc.contributions(S, row_df).iloc[0]
    items = svc.reasons(row_df.iloc[0], contrib, top=3)
    pills = [ui.pred_pill(r["pred"]), ui.agree_pill(f"판정 {r['agree']}"),
             ui.pill("유료" if r["level_last"] == "paid" else "무료", ui.VIOLET)]
    ui.playlist_header("👤", "고객 상세", str(uid),
                       "이 고객의 위험 점수, 위험을 올린 행동, 맞춤 유지 활동입니다.", [" ".join(pills)])
    left, right = st.columns([1, 1.3])
    with left:
        with st.container(border=True):
            color = ui.RED if r["score"] >= THR else ui.MINT
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=float(r["score"]), number=dict(suffix="점", font=dict(size=42, color=ui.TEXT)),
                title=dict(text="위험 점수 (0~100)", font=dict(size=14, color=ui.MUTED)),
                gauge=dict(axis=dict(range=[0, 100], tickcolor=ui.MUTED), bar=dict(color=color, thickness=.28),
                           bgcolor="#1C1C27", borderwidth=0,
                           steps=[dict(range=[0, THR], color="#18231F"), dict(range=[THR, 100], color="#2A1720")],
                           threshold=dict(line=dict(color="#FFFFFF", width=3), value=THR))))
            ui.plot(ui.style(fig, 260, legend=False))
            fig = go.Figure(go.Bar(x=[r["ml_score"], r["nn_score"]], y=["머신러닝", "신경망"], orientation="h",
                                   marker_color=[ui.VIOLET, ui.PINK], text=[f"{r['ml_score']:.1f}", f"{r['nn_score']:.1f}"],
                                   textposition="outside"))
            fig.update_layout(xaxis_range=[0, 110])
            ui.plot(ui.style(fig, 150, legend=False))
    with right:
        ui.section("이 고객에게 추천하는 유지 활동", "위험을 올린 행동 순서대로")
        if items:
            for it in items:
                detail = (f"점수 영향 +{it['contribution']:.1f}점" if it["feature"] == "level_last" else
                          f"{it['name']} {float(it['value']):,.1f} (유지 고객 중앙값 {float(it['reference']):,.1f}) · "
                          f"점수 영향 +{it['contribution']:.1f}점")
                ui.reason_card(it["icon"], it["reason"], detail, it["action"])
        else:
            ui.reason_card("🙂", "뚜렷한 위험 신호가 없어요", "현재 분석에서 두드러진 신호를 찾지 못했습니다.", C.DEFAULT_ACTION)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        with st.container(border=True):
            ui.section("행동별 위험 신호", "분홍색은 위험 증가 방향, 초록색은 감소 방향입니다.")
            order = contrib.reindex(contrib.abs().sort_values().index)
            fig = go.Figure(go.Bar(x=order.values, y=[C.FEATURE_KO[f] for f in order.index], orientation="h",
                                   marker_color=[ui.PINK if v > 0 else ui.MINT for v in order.values],
                                   hovertemplate="%{y}: %{x:+.1f}점<extra></extra>"))
            fig.update_layout(xaxis_title="위험 점수 영향 (점)")
            ui.plot(ui.style(fig, 430, legend=False))
    with c2:
        with st.container(border=True):
            ui.section("이용 행동이 바뀌면?", "유지 활동으로 행동이 바뀌었다고 가정")
            cand = [it["feature"] for it in items if it["feature"] in C.NUMERIC][:3]
            cand = (cand + [f for f in ["ad_count", "thumbs_down_count", "thumbs_up_count"] if f not in cand])[:3]
            changes = {}
            base_row = row_df.iloc[0]
            for f in cand:
                hi = float(max(svc.team[f].quantile(0.99), float(base_row[f]) * 1.5, 1))
                lo=0.0
                if f=='song_count_change':
                    hi=max(hi,abs(float(base_row[f]))*1.5);lo=-hi
                elif f=='active_days':lo,hi=1.0,28.0
                elif f=='session_count':lo=1.0
                elif f=='days_since_last_activity':hi=27.999
                changes[f] = st.slider(C.FEATURE_KO[f], lo, hi, float(base_row[f]), .1 if f=='days_since_last_activity' else 1.0, key=f"wi_{uid}_{f}")
            changes["level_last"] = st.radio("요금제", ["free", "paid"], index=0 if base_row["level_last"] == "free" else 1,
                                             horizontal=True, key=f"wi_lv_{uid}")
            try:
                before, after = svc.what_if(S, base_row, changes)
            except ValueError as error:
                st.warning(str(error))
                before,after=float(r['score']),float(r['score'])
            st.caption('시퀀스는 고정하고 정형 입력만 바꾸는 비교입니다. 실제 유지 활동의 효과를 뜻하지 않습니다.')
            dlt = after - before
            col = ui.MINT if dlt < 0 else (ui.PINK if dlt > 0 else ui.MUTED)
            st.markdown(f'<div class="kpi"><div class="label">바뀐 위험 점수</div><div class="value">{after:.1f}점</div>'
                        f'<div class="sub" style="color:{col};font-weight:800">{dlt:+.1f}점 (변경 전 {before:.1f}점) · '
                        f'{"이탈 예상" if after >= THR else "유지 예상"}</div></div>', unsafe_allow_html=True)
    player()

# ---------------------------------------------------------------
# 💰 캠페인
# ---------------------------------------------------------------
elif page == PAGES[2]:
    ui.playlist_header("💰", "캠페인 플래너", "몇 명에게 연락할까?",
                       "위험 점수가 높은 순으로 연락한다고 가정하고, 이익이 가장 큰 인원을 찾습니다.",
                       [f"대상 {N:,}명", f"운영 기준 {S.label}"])
    with st.container(border=True):
        a, b, c, d = st.columns(4)
        value = a.number_input("붙잡은 고객 1명의 가치 (원)", 0, 10_000_000, 120_000, 10_000,
                               help="예: 월 구독료 1만원 × 남은 기대 이용 12개월")
        cost = b.number_input("1명당 연락·혜택 비용 (원)", 0, 1_000_000, 5_000, 1_000)
        succ = c.slider("해지 예정 고객 방어 성공률", 0.0, 1.0, 0.30, 0.05, help="홈의 대응 결과 확인에서 관찰한 감소율을 참고하세요.")
        use_budget = d.toggle("예산 제한", value=False)
        budget = d.number_input("예산 (원)", 0, 1_000_000_000, 3_000_000, 100_000, disabled=not use_budget)
    curve, best = svc.campaign(S, N, value, cost, succ, budget if use_budget else None)
    view = curve[curve["frac"] <= max(0.3, best["frac"] * 3)]
    be = cost / (value * succ) if value * succ > 0 else float("inf")
    k = st.columns(5)
    for col, (lab, val, sub, hot) in zip(k, [
            ("추천 연락 인원", f"{best['contacts']:,}명", f"위험 점수 상위 {best['frac']:.0%}", True),
            ("예상 포착 해지자", f"{best['churners']:,.0f}명", f"정밀도 {best['precision']:.1%}", False),
            ("예상 방어 고객", f"{best['saved']:,.1f}명", f"성공률 {succ:.0%}", False),
            ("비용", f"{best['cost']:,.0f}원", "연락·혜택", False),
            ("예상 이익", f"{best['profit']:,.0f}원", "방어 가치 − 비용", False)]):
        with col:
            ui.kpi(lab, val, sub, hot=hot)
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            ui.section("연락 인원별 예상 이익", "★ = 이익이 가장 큰 지점")
            fig = go.Figure(go.Scatter(x=view["contacts"], y=view["profit"], mode="lines", line=dict(color=ui.VIOLET, width=3),
                                       fill="tozeroy", fillcolor="rgba(139,92,246,.18)",
                                       hovertemplate="%{x:,}명 → %{y:,.0f}원<extra></extra>"))
            fig.add_scatter(x=[best["contacts"]], y=[best["profit"]], mode="markers",
                            marker=dict(symbol="star", size=18, color=ui.PINK), hoverinfo="skip")
            fig.add_hline(y=0, line_color="#3A3A50")
            fig.update_layout(xaxis_title="연락 인원", yaxis_title="예상 이익 (원)")
            ui.plot(ui.style(fig, 330, legend=False))
    with c2:
        with st.container(border=True):
            ui.section("정밀도 vs 손익분기", f"손익분기 정밀도 = 비용 ÷ (가치 × 성공률) = {be:.1%}")
            fig = go.Figure(go.Scatter(x=view["contacts"], y=view["precision"], mode="lines", line=dict(color=ui.PINK, width=3)))
            fig.add_hline(y=min(be, 1), line_dash="dash", line_color="#FFFFFF", annotation_text="손익분기",
                          annotation_font_color="#FFFFFF")
            fig.add_hline(y=svc.reg["base_rate"]["valid"], line_dash="dot", line_color="#6B6B85", annotation_text="무작위",
                          annotation_font_color="#9A9AB0")
            fig.update_layout(xaxis_title="연락 인원", yaxis_title="정밀도 (검증셋)", yaxis_tickformat=".0%")
            ui.plot(ui.style(fig, 330, legend=False))
    if best["profit"] <= 0:
        st.warning("이 가정에서는 현재 탐색한 범위에서는 연락하지 않는 선택이 유리합니다. 비용을 낮추거나 효과가 큰 혜택을 찾아야 합니다.")
    elif has_truth:
        known = A.head(best["contacts"])["churn"].dropna()
        st.success(f"실제 결과가 있는 고객으로 확인: 상위 {best['contacts']:,}명에게 연락했다면 실제 해지자 {int(known.sum())}명을 "
                   f"잡았습니다 (정밀도 {known.mean():.1%}).")
    ui.note("정밀도는 검증셋에서 같은 비율을 골랐을 때의 실제 값입니다. 방어 성공률은 가정이므로, 실제 캠페인은 "
            "홈의 대응 결과 확인에서 같은 기간의 비교군과 함께 확인하세요.")
    player()

# ---------------------------------------------------------------
# 🗂️ 고객 자료
# ---------------------------------------------------------------
elif page == PAGES[3]:
    ui.playlist_header("🗂️", "라이브러리", "고객 자료 관리", "분석할 고객 자료를 추가·갱신하거나 직접 입력합니다.",
                       [f"현재 {N:,}명", data_desc])
    if "data_msg" in st.session_state:
        st.success(st.session_state.pop("data_msg"))
    operation_date=st.date_input('추가·교체할 이용 자료의 예측 기준일',value=active_date,key='operation_date')
    st.caption('날짜만 바꾸면 이용 기록이 다시 집계되지는 않습니다. 다른 기준일의 자료는 현재 자료와 섞지 말고 전체 교체하거나 저장 기록에서 불러오세요.')
    with st.expander('신경망 입력 자료 등록'):
        st.caption('학습 때와 같은 번호 체계로 만든 입력이 필요합니다. 준비된 압축 배열에 user_id와 tokens를 넣으세요. tokens는 고객당 정수 50개입니다.')
        sequence_upload=st.file_uploader('고객 번호가 포함된 시퀀스 파일',type=['npz'],key='sequence_upload')
        sequence_confirm=st.checkbox('학습과 같은 행동 번호·길이·관찰 기간이며 고객 번호 연결을 확인했습니다.')
        if st.button('이 기준일의 시퀀스 등록',disabled=sequence_upload is None or not sequence_confirm):
            try:
                count=register_inputs(io.BytesIO(sequence_upload.getvalue()),operation_date,C.STORE)
            except (ValueError,KeyError,OSError) as error:
                st.error(str(error))
            else:
                st.session_state.data_msg=f'신경망 입력 {count:,}명을 등록했습니다.'
                st.rerun()
    t1, t2, t3 = st.tabs(["파일 추가·갱신", "고객 직접 입력", "저장 기록"])
    with t1:
        st.caption("전처리된 고객별 자료(팀 공통 피처 14개 + user_id)가 필요합니다. 원본 행동 로그를 그대로 올리는 방식은 아직 연결하지 않았습니다.")
        up = st.file_uploader("고객별 이용 자료 파일", type=["csv"])
        mode = st.radio("파일 적용 방법", ["현재 자료에 추가·갱신", "현재 자료 교체"], horizontal=True)
        conflict = st.radio("같은 고객 번호가 있으면", ["기존 자료 유지", "새 자료로 갱신"], horizontal=True,
                            disabled=mode == "현재 자료 교체")
        tmpl = svc.team[svc.team["split"] == "valid"].head(5)[["user_id"] + C.FEATURES]
        st.download_button("입력 양식 내려받기", tmpl.to_csv(index=False).encode("utf-8-sig"), file_name="입력양식.csv",
                           mime="text/csv")
        if up is not None:
            raw = up.getvalue()
            try:
                new = C.read_customers(raw)
            except ValueError as error:
                st.error(str(error))
                st.stop()
            errors, warns = C.check_upload(new)
            for w in warns:
                st.warning(w)
            for er in errors:
                st.error(er)
            if not errors:
                st.caption(f"미리 보기: {len(new):,}행")
                st.dataframe(new.head(5), hide_index=True)
                if st.button("이 파일 적용", type="primary"):
                    if mode=='현재 자료에 추가·갱신' and operation_date!=active_date:
                        st.error('서로 다른 예측 기준일의 기록을 합칠 수 없습니다. 현재 자료 교체를 선택하세요.')
                        st.stop()
                    merged, info = C.merge_customers(customers, new, "replace" if mode == "현재 자료 교체" else "append",
                                                     "update" if conflict == "새 자료로 갱신" else "keep")
                    merged.attrs['as_of']=operation_date.isoformat()
                    try:svc.validate_population(S,merged)
                    except SequenceInputError as error:
                        st.error(str(error));st.stop()
                    C.save_customers(merged,as_of=operation_date)
                    detail = (f"{up.name} · 추가 {info['added']} · 갱신 {info['updated']} · 기존 유지 {info['kept']}"
                              if mode != "현재 자료 교체" else f"{up.name} · 전체 교체 {info['total']}명")
                    C.log_history("파일 " + ("교체" if mode == "현재 자료 교체" else "추가·갱신"), detail, info["total"])
                    st.session_state.data_msg = f"적용했습니다: {detail} → 현재 {info['total']:,}명"
                    st.rerun()
    with t2:
        with st.form("one"):
            uid_new = st.text_input("고객 번호", placeholder="예: NEW-0001")
            cols = st.columns(4)
            vals = {}
            for i, f in enumerate(C.NUMERIC):
                ref = svc.reference[f]
                if f=='song_count_change':
                    continue
                if f=='days_since_last_activity':
                    vals[f] = cols[i % 4].number_input(C.FEATURE_KO[f],min_value=0.0,max_value=27.999,value=float(ref),step=.1,key=f'in_{f}')
                else:
                    minimum=1 if f in ['active_days','session_count'] else 0
                    vals[f] = cols[i % 4].number_input(C.FEATURE_KO[f],min_value=minimum,value=max(minimum,int(round(ref))),step=1,key=f'in_{f}')
            vals['song_count_change']=vals['recent_7d_song_count']-vals['previous_7d_song_count']
            vals["level_last"] = cols[len(C.NUMERIC) % 4].selectbox("요금제", ["free", "paid"])
            conflict1 = st.radio("같은 고객 번호가 있으면", ["기존 자료 유지", "새 자료로 갱신"], horizontal=True, key="c1")
            ok = st.form_submit_button("고객 추가", type="primary")
        if ok:
            if operation_date!=active_date:
                st.error('직접 입력은 현재 자료와 같은 기준일에 추가하세요.')
                st.stop()
            if not uid_new.strip():
                st.error("고객 번호를 입력하세요.")
            else:
                one = pd.DataFrame([{"user_id": uid_new.strip(), **vals}])
                errors,_=C.check_upload(one)
                if errors:
                    st.error(' / '.join(errors))
                    st.stop()
                try:sc = svc.scores(S, one)["score"].iloc[0]
                except SequenceInputError as error:
                    st.error(str(error));st.stop()
                merged, info = C.merge_customers(customers, one, "append",
                                                 "update" if conflict1 == "새 자료로 갱신" else "keep")
                C.save_customers(merged)
                act = "추가" if info["added"] else ("갱신" if info["updated"] else "기존 유지")
                C.log_history("직접 입력", f"{uid_new.strip()} {act} · 위험 점수 {sc:.1f}", info["total"])
                st.session_state.data_msg = f"{uid_new.strip()} {act} · 위험 점수 {sc:.1f}점 ({'이탈 예상' if sc >= THR else '유지 예상'})"
                st.rerun()
        st.caption("기본값은 학습셋 유지 고객의 중앙값입니다.")
    with t3:
        st.subheader('기준일별 고객 자료 보관')
        snapshot_date=st.date_input('이용 기록의 예측 기준일',value=date(2018,11,1),key='snapshot_date')
        st.caption('날짜를 바꿔도 이용 기록이 자동으로 다시 집계되지는 않습니다. 해당 기준일 이전 28일의 자료를 저장하세요.')
        snapshot_policy=st.radio('같은 기준일·고객 번호가 있으면',['저장된 값 유지','현재 값으로 갱신'],horizontal=True,key='snapshot_policy')
        if st.button('현재 자료를 기준일별로 저장'):
            stats=C.snapshot_storage.save_snapshot(C.STORE/'customer_snapshots.sqlite3',snapshot_date,customers,'keep' if snapshot_policy=='저장된 값 유지' else 'update')
            st.success(f"저장 완료 · 추가 {stats['추가']}명 · 갱신 {stats['갱신']}명 · 기존 유지 {stats['유지']}명")
        dates=C.snapshot_storage.snapshot_dates(C.STORE/'customer_snapshots.sqlite3')
        if dates:
            st.dataframe(pd.DataFrame(dates),hide_index=True)
            selected_day=st.selectbox('불러올 기준일',[entry['기준일'] for entry in dates])
            if st.button('이 기준일 고객 자료 불러오기'):
                restored=C.snapshot_storage.load_snapshot(C.STORE/'customer_snapshots.sqlite3',selected_day)
                restored['churn']=np.nan
                C.save_customers(restored,as_of=selected_day)
                C.log_history('기준일 자료 복원',selected_day,len(restored))
                st.session_state.data_msg=f'{selected_day} 자료 {len(restored):,}명을 불러왔습니다.'
                st.rerun()
        st.caption('갱신 전 값은 데이터베이스의 수정 이력으로 보존합니다.')
        st.subheader('작업 기록')
        hist = C.load_history()
        if hist.empty:
            st.info("아직 저장 기록이 없습니다.")
        else:
            st.dataframe(hist.iloc[::-1], hide_index=True)
    st.divider()
    ui.section("현재 분석 자료", f"{N:,}명 · 새 자료를 추가해도 모델은 자동으로 재학습되지 않습니다.")
    x1, x2, _ = st.columns([1, 1, 2])
    x1.download_button("현재 고객 자료 전체 내려받기", customers.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"고객자료_{date.today():%Y%m%d}.csv", mime="text/csv")
    if x2.button("기본 자료(검증 예시 고객)로 되돌리기", disabled=is_default_data):
        C.reset_customers()
        C.log_history("초기화", "기본 자료(검증 예시 고객)로 되돌림", len(C.default_customers(svc.team)))
        st.session_state.data_msg = "기본 자료로 되돌렸습니다."
        st.rerun()
    player()

# ---------------------------------------------------------------
# ⚙️ 모델 설정
# ---------------------------------------------------------------
elif page == PAGES[4]:
    ui.playlist_header("⚙️", "운영 설정", "설정",
                       "확정한 모델의 연결 상태와 운영 기준을 관리합니다.",
                       [f"현재 운영: {S.label}", f"판정 기준 {THR:.1f}점"])
    if "set_msg" in st.session_state:
        st.success(st.session_state.pop("set_msg"))
    with st.container(border=True):
        ui.section("최적 탐색 결과", "팀원이 저장한 설정을 사용하며 앱에서 조합을 다시 탐색하지 않습니다.")
        st.markdown(f"**{PROFILE['name']}**")
        st.write(f"확률 결합 판정 기준: {PROFILE['best_threshold']:.6f}")
        st.success(PROFILE["status"])
        st.caption(PROFILE["reason"])
        def apply_optimal():
            C.save_setting(C.default_setting(svc))
            st.session_state.set_msg='팀원이 확정한 최적 조합을 적용했습니다.'
        st.button("최적 탐색 결과 적용",on_click=apply_optimal,disabled=S==C.default_setting(svc))
        st.caption(f"현재 운영: {REG[S.ml]['name']} / {REG[S.nn]['name']} · {S.label}")
    with st.expander("상세 옵션 · 모델과 판정 기준", expanded=False):
        ml_keys = [k for k, v in REG.items() if v["kind"] == "ml"]
        nn_keys = [k for k, v in REG.items() if v["kind"] == "nn"]
        with st.container(border=True):
            modes = ["optimal", "nn", "ml", "blend"]
            mode = st.radio("운영 기준", modes, index=modes.index(S.mode), format_func=C.MODE_NAMES.get, horizontal=True)
            a, b = st.columns(2)
            ml = a.selectbox("머신러닝 모델", ml_keys, index=ml_keys.index(S.ml), format_func=lambda k: REG[k]["name"], disabled=mode=='optimal')
            nn = b.selectbox("신경망 모델", nn_keys, index=nn_keys.index(S.nn), format_func=lambda k: REG[k]["name"], disabled=mode=='optimal')
            if mode=='optimal':
                fixed=C.default_setting(svc)
                ml,nn=fixed.ml,fixed.nn
            w = st.slider("결합할 때 머신러닝 모델 비중", 0, 100, int(round(S.ml_weight * 100)), 5, disabled=mode != "blend")
            st.caption(f"최적 조합은 저장된 확률 비중 {PROFILE['ml_weight']:.0%}:{PROFILE['dl_weight']:.0%}와 판정 기준을 그대로 사용합니다." if mode=='optimal' else f'수동 결합은 머신러닝 {w}% + 신경망 {100-w}%의 백분위 점수 결합입니다.')
            use_saved = st.checkbox("저장된 판정 기준 사용", value=S.use_saved,
                                    help="끄면 위험 점수 기준을 직접 정합니다")
            manual = st.slider("직접 정할 위험 점수 기준", 0.0, 100.0, float(S.manual_threshold), 0.5, disabled=use_saved)
            draft = C.Setting(ml=ml, nn=nn, mode=mode, ml_weight=PROFILE["ml_weight"] if mode=="optimal" else w / 100, use_saved=use_saved, manual_threshold=manual)
            st.caption(f"적용할 판정 기준: {svc.threshold(draft):.1f}점")

            def apply_setting(d=draft):
                C.save_setting(d)
                st.session_state.set_msg = f"적용했습니다: {d.label} · 기준 {svc.threshold(d):.1f}점"

            def reset_setting():
                (C.STORE / C.SETTING_FILE).unlink(missing_ok=True)
                st.session_state.set_msg = "저장된 최적 조합으로 되돌렸습니다."

            b1, b2, _ = st.columns([1, 1, 2])
            b1.button("상세 설정 적용", type="primary", on_click=apply_setting, disabled=draft == S)
            b2.button("최적 조합 기본값으로", on_click=reset_setting, disabled=not (C.STORE / C.SETTING_FILE).exists())
    with st.expander("두 모델의 현재 예측 비교", expanded=False):
        if input_issue is None:show_prediction_comparison()
        else:st.info(input_issue)
    with st.expander("보고서 작성용 자료 내려받기", expanded=False):
        st.caption("모델 성능 수치와 지표 설명은 보고서 파일에서 확인하세요.")
        report = O.model_report(svc, S)
        st.download_button("모델 평가·설명 보고서 내려받기", report.encode('utf-8-sig'),
                           file_name="모델평가_보고서자료.md", mime="text/markdown")
        results = O.load_campaigns()
        if results:
            import json
            st.download_button("대응 결과 상세 보고서 내려받기", json.dumps(results, ensure_ascii=False, indent=2).encode('utf-8-sig'),
                               file_name="대응결과_상세보고서.json", mime="application/json")
    player()
