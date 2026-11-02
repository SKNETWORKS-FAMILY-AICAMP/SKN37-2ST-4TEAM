"""🎧 Churn Care — 음악 스트리밍 이탈 위험 고객 관리 서비스 (Streamlit)

실행 (저장소 폴더에서):  streamlit run service/app.py
처음 실행하면 팀 공통 데이터로 머신러닝·신경망 모델을 학습해 model/service/ 에 저장합니다 (10초 안팎).
고객 자료를 추가·갱신하면 service_data/ 에 저장됩니다.
"""
from __future__ import annotations

import io
import sys
from dataclasses import astuple, replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import core as C  # noqa: E402
import ui  # noqa: E402

st.set_page_config(page_title="Churn Care · 이탈 위험 고객 관리", page_icon="🎧", layout="wide")
ui.inject_css()

PAGES = ["🏠 홈", "🎯 관리 목록", "👤 고객 상세", "💰 캠페인", "🗂️ 고객 자료", "⚙️ 모델 설정", "📊 성능 지표"]


@st.cache_resource(show_spinner=False)
def get_service() -> C.Service:
    bar = st.progress(0.0, text="처음 실행: 모델을 학습하는 중...")
    svc = C.load_service(progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
    bar.empty()
    return svc


@st.cache_data(show_spinner="고객별 위험 점수와 이유를 계산하는 중...")
def analyze(_svc: C.Service, setting: tuple, data_key: str, _df: pd.DataFrame) -> pd.DataFrame:
    out = _svc.analyze(C.Setting(*setting), _df)
    return out.join(_df[C.FEATURES + ["churn"]]).sort_values("score", ascending=False)


def go_to(page: str):
    st.session_state.nav = page


def wide_button(label: str, **kw):
    """Streamlit 버전에 따라 너비 옵션 이름이 달라서 둘 다 지원."""
    try:
        new = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 46)
    except ValueError:
        new = False
    return st.button(label, width="stretch", **kw) if new else st.button(label, use_container_width=True, **kw)


svc = get_service()
if "nav" not in st.session_state:
    st.session_state.nav = PAGES[0]
S: C.Setting = C.load_setting(svc)  # 운영 설정은 파일에 저장되어 모든 사용자에게 같게 적용
REG = svc.reg["models"]
THR = svc.threshold(S)

customers = C.load_customers(svc.team)
data_key = str(pd.util.hash_pandas_object(customers, index=False).sum())
A = analyze(svc, astuple(S), data_key, customers)
N = len(A)
flag = A["score"] >= THR
has_truth = A["churn"].notna().any()
is_default_data = not (C.STORE / C.CURRENT).exists()
data_desc = "테스트 고객 (학습에 쓰지 않은 고객)" if is_default_data else "저장된 고객 자료"

# ---------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">🎧</div><div class="name">Churn Care</div></div>'
                '<div style="color:#9A9AB0;font-size:.8rem;margin-bottom:6px">이탈 위험 고객 관리</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="lib">MENU</div>', unsafe_allow_html=True)
    st.radio("메뉴", PAGES, key="nav", label_visibility="collapsed")
    st.markdown('<div class="lib">LIBRARY</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-card">📁 <b>{N:,}명</b> · {data_desc}<br>🎚️ <b>{S.label}</b><br>'
        f'🤖 {REG[S.ml]["name"].split(" · ")[0]} + {REG[S.nn]["name"].split(" · ")[0]}<br>'
        f'🎯 기준 <b>{THR:.1f}점</b> 이상 = 이탈 예상<br><br>'
        f'관찰 2018-10-04 ~ 11-01 (28일)<br>예측 2018-11-01 ~ 11-15 (14일)</div>', unsafe_allow_html=True)

page = st.session_state.nav


def player():
    ui.player_bar(f"지금 운영 중 · {S.label}",
                  f"{REG[S.ml]['name']} + {REG[S.nn]['name']} · 기준 {THR:.1f}점",
                  float(flag.mean()), f"이탈 예상 {int(flag.sum()):,}명", f"{N:,}명",
                  f"분석 자료 {N:,}명 · {'테스트 고객' if is_default_data else '저장된 자료'}")


# ---------------------------------------------------------------
# 🏠 홈
# ---------------------------------------------------------------
if page == PAGES[0]:
    ui.topline([f"분석 자료: {data_desc} {N:,}명", "실제 해지 여부는 검증용으로만 표시합니다."], live="운영 중")
    ui.playlist_header("🔥", "오늘의 관리 플레이리스트", "오늘 먼저 확인할 고객",
                       "두 모델이 고객을 어떻게 판단하는지 보고, 실제 관리할 목록으로 이어가세요.",
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
        if has_truth:
            known = A["churn"].notna()
            caught = int(A.loc[flag & known, "churn"].sum())
            ui.kpi("실제 해지자 포착 (검증용)", f"{caught} / {int(A.loc[known, 'churn'].sum())}명",
                   f"재현율 {caught / max(int(A.loc[known, 'churn'].sum()), 1):.0%}")
        else:
            ui.kpi("평균 위험 점수", f"{A['score'].mean():.1f}점", "0~100점")

    st.write("")
    c1, c2 = st.columns([2.1, 1])
    with c1:
        with st.container(border=True):
            ui.section("두 모델이 바라본 고객", "점 하나가 고객 한 명 · 점선 = 각 모델의 판정 기준")
            t_ml, t_nn = svc.model_threshold(S.ml, "ml"), svc.model_threshold(S.nn, "nn")
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
            v = svc.evaluate(S, "valid")
            st.caption(f"검증셋 재현율 {v['recall']:.0%} · 정밀도 {v['precision']:.1%} · PR-AUC {v['pr_auc']:.3f}")
            wide_button("설정에서 모델 바꾸기", on_click=go_to, args=(PAGES[5],))
            wide_button("성능 지표 보기", on_click=go_to, args=(PAGES[6],))

    with st.container(border=True):
        ui.section("우선 확인할 고객 10명", "위험 점수가 높은 순 · 판정 비교 = 머신러닝과 신경망의 판정이 같은지")
        ui.track_list(A.head(10), show_truth=has_truth)
        st.write("")
        st.button("관리 목록으로 이동 ▶", on_click=go_to, args=(PAGES[1],), type="primary")
    player()

# ---------------------------------------------------------------
# 🎯 관리 목록
# ---------------------------------------------------------------
elif page == PAGES[1]:
    ui.playlist_header("🎯", "관리 플레이리스트", "관리 목록", "연락할 고객을 고르고, CRM에 바로 올릴 수 있는 목록을 내려받습니다.",
                       [f"{N:,}명 중 선택", f"운영 기준 {S.label}"])
    with st.container(border=True):
        a, b, c, d = st.columns([1.3, 1.2, 1, 1])
        method = a.radio("선정 방식", ["판정 기준", "상위 N명", "예산"], horizontal=True)
        if method == "판정 기준":
            thr = b.slider("위험 점수 기준", 0.0, 100.0, float(THR), 0.5, format="%.1f")
            mask = A["score"].to_numpy() >= thr
        else:
            if method == "상위 N명":
                topn = b.number_input("관리 인원", 1, N, max(1, N // 10), 10)
            else:
                budget = b.number_input("예산 (원)", 0, 1_000_000_000, 3_000_000, 100_000)
                unit = b.number_input("1명당 비용 (원)", 1, 1_000_000, 5_000, 1_000)
                topn = int(budget // unit)
            mask = np.zeros(N, dtype=bool)
            mask[: min(int(topn), N)] = True
        agree = c.selectbox("판정 비교", ["전체", "일치만", "다름만"])
        levels = d.multiselect("요금제", ["free", "paid"], default=["free", "paid"])
    m2 = mask & A["level_last"].isin(levels).to_numpy()
    if agree != "전체":
        m2 &= (A["agree"] == ("일치" if agree == "일치만" else "다름")).to_numpy()
    sel = A[m2]
    k = st.columns(4)
    with k[0]:
        ui.kpi("선택된 고객", f"{len(sel):,}명", f"전체의 {len(sel) / max(N, 1):.1%}", hot=True)
    with k[1]:
        ui.kpi("평균 위험 점수", f"{sel['score'].mean():.1f}점" if len(sel) else "-", "0~100점")
    with k[2]:
        ui.kpi("판정 다름", f"{int((sel['agree'] == '다름').sum()):,}명", "두 모델이 엇갈린 고객")
    with k[3]:
        known = sel["churn"].notna()
        if known.any():
            got = int(sel.loc[known, "churn"].sum())
            ui.kpi("실제 해지자 (검증용)", f"{got}명", f"정밀도 {got / max(int(known.sum()), 1):.1%}")
        else:
            ui.kpi("주요 활동", sel["action"].str.split(" / ").explode().mode().iloc[0] if len(sel) else "-", "가장 많이 추천")
    st.write("")
    cols = ["user_id", "score", "pred", "agree", "ml_score", "nn_score", "level_last", "reason_1", "action", "reason_2",
            "days_since_last_activity", "recent_7d_song_count"] + (["churn"] if has_truth else [])
    names = {"user_id": "고객 번호", "score": "위험 점수", "pred": "예측 결과", "agree": "판정 비교", "ml_score": "머신러닝 점수",
             "nn_score": "신경망 점수", "level_last": "요금제", "reason_1": "주요 이유 1", "reason_2": "주요 이유 2",
             "action": "추천 유지 활동", "days_since_last_activity": "마지막 활동(일 전)",
             "recent_7d_song_count": "최근 7일 재생", "churn": "실제 해지(검증용)"}
    show = sel[cols].rename(columns=names)
    st.dataframe(show, hide_index=True, height=470, column_config={
        "위험 점수": st.column_config.ProgressColumn("위험 점수", min_value=0.0, max_value=100.0, format="%.1f"),
        "머신러닝 점수": st.column_config.NumberColumn(format="%.1f"),
        "신경망 점수": st.column_config.NumberColumn(format="%.1f"),
        "마지막 활동(일 전)": st.column_config.NumberColumn(format="%.1f")})
    st.download_button("⬇ CRM 업로드용 CSV", show.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"관리목록_{date.today():%Y%m%d}.csv", mime="text/csv", type="primary")
    ui.note("이유 = 그 값만 일반 유지 고객 수준으로 바꾸면 위험 점수가 내려가는 행동입니다. 모델 판단의 설명이며 원인의 증명은 아닙니다. "
            "추천 활동은 규칙 기반 예시로, 효과는 '성능 지표 → 대응 효과' 방식의 A/B 테스트로 확인해야 합니다.")
    player()

# ---------------------------------------------------------------
# 👤 고객 상세
# ---------------------------------------------------------------
elif page == PAGES[2]:
    a, b = st.columns([1, 1])
    top = A.head(300)
    score_of = dict(zip(top["user_id"], top["score"]))
    typed = b.text_input("고객 번호로 찾기", placeholder="예: 1000035")
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
    if r["churn"] == r["churn"]:
        pills.append(ui.pill("실제: 해지" if r["churn"] == 1 else "실제: 유지", ui.MUTED))
    ui.playlist_header("👤", "NOW PLAYING · 고객 상세", str(uid),
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
            ui.reason_card("🙂", "뚜렷하게 위험을 올린 행동이 없어요", "모든 행동이 일반 유지 고객과 비슷합니다.", C.DEFAULT_ACTION)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        with st.container(border=True):
            ui.section("행동별 영향", "오른쪽(분홍) = 그 값 때문에 위험 점수가 올라감")
            order = contrib.reindex(contrib.abs().sort_values().index)
            fig = go.Figure(go.Bar(x=order.values, y=[C.FEATURE_KO[f] for f in order.index], orientation="h",
                                   marker_color=[ui.PINK if v > 0 else ui.MINT for v in order.values],
                                   hovertemplate="%{y}: %{x:+.1f}점<extra></extra>"))
            fig.update_layout(xaxis_title="위험 점수 영향 (점)")
            ui.plot(ui.style(fig, 430, legend=False))
    with c2:
        with st.container(border=True):
            ui.section("값을 바꿔 보면? (what-if)", "유지 활동으로 행동이 바뀌었다고 가정")
            cand = [it["feature"] for it in items if it["feature"] in C.NUMERIC][:3]
            cand = (cand + [f for f in ["ad_count", "thumbs_down_count", "thumbs_up_count"] if f not in cand])[:3]
            changes = {}
            base_row = row_df.iloc[0]
            for f in cand:
                hi = float(max(svc.team[f].quantile(0.99), float(base_row[f]) * 1.5, 1))
                changes[f] = st.slider(C.FEATURE_KO[f], 0.0, hi, float(base_row[f]), 1.0, key=f"wi_{uid}_{f}")
            changes["level_last"] = st.radio("요금제", ["free", "paid"], index=0 if base_row["level_last"] == "free" else 1,
                                             horizontal=True, key=f"wi_lv_{uid}")
            before, after = svc.what_if(S, base_row, changes)
            dlt = after - before
            col = ui.MINT if dlt < 0 else (ui.PINK if dlt > 0 else ui.MUTED)
            st.markdown(f'<div class="kpi"><div class="label">바뀐 위험 점수</div><div class="value">{after:.1f}점</div>'
                        f'<div class="sub" style="color:{col};font-weight:800">{dlt:+.1f}점 (변경 전 {before:.1f}점) · '
                        f'{"이탈 예상" if after >= THR else "유지 예상"}</div></div>', unsafe_allow_html=True)
    player()

# ---------------------------------------------------------------
# 💰 캠페인
# ---------------------------------------------------------------
elif page == PAGES[3]:
    ui.playlist_header("💰", "캠페인 플래너", "몇 명에게 연락할까?",
                       "위험 점수가 높은 순으로 연락한다고 가정하고, 이익이 가장 큰 인원을 찾습니다.",
                       [f"대상 {N:,}명", f"운영 기준 {S.label}"])
    with st.container(border=True):
        a, b, c, d = st.columns(4)
        value = a.number_input("붙잡은 고객 1명의 가치 (원)", 0, 10_000_000, 120_000, 10_000,
                               help="예: 월 구독료 1만원 × 남은 기대 이용 12개월")
        cost = b.number_input("1명당 연락·혜택 비용 (원)", 0, 1_000_000, 5_000, 1_000)
        succ = c.slider("해지 예정 고객 방어 성공률", 0.0, 1.0, 0.30, 0.05, help="A/B 테스트로 추정 (성능 지표 → 대응 효과)")
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
        st.warning("이 가정에서는 누구에게 연락해도 손해입니다. 비용을 낮추거나 효과가 큰 혜택을 찾아야 합니다.")
    elif has_truth:
        known = A.head(best["contacts"])["churn"].dropna()
        st.success(f"실제 결과가 있는 고객으로 확인: 상위 {best['contacts']:,}명에게 연락했다면 실제 해지자 {int(known.sum())}명을 "
                   f"잡았습니다 (정밀도 {known.mean():.1%}).")
    ui.note("정밀도는 검증셋에서 같은 비율을 골랐을 때의 실제 값입니다. 방어 성공률은 가정이므로, 실제 캠페인은 "
            "'성능 지표 → 대응 효과'처럼 대조군을 두고 측정하세요.")
    player()

# ---------------------------------------------------------------
# 🗂️ 고객 자료
# ---------------------------------------------------------------
elif page == PAGES[4]:
    ui.playlist_header("🗂️", "라이브러리", "고객 자료 관리", "분석할 고객 자료를 추가·갱신하거나 직접 입력합니다.",
                       [f"현재 {N:,}명", data_desc])
    if "data_msg" in st.session_state:
        st.success(st.session_state.pop("data_msg"))
    t1, t2, t3 = st.tabs(["파일 추가·갱신", "고객 직접 입력", "저장 기록"])
    with t1:
        st.caption("전처리된 고객별 자료(팀 공통 피처 14개 + user_id)가 필요합니다. 원본 행동 로그를 그대로 올리는 방식은 아직 연결하지 않았습니다.")
        up = st.file_uploader("고객별 이용 자료 파일", type=["csv"])
        mode = st.radio("파일 적용 방법", ["현재 자료에 추가·갱신", "현재 자료 교체"], horizontal=True)
        conflict = st.radio("같은 고객 번호가 있으면", ["기존 자료 유지", "새 자료로 갱신"], horizontal=True,
                            disabled=mode == "현재 자료 교체")
        tmpl = svc.team[svc.team["split"] == "test"].head(5)[["user_id"] + C.FEATURES]
        st.download_button("입력 양식 내려받기", tmpl.to_csv(index=False).encode("utf-8-sig"), file_name="입력양식.csv",
                           mime="text/csv")
        if up is not None:
            raw = up.getvalue()
            try:
                new = pd.read_csv(io.BytesIO(raw), dtype={"user_id": str}, encoding="utf-8-sig")
            except UnicodeDecodeError:
                new = pd.read_csv(io.BytesIO(raw), dtype={"user_id": str}, encoding="cp949")
            errors, warns = C.check_upload(new)
            for w in warns:
                st.warning(w)
            for er in errors:
                st.error(er)
            if not errors:
                st.caption(f"미리 보기: {len(new):,}행")
                st.dataframe(new.head(5), hide_index=True)
                if st.button("이 파일 적용", type="primary"):
                    merged, info = C.merge_customers(customers, new, "replace" if mode == "현재 자료 교체" else "append",
                                                     "update" if conflict == "새 자료로 갱신" else "keep")
                    C.save_customers(merged)
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
                vals[f] = cols[i % 4].number_input(C.FEATURE_KO[f], value=float(ref), step=1.0, key=f"in_{f}")
            vals["level_last"] = cols[len(C.NUMERIC) % 4].selectbox("요금제", ["free", "paid"])
            conflict1 = st.radio("같은 고객 번호가 있으면", ["기존 자료 유지", "새 자료로 갱신"], horizontal=True, key="c1")
            ok = st.form_submit_button("고객 추가", type="primary")
        if ok:
            if not uid_new.strip():
                st.error("고객 번호를 입력하세요.")
            else:
                one = pd.DataFrame([{"user_id": uid_new.strip(), **vals}])
                sc = svc.scores(S, one)["score"].iloc[0]
                merged, info = C.merge_customers(customers, one, "append",
                                                 "update" if conflict1 == "새 자료로 갱신" else "keep")
                C.save_customers(merged)
                act = "추가" if info["added"] else ("갱신" if info["updated"] else "기존 유지")
                C.log_history("직접 입력", f"{uid_new.strip()} {act} · 위험 점수 {sc:.1f}", info["total"])
                st.session_state.data_msg = f"{uid_new.strip()} {act} · 위험 점수 {sc:.1f}점 ({'이탈 예상' if sc >= THR else '유지 예상'})"
                st.rerun()
        st.caption("기본값은 학습셋 유지 고객의 중앙값입니다.")
    with t3:
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
    if x2.button("기본 자료(테스트 고객)로 되돌리기", disabled=is_default_data):
        (C.STORE / C.CURRENT).unlink(missing_ok=True)
        C.log_history("초기화", "기본 자료(테스트 고객)로 되돌림", len(C.default_customers(svc.team)))
        st.session_state.data_msg = "기본 자료로 되돌렸습니다."
        st.rerun()
    player()

# ---------------------------------------------------------------
# ⚙️ 모델 설정
# ---------------------------------------------------------------
elif page == PAGES[5]:
    ui.playlist_header("⚙️", "설정", "모델과 판정 기준 설정",
                       "모델 선택은 이 화면에서 관리합니다. 적용하면 홈과 관리 목록의 결과가 함께 바뀝니다.",
                       [f"현재: {S.label}", f"기준 {THR:.1f}점"])
    if "set_msg" in st.session_state:
        st.success(st.session_state.pop("set_msg"))
    ml_keys = [k for k, v in REG.items() if v["kind"] == "ml"]
    nn_keys = [k for k, v in REG.items() if v["kind"] == "nn"]
    with st.container(border=True):
        a, b = st.columns(2)
        ml = a.selectbox("머신러닝 모델", ml_keys, index=ml_keys.index(S.ml), format_func=lambda k: REG[k]["name"])
        nn = b.selectbox("신경망 모델", nn_keys, index=nn_keys.index(S.nn), format_func=lambda k: REG[k]["name"])
        modes = ["nn", "ml", "blend"]
        mode = st.radio("홈과 관리 목록에 적용할 운영 기준", modes, index=modes.index(S.mode),
                        format_func=C.MODE_NAMES.get, horizontal=True)
        w = st.slider("결합할 때 머신러닝 모델 비중", 0, 100, int(round(S.ml_weight * 100)), 5, disabled=mode != "blend")
        st.caption(f"머신러닝 {w}% + 신경망 {100 - w}%. 비중을 바꾸면 판정 기준과 성능을 검증셋에서 다시 계산합니다 (테스트셋은 쓰지 않음).")
        use_saved = st.checkbox("저장된 검증 판정 기준 사용", value=S.use_saved,
                                help="끄면 위험 점수 기준을 직접 정합니다")
        manual = st.slider("직접 정할 위험 점수 기준", 0.0, 100.0, float(S.manual_threshold), 0.5, disabled=use_saved)
        draft = C.Setting(ml=ml, nn=nn, mode=mode, ml_weight=w / 100, use_saved=use_saved, manual_threshold=manual)
        v = svc.evaluate(draft, "valid")
        p = st.columns(4)
        for col, (lab, val) in zip(p, [("판정 기준", f"{v['threshold']:.1f}점"), ("검증 재현율", f"{v['recall']:.1%}"),
                                       ("검증 정밀도", f"{v['precision']:.1%}"), ("검증 PR-AUC", f"{v['pr_auc']:.3f}")]):
            with col:
                ui.kpi(lab, val, "이 설정의 미리 보기")
        st.write("")

        def apply_setting(d=draft):
            C.save_setting(d)
            st.session_state.set_msg = f"적용했습니다: {d.label} · 기준 {svc.threshold(d):.1f}점"

        def reset_setting():
            (C.STORE / C.SETTING_FILE).unlink(missing_ok=True)
            st.session_state.set_msg = "기본 설정(검증 PR-AUC가 가장 좋은 조합)으로 되돌렸습니다."

        b1, b2, _ = st.columns([1, 1, 2])
        b1.button("모델 설정 적용", type="primary", on_click=apply_setting, disabled=draft == S)
        b2.button("기본 설정으로", on_click=reset_setting, disabled=not (C.STORE / C.SETTING_FILE).exists())
    with st.expander("개발 시점의 검증 성능과 점수 해석"):
        tbl = pd.DataFrame([{"모델": v_["name"], "종류": "머신러닝" if v_["kind"] == "ml" else "신경망",
                             "검증 PR-AUC": v_["valid_pr_auc"], "검증 ROC-AUC": v_["valid_roc_auc"],
                             "테스트 PR-AUC": v_["test_pr_auc"], "테스트 ROC-AUC": v_["test_roc_auc"],
                             "반복 수": v_["iterations"]} for v_ in REG.values()])
        st.dataframe(tbl.round(4), hide_index=True)
        ui.explain(
            "<b>위험 점수</b>: 모델마다 확률 크기가 달라서, 검증 고객 확률 분포에서의 순위(백분위)로 바꿉니다. "
            "예) 90점 = 검증 고객 90%보다 위험. 그래서 머신러닝과 신경망 점수를 같은 눈금으로 섞을 수 있습니다.<br>"
            "<b>판정 기준</b>: 운영 설정마다 검증셋에서 F2(재현율을 정밀도보다 2배 중시)가 최대인 점수.<br>"
            f"<b>무작위 기준선</b>: PR-AUC {svc.reg['base_rate']['valid']:.3f} (이탈률). 모델 사이 차이 0.01~0.02는 "
            "검증 이탈자 156명 규모에서는 우연 범위입니다.<br>"
            "<b>팀 하이브리드(LightGBM + LSTM, AUC 0.891)를 넣지 않은 이유</b>: LSTM 입력 텐서는 마지막 토큰 하나로 AUC 0.846이 나오고, "
            "그 토큰만 바꾸면 LSTM AUC가 0.855 → 0.568로 떨어집니다. 관찰 기간 밖(해지 직전) 정보로 보여 제외했습니다.")
    player()

# ---------------------------------------------------------------
# 📊 성능 지표
# ---------------------------------------------------------------
else:
    E = svc.evaluate(S, "test")
    ui.playlist_header("📊", "차트", "성능 지표",
                       "모델이 이탈 고객을 얼마나 잘 골랐는지, 대응 후 이탈이 실제로 줄었는지 나눠서 봅니다.",
                       [f"테스트셋 {E['n']:,}명 (학습에 안 쓴 고객)", f"해지자 {E['positives']}명", f"운영 기준 {S.label}"])
    t1, t2 = st.tabs(["① 모델이 이탈 고객을 잘 골랐나", "② 대응 후 이탈이 실제로 줄었나"])
    with t1:
        k = st.columns(4)
        with k[0]:
            ui.kpi("1. 재현율 · 가장 중요", f"{E['recall']:.1%}", f"해지자 {E['positives']}명 중 {E['tp']}명을 찾음", hot=True)
        with k[1]:
            ui.kpi("2. 정밀도", f"{E['precision']:.1%}", f"이탈 예상 {E['tp'] + E['fp']}명 중 실제 해지 {E['tp']}명")
        with k[2]:
            ui.kpi("4. PR-AUC", f"{E['pr_auc']:.3f}", f"무작위 {E['base_rate']:.3f} 대비 {E['pr_auc'] / E['base_rate']:.1f}배")
        with k[3]:
            ui.kpi("ROC-AUC", f"{E['roc_auc']:.3f}", "0.5 = 무작위")
        st.write("")
        ui.explain(
            "<b>1. 재현율(Recall)</b> — 실제 이탈 고객 중 모델이 찾아낸 비율. 이탈 고객을 놓치지 않는 목적에 직접 맞는 지표입니다.<br>"
            "<b>2. 정밀도(Precision)</b> — 위험하다고 표시한 고객 중 실제 이탈 고객의 비율. 낮으면 이탈하지 않을 고객에게도 연락·혜택이 나갑니다.<br>"
            "<b>3. 재현율@관리 가능 인원</b> — 연락할 수 있는 인원이 정해져 있을 때, 위험 점수 상위 N명 안에 실제 이탈자가 몇 명이고 "
            "전체 이탈자의 몇 %인지. 실무에서 특히 유용합니다.<br>"
            "<b>4. PR-AUC</b> — 판정 기준을 바꿔 가며 정밀도와 재현율이 어떻게 변하는지 종합. 이탈 비율이 약 5.7%로 낮아 ROC-AUC와 함께 봅니다.<br>"
            "<b>5. 오경고 수와 혼동행렬</b> — 이탈자를 몇 명 찾고 놓쳤는지, 이탈하지 않을 고객을 몇 명 위험으로 표시했는지. 대응 부담을 숫자로 봅니다.")
        st.write("")
        c1, c2 = st.columns([1.25, 1])
        with c1:
            with st.container(border=True):
                ui.section("3. 재현율@관리 가능 인원", "위험 점수 상위 N명에게만 연락할 수 있을 때")
                cap = st.number_input("관리 가능 인원 (명)", 1, E["n"], min(500, E["n"]), 50)
                caught = int(E["cum_caught"][cap - 1])
                m = st.columns(3)
                with m[0]:
                    ui.kpi("상위 N명 중 실제 이탈", f"{caught}명", f"N = {cap:,}명", hot=True)
                with m[1]:
                    ui.kpi("전체 이탈자 중", f"{caught / E['positives']:.1%}", "재현율@N")
                with m[2]:
                    ui.kpi("상위 N명의 정밀도", f"{caught / cap:.1%}", f"무작위 {E['base_rate']:.1%}")
                n_ax = np.arange(1, E["n"] + 1)
                fig = go.Figure()
                fig.add_scatter(x=n_ax, y=E["cum_caught"] / E["positives"], name="재현율@N", line=dict(color=ui.VIOLET, width=3))
                fig.add_scatter(x=n_ax, y=E["cum_caught"] / n_ax, name="정밀도@N", line=dict(color=ui.PINK, width=2))
                fig.add_vline(x=cap, line_dash="dash", line_color="#FFFFFF")
                fig.update_layout(xaxis_title="관리 인원 N", yaxis_tickformat=".0%", yaxis_range=[0, 1.02])
                ui.plot(ui.style(fig, 300))
        with c2:
            with st.container(border=True):
                ui.section("5. 오경고 수와 혼동행렬", f"판정 기준 {E['threshold']:.1f}점")
                z = [[E["tn"], E["fp"]], [E["fn"], E["tp"]]]
                fig = go.Figure(go.Heatmap(z=z, x=["유지 예상", "이탈 예상"], y=["실제 유지", "실제 해지"],
                                           colorscale=[[0, "#1C1C27"], [1, "#8B5CF6"]], showscale=False,
                                           text=[[f"정상 {E['tn']:,}", f"오경고 {E['fp']:,}"], [f"놓침 {E['fn']:,}", f"찾음 {E['tp']:,}"]],
                                           texttemplate="%{text}", textfont=dict(size=16, color="#FFFFFF")))
                fig.update_yaxes(autorange="reversed")
                ui.plot(ui.style(fig, 260, legend=False))
                st.caption(f"오경고 {E['fp']:,}명 = 이탈하지 않을 고객에게 나가는 연락·혜택 · 놓친 이탈자 {E['fn']}명")
        with st.container(border=True):
            ui.section("4. 정밀도-재현율 곡선 (PR-AUC)", "곡선 아래 넓이가 PR-AUC · 점선 = 무작위")
            fig = go.Figure(go.Scatter(x=E["rc"], y=E["pr"], line=dict(color=ui.PINK, width=3), fill="tozeroy",
                                       fillcolor="rgba(236,72,153,.12)"))
            fig.add_hline(y=E["base_rate"], line_dash="dot", line_color="#6B6B85")
            pt_r, pt_p = E["recall"], E["precision"]
            fig.add_scatter(x=[pt_r], y=[pt_p], mode="markers+text", text=["현재 기준"], textposition="top right",
                            marker=dict(size=12, color="#FFFFFF"), textfont=dict(color="#FFFFFF"))
            fig.update_layout(xaxis_title="재현율", yaxis_title="정밀도", yaxis_range=[0, min(1, max(E["pr"]) + .05)])
            ui.plot(ui.style(fig, 300, legend=False))
    with t2:
        ui.explain(
            "모델이 잘 골랐어도 <b>연락·혜택이 이탈을 줄였는지</b>는 따로 확인해야 합니다. 방법:<br>"
            "① 관리 대상 고객을 무작위로 <b>대응군</b>(연락·혜택)과 <b>대조군</b>(아무것도 안 함)으로 나눕니다.<br>"
            "② 14일 뒤 두 그룹의 이탈률을 비교합니다. 차이가 곧 대응 효과입니다 (모델 성능과 무관하게 측정).<br>"
            "③ 차이가 우연인지 검정하고(p-값), 방어한 고객 수와 캠페인 이익을 계산합니다.<br>"
            "볼 수치: <b>이탈률 차이(%p)</b>, <b>상대 감소율</b>, <b>방어 고객 수</b>, <b>캠페인 이익·ROI</b>, <b>p-값</b>.")
        st.write("")
        with st.container(border=True):
            ui.section("대응 효과 계산기", "캠페인 결과를 넣으면 효과와 이익을 계산합니다 (기본값은 예시)")
            a, b, c, d = st.columns(4)
            n_t = a.number_input("대응군 고객 수", 1, 10_000_000, 400, 10)
            ch_t = b.number_input("대응군 중 이탈", 0, 10_000_000, 40, 1)
            n_c = c.number_input("대조군 고객 수", 1, 10_000_000, 400, 10)
            ch_c = d.number_input("대조군 중 이탈", 0, 10_000_000, 60, 1)
            a2, b2, _, _ = st.columns(4)
            val = a2.number_input("붙잡은 고객 1명의 가치 (원)", 0, 10_000_000, 120_000, 10_000, key="u_val")
            cst = b2.number_input("1명당 연락·혜택 비용 (원)", 0, 1_000_000, 5_000, 1_000, key="u_cost")
            if ch_t > n_t or ch_c > n_c:
                st.error("이탈 수가 고객 수보다 클 수 없습니다.")
            else:
                u = C.uplift_test(int(n_t), int(ch_t), int(n_c), int(ch_c))
                profit = u["saved"] * val - n_t * cst
                k = st.columns(5)
                for col, (lab, v_, sub, hot) in zip(k, [
                        ("대응군 이탈률", f"{u['rate_t']:.1%}", f"{ch_t}/{n_t}명", False),
                        ("대조군 이탈률", f"{u['rate_c']:.1%}", f"{ch_c}/{n_c}명", False),
                        ("이탈률 차이", f"{u['diff'] * 100:+.1f}%p", f"상대 {u['rel']:.0%} 감소" if u["diff"] > 0 else "감소 없음", True),
                        ("방어 고객 수", f"{u['saved']:,.1f}명", "대응군 기준 추정", False),
                        ("캠페인 이익", f"{profit:,.0f}원", f"ROI {profit / max(n_t * cst, 1):.0%}" if cst else "비용 0", False)]):
                    with col:
                        ui.kpi(lab, v_, sub, hot=hot)
                st.write("")
                lo, hi = u["ci"]
                if u["p"] < 0.05 and u["diff"] > 0:
                    st.success(f"대응으로 이탈이 줄었다고 볼 수 있습니다 (p = {u['p']:.3f}). 이탈률 차이 95% 구간 "
                               f"{lo * 100:+.1f}%p ~ {hi * 100:+.1f}%p")
                else:
                    st.warning(f"아직 우연과 구분하기 어렵습니다 (p = {u['p']:.3f}). 이탈률 차이 95% 구간 {lo * 100:+.1f}%p ~ "
                               f"{hi * 100:+.1f}%p. 고객 수를 늘리거나 기간을 두고 다시 측정하세요.")
                fig = go.Figure(go.Bar(x=["대응군", "대조군"], y=[u["rate_t"], u["rate_c"]], marker_color=[ui.MINT, ui.PINK],
                                       text=[f"{u['rate_t']:.1%}", f"{u['rate_c']:.1%}"], textposition="outside"))
                fig.update_layout(yaxis_tickformat=".0%", yaxis_title="14일 이탈률",
                                  yaxis_range=[0, max(u["rate_t"], u["rate_c"]) * 1.35 + 1e-6])
                ui.plot(ui.style(fig, 280, legend=False))
    ui.note("한계: 같은 기준일(2018-11-01)의 다른 고객으로 평가했습니다. 다른 달에도 같은 성능을 보장하지 않으며, "
            "이유·추천 활동은 원인의 증명이 아닙니다.")
    player()
