"""음악 스트리밍 사이트 느낌의 화면 꾸밈: 다크 테마 CSS, 플레이리스트 헤더, 트랙 리스트, 하단 플레이어 바."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

BG, PANEL, PANEL2, LINE = "#0D0D12", "#15151D", "#1C1C27", "#2A2A3A"
TEXT, MUTED = "#F2F2F7", "#9A9AB0"
VIOLET, PINK, MINT, AMBER, RED = "#8B5CF6", "#EC4899", "#2DD4A7", "#F5A524", "#F43F5E"
FONT = 'Pretendard, "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", "Noto Sans CJK KR", sans-serif'

CSS = """
<style>
@import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");
:root { --bg:%(BG)s; --panel:%(PANEL)s; --panel2:%(PANEL2)s; --line:%(LINE)s; --text:%(TEXT)s; --muted:%(MUTED)s;
  --violet:%(VIOLET)s; --pink:%(PINK)s; --mint:%(MINT)s; --amber:%(AMBER)s; --red:%(RED)s; }
html, body, [class*="css"], .stMarkdown, button, input, textarea, select, label { font-family: %(FONT)s !important; }
.stApp { background: radial-gradient(900px 420px at 18%% -8%%, rgba(139,92,246,.22), transparent 60%%),
                     radial-gradient(700px 380px at 95%% 0%%, rgba(236,72,153,.14), transparent 60%%), var(--bg); }
.block-container { padding-top: 2.6rem; padding-bottom: 7.5rem; max-width: 1320px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], .stDeployButton { display: none; }
section[data-testid="stSidebar"] { background: #08080C; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] > div { padding-bottom: 90px; }
/* 메뉴 라디오: 동그라미 숨기고 음악 앱 메뉴처럼 (Streamlit 신·구 버전 모두) */
section[data-testid="stSidebar"] [role="radiogroup"] { width: 100%%; gap: 2px; align-items: stretch !important; }
section[data-testid="stSidebar"] [data-testid="stRadio"], section[data-testid="stSidebar"] [data-testid="stRadio"] > div { width: 100%%; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child,
section[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { display: none !important; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"],
section[data-testid="stSidebar"] [data-baseweb="radio"] { display: flex !important; width: 100%% !important; box-sizing: border-box;
  padding: 10px 14px; border-radius: 10px; margin: 0;
  transition: background .15s; cursor: pointer; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover,
section[data-testid="stSidebar"] [data-baseweb="radio"]:hover { background: #17171F; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"],
section[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) { background: linear-gradient(90deg, #241D3D, #17171F);
  box-shadow: inset 3px 0 0 var(--violet); }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p,
section[data-testid="stSidebar"] [data-baseweb="radio"] p { font-size: .95rem; font-weight: 600; color: #D9D9E6; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p,
section[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) p { color: #FFFFFF; font-weight: 800; }
.brand { display:flex; align-items:center; gap:10px; margin: 2px 0 4px; }
.brand .logo { width: 36px; height: 36px; border-radius: 10px; display:grid; place-items:center; font-size: 18px;
  background: linear-gradient(135deg, var(--violet), var(--pink)); box-shadow: 0 6px 18px rgba(139,92,246,.45); }
.brand .name { font-size: 1.22rem; font-weight: 800; letter-spacing: -.02em; color: var(--text); }
.lib { font-size: .74rem; letter-spacing: .12em; color: var(--muted); font-weight: 700; margin: 14px 0 6px; }
.side-card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px;
  font-size: .8rem; color: var(--muted); line-height: 1.55; }
.side-card b { color: var(--text); }

.topline { color: var(--muted); font-size: .86rem; display:flex; gap:14px; align-items:center; flex-wrap: wrap; margin-bottom: 10px; }
.live { display:inline-flex; align-items:center; gap:6px; color: var(--mint); font-weight: 700; }
.eq { display:inline-flex; gap:2px; align-items:flex-end; height: 14px; }
.eq i { width: 3px; background: currentColor; border-radius: 2px; animation: eq 1s ease-in-out infinite; }
.eq i:nth-child(1) { height: 40%%; animation-delay: -.2s } .eq i:nth-child(2) { height: 90%%; animation-delay: -.5s }
.eq i:nth-child(3) { height: 60%%; animation-delay: -.8s } .eq i:nth-child(4) { height: 75%%; animation-delay: -.35s }
@keyframes eq { 0%%,100%% { transform: scaleY(.35) } 50%% { transform: scaleY(1) } }
.eq i { transform-origin: bottom; }

.pl-head { display:flex; gap:26px; align-items:flex-end; padding: 22px; border-radius: 22px; margin-bottom: 18px;
  background: linear-gradient(180deg, rgba(139,92,246,.28), rgba(21,21,29,.6)); border: 1px solid rgba(139,92,246,.25); }
.cover { width: 150px; height: 150px; flex: 0 0 150px; border-radius: 16px; position: relative; overflow: hidden;
  display:grid; place-items:center; font-size: 58px; box-shadow: 0 18px 40px rgba(0,0,0,.45);
  background: conic-gradient(from 210deg, #4C1D95, #8B5CF6, #EC4899, #F59E0B, #4C1D95); }
.cover .bars { position:absolute; bottom: 10px; left: 12px; color: rgba(255,255,255,.85); }
.cover .bars .eq { height: 22px; } .cover .bars .eq i { width: 4px; }
.pl-meta .kicker { font-size: .78rem; font-weight: 800; letter-spacing: .14em; color: #D8CCFF; text-transform: uppercase; }
.pl-meta h1 { font-size: 2.6rem; line-height: 1.1; margin: 6px 0 8px; font-weight: 900; letter-spacing: -.04em; color: var(--text); }
.pl-meta p { color: #CFCFE0; margin: 0 0 10px; font-size: .98rem; }
.pl-meta .meta { color: var(--muted); font-size: .86rem; }
.pl-meta .meta b { color: var(--text); }

.kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; height: 100%%; }
.kpi .label { color: var(--muted); font-size: .82rem; font-weight: 600; }
.kpi .value { color: var(--text); font-size: 1.9rem; font-weight: 800; letter-spacing: -.03em; margin-top: 4px; }
.kpi .sub { color: var(--muted); font-size: .78rem; margin-top: 2px; }
.kpi.hot { background: linear-gradient(135deg, rgba(139,92,246,.22), rgba(236,72,153,.16)); border-color: rgba(139,92,246,.45); }

.sec { font-size: 1.15rem; font-weight: 800; color: var(--text); margin: 6px 0 2px; letter-spacing: -.02em; }
.sec-sub { color: var(--muted); font-size: .84rem; margin-bottom: 10px; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--panel); border-color: var(--line) !important; border-radius: 18px; }

.pill { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: .76rem; font-weight: 700; white-space: nowrap; }
.setting { border-left: 4px solid var(--violet); background: var(--panel2); border-radius: 12px; padding: 12px 14px; margin: 6px 0 12px; }
.setting .k { color: var(--muted); font-size: .78rem; } .setting .v { color: var(--text); font-weight: 800; font-size: 1.02rem; margin-bottom: 6px; }

.tracks { width: 100%%; border-collapse: separate; border-spacing: 0; font-size: .9rem; }
.tracks th { color: var(--muted); font-weight: 600; font-size: .78rem; text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
.tracks td { padding: 9px 10px; color: var(--text); border-bottom: 1px solid rgba(42,42,58,.55); vertical-align: middle; }
.tracks tr:hover td { background: #1A1A24; }
.tracks .no { color: var(--muted); width: 28px; text-align: right; font-variant-numeric: tabular-nums; }
.tracks .art { width: 40px; height: 40px; border-radius: 8px; display:grid; place-items:center; font-size: 18px; }
.tracks .ttl { font-weight: 700; } .tracks .sub { color: var(--muted); font-size: .78rem; margin-top: 2px; }
.tracks .num { text-align: right; font-variant-numeric: tabular-nums; color: #D9D9E6; }
.bar { height: 6px; border-radius: 6px; background: #2A2A3A; overflow: hidden; width: 120px; display:inline-block; vertical-align: middle; }
.bar > span { display:block; height: 100%%; border-radius: 6px; background: linear-gradient(90deg, var(--violet), var(--pink)); }

.reason { background: var(--panel2); border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; margin-bottom: 10px; }
.reason .t { font-weight: 800; color: var(--text); } .reason .d { color: var(--muted); font-size: .8rem; margin-top: 3px; }
.reason .a { margin-top: 8px; display:inline-block; padding: 6px 10px; border-radius: 10px; font-weight: 700; font-size: .86rem;
  color: #E9E1FF; background: linear-gradient(90deg, rgba(139,92,246,.35), rgba(236,72,153,.25)); }
.note { background: rgba(245,165,36,.10); border: 1px solid rgba(245,165,36,.35); color: #F8D9A0; border-radius: 12px;
  padding: 10px 14px; font-size: .85rem; }
.explain { color: #CFCFE0; font-size: .88rem; line-height: 1.6; }
.explain b { color: var(--text); }

.player { position: fixed; left: 0; right: 0; bottom: 0; height: 74px; z-index: 1000000; display:flex; align-items:center;
  gap: 18px; padding: 0 22px; background: rgba(10,10,14,.96); border-top: 1px solid var(--line); backdrop-filter: blur(8px); }
.player .now { display:flex; align-items:center; gap: 12px; min-width: 300px; }
.player .mini { width: 46px; height: 46px; border-radius: 8px; display:grid; place-items:center; color: #fff;
  background: conic-gradient(from 210deg, #4C1D95, #8B5CF6, #EC4899, #F59E0B, #4C1D95); }
.player .t1 { color: var(--text); font-weight: 800; font-size: .92rem; } .player .t2 { color: var(--muted); font-size: .78rem; }
.player .mid { flex: 1; display:flex; flex-direction: column; align-items: center; gap: 6px; }
.player .ctrl { color: #D9D9E6; letter-spacing: 14px; font-size: .95rem; }
.player .ctrl b { display:inline-grid; place-items:center; width: 30px; height: 30px; border-radius: 50%%; background: #fff; color: #0D0D12;
  letter-spacing: 0; font-size: .8rem; margin: 0 6px; }
.player .prog { width: min(520px, 90%%); display:flex; align-items:center; gap: 10px; color: var(--muted); font-size: .72rem; }
.player .prog .bar { flex: 1; width: auto; height: 4px; }
.player .right { min-width: 220px; text-align: right; color: var(--muted); font-size: .8rem; }
@media (max-width: 900px) { .player .right { display: none; } .player .now { min-width: 0; } .pl-meta h1 { font-size: 1.8rem; } .cover { width: 96px; height: 96px; flex-basis: 96px; font-size: 38px; } }
</style>
""" % dict(BG=BG, PANEL=PANEL, PANEL2=PANEL2, LINE=LINE, TEXT=TEXT, MUTED=MUTED, VIOLET=VIOLET, PINK=PINK,
           MINT=MINT, AMBER=AMBER, RED=RED, FONT=FONT)

EQ = '<span class="eq"><i></i><i></i><i></i><i></i></span>'


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def e(x) -> str:
    return html.escape(str(x))


def topline(items: list[str], live: str = ""):
    live_html = f'<span class="live">{EQ} {e(live)}</span>' if live else ""
    st.markdown(f'<div class="topline">{live_html}' + "".join(f"<span>{e(i)}</span>" for i in items) + "</div>",
                unsafe_allow_html=True)


def playlist_header(icon: str, kicker: str, title: str, subtitle: str, meta: list[str] = ()):
    meta_html = " · ".join(meta)
    st.markdown(
        f'<div class="pl-head"><div class="cover">{icon}<div class="bars">{EQ}</div></div>'
        f'<div class="pl-meta"><div class="kicker">{e(kicker)}</div><h1>{e(title)}</h1><p>{e(subtitle)}</p>'
        f'<div class="meta">{meta_html}</div></div></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", hot: bool = False):
    st.markdown(f'<div class="kpi{" hot" if hot else ""}"><div class="label">{e(label)}</div>'
                f'<div class="value">{e(value)}</div><div class="sub">{e(sub)}</div></div>', unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    st.markdown(f'<div class="sec">{e(title)}</div>' + (f'<div class="sec-sub">{e(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}55">{e(text)}</span>'


def pred_pill(pred: str) -> str:
    return pill(pred, RED if pred == "이탈 예상" else MINT)


def agree_pill(a: str) -> str:
    return pill(a, MUTED if a == "일치" else AMBER)


ART = ["linear-gradient(135deg,#8B5CF6,#EC4899)", "linear-gradient(135deg,#06B6D4,#8B5CF6)",
       "linear-gradient(135deg,#F59E0B,#EC4899)", "linear-gradient(135deg,#10B981,#06B6D4)",
       "linear-gradient(135deg,#F43F5E,#F59E0B)"]


def track_list(df, show_truth: bool = False):
    """고객 목록을 음악 앱의 트랙 리스트처럼."""
    head = ("<tr><th></th><th></th><th>고객</th><th>위험 점수</th><th>예측 결과</th><th>판정 비교</th>"
            "<th style='text-align:right'>머신러닝</th><th style='text-align:right'>신경망</th>"
            + ("<th>실제</th>" if show_truth else "") + "</tr>")
    rows = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        icon = (r.get("reason_1") or "🎧").split(" ")[0] if r.get("reason_1") not in (None, "-") else "🎧"
        sub = f"{'유료' if str(r.get('level_last')) == 'paid' else '무료'} · {str(r.get('reason_1', '-'))[2:]}"
        truth = ""
        if show_truth:
            c = r.get("churn")
            truth = "<td>" + ("-" if c != c or c is None else (pill("해지", RED) if c == 1 else pill("유지", MUTED))) + "</td>"
        rows.append(
            f"<tr><td class='no'>{i}</td><td><div class='art' style='background:{ART[i % len(ART)]}'>{e(icon)}</div></td>"
            f"<td><div class='ttl'>{e(r['user_id'])}</div><div class='sub'>{e(sub)}</div></td>"
            f"<td><span class='bar'><span style='width:{max(2, float(r['score'])):.0f}%'></span></span> "
            f"<b style='margin-left:8px'>{float(r['score']):.1f}</b></td>"
            f"<td>{pred_pill(r['pred'])}</td><td>{agree_pill(r['agree'])}</td>"
            f"<td class='num'>{float(r['ml_score']):.1f}</td><td class='num'>{float(r['nn_score']):.1f}</td>{truth}</tr>")
    st.markdown(f"<table class='tracks'>{head}{''.join(rows)}</table>", unsafe_allow_html=True)


def setting_card(ml_name: str, nn_name: str):
    st.markdown(f'<div class="setting"><div class="k">머신러닝</div><div class="v">{e(ml_name)}</div>'
                f'<div class="k">신경망</div><div class="v" style="margin-bottom:0">{e(nn_name)}</div></div>',
                unsafe_allow_html=True)


def reason_card(icon: str, title: str, detail: str, action: str):
    st.markdown(f'<div class="reason"><div class="t">{icon} {e(title)}</div><div class="d">{e(detail)}</div>'
                f'<div class="a">▶ {e(action)}</div></div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def explain(text: str):
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def player_bar(title: str, sub: str, progress: float, left_label: str, right_label: str, right: str):
    st.markdown(
        f'<div class="player"><div class="now"><div class="mini">{EQ}</div><div><div class="t1">{e(title)}</div>'
        f'<div class="t2">{e(sub)}</div></div></div>'
        f'<div class="mid"><div class="ctrl">⤮ ⏮ <b>▶</b> ⏭ ↻</div><div class="prog"><span>{e(left_label)}</span>'
        f'<span class="bar"><span style="width:{max(1, min(100, progress * 100)):.0f}%"></span></span>'
        f'<span>{e(right_label)}</span></div></div><div class="right">{e(right)}</div></div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, size=13, color="#D9D9E6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color="#D9D9E6")) if legend else None,
        showlegend=legend, hoverlabel=dict(font=dict(family=FONT), bgcolor="#1C1C27", bordercolor=LINE),
    )
    fig.update_xaxes(gridcolor="#24243A", zerolinecolor="#30304A", linecolor=LINE)
    fig.update_yaxes(gridcolor="#24243A", zerolinecolor="#30304A", linecolor=LINE)
    return fig


def plot(fig: go.Figure):
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except ValueError:
        major, minor = 1, 0
    if (major, minor) >= (1, 46):
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
