"""5단계: 학습 결과서(md)를 그림이 들어 있는 HTML 파일 하나로 만듭니다.

- 그림을 파일 안에 넣어(base64) 파일 하나만 보내도 어느 브라우저에서든 그림과 함께 열립니다.
- 브라우저에서 인쇄(Ctrl+P) → 'PDF로 저장'을 고르면 바로 PDF가 됩니다.
- 추가 설치가 필요 없도록 결과서에 쓰는 마크다운 문법(제목, 목록, 표, 그림, 굵게, 코드)만 직접 변환합니다.
"""
from __future__ import annotations

import base64
import html
import re
from datetime import datetime
from pathlib import Path

from .config import Config
from .utils import log

CSS = """
:root { --ink:#1f2328; --muted:#59636e; --line:#d1d9e0; --soft:#f6f8fa; --accent:#2f6fde; }
* { box-sizing: border-box; }
body { margin: 0; background: #fff; color: var(--ink);
       font: 15px/1.65 "Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR","Noto Sans CJK KR",sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 24px 64px; }
header { border-bottom: 2px solid var(--ink); padding-bottom: 12px; margin-bottom: 20px; }
header h1 { margin: 0 0 4px; font-size: 26px; }
header .meta { color: var(--muted); font-size: 13px; }
nav { background: var(--soft); border: 1px solid var(--line); border-radius: 8px; padding: 10px 16px; margin: 0 0 24px; }
nav b { font-size: 13px; color: var(--muted); }
nav ul { margin: 6px 0 0; padding-left: 0; list-style: none; columns: 2; column-gap: 32px; font-size: 14px; }
nav a { color: var(--accent); text-decoration: none; }
h2 { font-size: 20px; margin: 36px 0 10px; padding-top: 8px; border-top: 1px solid var(--line); }
h3 { font-size: 16px; margin: 22px 0 8px; }
p { margin: 8px 0; }
ul { margin: 6px 0 10px; padding-left: 22px; }
li { margin: 3px 0; }
code { font-family: Consolas,"D2Coding",monospace; font-size: 0.9em; background: var(--soft);
       border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; }
.table-wrap { overflow-x: auto; margin: 10px 0 14px; }
table { border-collapse: collapse; font-size: 13px; width: auto; min-width: 50%; }
th, td { border: 1px solid var(--line); padding: 5px 9px; vertical-align: top; }
th { background: var(--soft); font-weight: 600; white-space: nowrap; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tbody tr:nth-child(even) td { background: #fbfcfd; }
figure { margin: 12px 0 18px; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; }
.missing { color: #b42318; font-size: 13px; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--line); padding-top: 10px; }
@media (max-width: 640px) { main { padding: 20px 16px 48px; } nav ul { columns: 1; } }
@media print {
  @page { size: A4; margin: 14mm 12mm; }
  body { font-size: 11.5px; }
  main { max-width: none; padding: 0; }
  nav { break-inside: avoid; }
  nav ul { font-size: 11px; }
  .hint { display: none; }
  table { font-size: 9.5px; min-width: 0; }
  th, td { padding: 3px 5px; }
  th { white-space: normal; }
  .table-wrap { overflow: visible; }
  h2 { break-after: avoid; page-break-after: avoid; }
  table, figure, li { break-inside: avoid; page-break-inside: avoid; }
  figure { margin: 6px 0 10px; }
  figure img { max-height: 82mm; width: auto; max-width: 100%; }
  a { color: inherit; text-decoration: none; }
}
"""

_NUM = re.compile(r"^[-+]?\d[\d,]*(\.\d+)?%?$")


def _inline(text: str) -> str:
    """`코드`, **굵게** 처리. 코드 안은 그대로 두고, 굵게 안에 코드가 있어도 처리."""
    codes: list[str] = []

    def keep(m):
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    s = re.sub(r"`([^`]+)`", keep, text)
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], s)


def _header_cell(text: str) -> str:
    """긴 컬럼 이름은 밑줄(_)에서 줄바꿈되도록."""
    return _inline(text).replace("_", "_<wbr>")


def _image(path: Path, alt: str) -> str:
    if not path.exists():
        return f'<p class="missing">그림 파일 없음: {html.escape(str(path.name))}</p>'
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml",
            "gif": "image/gif"}.get(path.suffix.lower().lstrip("."), "image/png")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img src="data:{mime};base64,{data}" alt="{html.escape(alt)}"></figure>'


def _table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], [r for r in cells[1:] if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]
    th = "".join(f"<th>{_header_cell(c)}</th>" for c in head)
    trs = []
    for r in body:
        tds = "".join(f'<td class="num">{_inline(c)}</td>' if _NUM.match(c.replace(" ", "")) else f"<td>{_inline(c)}</td>"
                      for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


def markdown_to_html(md: str, base_dir: Path) -> tuple[str, str, list[tuple[str, str]]]:
    """(제목, 본문 HTML, 목차[(id, 제목)])"""
    lines = md.splitlines()
    out, toc, title = [], [], "학습 결과서"
    para: list[str] = []
    stack: list[int] = []  # 열린 <ul>의 들여쓰기
    i = 0

    def flush_para():
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    def close_lists(level: int = -1):
        while stack and stack[-1] > level:
            out.append("</li></ul>")
            stack.pop()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush_para()
            i += 1
            continue
        if s.startswith("|"):
            flush_para(); close_lists()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i]); i += 1
            out.append(_table(rows))
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            flush_para(); close_lists()
            level, text = len(m.group(1)), m.group(2)
            if level == 1:
                title = text
            else:
                hid = f"s{len(toc) + 1}"
                if level == 2:
                    toc.append((hid, text))
                out.append(f'<h{level} id="{hid}">{_inline(text)}</h{level}>')
            i += 1
            continue
        m = re.match(r"^!\[(.*?)\]\((.+?)\)$", s)
        if m:
            flush_para(); close_lists()
            out.append(_image(base_dir / m.group(2), m.group(1)))
            i += 1
            continue
        m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if m:
            flush_para()
            indent = len(m.group(1).replace("\t", "  "))
            if not stack or indent > stack[-1]:
                out.append("<ul><li>")
                stack.append(indent)
            else:
                close_lists(indent)
                if stack:
                    out.append("</li><li>")
                else:
                    out.append("<ul><li>")
                    stack.append(indent)
            out.append(_inline(m.group(2)))
            i += 1
            continue
        close_lists()
        para.append(s)
        i += 1
    flush_para(); close_lists()
    return title, "\n".join(out), toc


def build_html(md_path: Path, out_path: Path | None = None) -> Path:
    md_path = Path(md_path)
    out_path = out_path or md_path.with_suffix(".html")
    title, body, toc = markdown_to_html(md_path.read_text(encoding="utf-8"), md_path.parent)
    nav = "".join(f'<li><a href="#{hid}">{_inline(t)}</a></li>' for hid, t in toc)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<header><h1>{_inline(title)}</h1><div class="meta">생성 {stamp} · 원본 {html.escape(md_path.name)}<span class="hint"> · 인쇄(Ctrl+P) → PDF로 저장 가능</span></div></header>
<nav><b>목차</b><ul>{nav}</ul></nav>
{body}
<footer>python -m churn 으로 자동 생성한 문서입니다.</footer>
</main>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def run_report(cfg: Config) -> Path:
    md = cfg.paths.reports_dir / "학습_결과서.md"
    if not md.exists():
        raise FileNotFoundError(f"{md} 가 없습니다. evaluate 단계를 먼저 실행하세요.")
    out = build_html(md)
    log.info(f"[report] HTML 결과서 저장: {out} ({out.stat().st_size / 1e6:.1f} MB, 그림 포함)")
    return out
