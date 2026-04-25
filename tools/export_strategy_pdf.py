from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "十年发展战略说明书（家庭安心版+学术规划版）.md"
OUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
TEX_PATH = TMP_DIR / "strategy_book.tex"
PDF_NAME = "family_strategy_book_2026.pdf"
PDF_PATH = OUT_DIR / PDF_NAME


SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    """Escape text for LaTeX while preserving clickable URLs."""
    text = text.replace("  ", " ")
    pieces: list[str] = []
    last = 0
    for match in re.finditer(r"https?://[^\s，。；、）)]+", text):
        pieces.append(_escape_plain(text[last : match.start()]))
        pieces.append(r"\url{" + match.group(0).replace("\\", r"\textbackslash{}") + "}")
        last = match.end()
    pieces.append(_escape_plain(text[last:]))
    return "".join(pieces)


def _escape_plain(text: str) -> str:
    return "".join(SPECIALS.get(ch, ch) for ch in text)


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    table: list[list[str]] = []
    i = start
    while i < len(lines):
        raw = lines[i].strip()
        if not raw.startswith("|") or not raw.endswith("|"):
            break
        cells = [cell.strip() for cell in raw.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            i += 1
            continue
        table.append(cells)
        i += 1
    return table, i


def table_colspec(columns: int, wide: bool) -> str:
    if columns <= 2:
        widths = [0.22, 0.74]
    elif columns == 3:
        widths = [0.19, 0.37, 0.40]
    elif columns == 4:
        widths = [0.15, 0.24, 0.38, 0.17]
    else:
        widths = [0.10] + [0.122] * (columns - 1)

    if wide:
        widths = [w * 0.98 for w in widths]

    specs = [
        r">{\RaggedRight\arraybackslash}p{%.3f\textwidth}" % width
        for width in widths[:columns]
    ]
    return "@{}" + "".join(specs) + "@{}"


def render_wide_table(rows: list[list[str]]) -> list[str]:
    headers = rows[0]
    out: list[str] = []
    for row in rows[1:]:
        cells = row + [""] * (len(headers) - len(row))
        title = latex_escape(cells[0])
        out.extend(
            [
                rf"\begin{{tcolorbox}}[milestonebox,title={{{title}}}]",
                r"\small",
                r"\renewcommand{\arraystretch}{1.18}",
                r"\begin{tabular}{@{}>{\bfseries\color{AccentDark}\RaggedRight\arraybackslash}p{0.23\textwidth}>{\RaggedRight\arraybackslash}p{0.66\textwidth}@{}}",
            ]
        )
        for header, cell in zip(headers[1:], cells[1:]):
            out.append(latex_escape(header) + " & " + latex_escape(cell) + r" \\")
        out.extend([r"\end{tabular}", r"\end{tcolorbox}", r"\normalsize"])
    return out


def render_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []

    columns = max(len(row) for row in rows)
    rows = [row + [""] * (columns - len(row)) for row in rows]
    if columns >= 6:
        return render_wide_table(rows)

    size = r"\small"
    colspec = table_colspec(columns, False)

    out: list[str] = []
    out.extend(
        [
            r"\begin{center}",
            size,
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{1.35}",
            r"\arrayrulecolor{TableRule}",
            rf"\begin{{longtable}}{{{colspec}}}",
            r"\hline",
            r"\rowcolor{TableHead} "
            + " & ".join(r"\textbf{" + latex_escape(cell) + "}" for cell in rows[0])
            + r" \\ \hline",
            r"\endfirsthead",
            r"\hline",
            r"\rowcolor{TableHead} "
            + " & ".join(r"\textbf{" + latex_escape(cell) + "}" for cell in rows[0])
            + r" \\ \hline",
            r"\endhead",
        ]
    )

    for index, row in enumerate(rows[1:]):
        color = r"\rowcolor{TableAlt} " if index % 2 else ""
        out.append(color + " & ".join(latex_escape(cell) for cell in row) + r" \\ \hline")

    out.extend([r"\end{longtable}", r"\end{center}", r"\normalsize"])
    return out


def render_code_block(code_lines: list[str]) -> list[str]:
    return [
        r"\begin{tcolorbox}[codebox]",
        r"\begin{Verbatim}[fontsize=\small]",
        *[line.rstrip("\n") for line in code_lines],
        r"\end{Verbatim}",
        r"\end{tcolorbox}",
    ]


def render_quote(lines: list[str]) -> list[str]:
    cleaned = [latex_escape(line.lstrip(">").strip()) for line in lines]
    body = r"\\[0.25em]".join(cleaned)
    return [r"\begin{tcolorbox}[quotebox]", body, r"\end{tcolorbox}"]


def render_markdown_body(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    in_enumerate = False

    def close_list() -> None:
        nonlocal in_enumerate
        if in_enumerate:
            out.append(r"\end{enumerate}")
            in_enumerate = False

    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            close_list()
            out.append("")
            i += 1
            continue

        if stripped.startswith("```"):
            close_list()
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            if i < len(lines):
                i += 1
            out.extend(render_code_block(code_lines))
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            close_list()
            table, i = parse_table(lines, i)
            out.extend(render_table(table))
            continue

        if stripped.startswith(">"):
            close_list()
            quote_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].rstrip("\n"))
                i += 1
            out.extend(render_quote(quote_lines))
            continue

        if stripped == "---":
            close_list()
            out.append(r"\bigskip\hrule\bigskip")
            i += 1
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            close_list()
            level = len(heading_match.group(1))
            title = latex_escape(heading_match.group(2))
            if level == 1:
                out.append(r"\section{" + title + "}")
            elif level == 2:
                out.append(r"\section{" + title + "}")
            elif level == 3:
                out.append(r"\subsection{" + title + "}")
            else:
                out.append(r"\subsubsection{" + title + "}")
            i += 1
            continue

        item_match = re.match(r"^\d+\.\s+(.+)$", stripped)
        if item_match:
            if not in_enumerate:
                out.append(r"\begin{enumerate}")
                in_enumerate = True
            out.append(r"\item " + latex_escape(item_match.group(1)))
            i += 1
            continue

        close_list()
        out.append(latex_escape(stripped) + r"\par")
        i += 1

    close_list()
    return out


def split_front_matter(lines: list[str]) -> tuple[str, list[str], list[str]]:
    title = SOURCE.stem
    meta: list[str] = []
    start = 0

    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        start = 1

    while start < len(lines) and not lines[start].strip():
        start += 1

    while start < len(lines) and lines[start].strip().startswith(">"):
        meta.append(lines[start].strip().lstrip(">").strip())
        start += 1

    while start < len(lines) and not lines[start].strip():
        start += 1

    if start < len(lines) and lines[start].strip() == "---":
        start += 1

    return title, meta, lines[start:]


def make_tex(title: str, meta: list[str], body: list[str]) -> str:
    escaped_title = latex_escape(title)
    title_main = title
    title_sub = ""
    if "（" in title and title.endswith("）"):
        title_main, rest = title.split("（", 1)
        title_main = title_main.strip()
        title_sub = "（" + rest
    escaped_title_main = latex_escape(title_main)
    escaped_title_sub = latex_escape(title_sub)
    meta_lines = [latex_escape(line) for line in meta]
    meta_tex = r"\\[0.35em]".join(meta_lines)

    preamble = rf"""
\documentclass[UTF8,fontset=none,zihao=-4,a4paper]{{ctexart}}
\usepackage[margin=2.15cm,headheight=16pt,footskip=1.1cm]{{geometry}}
\usepackage{{fontspec}}
\setmainfont{{Times New Roman}}
\setsansfont{{Arial}}
\setmonofont{{Consolas}}
\setCJKmainfont[BoldFont={{Noto Sans SC Bold}}]{{Noto Serif SC}}
\setCJKsansfont[BoldFont={{Noto Sans SC Bold}}]{{Noto Sans SC}}
\setCJKmonofont{{Noto Sans SC}}
\usepackage[table]{{xcolor}}
\definecolor{{Ink}}{{HTML}}{{22313F}}
\definecolor{{Accent}}{{HTML}}{{0F6B68}}
\definecolor{{AccentDark}}{{HTML}}{{164B60}}
\definecolor{{SoftBack}}{{HTML}}{{F5F8F7}}
\definecolor{{QuoteBack}}{{HTML}}{{EEF6F4}}
\definecolor{{CodeBack}}{{HTML}}{{F7F7F7}}
\definecolor{{TableHead}}{{HTML}}{{D9ECE9}}
\definecolor{{TableAlt}}{{HTML}}{{F7FBFA}}
\definecolor{{TableRule}}{{HTML}}{{B8D0CC}}
\usepackage{{array}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{ragged2e}}
\usepackage{{pdflscape}}
\usepackage{{enumitem}}
\usepackage{{fancyhdr}}
\usepackage{{titlesec}}
\usepackage[most]{{tcolorbox}}
\usepackage{{fancyvrb}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true,linkcolor=AccentDark,urlcolor=AccentDark}}
\setcounter{{secnumdepth}}{{0}}
\setcounter{{tocdepth}}{{2}}
\linespread{{1.18}}
\setlength{{\parindent}}{{2em}}
\setlength{{\parskip}}{{0.35em}}
\setlist[enumerate]{{leftmargin=2.3em,itemsep=0.18em,topsep=0.35em}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small {escaped_title}}}
\fancyfoot[C]{{\small\thepage}}
\renewcommand{{\headrulewidth}}{{0.3pt}}
\renewcommand{{\footrulewidth}}{{0pt}}
\titleformat{{\section}}{{\Large\bfseries\color{{AccentDark}}}}{{}}{{0pt}}{{}}
\titlespacing*{{\section}}{{0pt}}{{1.0em}}{{0.45em}}
\titleformat{{\subsection}}{{\large\bfseries\color{{Accent}}}}{{}}{{0pt}}{{}}
\titlespacing*{{\subsection}}{{0pt}}{{0.85em}}{{0.35em}}
\titleformat{{\subsubsection}}{{\normalsize\bfseries\color{{Ink}}}}{{}}{{0pt}}{{}}
\titlespacing*{{\subsubsection}}{{0pt}}{{0.65em}}{{0.25em}}
\tcbset{{
  quotebox/.style={{
    enhanced,breakable,colback=QuoteBack,colframe=Accent,
    boxrule=0.6pt,left=8pt,right=8pt,top=7pt,bottom=7pt,arc=2pt
  }},
  codebox/.style={{
    enhanced,breakable,colback=CodeBack,colframe=TableRule,
    boxrule=0.4pt,left=7pt,right=7pt,top=6pt,bottom=6pt,arc=2pt
  }},
  milestonebox/.style={{
    enhanced,colback=SoftBack,colframe=TableRule,
    coltitle=white,colbacktitle=AccentDark,fonttitle=\bfseries,
    boxrule=0.45pt,left=8pt,right=8pt,top=6pt,bottom=6pt,arc=2pt
  }}
}}
\begin{{document}}
\begin{{titlepage}}
\centering
\vspace*{{2.2cm}}
\sffamily
{{\Huge\bfseries\color{{AccentDark}} {escaped_title_main}\par}}
\vspace{{0.35cm}}
{{\LARGE\bfseries\color{{AccentDark}} {escaped_title_sub}\par}}
\vspace{{1.1cm}}
\begin{{tcolorbox}}[quotebox,width=0.86\textwidth]
{meta_tex}
\end{{tcolorbox}}
\vfill
{{\large 为家庭沟通与学术规划准备的阅读版\par}}
\vspace{{0.4cm}}
{{\small 生成日期：2026 年 4 月 14 日\par}}
\rmfamily
\end{{titlepage}}
\tableofcontents
\newpage
"""
    ending = "\n\\end{document}\n"
    return preamble + "\n".join(body) + ending


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    title, meta, body_lines = split_front_matter(lines)
    tex = make_tex(title, meta, render_markdown_body(body_lines))
    TEX_PATH.write_text(tex, encoding="utf-8")

    for _ in range(2):
        subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory=.",
                TEX_PATH.name,
            ],
            cwd=TMP_DIR,
            check=True,
        )

    built_pdf = TMP_DIR / "strategy_book.pdf"
    if built_pdf.exists():
        PDF_PATH.write_bytes(built_pdf.read_bytes())
    print(PDF_PATH)


if __name__ == "__main__":
    main()
