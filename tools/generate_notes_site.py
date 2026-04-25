from __future__ import annotations

from dataclasses import dataclass
from html import escape, unescape
from pathlib import Path
import re
import shutil

import markdown


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
NOTES_DIR = DOCS_DIR / "notes"
ASSET_COPY_ROOT = DOCS_DIR / "assets" / "knowledge"
REPO_BLOB_BASE = "https://github.com/ahaxxxx/Informed_machine_learning/blob/main/"


@dataclass(frozen=True)
class NoteConfig:
    source: Path
    slug: str
    title: str
    eyebrow: str
    summary: str
    source_label: str


NOTES = [
    NoteConfig(
        source=ROOT / "Knowledge" / "logic_net_notes_zh.md",
        slug="logic-net-zh",
        title="Logic-Net 深度结构解析",
        eyebrow="Logic Rules / Chinese Notes",
        summary="围绕 Logic-Net-type methods 的 posterior、loss、feasible set 三条约束进入路径，补齐优化严格性、局限机制和研究延展。",
        source_label="Knowledge/logic_net_notes_zh.md",
    ),
    NoteConfig(
        source=ROOT / "Knowledge" / "survey_notes_zh.md",
        slug="survey-zh",
        title="Informed ML Survey 结构化中文笔记",
        eyebrow="Survey / Chinese Notes",
        summary="把 informed machine learning taxonomy 拆成 knowledge source、representation、integration 三层接口，并沿主线展开逻辑规则、知识图谱与约束学习。",
        source_label="Knowledge/survey_notes_zh.md",
    ),
]


MATH_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
MATH_INLINE_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", re.DOTALL)
MD_LINK_RE = re.compile(r"(!?\[[^\]]*\])\(([^)]+)\)")
HEADING_RE = re.compile(r"<h([1-6]) id=\"([^\"]+)\">(.*?)</h\1>", re.DOTALL)


def repo_blob_url(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    return REPO_BLOB_BASE + rel


def protect_math(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}

    def store(prefix: str, raw: str) -> str:
        key = f"@@{prefix}_{len(placeholders)}@@"
        placeholders[key] = raw
        return key

    text = MATH_BLOCK_RE.sub(lambda m: store("MATH_BLOCK", m.group(0)), text)
    text = MATH_INLINE_RE.sub(lambda m: store("MATH_INLINE", m.group(0)), text)
    return text, placeholders


def restore_math(text: str, placeholders: dict[str, str]) -> str:
    for key, raw in placeholders.items():
        text = text.replace(key, raw)
    return text


def copy_local_asset(path: Path) -> str:
    rel = path.relative_to(ROOT / "Knowledge")
    target = ASSET_COPY_ROOT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return "../assets/knowledge/" + rel.as_posix()


def rewrite_markdown_links(text: str, source_path: Path, slug_map: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        label = match.group(1)
        raw_target = match.group(2).strip()
        target = raw_target.split("#", 1)[0]
        anchor = ""
        if "#" in raw_target:
            _, anchor_part = raw_target.split("#", 1)
            anchor = "#" + anchor_part

        if raw_target.startswith(("http://", "https://", "mailto:")):
            return match.group(0)

        resolved = (source_path.parent / target).resolve()

        if raw_target.startswith("./images/") or (
            resolved.exists() and resolved.is_file() and resolved.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
        ):
            new_target = copy_local_asset(resolved)
            return f"{label}({new_target}{anchor})"

        if resolved.suffix.lower() == ".md":
            try:
                rel = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                rel = ""
            if rel in slug_map:
                return f"{label}(./{slug_map[rel]}.html{anchor})"

        if resolved.exists():
            return f"{label}({repo_blob_url(resolved)}{anchor})"

        return match.group(0)

    return MD_LINK_RE.sub(replace, text)


def convert_markdown_to_html(text: str) -> str:
    protected, placeholders = protect_math(text)
    md = markdown.Markdown(
        extensions=[
            "extra",
            "toc",
            "sane_lists",
            "smarty",
        ],
        extension_configs={
            "toc": {
                "permalink": False,
            }
        },
        output_format="html5",
    )
    html = md.convert(protected)
    return restore_math(html, placeholders)


def strip_tags(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", text))


def drop_leading_h1(article_html: str) -> str:
    return re.sub(r"^\s*<h1[^>]*>.*?</h1>\s*", "", article_html, count=1, flags=re.DOTALL)


def build_toc(article_html: str) -> str:
    items: list[tuple[int, str, str]] = []
    for level, heading_id, inner_html in HEADING_RE.findall(article_html):
        depth = int(level)
        if depth == 1 or depth > 3:
            continue
        items.append((depth, heading_id, strip_tags(inner_html).strip()))

    if not items:
        return "<p class=\"note-toc-empty\">本文未生成目录。</p>"

    parts = ["<ul class=\"note-toc-list\">"]
    for depth, heading_id, title in items:
        parts.append(
            f'<li class="toc-level-{depth}"><a href="#{escape(heading_id)}">{escape(title)}</a></li>'
        )
    parts.append("</ul>")
    return "\n".join(parts)


def render_note_page(note: NoteConfig, article_html: str, toc_html: str) -> str:
    source_rel = note.source.relative_to(ROOT).as_posix()
    source_url = repo_blob_url(note.source)
    return f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{escape(note.title)} | Informed Machine Learning</title>
    <meta name="description" content="{escape(note.summary)}">
    <link rel="stylesheet" href="../assets/site.css">
    <script defer src="../assets/site.js"></script>
    <script>
      window.MathJax = {{
        tex: {{
          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
        }},
        options: {{
          skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
        }}
      }};
    </script>
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  </head>
  <body data-page="notes">
    <div class="page-shell">
      <header class="site-header">
        <a class="brand" href="../index.html">
          <span class="brand-mark">IML</span>
          <span class="brand-text">Informed Machine Learning</span>
        </a>
        <button class="nav-toggle" type="button" aria-expanded="false" aria-label="切换导航">
          <span></span>
          <span></span>
        </button>
        <nav class="site-nav">
          <a href="../index.html">首页</a>
          <a href="../reading-map.html">阅读路线</a>
          <a href="../notes.html">研究笔记</a>
          <a href="../toys.html">Toy 项目</a>
          <a href="../deploy.html">部署说明</a>
        </nav>
      </header>

      <main class="content-page note-page">
        <section class="page-intro note-intro">
          <p class="eyebrow">{escape(note.eyebrow)}</p>
          <h1>{escape(note.title)}</h1>
          <p>{escape(note.summary)}</p>
        </section>

        <section class="note-layout">
          <aside class="note-sidebar">
            <article class="note-block">
              <h2>笔记信息</h2>
              <p>源文件：<code>{escape(source_rel)}</code></p>
              <p>发布方式：由本地 Markdown 自动生成站内 HTML。</p>
              <div class="link-list">
                <a href="../notes.html">返回笔记入口</a>
                <a href="{escape(source_url)}">在 GitHub 看源文件</a>
              </div>
            </article>

            <article class="note-block">
              <h2>目录</h2>
              {toc_html}
            </article>
          </aside>

          <article class="note-article">
            {article_html}
          </article>
        </section>
      </main>

      <footer class="site-footer">
        <p>Generated from <code>{escape(note.source_label)}</code>. Publish from <code>main /docs</code>.</p>
      </footer>
    </div>
  </body>
</html>
"""


def generate_note_pages() -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    slug_map = {
        note.source.relative_to(ROOT).as_posix(): note.slug
        for note in NOTES
    }

    for note in NOTES:
        text = note.source.read_text(encoding="utf-8").lstrip("\ufeff")
        text = rewrite_markdown_links(text, note.source, slug_map)
        article_html = convert_markdown_to_html(text)
        article_html = drop_leading_h1(article_html)
        toc_html = build_toc(article_html)
        page_html = render_note_page(note, article_html, toc_html)
        (NOTES_DIR / f"{note.slug}.html").write_text(page_html, encoding="utf-8")


if __name__ == "__main__":
    generate_note_pages()
