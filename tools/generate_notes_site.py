from __future__ import annotations

from dataclasses import dataclass
from html import escape, unescape
from pathlib import Path
import json
import re
import shutil

import markdown


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
NOTES_DIR = DOCS_DIR / "notes"
ASSET_COPY_ROOT = DOCS_DIR / "assets" / "knowledge"
REPO_BLOB_BASE = "https://github.com/ahaxxxx/Informed_machine_learning/blob/main/"
MANIFEST_PATH = ROOT / "tools" / "notes_manifest.json"
LANGUAGE_LABELS = {"en": "EN", "zh": "ZH"}


@dataclass(frozen=True)
class NoteVariant:
    source: Path
    slug: str
    language: str
    legacy_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class NoteCard:
    key: str
    tag: str
    title: str
    summary: str
    variants: tuple[NoteVariant, ...]


MATH_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
MATH_INLINE_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", re.DOTALL)
MD_LINK_RE = re.compile(r"(!?\[[^\]]*\])\(([^)]+)\)")
HEADING_RE = re.compile(r"<h([1-6]) id=\"([^\"]+)\">(.*?)</h\1>", re.DOTALL)
FIRST_H1_RE = re.compile(r"(?m)^#\s+(.+?)\s*$")


def load_notes() -> list[NoteCard]:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    notes: list[NoteCard] = []
    for item in raw["notes"]:
        variants: list[NoteVariant] = []
        for variant in item["variants"]:
            source = ROOT / variant["source"]
            source_html = source.relative_to(ROOT).with_suffix(".html").as_posix()
            legacy_paths = [source_html]
            legacy_paths.extend(variant.get("legacy_paths", []))
            variants.append(
                NoteVariant(
                    source=source,
                    slug=variant["slug"],
                    language=variant["language"],
                    legacy_paths=tuple(dict.fromkeys(legacy_paths)),
                )
            )
        notes.append(
            NoteCard(
                key=item["key"],
                tag=item["tag"],
                title=item["title"],
                summary=item["summary"],
                variants=tuple(variants),
            )
        )
    return notes


def language_label(language: str) -> str:
    return LANGUAGE_LABELS.get(language.lower(), language.upper())


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
        extensions=["extra", "toc", "sane_lists", "smarty"],
        extension_configs={"toc": {"permalink": False}},
        output_format="html5",
    )
    html = md.convert(protected)
    return restore_math(html, placeholders)


def strip_tags(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", text))


def drop_leading_h1(article_html: str) -> str:
    return re.sub(r"^\s*<h1[^>]*>.*?</h1>\s*", "", article_html, count=1, flags=re.DOTALL)


def extract_first_h1(text: str) -> str:
    match = FIRST_H1_RE.search(text)
    if not match:
        return ""
    return match.group(1).strip().strip("#").strip()


def build_toc(article_html: str) -> str:
    items: list[tuple[int, str, str]] = []
    for level, heading_id, inner_html in HEADING_RE.findall(article_html):
        depth = int(level)
        if depth == 1 or depth > 3:
            continue
        items.append((depth, heading_id, strip_tags(inner_html).strip()))

    if not items:
        return '<p class="note-toc-empty">No table of contents was generated.</p>'

    parts = ['<ul class="note-toc-list">']
    for depth, heading_id, title in items:
        parts.append(f'<li class="toc-level-{depth}"><a href="#{escape(heading_id)}">{escape(title)}</a></li>')
    parts.append("</ul>")
    return "\n".join(parts)


def render_note_page(note: NoteVariant, card: NoteCard, page_title: str, eyebrow: str, summary: str, article_html: str, toc_html: str) -> str:
    source_rel = note.source.relative_to(ROOT).as_posix()
    source_url = repo_blob_url(note.source)
    page_lang = note.language.lower()
    return f"""<!doctype html>
<html lang=\"{escape(page_lang)}\">
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>{escape(page_title)} | Informed Machine Learning</title>
    <meta name=\"description\" content=\"{escape(summary)}\">
    <link rel=\"stylesheet\" href=\"../assets/site.css\">
    <script defer src=\"../assets/site.js\"></script>
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
    <script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js\"></script>
  </head>
  <body data-page=\"notes\">
    <div class=\"page-shell\">
      <header class=\"site-header\">
        <a class=\"brand\" href=\"../index.html\">
          <span class=\"brand-mark\">IML</span>
          <span class=\"brand-text\">Informed Machine Learning</span>
        </a>
        <button class=\"nav-toggle\" type=\"button\" aria-expanded=\"false\" aria-label=\"Toggle navigation\">
          <span></span>
          <span></span>
        </button>
        <nav class=\"site-nav\">
          <a href=\"../index.html\">Home</a>
          <a href=\"../reading-map.html\">Reading Map</a>
          <a href=\"../notes.html\">Notes</a>
          <a href=\"../toys.html\">Reproductions</a>
          <a href=\"../deploy.html\">Site Setup</a>
        </nav>
      </header>

      <main class=\"content-page note-page\">
        <section class=\"page-intro note-intro\">
          <p class=\"eyebrow\">{escape(eyebrow)}</p>
          <h1>{escape(page_title)}</h1>
          <p>{escape(summary)}</p>
        </section>

        <section class=\"note-layout\">
          <aside class=\"note-sidebar\">
            <article class=\"note-block\">
              <h2>Note Info</h2>
              <p>Source file: <code>{escape(source_rel)}</code></p>
              <p>Published as: Generated from local Markdown into in-site HTML.</p>
              <div class=\"link-list\">
                <a href=\"../notes.html\">Back to notes</a>
                <a href=\"{escape(source_url)}\">View source on GitHub</a>
              </div>
            </article>

            <article class=\"note-block\">
              <h2>Contents</h2>
              {toc_html}
            </article>
          </aside>

          <article class=\"note-article\">
            {article_html}
          </article>
        </section>
      </main>

      <footer class=\"site-footer\">
        <p>Generated from <code>{escape(source_rel)}</code>. Publish from <code>main /docs</code>.</p>
      </footer>
    </div>
  </body>
</html>
"""


def render_redirect_page(target_href: str, title: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <meta http-equiv=\"refresh\" content=\"0; url={escape(target_href)}\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>{escape(title)} | Redirect</title>
    <script>
      window.location.replace({target_href!r});
    </script>
  </head>
  <body>
    <p>Redirecting to the new page: <a href=\"{escape(target_href)}\">{escape(target_href)}</a></p>
  </body>
</html>
"""


def render_notes_index(cards: list[NoteCard]) -> str:
    card_html: list[str] = []
    for card in cards:
        actions: list[str] = []
        if len(card.variants) == 1:
            variant = card.variants[0]
            actions.append(f'<a href="./notes/{escape(variant.slug)}.html">Open note</a>')
            actions.append(f'<a href="{escape(repo_blob_url(variant.source))}">Source</a>')
        else:
            for variant in card.variants:
                label = language_label(variant.language)
                actions.append(f'<a href="./notes/{escape(variant.slug)}.html">Open {escape(label)}</a>')
            for variant in card.variants:
                label = language_label(variant.language)
                actions.append(f'<a href="{escape(repo_blob_url(variant.source))}">Source {escape(label)}</a>')

        actions_html = "\n              ".join(actions)
        card_html.append(
            f"""          <article class="note-card">
            <p class="card-tag">{escape(card.tag)}</p>
            <h3>{escape(card.title)}</h3>
            <p>
              {escape(card.summary)}
            </p>
            <div class="card-actions">
              {actions_html}
            </div>
          </article>"""
        )

    cards_markup = "\n\n".join(card_html)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Reading Notes | Informed Machine Learning</title>
    <meta
      name="description"
      content="Structured reading notes on informed machine learning, published from local Markdown and organized as bilingual note pages where available."
    >
    <link rel="stylesheet" href="./assets/site.css">
    <script defer src="./assets/site.js"></script>
  </head>
  <body data-page="notes">
    <div class="page-shell">
      <header class="site-header">
        <a class="brand" href="./index.html">
          <span class="brand-mark">IML</span>
          <span class="brand-text">Informed Machine Learning</span>
        </a>

        <div class="header-controls">
          <nav class="site-nav">
            <a href="./index.html">Home</a>
            <a href="./reading-map.html">Reading Map</a>
            <a href="./notes.html">Notes</a>
            <a href="./toys.html">Reproductions</a>
            <a href="./deploy.html">Site Setup</a>
          </nav>

          <button class="nav-toggle" type="button" aria-expanded="false" aria-label="Toggle navigation">
            <span></span>
            <span></span>
          </button>
        </div>
      </header>

      <main class="content-page">
        <section class="page-intro">
          <p class="eyebrow">Reading Notes</p>
          <h1>Published notes with equations, figures, and implementation hooks preserved</h1>
          <p>
            These pages are generated from the Markdown notes in the repository. They document my ongoing reading and current understanding,
            with emphasis on mechanism, notation, and code-facing questions rather than final claims.
          </p>
        </section>

        <section class="cards-grid notes-grid">
{cards_markup}
        </section>

        <section class="resource-block">
          <h2>How these note pages are maintained</h2>
          <p>
            The writing stays in the <code>Knowledge/</code> directory as Markdown, while the publication list lives in
            <code>tools/notes_manifest.json</code>. The generator builds this overview page and the per-note HTML pages from that manifest.
          </p>
          <div class="link-list">
            <a href="{escape(repo_blob_url(ROOT / 'tools' / 'generate_notes_site.py'))}">Site generator</a>
            <a href="{escape(repo_blob_url(MANIFEST_PATH))}">Notes manifest</a>
            <a href="./deploy.html">Site setup</a>
          </div>
        </section>
      </main>

      <footer class="site-footer">
        <p>The note pages are meant to be readable first and repository-traceable second.</p>
      </footer>
    </div>
  </body>
</html>
"""


def generate_note_pages() -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    cards = load_notes()
    variants = [variant for card in cards for variant in card.variants]
    slug_map = {variant.source.relative_to(ROOT).as_posix(): variant.slug for variant in variants}

    for card in cards:
        for note in card.variants:
            text = note.source.read_text(encoding="utf-8").lstrip("\ufeff")
            text = rewrite_markdown_links(text, note.source, slug_map)
            article_html = convert_markdown_to_html(text)
            article_html = drop_leading_h1(article_html)
            toc_html = build_toc(article_html)
            page_title = extract_first_h1(text) or f"{card.title} ({language_label(note.language)})"
            eyebrow = f"{card.tag} / {language_label(note.language)} Notes"
            page_html = render_note_page(note, card, page_title, eyebrow, card.summary, article_html, toc_html)
            output_path = NOTES_DIR / f"{note.slug}.html"
            output_path.write_text(page_html, encoding="utf-8")

            for legacy_rel in note.legacy_paths:
                legacy_path = DOCS_DIR / legacy_rel
                legacy_path.parent.mkdir(parents=True, exist_ok=True)
                target_href = "../notes/" + f"{note.slug}.html"
                redirect_html = render_redirect_page(target_href, page_title)
                legacy_path.write_text(redirect_html, encoding="utf-8")

    (DOCS_DIR / "notes.html").write_text(render_notes_index(cards), encoding="utf-8")


if __name__ == "__main__":
    generate_note_pages()



