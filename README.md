# Informed Machine Learning Research Workspace

This repository is the working source behind [research.bozhanli.com](https://research.bozhanli.com), a structured research site for informed machine learning.

The project combines:
- bilingual reading notes
- small-scale reproductions
- roadmap documents
- the generated GitHub Pages site in `docs/`

The emphasis is on mechanism-level understanding, traceable note revision, and code-facing comparison rather than polished benchmark packaging.

## Live Entry Points

- Website: [research.bozhanli.com](https://research.bozhanli.com)
- Notes index: [research.bozhanli.com/notes.html](https://research.bozhanli.com/notes.html)
- Reproductions: [research.bozhanli.com/toys.html](https://research.bozhanli.com/toys.html)
- Reading map: [research.bozhanli.com/reading-map.html](https://research.bozhanli.com/reading-map.html)

## Published Notes

### Constraint-oriented papers

#### 1. Informed ML Survey

- PDF: [taxonomy_survey_2021_tkde.pdf](papers/survey/taxonomy_survey_2021_tkde.pdf)
- Notes EN: [research.bozhanli.com/notes/survey-en.html](https://research.bozhanli.com/notes/survey-en.html)
- Notes ZH: [research.bozhanli.com/notes/survey-zh.html](https://research.bozhanli.com/notes/survey-zh.html)
- Source: [Knowledge/survey_notes_en.md](Knowledge/survey_notes_en.md), [Knowledge/survey_notes_zh.md](Knowledge/survey_notes_zh.md)

#### 2. Logic-Net

- PDF: [logic_net_2016_arxiv.pdf](papers/logic_net/logic_net_2016_arxiv.pdf)
- Notes EN: [research.bozhanli.com/notes/logic-net-en.html](https://research.bozhanli.com/notes/logic-net-en.html)
- Notes ZH: [research.bozhanli.com/notes/logic-net-zh.html](https://research.bozhanli.com/notes/logic-net-zh.html)
- Reproduction: [repro/01_logic_net_toy/README.md](repro/01_logic_net_toy/README.md)
- Source: [Knowledge/logic_net_notes_en.md](Knowledge/logic_net_notes_en.md), [Knowledge/logic_net_notes_zh.md](Knowledge/logic_net_notes_zh.md)

#### 3. Semantic Loss

- PDF: [semantic_loss_2018.pdf](papers/semantic_loss/semantic_loss_2018.pdf)
- Notes EN: [research.bozhanli.com/notes/semantic-loss-en.html](https://research.bozhanli.com/notes/semantic-loss-en.html)
- Notes ZH: [research.bozhanli.com/notes/semantic-loss-zh.html](https://research.bozhanli.com/notes/semantic-loss-zh.html)
- Reproduction: [repro/03_semantic_loss_toy/README.md](repro/03_semantic_loss_toy/README.md)
- Source: [Knowledge/semantic_loss_notes_en.md](Knowledge/semantic_loss_notes_en.md), [Knowledge/semantic_loss_notes_zh.md](Knowledge/semantic_loss_notes_zh.md)

#### 4. DL2

- PDF: [dl2_2019_icml.pdf](papers/dl2/dl2_2019_icml.pdf)
- Notes EN: [research.bozhanli.com/notes/dl2-en.html](https://research.bozhanli.com/notes/dl2-en.html)
- Notes ZH: [research.bozhanli.com/notes/dl2-zh.html](https://research.bozhanli.com/notes/dl2-zh.html)
- Source: [Knowledge/dl2_notes_en.md](Knowledge/dl2_notes_en.md), [Knowledge/dl2_notes_zh.md](Knowledge/dl2_notes_zh.md)

### Granular and knowledge-guided papers

#### 5. Granular Computing for Machine Learning

- PDF: [granular_computing_2025_tcyb.pdf](papers/granular/granular_computing_2025_tcyb.pdf)
- Notes EN: [research.bozhanli.com/notes/granular-ml-en.html](https://research.bozhanli.com/notes/granular-ml-en.html)
- Notes ZH: [research.bozhanli.com/notes/granular-ml-zh.html](https://research.bozhanli.com/notes/granular-ml-zh.html)
- Source: [Knowledge/granular_ml_notes_en.md](Knowledge/granular_ml_notes_en.md), [Knowledge/granular_ml_notes_zh.md](Knowledge/granular_ml_notes_zh.md)

#### 6. From Fuzzy Rule-Based Models to Granular Models

- PDF: [fuzzy_to_granular_models_2025_tfuzz.pdf](papers/granular/fuzzy_to_granular_models_2025_tfuzz.pdf)
- Notes EN: [research.bozhanli.com/notes/fuzzy-to-granular-en.html](https://research.bozhanli.com/notes/fuzzy-to-granular-en.html)
- Notes ZH: [research.bozhanli.com/notes/fuzzy-to-granular-zh.html](https://research.bozhanli.com/notes/fuzzy-to-granular-zh.html)
- Source: [Knowledge/fuzzy_to_granular_notes_en.md](Knowledge/fuzzy_to_granular_notes_en.md), [Knowledge/fuzzy_to_granular_notes_zh.md](Knowledge/fuzzy_to_granular_notes_zh.md)

#### 7. Knowledge Landmarks

- PDF: [knowledge_landmarks_2026.pdf](papers/knowledge_landmarks/knowledge_landmarks_2026.pdf)
- Notes EN: [research.bozhanli.com/notes/knowledge-landmarks-en.html](https://research.bozhanli.com/notes/knowledge-landmarks-en.html)
- Notes ZH: [research.bozhanli.com/notes/knowledge-landmarks-zh.html](https://research.bozhanli.com/notes/knowledge-landmarks-zh.html)
- Reproduction: [repro/02_knowledge_landmarks_toy/README.md](repro/02_knowledge_landmarks_toy/README.md)
- Source: [Knowledge/knowledge_landmarks_notes_en.md](Knowledge/knowledge_landmarks_notes_en.md), [Knowledge/knowledge_landmarks_notes_zh.md](Knowledge/knowledge_landmarks_notes_zh.md)

## Reproduction Projects

- [logic_net_toy](repro/01_logic_net_toy/README.md): teacher distribution + distillation under logic rules
- [knowledge_landmarks_toy](repro/02_knowledge_landmarks_toy/README.md): knowledge landmarks as a softer global regularizer
- [semantic_loss_toy](repro/03_semantic_loss_toy/README.md): satisfying-world probability mass as a symbolic loss

## Repository Layout

```text
papers/       original PDFs
Knowledge/    bilingual Markdown notes
repro/        toy reproductions and analysis notes
docs/         generated GitHub Pages site
tools/        note-manifest and site-generation scripts
README.md     repository entry point
```

## Site Maintenance

- Source notes live in `Knowledge/`
- Notes publication is controlled by `tools/notes_manifest.json`
- Site generation is handled by `tools/generate_notes_site.py`
- Homepage status and progress text are maintained directly in `docs/index.html`

Typical local regeneration command:

```bash
python tools/generate_notes_site.py
```
