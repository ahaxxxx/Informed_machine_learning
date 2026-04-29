---
name: research-companion
description: Use when tasks involve reading papers, integrating them into an existing note system, writing literature reviews, comparing methods, extracting research gaps, or building study/review plans. Especially useful when the user wants self-contained notes, explicit formulas, symbol harmonization, and critical analysis rather than shallow summaries.
---

# Research Companion

This skill is for sustained research work, not one-off summarization.

Use it when the user wants any of the following:

- read a new paper and connect it to an existing knowledge map;
- write or revise a literature review;
- compare multiple papers under a shared framework;
- identify assumptions, costs, failure modes, and open research interfaces;
- build study, review, or research lookup notes.

## First read the local map

Before synthesizing papers, locate the project's existing coordinate system.

Look for:

1. a roadmap or survey index such as `roadmap.md`, `roadmap_zh.md`, `survey_notes.md`;
2. a paper intake template such as `new_paper_integration_template_zh.md`;
3. existing per-paper notes;
4. code or toy reproductions that show which papers have already been operationalized.

If the workspace is this Informed Machine Learning project, start with:

- `Knowledge/roadmap_zh.md`
- `Knowledge/new_paper_integration_template_zh.md`
- the relevant `Knowledge/*_notes_zh.md` files

Do not treat a new paper as isolated until you have checked whether a local roadmap already exists.

## Default workflow

### 1. Establish the coordinate system

Extract the current project axes before reading deeply.

At minimum, write down:

- the main problem split or research lines;
- the project's stable symbols and local notation conventions;
- what counts as a meaningful comparison axis in this workspace.

### 2. Integrate each paper with fixed questions

For every new paper, answer the same eight questions:

1. What gap is the paper actually closing?
2. What object is being treated as "knowledge"?
3. How is that knowledge formalized?
4. Where is the knowledge injected into the learning system?
5. What is the system truly optimizing?
6. Where does the paper sit relative to the existing map?
7. What are its assumptions, costs, and failure points?
8. What is its value for learning, reproduction, and future research?

If a local intake template exists, use it rather than inventing a new format.

### 3. Harmonize symbols before explaining mechanisms

Never explain formulas with undefined symbols.

When papers reuse overloaded symbols such as `A_i`, `B_i`, `lambda`, `phi`, or `q`, do the following:

- keep the paper's original symbol when needed for fidelity;
- define it locally before using it;
- explicitly mark when the same symbol name means different things across papers;
- add a short local symbol table when the notation is dense.

### 4. Keep critique inside the reading, not after it

Do not write ten paragraphs of neutral summary and then append one sentence of criticism.

At every major mechanism, ask:

- what assumption is being introduced here;
- what cost is being paid for this choice;
- where would this break first;
- what simpler baseline would already solve part of the problem;
- what research room is left open.

### 5. Choose the correct output mode

Do not use the same output shape for all tasks.

Read `references/output-modes.md` and pick the mode that matches the request.

### 6. Preserve self-contained mathematical writing

When the user prefers math-heavy but beginner-readable notes:

- define symbols before formulas;
- give the minimal complete formulas needed to follow the mechanism;
- follow formulas with one short intuition or toy explanation;
- avoid saying "details omitted" when those details are required for comprehension.

### 7. Separate paper claims from your own inferences

Use paper-backed language for what the authors actually do.

Clearly mark your own synthesis when you infer:

- which existing paper it is closest to;
- whether its contribution is mainly representational or algorithmic;
- whether its claimed novelty is narrow or system-level.

## Literature review rules

When writing a literature review:

- do not default to chronological narration;
- organize by problem, interface, or tradeoff when possible;
- use comparison axes that survive across papers;
- make assumptions and failure modes first-class content, not side notes;
- end with a research gap section that is narrower than "future work exists".

For the recommended structures, read `references/output-modes.md`.

## Quality bar

Before finishing, run the checklist in `references/checklists.md`.

The output is not ready if any of the following are true:

- a central formula uses undefined symbols;
- a comparison mixes together knowledge object, injection point, and optimization target;
- a paper is summarized without stating its most likely failure mode;
- a literature review reads like stacked mini-abstracts;
- a claimed connection to the local roadmap is only intuitive and not written explicitly.

## Notes

- Prefer local primary materials in the workspace before looking elsewhere.
- If the user asks for newer papers, verify with fresh browsing before making claims about recency.
- If the workspace has no roadmap yet, create a lightweight one before scaling into a full review.

## References

- Output formats and deliverables: `references/output-modes.md`
- Consistency and critique checklist: `references/checklists.md`
