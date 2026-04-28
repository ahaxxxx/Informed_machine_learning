# Notes 页面维护手册（中文）

这份文档是给你自己以后手动维护 `research.bozhanli.com` 的 `Notes` 页面用的。

目标只有一个：

> 以后你写完一篇新 note，或者想改 `Notes` 页的显示，不需要再找我，直接按这里的步骤做。

---

## 0. 先记住这个总关系

站点里和 note 相关的东西，关系是这样的：

```text
Knowledge/*.md
        +
tools/notes_manifest.json
        |
        v
python tools/generate_notes_site.py
        |
        v
docs/notes.html
docs/notes/*.html
docs/Knowledge/*.html
```

意思是：

- `Knowledge/*.md` 放的是你的原始笔记内容；
- `tools/notes_manifest.json` 决定哪些 note 出现在 `Notes` 页、顺序如何、卡片标题是什么、有哪些 EN/ZH 按钮；
- `python tools/generate_notes_site.py` 会把它们生成成站点页面；
- `docs/notes.html`、`docs/notes/*.html`、`docs/Knowledge/*.html` 都是生成产物。

所以有一个最重要的原则：

> 不要手改 `docs/notes.html`、`docs/notes/*.html`、`docs/Knowledge/*.html`。

因为它们下次生成时会被覆盖。

---

## 1. 每个文件各自控制什么

### 1.1 你最常改的文件

- `Knowledge/xxx_notes_zh.md`
  - 控制中文 note 正文内容。
- `Knowledge/xxx_notes_en.md`
  - 控制英文 note 正文内容。
- `tools/notes_manifest.json`
  - 控制 `Notes` 页卡片是否出现；
  - 控制卡片顺序；
  - 控制卡片标题、tag、summary；
  - 控制每张卡显示哪些按钮；
  - 控制按钮对应的 note URL 和 source URL。

### 1.2 首页手改区

首页这些内容不是自动生成的，而是直接手写在 `docs/index.html` 里：

- `Current Status`
- `Current research progress and next analytical steps`
- `Small-scale implementations used to examine paper-level mechanisms`
- `Research Interests`

所以如果你想改这些文案、顺序、卡片数量，直接改：

- `docs/index.html`

这些区块不会被 `generate_notes_site.py` 覆盖。

#### 首页这几个块分别对应哪里

- `Current Status`
  - 首页右侧状态面板；
  - 包括 `Working question`、`Current reading`、`Approach`；
  - 改 `docs/index.html` 里 `profile-card` 那一段。
- `Small-scale implementations used to examine paper-level mechanisms`
  - 首页的 toy / reproduction 项目卡片区；
  - 改 `docs/index.html` 里 `Reproduction Projects` 那一段。
- `Current research progress and next analytical steps`
  - 首页下方的进度时间线区；
  - 改 `docs/index.html` 里 `Current Progress` 那一段。

如果你只改文案，不需要跑 note 生成脚本。

### 1.3 偶尔会改的文件

- `tools/generate_notes_site.py`
  - 控制生成逻辑本身。
  - 例如：按钮文案怎么写、note 页面模板怎么排、旧链接怎么跳转。
- `docs/index.html`
  - 控制首页是否手动展示某篇 note。
- `docs/reading-map.html`
  - 控制阅读路线页里的入口。
- `docs/assets/site.css`
  - 控制样式。

### 1.4 不要手改的生成文件

- `docs/notes.html`
- `docs/notes/*.html`
- `docs/Knowledge/*.html`

这些文件只看，不手改。

---

## 2. 最常见的 4 种任务

### 2.1 任务 A：新增一篇全新的论文 note

适用场景：

- 以前站点上没有这篇论文；
- 你现在想让它作为一张新卡片出现在 `Notes` 页。

你要改：

1. `Knowledge/` 里的 note Markdown 文件。
2. `tools/notes_manifest.json`。
3. 运行生成脚本。

---

### 2.2 任务 B：给已有论文补一个新语言版本

适用场景：

- 这篇论文原来已有 EN；
- 你现在又写了 ZH；
- 或者反过来。

你要改：

1. 新增对应语言的 Markdown 文件；
2. 在这篇论文原有卡片的 `variants` 里补一个对象；
3. 运行生成脚本。

---

### 2.3 任务 C：只改卡片显示，不改正文

适用场景：

- 想改卡片标题；
- 想改 tag；
- 想改 summary；
- 想调顺序。

你只要改：

- `tools/notes_manifest.json`

然后重新生成。

---

### 2.4 任务 D：想改按钮、版式、生成规则

适用场景：

- 想把 `Open EN / Open ZH` 改成别的文案；
- 想改 note 页面顶部布局；
- 想改旧链接跳转行为；
- 想改 `Notes` 页卡片结构。

你要改：

- `tools/generate_notes_site.py`

如果只是样式问题，再加：

- `docs/assets/site.css`

---

### 2.5 任务 E：想改首页的 Current Status / Current Progress / Reproduction Projects

适用场景：

- 想改首页 `Current Status` 文案；
- 想改 `Current research progress and next analytical steps` 的标题或时间线内容；
- 想改 `Small-scale implementations used to examine paper-level mechanisms` 的标题、项目卡片、顺序；
- 想新增一个 toy 项目卡片到首页。

你要改：

- `docs/index.html`

如果你还想改对应样式，再加：

- `docs/assets/site.css`

如果你新增了一个项目卡片，并且希望它点进去有单独页面，再额外处理：

- `docs/projects/xxx.html`

---

## 3. 标准操作流程

这是以后最推荐你照做的一套流程。

### Step 1：写 note 正文

把正文写到 `Knowledge/` 目录里。

例如：

```text
Knowledge/my_new_paper_notes_zh.md
Knowledge/my_new_paper_notes_en.md
```

建议命名规则：

- 全小写；
- 单词之间用下划线；
- 结尾统一用 `_notes_zh.md` 或 `_notes_en.md`。

例如：

```text
Knowledge/graph_constraints_notes_zh.md
Knowledge/graph_constraints_notes_en.md
```

---

### Step 2：修改 `tools/notes_manifest.json`

这是最关键的一步。

你可以把它理解成：

> `Notes` 页的总目录清单。

#### 2.1 如果是新增一张卡片

复制一个现有条目，改成类似这样：

```json
{
  "key": "graph-constraints",
  "tag": "Graph Constraints",
  "title": "Graph Constraints for Informed ML",
  "summary": "This note focuses on how graph-structured prior knowledge is encoded, injected, and optimized inside an informed machine learning pipeline.",
  "variants": [
    {
      "source": "Knowledge/graph_constraints_notes_en.md",
      "slug": "graph-constraints-en",
      "language": "en"
    },
    {
      "source": "Knowledge/graph_constraints_notes_zh.md",
      "slug": "graph-constraints-zh",
      "language": "zh"
    }
  ]
}
```

字段含义：

- `key`
  - 这张卡片的内部标识，尽量简短稳定。
- `tag`
  - 卡片左上角的小标签。
- `title`
  - 卡片标题。
- `summary`
  - 卡片摘要，同时也会作为 note 页顶部说明。
- `variants`
  - 这篇论文有哪些语言版本。

每个 `variant` 里：

- `source`
  - 对应的 Markdown 源文件。
- `slug`
  - 生成后的页面名。
  - 会生成成 `docs/notes/<slug>.html`。
- `language`
  - 目前用 `en` 或 `zh`。

#### 2.2 如果只是给已有卡片补一个中文版本

比如原来只有：

```json
"variants": [
  {
    "source": "Knowledge/graph_constraints_notes_en.md",
    "slug": "graph-constraints-en",
    "language": "en"
  }
]
```

补成：

```json
"variants": [
  {
    "source": "Knowledge/graph_constraints_notes_en.md",
    "slug": "graph-constraints-en",
    "language": "en"
  },
  {
    "source": "Knowledge/graph_constraints_notes_zh.md",
    "slug": "graph-constraints-zh",
    "language": "zh"
  }
]
```

#### 2.3 如果只是改顺序

直接在 `notes` 数组里上下移动整个条目即可。

数组里谁排前面，`Notes` 页就谁先显示。

#### 2.4 如果只是改卡片标题、tag、summary

只改：

- `tag`
- `title`
- `summary`

不用动 Markdown 正文。

---

## 4. 运行生成脚本

改完以后，在仓库根目录执行：

```bash
python tools/generate_notes_site.py
```

这一步会自动生成或更新：

- `docs/notes.html`
- `docs/notes/*.html`
- `docs/Knowledge/*.html`
- `docs/assets/knowledge/...` 里 note 用到的本地图片

如果你的 note 里用到了本地图片，尽量用相对路径，例如：

```md
![caption](./images/paper_screenshots/my_figure.png)
```

生成脚本会自动把它复制到站点资源目录。

---

## 5. 如何检查自己改对了

生成后，至少检查这几件事：

### 5.1 `docs/notes.html`

看：

- 新卡片有没有出现；
- 顺序对不对；
- `Open EN / Open ZH / Source EN / Source ZH` 按钮是否都对；
- 卡片摘要是不是你想要的那句。

### 5.2 `docs/notes/<slug>.html`

看：

- 页面能不能打开；
- 顶部标题是否正确；
- 左侧 `View source on GitHub` 是否对应正确 Markdown；
- 图片是否正常显示；
- 数学公式是否正常渲染。

### 5.3 `docs/Knowledge/*.html`

看旧链接跳转是否还在。

这类文件一般不用深入看，只要确认存在即可。

---

## 6. Git 提交建议

### 6.1 最稳妥的本地流程

```bash
python tools/generate_notes_site.py
git status
git add Knowledge/你的note文件.md tools/notes_manifest.json docs/notes.html docs/notes docs/Knowledge docs/assets/knowledge docs/deploy.html tools/generate_notes_site.py
git commit -m "notes: publish new note page"
git pull origin main
git push origin main
```

### 6.2 更实际的提交说明

如果这次只是新增一篇 note，通常要提交的至少包括：

- `Knowledge/...`
- `tools/notes_manifest.json`
- `docs/notes.html`
- `docs/notes/...`
- `docs/Knowledge/...`
- `docs/assets/knowledge/...`（如果有新图片）

如果这次还改了规则或维护说明，再加：

- `tools/generate_notes_site.py`
- `docs/deploy.html`
- 本文档本身

---

## 7. 你以后最容易踩的坑

### 7.1 只写了 Markdown，但没改 manifest

结果：

- 站点不会出现这篇 note。

原因：

- `Notes` 页不自动扫描所有 Markdown；
- 它只认 `tools/notes_manifest.json`。

---

### 7.2 手改了 `docs/notes.html`

结果：

- 你下次一运行生成脚本，手改内容全没了。

原因：

- 它是生成产物，不是源文件。

---

### 7.3 `slug` 改得不规范

建议统一：

```text
paper-name-en
paper-name-zh
```

例如：

```text
graph-constraints-en
graph-constraints-zh
```

不要混用：

```text
GraphConstraintsEN
graph_constraints_zh
paper-final2
```

---

### 7.4 `source` 路径写错

结果：

- 生成脚本找不到 Markdown 文件；
- GitHub Pages workflow 会失败。

所以每次都检查：

- `Knowledge/...md` 文件是否真实存在；
- `manifest` 里的路径拼写是否一致。

---

### 7.5 图片路径不是相对路径

建议在 note 里尽量写：

```md
![figure](./images/xxx.png)
```

不要写系统绝对路径。

---

## 8. 一页压缩版

以后你如果只想看最短版，就照这个做：

### 8.1 如果你在改 Notes 页面

1. 在 `Knowledge/` 写好 note。
2. 在 `tools/notes_manifest.json` 增加或修改条目。
3. 运行：

```bash
python tools/generate_notes_site.py
```

4. 检查：
   - `docs/notes.html`
   - `docs/notes/<slug>.html`
5. 提交并 push。

### 8.2 如果你在改首页的 Current Status / Progress / Reproduction

1. 直接改 `docs/index.html`。
2. 如果只是改文字，不用跑 `python tools/generate_notes_site.py`。
3. 如果只是改样式，改 `docs/assets/site.css`。
4. 检查首页显示是否正常。
5. 提交并 push。

最重要的记忆句：

> 改正文，去 `Knowledge/`；  
> 改是否显示、顺序、按钮和标题，去 `tools/notes_manifest.json`；  
> 改生成逻辑，去 `tools/generate_notes_site.py`；  
> 改首页的 `Current Status / Progress / Reproduction`，去 `docs/index.html`；  
> 不要手改 `docs/notes*.html` 这些生成产物。
