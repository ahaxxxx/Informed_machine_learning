# Informed Machine Learning

这个仓库整理的是我当前在学的 `Informed Machine Learning` 相关论文、笔记和 toy reproduction。

最适合先看的入口是：
- [总路线文档](Knowledge/informed_ml_论文解读与复现路线.md)

当前已经落地的 toy 项目有：
- [01_logic_net_toy](repro/01_logic_net_toy/README.md)
- [02_knowledge_landmarks_toy](repro/02_knowledge_landmarks_toy/README.md)
- [03_semantic_loss_toy](repro/03_semantic_loss_toy/README.md)

GitHub Pages 版文档站骨架在：
- [docs/index.html](docs/index.html)

如果你准备直接发布这个站点：
1. 把仓库推到 GitHub
2. 进入 `Settings -> Pages`
3. 选择 `Deploy from a branch`
4. 选择 `main` 分支和 `/docs` 目录
5. 保存后等待 Pages 发布

如果你主要想在手机上看，建议顺序是：
1. 先看 [总路线文档](Knowledge/informed_ml_论文解读与复现路线.md)
2. 再按文档里的“读代码顺序总导航”进入具体 toy
3. 每个 toy 先看 `README.md`，再看对应的“代码逐行解析.md”

仓库内容大致分成两部分：
- `Knowledge/`
  - 论文解读、阅读路线、截图和笔记
- `repro/`
  - 逐步落地的 toy code

当前三条主线分别对应：
- `logic_net_toy`
  - 规则如何通过 teacher-student 方式进入训练
- `semantic_loss_toy`
  - 逻辑知识如何直接变成输出分布上的 semantic loss
- `knowledge_landmarks_toy`
  - 局部数据和全局知识 landmarks 如何联合建模

后续计划还包括：
- `dl2_toy`
- `granular_interval_toy`

说明：
- 这个仓库更偏学习记录和 toy reproduction，不追原论文的完整大规模实验。
- 文档里的链接已经改成仓库内相对路径，适合直接在 GitHub 网页和手机端阅读。
