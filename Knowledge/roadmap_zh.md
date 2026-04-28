# Informed Machine Learning 阅读路线图（中文索引）

## 0. 这份索引怎么用

这份文件是当前仓库中文笔记的总导航。它本身不是技术细节笔记，因此不追求像单篇论文笔记那样完全自包含；它只负责三件事：

1. 帮你快速定位两条主线各自的论文顺序。
2. 帮你把“单篇深度笔记”和“总路线长文”连起来。
3. 帮你判断当前最适合先读哪篇、先做哪个 toy。

如果要看完整的逐篇精读、截图解析和渐进式复现路线，主文档仍然是：

- [Informed Machine Learning 逐篇精读、逻辑关系与渐进式复现路线](./informed_ml_论文解读与复现路线.md)

---

## 0.1 七篇论文的逻辑闭环

把这七篇论文串成一条真正闭合的学习链，可以写成下面这个结构：

### 0.1.1 母图：Survey 先给坐标系

Survey 先回答三个最基础的问题：

1. 知识从哪里来？
2. 知识怎样表示？
3. 知识注入到学习系统的哪里？

后面六篇，如果不放回这个三轴 taxonomy 里，很容易被误读成互不相关的技巧。

### 0.1.2 逻辑约束主线：从“知识存在”到“知识怎样约束优化”

这条线的闭环是：

- [survey_notes_zh.md](./survey_notes_zh.md)：先确认 logic rules 在 taxonomy 里的位置；
- [logic_net_notes_zh.md](./logic_net_notes_zh.md)：知识先进入 posterior，形成 `rule -> teacher -> student`；
- [semantic_loss_notes_zh.md](./semantic_loss_notes_zh.md)：知识不再经 teacher，而是直接进入精确语义 loss；
- [dl2_notes_zh.md](./dl2_notes_zh.md)：知识进一步进入统一声明式接口，不只用于 training，也用于 querying。

这条线最终回答的是：

> 如果知识是逻辑结构，怎样让模型的训练与推理显式面对这些结构？

### 0.1.3 粒化主线：从“知识有效”到“知识和结果是否可信”

这条线的闭环是：

- [survey_notes_zh.md](./survey_notes_zh.md)：先给出“知识可以进入 ML”的总框架；
- [granular_ml_notes_zh.md](./granular_ml_notes_zh.md)：指出模型不能只给一个看似精确的点值；
- [fuzzy_to_granular_notes_zh.md](./fuzzy_to_granular_notes_zh.md)：把这种视角落到具体规则模型，形成 granular output；
- [knowledge_landmarks_notes_zh.md](./knowledge_landmarks_notes_zh.md)：再把 granules 从“输出形式”推进到“知识 regularizer”。

这条线最终回答的是：

> 当知识本身就是模糊的、局部有效的、带范围的时，怎样让模型输出和训练目标一起变得更诚实？

### 0.1.4 两条主线最后汇到哪里

如果你后面要做创造性工作，那么这七篇的真正闭环，不是“把七篇都读完”，而是把它们汇到同一个更高层的问题：

> 怎样把知识的**有效性**与输出的**可信性**一起纳入学习系统？

可以把这个总问题写成一个抽象目标：
$$
\min \; L_{\text{task}}
\;+\;
\lambda_{\phi}\,\Psi_{\phi}
\;+\;
\lambda_{G}\,\Psi_{G}.
$$

这里：

- $L_{\text{task}}$ 表示普通任务损失；
- $\Psi_{\phi}$ 表示逻辑/规则/结构约束项；
- $\Psi_{G}$ 表示粒化、可信度、局部知识或范围表达项。

前四篇更偏怎样设计 $\Psi_{\phi}$；  
后三篇更偏怎样设计 $\Psi_{G}$。  
真正有创造性的工作，往往就发生在这两个项怎样同时存在、彼此制衡、彼此增益上。

---

## 0.2 跨文统一符号约定

下面这张表不是要求七篇完全用同一套符号，而是帮你避免“同一个字母在不同论文里被误当成同一个对象”。

| 符号 | 默认含义 | 主要出现位置 | 读笔记时要注意什么 |
| --- | --- | --- | --- |
| $x$ | 输入 | 七篇几乎都有 | 一般默认是输入变量，但在 querying / inner optimization 中也可能是被搜索对象的一部分 |
| $y$ | 输出、标签或候选状态 | 七篇几乎都有 | 在 Logic / Semantic Loss 里常是离散输出状态；在 granular 线里也可能是连续输出值 |
| $\mathcal D$ | 训练数据集 | Survey、Logic-Net、Semantic Loss、granular papers | 若写成 $\mathcal D^\ast$，通常强调“只覆盖局部区域”的数据 |
| $\theta$ | 神经网络参数 | Logic-Net、Semantic Loss、DL2 | 逻辑线更偏用 $\theta$ |
| $a$ | 一般模型参数 | granular_ml、knowledge_landmarks | 粒化主线更偏用 $a$；不要和逻辑线里的 $\theta$ 混看成不同思想，它们都只是“参数” |
| $\phi$ | 逻辑约束、公式 | Logic-Net、Semantic Loss、DL2 | 只在逻辑线里是核心对象；粒化主线通常不以 $\phi$ 为中心 |
| $q^\star$ | 规则投影后的教师分布 | Logic-Net | 这是 Hu 2016 路线特有对象，后两篇逻辑论文不再以它为核心 |
| $A_i$ | 局部对象，含义随文变化 | Fuzzy-to-Granular、Knowledge Landmarks | 在 fuzzy-to-granular 里常是规则激活度；在 knowledge landmarks 里是输入粒，不能直接等同 |
| $B_i$ | 局部对象，含义随文变化 | Fuzzy-to-Granular、Knowledge Landmarks | 在 fuzzy-to-granular 里是 consequent granule；在 knowledge landmarks 里是输出粒 |
| $\lambda$ | 权重系数，但语义随文变化 | 几乎所有笔记 | 这是最容易误读的符号：在 Logic-Net 里常是规则置信度，在 KD-ML 里是 data/knowledge trade-off，在其它地方也可能只是正则强度 |
| $\operatorname{cov}$ | 覆盖度 | granular 三篇 | 在粒化主线里表示“数据或预测是否被某个粒覆盖/匹配” |
| $\operatorname{sp}$ | 特异性、尖锐度 | granular 三篇 | 在粒化主线里和 coverage 构成基本张力：越宽越覆盖，越窄越具体 |
| $M(x;a)$ | 一般预测模型 | granular_ml、knowledge_landmarks | 用来表示“先别管具体架构，只把模型当映射” |

一条最重要的读法规则是：

> 跨文阅读时，默认只把 `x / y / 数据集 / 参数` 视为全局稳定对象；其余符号一律先按“局部重新绑定”理解。

特别是下面三个符号，永远不要跨文直接对齐：

- $A_i$
- $B_i$
- $\lambda$

---

## 0.3 跨文批评性阅读清单

如果你读这七篇是为了后期做创造性工作，而不是为了考试，那么每篇都应该反复问同一组问题：

### 0.3.1 这篇论文把“知识”当成什么对象

- 是逻辑公式？
- 是后验偏置？
- 是满足世界集合？
- 是连续 surrogate？
- 是区间、模糊集或概率粒？
- 是局部数据外的全局 landmarks？

### 0.3.2 这篇论文到底把知识注入到哪里

- posterior；
- loss；
- feasible set；
- output representation；
- knowledge regularizer；
- hypothesis set。

如果这个接口不看清，后面所有“它和上一篇有什么不同”都会变模糊。

### 0.3.3 这篇论文真正优化的对象是什么

- 参数 $\theta$ 或 $a$；
- 教师分布 $q^\star$；
- 反例变量 $z$；
- granule 的边界或宽度；
- data/knowledge trade-off 参数。

真正有研究价值的批评，必须落到“它到底在优化谁”。

### 0.3.4 这篇论文最可能在哪里失效

- 规则写错；
- 状态空间爆炸；
- surrogate 太平；
- 内层搜索太贵；
- granule 太宽或太窄；
- 局部数据和全局知识互相冲突。

如果一篇笔记只写优点，不写失效链，那它更像摘要，不像研究入口。

### 0.3.5 这篇论文留下的可创造性空间是什么

- 把全局知识改成局部知识；
- 把硬约束改成带可信度的约束；
- 把输出粒化；
- 把知识粒化；
- 把逻辑有效性和 granular 可信性同时放进统一目标；
- 把规则/图谱/粒化对象下沉到表征层。

这一步最重要，因为它直接决定你读完之后是“知道了”，还是“能往前做了”。

---

## 0.4 新论文接入入口

以后每读一篇新论文，建议先直接套用：

- [new_paper_integration_template_zh.md](./new_paper_integration_template_zh.md)

它的作用不是重复做摘要，而是强制回答 8 个固定问题，把新论文接回当前七篇体系。

---

## 1. 先把材料分成两条主线

### 1.1 逻辑约束主线

这条线关心的是：

> 已知一些规则、逻辑或结构约束，怎样把它们显式注入神经网络训练与推理？

推荐顺序：

1. [survey_notes_zh.md](./survey_notes_zh.md)
2. [logic_net_notes_zh.md](./logic_net_notes_zh.md)
3. [semantic_loss_notes_zh.md](./semantic_loss_notes_zh.md)
4. [dl2_notes_zh.md](./dl2_notes_zh.md)

这条线的推进关系可以压缩成：

- Survey 给总 taxonomy。
- Logic-Net 给 `rule -> teacher -> student`。
- Semantic Loss 给 `rule -> exact semantic loss`。
- DL2 给 `declarative constraints -> training + querying`。

### 1.2 粒化知识 / 可信表达主线

这条线关心的是：

> 知识不一定是硬逻辑，输出也不一定只能是点值；那怎样把区间、模糊、概率粒和全局知识一起放进学习系统？

推荐顺序：

1. [survey_notes_zh.md](./survey_notes_zh.md)
2. [granular_ml_notes_zh.md](./granular_ml_notes_zh.md)
3. [fuzzy_to_granular_notes_zh.md](./fuzzy_to_granular_notes_zh.md)
4. [knowledge_landmarks_notes_zh.md](./knowledge_landmarks_notes_zh.md)

这条线的推进关系可以压缩成：

- Survey 给“知识可以进入 ML”的母图。
- Granular Computing for ML 说明为什么点值输出不够，为什么需要可信粒化表达。
- From Fuzzy Rule-Based Models to Granular Models 说明如何把数值规则模型升级为 interval / fuzzy / probabilistic 输出。
- Knowledge Landmarks 则把“局部数据 + 全局粒化知识”真正做成一个统一训练框架。

### 1.3 旁支：rough set / attribute reduction

这部分更像方法论补充，而不是当前主链的下一步：

- `rough set` 更偏数据粒化、属性约简和不确定性处理；
- `knowledge landmarks` 更偏知识-数据联合学习；
- 两者有共鸣，但不是简单前后继承关系。

---

## 2. 当前最建议的阅读顺序

如果按“先稳，再扩，再做自己方向”的节奏，建议这么看：

### 2.1 第一轮：先建立 informed ML 地图

1. [survey_notes_zh.md](./survey_notes_zh.md)
2. [logic_net_notes_zh.md](./logic_net_notes_zh.md)
3. [semantic_loss_notes_zh.md](./semantic_loss_notes_zh.md)
4. [dl2_notes_zh.md](./dl2_notes_zh.md)

这一轮的目标不是记住所有公式，而是建立三个判断：

- 知识来自哪里；
- 知识如何表示；
- 知识被注入到哪里。

### 2.2 第二轮：切到可信表达与粒化知识视角

1. [granular_ml_notes_zh.md](./granular_ml_notes_zh.md)
2. [fuzzy_to_granular_notes_zh.md](./fuzzy_to_granular_notes_zh.md)
3. [knowledge_landmarks_notes_zh.md](./knowledge_landmarks_notes_zh.md)

这一轮的目标是把视角从“满足约束”扩展到：

- 输出是否过于尖锐和自信；
- 知识是否只能写成硬方程或硬规则；
- 数据局部、知识全局时怎样联合学习。

### 2.3 第三轮：回头做连接

这时再去想下面这些问题会更自然：

- 逻辑约束和 granular output 能不能结合？
- teacher / semantic loss / DL2 能不能处理局部有效、带粒度的知识？
- rough set 或 feature grouping 能不能先做颗粒化预处理，再接 informed ML？

---

## 3. 仓库里的笔记与代码怎么对应

### 3.1 已经有可运行 toy 的部分

- [logic_net_notes_zh.md](./logic_net_notes_zh.md) 对应 [repro/01_logic_net_toy/README.md](../repro/01_logic_net_toy/README.md)
- [semantic_loss_notes_zh.md](./semantic_loss_notes_zh.md) 对应 [repro/03_semantic_loss_toy/README.md](../repro/03_semantic_loss_toy/README.md)
- [knowledge_landmarks_notes_zh.md](./knowledge_landmarks_notes_zh.md) 对应 [repro/02_knowledge_landmarks_toy/README.md](../repro/02_knowledge_landmarks_toy/README.md)

### 3.2 适合下一步补 toy 的部分

- `dl2_toy`
  目标：先做一个小型约束翻译 + 反例搜索框架。
- `granular_interval_toy`
  目标：先做一个简单回归器，把点值输出改成区间输出。
- `logic + granular hybrid toy`
  目标：试探“输出既满足规则，又给出可信范围”的组合形式。

---

## 4. 如果只想抓最短主线

最短且最稳的一条是：

1. [survey_notes_zh.md](./survey_notes_zh.md)
2. [logic_net_notes_zh.md](./logic_net_notes_zh.md)
3. [semantic_loss_notes_zh.md](./semantic_loss_notes_zh.md)
4. [knowledge_landmarks_notes_zh.md](./knowledge_landmarks_notes_zh.md)

原因很简单：

- 前三篇让你建立 informed ML 最经典的逻辑注入链；
- 最后一篇让你把视角拉到“局部数据 + 全局知识 + 粒化 regularizer”这个更接近后续研究的问题上。

---

## 5. 一页压缩总结

如果把当前仓库的材料压成一句话，那么它其实是在回答两个层层推进的问题：

1. 已知规则时，怎样让模型不只靠数据硬学，而能显式遵守知识？
2. 当知识本身就是模糊的、区间式的、局部有效的、全局抽象的时，怎样继续把它放进学习系统？

前一个问题对应 `Survey -> Logic-Net -> Semantic Loss -> DL2`。  
后一个问题对应 `Survey -> Granular Computing -> Fuzzy to Granular -> Knowledge Landmarks`。

这两条线最后会汇到同一个更成熟的研究视角上：

> informed ML 不只是“把规则加进 loss”，而是“在知识形式、输出表达、训练机制和泛化行为上，一起重新设计学习系统”。
