# semantic_loss_toy 代码逐行解析

这份文档对应当前目录下的 `03_semantic_loss_toy` 项目，目标不是只告诉你“它能跑”，而是把它拆成“你能自己改、自己讲”的程度。

说明：
- 我按执行路径和文件顺序来写。
- 我尽量细到逐行粒度，但连续几行属于同一件事时，会按一个小块合并解释。
- 这里解析的是代码文件，不包括 `README.md` 和 `notes.md`。

---

## 1. 先看整个 toy 的执行链

这个 toy 的执行路径是：

1. `run.py`
   - 解析命令行参数
   - 生成 `ExperimentConfig`
   - 调用 `experiment.py` 里的 `run_experiment`

2. `experiment.py`
   - 设随机种子
   - 生成 labeled / unlabeled / val / test 数据
   - 读取约束定义
   - 分别训练 baseline 和 semantic-guided 两个模型
   - 保存 `metrics.json` 和图

3. `trainer.py`
   - baseline：只用监督信号训练
   - semantic-guided：监督项 + semantic loss
   - 评估 accuracy、约束满足概率、硬满足率
   - 画训练曲线和决策区域图

4. `constraints.py`
   - 定义“哪些二值赋值是合法的”
   - 计算满足约束的概率质量
   - 把它变成 semantic loss

5. `data.py`
   - 生成 4 类二维 toy 数据
   - 人工保证各类大致平衡
   - 分出 labeled、unlabeled、val、test

6. `model.py`
   - 定义最小 MLP
   - 输出 4 个独立 logits

所以这整个项目压成一句话就是：

> 先造一个 4 类分类问题，再把“合法输出结构”写成一组二值赋值集合，然后让模型在无标签样本上尽量把概率质量压到这些合法赋值上。

---

## 2. [run.py](./run.py)

这个文件是命令行入口。

### 2.1 导入部分

- `1`: `import argparse`
  - 导入命令行参数解析器。
- `2`: `from pathlib import Path`
  - 用来拼接结果目录路径。
- `4`: `from config import ExperimentConfig`
  - 导入统一实验配置类。
- `5`: `from constraints import available_constraint_sets`
  - 让命令行参数的 `choices=` 只能取合法约束名。
- `6`: `from experiment import run_experiment`
  - 导入真正执行实验的主函数。

### 2.2 `parse_args`

- `9`: `def parse_args():`
  - 定义命令行参数解析函数。
- `10`: `ArgumentParser(description="Semantic loss toy experiment")`
  - 给脚本加一段简短说明。
- `11`: `--seed`
  - 随机种子。
- `12`: `--epochs`
  - 训练轮数。
- `13`: `--num-labeled`
  - 有标签样本数。
- `14`: `--num-unlabeled`
  - 无标签样本数。
- `15`: `--lambda-semantic`
  - semantic loss 的最大权重。注意它不是一上来就直接乘上去，真正训练时还会做 ramp-up。
- `16-21`
  - 定义 `--constraint-set` 参数。
  - `default="exactly_one"` 表示默认用正确约束。
  - `choices=available_constraint_sets()` 强制用户只能在预设集合里选。
- `22`: `--experiment-name`
  - 结果输出目录名。
- `23`: `--skip-plots`
  - 是否跳过绘图。
- `24`: `return parser.parse_args()`
  - 返回解析后的参数对象。

### 2.3 `build_config`

- `27`: `def build_config(args) -> ExperimentConfig:`
  - 把命令行参数映射成统一配置对象。
- `28-37`
  - 构造 `ExperimentConfig(...)`。
  - 这里没有把所有字段都显式传进来，只覆盖了最常改的几项。
  - 没写出来的字段继续用 `config.py` 里的默认值。
- `36`
  - `results_dir=Path(__file__).resolve().parent / "results" / args.experiment_name`
  - 把结果目录固定成当前 toy 目录下的 `results/<实验名>/`。
- `38`
  - 立刻创建输出目录，避免后续保存文件时报路径不存在。
- `39`
  - 返回配置对象。

### 2.4 `main`

- `42`: `def main():`
  - 脚本主入口。
- `43`
  - 先解析命令行参数。
- `44`
  - 再构造配置对象。
- `45`
  - 调 `run_experiment(...)` 执行整个实验。
  - `save_artifacts=True` 表示保存 `metrics.json`。
  - `save_plots=not args.skip_plots` 表示默认会画图，除非显式跳过。
- `47-55`
  - 在终端打印最关键的结果：
  - 输出目录
  - 当前约束名
  - baseline 准确率
  - semantic-guided 准确率
  - accuracy 差值
  - 满足概率差值

### 2.5 入口保护

- `58-59`
  - 只有直接运行这个文件时才执行 `main()`。
  - 如果它被别的模块 import，不会自动开跑。

这个文件的核心作用很纯粹：

> 它不管训练细节，只负责把“命令行世界”转成“实验配置世界”。

---

## 3. [config.py](./config.py)

这个文件集中定义所有实验超参数。

### 3.1 导入部分

- `1`: `from dataclasses import dataclass`
  - 用 `dataclass` 简化配置类的书写。
- `2`: `from pathlib import Path`
  - 处理结果目录。
- `3`: `from typing import Optional`
  - 说明 `results_dir` 可以先为空。

### 3.2 配置类声明

- `6`: `@dataclass`
  - 告诉 Python：这个类主要是“存配置”的，不是复杂业务对象。
- `7`: `class ExperimentConfig:`
  - 定义整个 toy 的统一配置对象。

### 3.3 数据规模相关参数

- `8`: `seed: int = 17`
  - 随机种子。
- `9`: `num_classes: int = 4`
  - 类别数固定为 4。
- `10`: `num_labeled: int = 48`
  - 有标签样本很少，故意制造 low-label 场景。
- `11`: `num_unlabeled: int = 768`
  - 无标签样本明显更多，体现 semantic loss 的用武之地。
- `12`: `num_val: int = 256`
  - 验证集大小。
- `13`: `num_test: int = 512`
  - 测试集大小。

### 3.4 模型和 batch 参数

- `14`: `hidden_dim: int = 48`
  - MLP 隐藏层宽度。
- `15`: `batch_size_labeled: int = 24`
  - 有标签 batch 大小。
- `16`: `batch_size_unlabeled: int = 96`
  - 无标签 batch 更大，因为它主要用来提供约束信号。

### 3.5 优化相关参数

- `17`: `epochs: int = 160`
  - 训练总轮数。
- `18`: `learning_rate: float = 1e-2`
  - 学习率。
- `19`: `weight_decay: float = 1e-4`
  - 权重衰减。

### 3.6 semantic loss 相关参数

- `20`: `lambda_semantic: float = 0.8`
  - semantic loss 的目标最大权重。
- `21`: `ramp_up_epochs: int = 40`
  - 前 40 个 epoch 里逐步把 semantic loss 权重拉起来。
- `22`: `constraint_set_name: str = "exactly_one"`
  - 当前使用的约束集名称。
- `23`: `constraint_threshold: float = 0.5`
  - 只用于“硬满足率”评估，不参与训练。
  - 也就是把 `sigmoid(logit) >= 0.5` 看成该输出位取 1。

### 3.7 绘图和设备参数

- `24`: `mesh_step: float = 0.05`
  - 绘制二维网格图时的步长。
  - 越小图越细，但也越慢。
- `25`: `device: str = "cpu"`
  - 当前固定用 CPU。

### 3.8 输出目录参数

- `26`: `experiment_name: str = "default"`
  - 本次实验名。
- `27`: `results_root: Path = Path(__file__).resolve().parent / "results"`
  - 所有实验结果默认都放当前目录下的 `results/`。
- `28`: `results_dir: Optional[Path] = None`
  - 具体实验目录可以稍后再决定。

### 3.9 `ensure_results_dir`

- `30`: `def ensure_results_dir(self) -> Path:`
  - 保证结果目录存在。
- `31-32`
  - 如果外部没传 `results_dir`，就自动拼成 `results_root / experiment_name`。
- `33`
  - 真的去创建目录。
  - `parents=True` 允许递归创建多级目录。
  - `exist_ok=True` 表示目录已存在也不报错。
- `34`
  - 返回路径，方便外部直接拿来用。

### 3.10 `to_dict`

- `36`: `def to_dict(self) -> dict:`
  - 把配置对象转成普通字典，方便写进 `metrics.json`。
- `37-58`
  - 逐项展开所有关键超参数。
- `57`
  - `results_dir` 这里转成字符串，因为 JSON 不能直接序列化 `Path`。

这个文件的核心思想是：

> 所有模块都只接收一个 `config`，不各自偷偷维护一套参数。

---

## 4. [data.py](./data.py)

这个文件负责定义 toy 数据分布。

### 4.1 导入和数据打包结构

- `1`: 导入 `dataclass`
- `3`: 导入 `torch`
- `4`: 导入 `TensorDataset`

- `7`: `@dataclass`
- `8`: `class DatasetBundle:`
  - 用一个对象统一打包 4 份数据集。
- `9`: `labeled`
  - 有标签训练集。
- `10`: `unlabeled`
  - 无标签训练集。
- `11`: `val`
  - 验证集。
- `12`: `test`
  - 测试集。

### 4.2 随机种子

- `15-16`
  - `set_seed(seed)` 只做一件事：调用 `torch.manual_seed(seed)`。

### 4.3 `class_scores`

- `19`: `def class_scores(x: torch.Tensor) -> torch.Tensor:`
  - 定义“真实世界”的 4 类打分函数。
- `20`: `x1 = x[:, 0]`
  - 拿第一维输入。
- `21`: `x2 = x[:, 1]`
  - 拿第二维输入。
- `22-25`
  - 分别构造 4 个类别的分数 `s0, s1, s2, s3`。
  - 每个分数都不是简单线性函数，而是线性项、平方项、正弦项、余弦项的组合。
  - 这样决策边界就不会太简单，也不会复杂到完全看不懂。
- `26`
  - 把 4 个类别分数组成形状为 `[batch, 4]` 的张量。

这几行的本质是：

> 先人为定义 4 个“类势能面”，哪个分数最大，样本就属于哪个类。

### 4.4 `make_labels`

- `29-30`
  - 直接取 `class_scores(x).argmax(dim=1)`。
  - 也就是说，真实标签是 0 到 3 中分数最高的那个类。

### 4.5 `_sample_candidate_pool`

- `33`: 内部辅助函数。
- `34`
  - 在二维平面 `[-2, 2] x [-2, 2]` 上均匀采样点。
- `35`
  - 用刚才的真实规则给这些点赋标签。
- `36`
  - 返回候选输入和标签。

### 4.6 `make_balanced_dataset`

这个函数很关键，它不是随便采样，而是主动让每一类样本数大致平衡。

- `39-43`
  - 函数参数是：
  - 目标样本数 `num_points`
  - 类别数 `num_classes`
  - 随机数生成器 `generator`
- `44`
  - 先让每一类都拿到 `num_points // num_classes` 个样本。
- `45-46`
  - 如果除不尽，把余数分给前几类。

- `48-50`
  - 初始化样本缓存 `xs`、标签缓存 `ys`，以及每类当前已有数量 `current_counts`。

- `52`
  - 只要还有某一类样本没凑够，就继续采样。
- `53`
  - 一次性采一个比较大的候选池。
  - `max(6 * num_points, 512)` 的意思是：候选池不要太小，不然类别不够平衡时会频繁重采。

- `55`
  - 依次检查每个类别。
- `56`
  - 算这个类别还缺多少样本。
- `57-58`
  - 如果已经够了，跳过。

- `60`
  - 用 `class_mask = y_pool == class_idx` 找出当前候选池里属于这个类别的样本。
- `61`
  - 算当前类别在候选池里实际有多少可用样本。
- `62`
  - 本轮最多取 `remaining` 和 `available` 里的较小值。
- `63-64`
  - 如果这类在这轮候选池里一个都没有，就跳过。

- `66-68`
  - 把当前类别取到的样本和标签加到缓存里，并更新计数。

- `70-71`
  - 把多轮缓存的张量拼接起来。
- `72`
  - 再随机打乱顺序，避免不同类别按块堆在一起。
- `73`
  - 返回平衡后的数据和标签。

这整个函数的目的就是：

> 尽量避免“某一类太少”，否则很容易把后面 accuracy 和 semantic effect 的比较搅乱。

### 4.7 `create_datasets`

- `76`: `def create_datasets(config) -> DatasetBundle:`
  - 根据配置统一生成 4 份数据。
- `77`
  - 创建带固定种子的 `torch.Generator`。
- `79-82`
  - 分别生成 labeled、unlabeled、val、test。
  - 它们都是平衡多分类数据，只是样本数不同。
- `84-89`
  - 用 `TensorDataset` 包装后打包成 `DatasetBundle` 返回。

这个文件的核心思想是：

> 故意制造“少量标注 + 大量无标签 + 类别平衡”的训练环境，让你能清楚观察 semantic loss 的作用。

---

## 5. [constraints.py](./constraints.py)

这是整个 toy 最关键的文件，因为 semantic loss 真正定义在这里。

### 5.1 导入

- `1`: 导入 `dataclass`
- `2`: 导入 `itertools`
  - 后面要用它枚举所有二值赋值。
- `4`: 导入 `torch`
- `5`: 导入 `torch.nn.functional as F`

### 5.2 `ConstraintSpec`

- `8`: `@dataclass(frozen=True)`
  - `frozen=True` 表示约束对象创建后不应再被修改。
- `9`: `class ConstraintSpec:`
  - 表示一条“结构约束”的完整定义。
- `10`: `name`
  - 约束名。
- `11`: `description`
  - 人能读懂的文字说明。
- `12`: `num_classes`
  - 输出位数。
- `13`: `valid_assignments`
  - 最关键字段。
  - 它直接存“哪些二值向量是合法的”。

### 5.3 `_all_binary_assignments`

- `16`: `def _all_binary_assignments(num_classes: int) -> torch.Tensor:`
  - 枚举所有可能的 0/1 赋值。
- `17`
  - `itertools.product([0.0, 1.0], repeat=num_classes)` 会生成全部 `2^num_classes` 个二值组合。
  - 这里 `num_classes=4`，所以一共是 16 个。
- `18`
  - 把这些组合转成 `torch.Tensor`。

### 5.4 `available_constraint_sets`

- `21-22`
  - 返回可选的约束名列表。
  - 当前支持：
  - `exactly_one`
  - `at_least_one`
  - `exactly_two_bad`

### 5.5 `get_constraint_spec`

- `25`
  - 外部获取约束定义的统一入口。
- `26`
  - 先枚举所有二值赋值。
- `27`
  - 计算每个赋值里有多少个 1。

#### `exactly_one`

- `29-31`
  - 只保留“恰好 1 个位置为 1”的赋值。
  - 在 4 类情况下，合法赋值就是：
  - `1000`
  - `0100`
  - `0010`
  - `0001`

#### `at_least_one`

- `32-34`
  - 保留“至少 1 个位置为 1”的赋值。
  - 这比 `exactly_one` 弱很多。
  - 像 `1100`、`1111` 这样的多激活结构也会被视为合法。

#### `exactly_two_bad`

- `35-37`
  - 故意保留“恰好 2 个位置为 1”的赋值。
  - 这是错误约束，用来做坏知识对照实验。

#### 非法名字

- `38-39`
  - 如果名字不认识，直接抛错。

#### 返回 `ConstraintSpec`

- `41-46`
  - 把约束名、说明、类别数和合法赋值表打包返回。

### 5.6 `log_satisfaction_mass`

这是 semantic loss 的数学核心。

- `49`: `def log_satisfaction_mass(logits: torch.Tensor, spec: ConstraintSpec) -> torch.Tensor:`
  - 输入是模型输出 logits 和约束定义。
  - 输出是每个样本“满足约束的总概率质量”的对数。

- `50`
  - 把合法赋值表搬到和 `logits` 同样的设备、同样的数据类型上。

- `51`
  - `F.logsigmoid(logits)` 等价于 `log(sigmoid(logits))`。
  - 这表示每个输出位取 1 的对数概率。
  - `.unsqueeze(1)` 把形状从 `[B, C]` 变成 `[B, 1, C]`，为了后面和所有合法赋值做广播。

- `52`
  - `F.logsigmoid(-logits)` 等价于 `log(1 - sigmoid(logits))`。
  - 这表示每个输出位取 0 的对数概率。
  - 同样扩成 `[B, 1, C]`。

- `53`
  - 这是整段代码最重要的一行。
  - `assignments.unsqueeze(0)` 形状是 `[1, K, C]`，这里 `K` 是合法赋值个数。
  - 如果某一位合法赋值是 1，就取 `log_prob_one`。
  - 如果某一位合法赋值是 0，就取 `log_prob_zero`。
  - 所以 `log_terms` 的形状是 `[B, K, C]`。
  - 它表示：对每个样本、每个合法赋值、每个输出位，对应的 log 概率项。

- `54`
  - `log_terms.sum(dim=2)` 先把每个合法赋值在各个输出位上的 log 概率加起来，得到 `[B, K]`。
  - 这相当于每个合法赋值的联合 log 概率。
  - 再 `torch.logsumexp(..., dim=1)`，把所有合法赋值的概率质量加起来，并且保持数值稳定。
  - 最终得到每个样本满足约束的总 log 概率，形状是 `[B]`。

这一段对应的直觉公式就是：

```text
log P(约束成立) = log Σ_{a 属于合法赋值集合} P(a)
```

而单个赋值 `a` 的概率由各个输出位独立 sigmoid 给出。

### 5.7 `satisfaction_probability`

- `57-58`
  - 直接对 `log_satisfaction_mass` 取指数。
  - 得到普通概率而不是对数概率。

### 5.8 `semantic_loss`

- `61-62`
  - semantic loss 就是：

```text
- mean(log P(约束成立))
```

  - 如果模型把更多概率质量放在合法赋值上，这个 loss 就更小。

### 5.9 `hard_constraint_satisfaction`

这部分不参与训练，只用于评估“硬满足率”。

- `65-69`
  - 输入 logits、约束定义和阈值。
- `70`
  - 合法赋值转成整数张量。
- `71`
  - 先把 logits 过 sigmoid，再按阈值二值化。
  - 默认阈值是 0.5。
- `72`
  - 比较当前预测的 bit 向量是否和任意一个合法赋值完全一致。
  - `all(dim=2)` 表示每一位都要一样。
- `73`
  - 如果匹配任意一个合法赋值，就记作满足，返回 1；否则返回 0。

### 5.10 `serialize_constraint_spec`

- `76-83`
  - 把 `ConstraintSpec` 变成普通字典，方便写进 `metrics.json`。
  - 特别是把 `valid_assignments` 也一起保存下来，这对你后面复盘非常有用。

这个文件的真正核心思想是：

> 语义知识不是一句口号，而是一个合法赋值集合；semantic loss 就是在问模型当前到底给这个集合分了多少概率质量。

---

## 6. [model.py](./model.py)

这个文件很短，但它的设计选择很重要。

### 6.1 导入

- `1`: `import torch.nn as nn`
  - 导入 PyTorch 神经网络模块。

### 6.2 `TinySemanticMLP`

- `4`: `class TinySemanticMLP(nn.Module):`
  - 定义最小 MLP。
- `5`
  - 构造函数允许外部指定隐藏层宽度和类别数。
- `6`
  - 初始化父类。
- `7-13`
  - 用 `nn.Sequential` 堆一个两层隐藏层网络：
  - `8`: 输入 2 维 -> 隐藏层
  - `9`: `Tanh`
  - `10`: 隐藏层 -> 隐藏层
  - `11`: `Tanh`
  - `12`: 隐藏层 -> `num_classes` 个输出 logits

- `15-16`
  - 前向传播就是把输入丢进 `self.net`。

这里最重要的不是“网络有多复杂”，而是：

> 输出层给的是 4 个独立 logits，而不是 `softmax` 概率。

这件事非常关键，因为 semantic loss 要做的是对所有二值赋值建模。
如果直接用 `softmax`，one-hot 结构会被模型结构本身部分硬编码掉，演示效果就没那么干净。

---

## 7. [trainer.py](./trainer.py)

这是训练、评估、画图的主战场。

### 7.1 导入部分

- `1-5`
  - 导入 `copy`、`itertools`、`json`、`math`、`Path`。
- `7-12`
  - 导入 `matplotlib`、`numpy`、`torch`、`F`、`ListedColormap`、`DataLoader`。
- `14`
  - 从 `constraints.py` 导入三个核心函数：
  - `hard_constraint_satisfaction`
  - `satisfaction_probability`
  - `semantic_loss`
- `15`
  - 导入模型。
- `17`
  - `matplotlib.use("Agg")`
  - 强制使用无图形界面后端，这样在终端环境也能保存图片。
- `18`
  - 再导入 `matplotlib.pyplot`。

### 7.2 `ManualAdamW`

这部分是手写的 AdamW 优化器。

#### 初始化

- `21`: `class ManualAdamW:`
  - 自定义优化器类。
- `22`
  - 构造函数接收参数列表、学习率、权重衰减、beta 和 eps。
- `23`
  - 只保留 `requires_grad=True` 的参数。
- `24-28`
  - 保存优化器超参数和步数计数器。
- `29-35`
  - 给每个参数建立状态：
  - `exp_avg` 是一阶动量
  - `exp_avg_sq` 是二阶动量

#### `zero_grad`

- `37-40`
  - 把所有参数梯度清零。

#### `step`

- `42`
  - 执行一次参数更新。
- `43-45`
  - 更新步数，并算 bias correction。
- `47`
  - 在 `torch.no_grad()` 里更新参数，避免把更新过程也记进计算图。
- `48-50`
  - 遍历所有参数；没有梯度的跳过。
- `52-55`
  - 取出当前参数梯度和对应状态。
- `57`
  - 更新一阶动量。
- `58`
  - 更新二阶动量。
- `60-62`
  - 计算归一化分母和实际步长。
- `64-65`
  - 如果设置了权重衰减，先做 AdamW 风格的 decoupled weight decay。
- `67`
  - 按 Adam 的规则更新参数。

对你理解这个 toy 来说，这段不是 semantic loss 的重点，但它承担了一个工程角色：

> 让这个项目不依赖外部优化器实现细节，自己就能完成稳定训练。

### 7.3 小工具函数

#### `_to_device`

- `70-71`
  - 把一个 batch 里的所有张量都搬到目标设备。

#### `_one_hot_targets`

- `74-75`
  - 把类别标签转成 one-hot。
  - 这里后面会配合 `BCEWithLogits` 一起用。

#### `semantic_weight_at`

- `78-80`
  - 计算当前 epoch 的 semantic loss 权重。
  - 先按 `epoch / ramp_up_epochs` 线性升高，再乘 `lambda_semantic`。

这意味着：
- 前期让模型先学基础分类
- 后期再更强地推它满足语义约束

### 7.4 `evaluate`

这个函数负责统一评估。

- `83`
  - 定义评估函数。
- `84`
  - 切到 `eval()` 模式。
- `85-87`
  - 从 `TensorDataset` 中取出整份数据并放到设备上。

- `89`
  - 评估阶段不用梯度。
- `90`
  - 前向得到 logits。
- `91`
  - 对每个输出位做 `sigmoid`，得到独立激活概率。
- `92`
  - 用 `argmax` 取预测类别。
  - 注意这里仍然把分类输出看作“最大概率的那个类”。
- `93`
  - 真实标签转 one-hot。

- `95`
  - 用 `BCEWithLogits` 计算监督损失。
  - 这说明模型是按“4 个独立二值输出”训练的，不是按 `softmax + cross entropy` 训练的。
- `96`
  - 计算平均满足概率。
- `97-101`
  - 计算硬满足率。

- `103-110`
  - 返回 6 个指标：
  - `loss`
  - `accuracy`
  - `mean_confidence`
  - `mean_satisfaction_probability`
  - `hard_constraint_satisfaction_rate`
  - `hard_constraint_violation_rate`

这里有一个很重要的理解点：

> 在这个 toy 里，accuracy 和 constraint satisfaction 是两条不同的轴。

也就是说：
- 一个模型可能分类准确，但结构不规整
- 也可能结构满足率高，但如果约束错了，准确率反而掉

### 7.5 `_make_optimizer`

- `113-118`
  - 用当前配置给模型构造一个 `ManualAdamW`。

### 7.6 `train_baseline`

这是纯监督版本。

- `121`
  - 定义 baseline 训练函数。
- `122`
  - 新建模型，并放到设备上。
- `123`
  - 构建优化器。
- `124`
  - 只给 labeled 数据建 `DataLoader`。

- `126-128`
  - 保存当前最佳模型状态，按验证集准确率选最优。
  - `history` 里记录训练损失、验证准确率和验证满足概率。

- `130`
  - 开始 epoch 循环。
- `131`
  - 切到训练模式。
- `132`
  - 累积 epoch 内训练损失。

- `134`
  - 遍历 labeled batch。
- `135`
  - 把 batch 放到设备上。
- `136`
  - 清梯度。
- `137`
  - 前向。
- `138`
  - 监督损失是 `BCEWithLogits(logits, one_hot_targets)`。
- `139`
  - 反向传播。
- `140`
  - 参数更新。
- `141`
  - 累积总损失。

- `143`
  - 每个 epoch 结束后在验证集上评估。
- `144-146`
  - 把训练曲线和验证曲线存到 `history`。
- `148-150`
  - 如果验证准确率更好，就把当前参数保存成最佳状态。

- `152-153`
  - 训练结束后恢复最佳模型，并返回模型和训练历史。

这个 baseline 很重要，因为它让你能回答一个干净的问题：

> 不加语义知识时，单靠少量标注，模型能学到什么程度？

### 7.7 `train_semantic_guided`

这是整份 toy 里最重要的训练函数。

- `156`
  - 定义 semantic-guided 训练函数。
- `157`
  - 新建模型。
- `158`
  - 新建优化器。
- `159`
  - labeled 数据 loader。
- `160`
  - unlabeled 数据 loader。

- `162-170`
  - 和 baseline 一样保存最佳状态和训练历史，只是这次要额外记录 `semantic_loss` 曲线。

- `172`
  - 进入 epoch 循环。
- `173`
  - 切训练模式。
- `174`
  - 计算当前 epoch 的 semantic loss 权重 `lambda_t`。
- `175-176`
  - 用 `itertools.cycle(...)` 把两个 loader 都变成可循环迭代器。
  - 这样如果一个 loader 更短，它会自动从头再来。
- `177`
  - 每轮 epoch 跑 `max(len(labeled_loader), len(unlabeled_loader))` 步。
  - 这样两个数据源都会被充分用到。

- `179-182`
  - 初始化统计量。

- `184`
  - 开始 step 循环。
- `185`
  - 取一个 labeled batch。
- `186`
  - 取一个 unlabeled batch。
  - 这里虽然 `TensorDataset` 里也有标签，但训练时故意不使用它。

- `188`
  - 清梯度。
- `189`
  - 对 labeled batch 前向。
- `190`
  - 对 unlabeled batch 前向。

- `192-195`
  - 监督项还是普通 one-hot BCE。
- `196`
  - semantic 项直接调用 `semantic_loss(unlabeled_logits, constraint_spec)`。
  - 注意这一步完全没看 unlabeled 的真实标签。
- `197`
  - 总损失就是：

```text
total_loss = supervised_term + lambda_t * semantic_term
```

  - 和 `logic_net_toy` 不同，这里没有 teacher-student 蒸馏，也没有 KL distillation。
  - 它是更直接的“监督项 + 语义项”。

- `198`
  - 反向传播。
- `199`
  - 参数更新。

- `201-205`
  - 累积各项损失统计。
  - 这里用的是 labeled batch 大小做平均尺度，主要是为了让曲线量级稳定可读。

- `207`
  - epoch 末在验证集上评估。
- `208-212`
  - 记录总损失、监督损失、semantic 损失、验证准确率、验证满足概率。
- `214-216`
  - 按验证准确率保存最佳模型。

- `218-219`
  - 恢复最佳状态并返回。

这整个函数最值得你记住的是：

> semantic loss 并不是替代监督项，而是在无标签数据上补一条“输出结构应该像什么样”的训练信号。

### 7.8 `save_metrics`

- `222-224`
  - 把实验指标写成 `metrics.json`。
  - `ensure_ascii=False` 保证中文字段也能正常写。

### 7.9 `plot_training_curves`

- `227`
  - 定义训练曲线绘图函数。
- `228`
  - 创建左右两个子图。

#### 左图：训练损失

- `230`
  - baseline 训练损失。
- `231`
  - semantic 模型总损失。
- `232`
  - semantic 模型监督项。
- `233`
  - semantic 模型约束项。
- `234-236`
  - 设置标题、横轴和图例。

#### 右图：验证指标

- `238`
  - baseline 验证准确率。
- `239`
  - semantic 验证准确率。
- `240`
  - baseline 验证满足概率。
- `241`
  - semantic 验证满足概率。
- `242-245`
  - 设置标题、横轴、纵轴范围和图例。

- `247-249`
  - 调整布局、保存图片、关闭图对象。

### 7.10 `plot_decision_and_constraint_maps`

这个函数把“分类区域”和“约束满足概率区域”画在一张图里。

- `252`
  - 定义画图函数。
- `253`
  - 拿到 labeled 数据。
- `254-255`
  - 指定二维显示范围。

- `257-260`
  - 在平面上生成规则网格。
  - `grid` 是所有网格点拼成的 `[N, 2]` 张量。

- `262`
  - 创建 `2 x 2` 子图。
  - 两列对应 baseline 和 semantic-guided。
  - 上排画决策区域，下排画约束满足概率。
- `263`
  - 组织模型列表。
- `264`
  - 定义 4 类的颜色映射。

- `266`
  - 逐列处理两个模型。
- `267-272`
  - 对整张网格做前向：
  - `logits`
  - `sigmoid` 概率
  - `argmax` 得到预测类别
  - `satisfaction_probability` 得到每个网格点的约束满足概率

- `274-275`
  - 取出这一列的上下两个子图。

#### 上排：决策区域

- `277-284`
  - 用 `contourf` 画预测类别区域。
  - `levels=np.arange(config.num_classes + 1) - 0.5` 用来让 4 个离散类别正好各占一个颜色区间。
- `285`
  - 把 labeled 样本点叠加上去。
- `286-287`
  - 标题和横轴。

#### 下排：约束满足概率

- `289`
  - 用 `contourf` 画满足概率热力图。
- `290`
  - 同样叠加 labeled 样本点。
- `291-292`
  - 标题和横轴。
- `294`
  - 给满足概率图单独加 colorbar。

- `296-297`
  - 左列加 y 轴标签。
- `298-299`
  - 保存图片并关闭图对象。

这个函数很有价值，因为它同时把两件事画出来了：

> 模型“把哪里判成哪一类”和“模型在什么区域更满足约束”，并不总是同一件事。

---

## 8. [experiment.py](./experiment.py)

这个文件负责把前面的模块串起来。

### 8.1 导入

- `1`: 导入 `get_constraint_spec` 和 `serialize_constraint_spec`
- `2`: 导入 `create_datasets` 和 `set_seed`
- `3-10`
  - 从 `trainer.py` 导入训练、评估、绘图、保存指标这些主函数

### 8.2 `run_experiment`

- `13`
  - 定义实验总入口。
- `14`
  - 保证结果目录存在。
- `15`
  - 固定随机种子。
- `16`
  - 生成数据集。
- `17`
  - 根据配置读取当前约束定义。

- `19`
  - 训练 baseline。
- `20`
  - 训练 semantic-guided 模型。

- `22`
  - 在测试集上评估 baseline。
- `23`
  - 在测试集上评估 semantic-guided。

- `25-37`
  - 组装最终 `metrics` 字典。
  - 里面包含：
  - 配置
  - 约束定义
  - baseline 指标
  - semantic-guided 指标
  - `delta_accuracy`
  - `delta_satisfaction_probability`
  - `delta_violation_rate`

这里要特别注意：

- `30`
  - `delta_accuracy = semantic - baseline`
  - 正数表示 semantic-guided 更好。
- `31-33`
  - `delta_satisfaction_probability` 也是 semantic 减 baseline。
  - 正数表示满足概率提升。
- `34-36`
  - `delta_violation_rate` 仍然是 semantic 减 baseline。
  - 所以如果这个值是负数，反而代表 semantic-guided 更好，因为 violation rate 降下去了。

- `39`
  - 如果要求保存结果，就进入保存分支。
- `40`
  - 先写 `metrics.json`。
- `41`
  - 如果允许画图，再保存图像。
- `42-49`
  - 保存决策区域 + 满足概率图。
- `50-54`
  - 保存训练曲线图。

- `56`
  - 返回最终指标字典。

这个文件的作用可以概括成：

> 它不定义新算法，只做一件事：把配置、数据、约束、训练、评估、可视化组织成一个完整实验。

---

## 9. 这份 toy 最值得你真正看懂的 5 个点

### 9.1 为什么这里不用 `softmax + cross entropy`？

因为这里要演示的是“输出结构约束”。
如果直接用 `softmax`，模型天生就会把输出压成总和为 1 的分布，`exactly-one` 的一部分结构会被模型形式提前吸收掉。

这里故意写成：
- 输出 4 个独立 logits
- 监督用 one-hot BCE
- 再额外用 semantic loss 去约束合法结构

这样你才能清楚看到 semantic loss 真正在补什么。

### 9.2 这个 toy 的 semantic loss 本质上在算什么？

它不是算“哪个类最大”。
它算的是：

```text
当前模型输出分布，在所有合法二值赋值上的总概率质量
```

如果这个总质量大，说明模型更相信合法结构。
如果这个总质量小，说明模型把概率分到了很多不合法结构上。

### 9.3 为什么训练时 semantic loss 只用 unlabeled 样本？

因为这正是它最有代表性的使用方式：
- labeled 样本给监督信号
- unlabeled 样本给结构信号

这能最清楚地体现：

> 就算没有标签，只要你知道输出该长什么样，也能给模型额外训练信息。

### 9.4 为什么还要算 `hard_constraint_satisfaction_rate`？

因为 `mean_satisfaction_probability` 是软指标。
它告诉你“平均来看合法概率多不多”。

但有时你还想知道：

> 真要把每一位硬阈值化，最终有多少输出真的落在合法集合里？

这就是硬满足率的作用。

### 9.5 这份 toy 和 `logic_net_toy` 最本质的区别是什么？

`logic_net_toy`：
- 规则先变 teacher
- 再做 student 蒸馏

`semantic_loss_toy`：
- 不造 teacher
- 直接对输出分布本身加约束

也就是说，这个 toy 更接近“语义直接进 loss”的思路。

---

## 10. 建议你接下来怎么读这份代码

如果你是第一次看，建议顺序是：

1. 先看 `run.py`
   - 先知道脚本怎么启动
2. 再看 `experiment.py`
   - 先抓住总体流程
3. 再看 `constraints.py`
   - 这是论文思想真正落地的地方
4. 再看 `trainer.py`
   - 看 semantic loss 是怎样接进训练循环的
5. 最后再看 `data.py` 和 `model.py`
   - 理解 toy 任务本身和模型结构

如果你是准备改代码，最值得先动的地方是：

- `constraints.py`
  - 加新约束或错误约束
- `trainer.py`
  - 改 loss 权重策略
- `run.py`
  - 暴露更多命令行参数
- `data.py`
  - 把 4 类 toy 换成你自己更想看的结构

这份解析读完后，你应该至少能自己说清楚三句话：

1. 这个 toy 为什么用独立 sigmoid 而不是 softmax。
2. `log_satisfaction_mass` 那一行广播到底在算什么。
3. semantic loss 为什么能在无标签样本上提供训练信号。
