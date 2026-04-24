# logic_net_toy 代码逐行解析

这份文档对应当前目录下的 toy 项目代码，目标是把代码从“能跑”解释成“你能自己改”。

说明：

- 我按文件顺序讲。
- 我尽量细到逐行粒度，但对连续几行完全同类的声明，会合并成一个小块解释。
- 这里解析的是代码文件，不包括 `README.md` 和 `notes.md`。

---

## 1. 先看整个项目是怎么串起来的

这个 toy 的执行路径是：

1. `run.py`
   - 解析命令行参数
   - 生成 `ExperimentConfig`
   - 调用 `experiment.py` 里的 `run_experiment`

2. `experiment.py`
   - 设置随机种子
   - 生成数据集
   - 读取规则集
   - 分别训练 baseline 和 logic-guided model
   - 调用 `trainer.py` 评估和画图

3. `trainer.py`
   - baseline：纯监督训练
   - logic-guided：监督损失 + 蒸馏损失
   - teacher 不是一个单独网络，而是由 `rules.py` 动态构造的 soft target

4. `rules.py`
   - 定义 rule set
   - 把规则转成概率分布
   - 把多条规则聚合成 teacher

5. `data.py`
   - 生成二维二分类 toy 数据
   - 保证类别大致平衡

所以你可以把整个项目压成一句话：

> 先造二维数据，再把规则转成 teacher 分布，然后让 student 同时学标签和 teacher。

---

## 2. [config.py](./config.py)

这个文件只做一件事：集中定义实验配置。

### 2.1 导入部分

- `1`: `from dataclasses import dataclass`
  - 导入 `dataclass`，后面用它把配置类写成简洁的“字段集合”。
- `2`: `from pathlib import Path`
  - 用 `Path` 处理结果目录路径。
- `3`: `from typing import Optional`
  - 说明某些字段可以为空，比如 `results_dir` 初始可以是 `None`。

### 2.2 配置类声明

- `6`: `@dataclass`
  - 告诉 Python：下面这个类主要是“存数据”的，不是复杂逻辑类。
- `7`: `class ExperimentConfig:`
  - 定义整个实验的统一配置对象。

### 2.3 随机性和数据规模

- `8`: `seed: int = 7`
  - 随机种子，控制数据采样和训练可复现。
- `9`: `num_labeled: int = 64`
  - 有标签数据量。
- `10`: `num_unlabeled: int = 512`
  - 无标签数据量，logic distillation 会用到。
- `11`: `num_val: int = 256`
  - 验证集大小。
- `12`: `num_test: int = 512`
  - 测试集大小。

### 2.4 模型和 batch 配置

- `13`: `hidden_dim: int = 32`
  - 两层 MLP 的隐藏层维度。
- `14`: `batch_size_labeled: int = 32`
  - 有标签 batch 大小。
- `15`: `batch_size_unlabeled: int = 64`
  - 无标签 batch 大小。

### 2.5 优化相关配置

- `16`: `epochs: int = 140`
  - 总训练轮数。
- `17`: `learning_rate: float = 1e-2`
  - 学习率。
- `18`: `weight_decay: float = 1e-4`
  - 权重衰减，用于简单正则化。

### 2.6 规则相关配置

- `19`: `rule_temperature: float = 2.0`
  - 规则 sigmoid 的温度，越大越接近硬规则边界。
- `20`: `rule_strength: float = 1.25`
  - 规则在 teacher 构造中的强度。
- `21`: `rule_set_name: str = "single_good"`
  - 当前使用哪一个规则集。
- `22`: `max_distill_weight: float = 0.65`
  - 蒸馏项最大占比。
- `23`: `ramp_up_epochs: int = 40`
  - 蒸馏权重不是一开始就拉满，而是前 40 个 epoch 渐进增大。

### 2.7 画图和设备相关

- `24`: `mesh_step: float = 0.04`
  - 决策边界图的网格步长。
- `25`: `device: str = "cpu"`
  - 当前固定 CPU。

### 2.8 输出目录相关

- `26`: `experiment_name: str = "default"`
  - 当前实验名称，决定结果目录名。
- `27`: `results_root: Path = Path(__file__).resolve().parent / "results"`
  - 默认结果根目录是当前文件夹下的 `results/`。
- `28`: `results_dir: Optional[Path] = None`
  - 具体实验目录可以后面再生成。

### 2.9 目录创建函数

- `30`: `def ensure_results_dir(self) -> Path:`
  - 保证实验输出目录存在。
- `31-32`
  - 如果用户没有显式传入 `results_dir`，就用 `results_root / experiment_name` 作为默认目录。
- `33`
  - 真正创建目录，`parents=True` 允许自动创建多级目录。
- `34`
  - 返回目录路径，方便外部调用。

### 2.10 序列化函数

- `36`: `def to_dict(self) -> dict:`
  - 把配置对象转成普通字典，用于写入 `metrics.json`。
- `37-58`
  - 把所有核心超参数逐个展开。
  - `results_dir` 这里转成字符串，因为 JSON 不能直接序列化 `Path`。

这个文件的核心思想很简单：

> 所有超参数统一放在一个对象里，训练、评估、画图都只依赖这个对象。

---

## 3. [data.py](./data.py)

这个文件负责生成 toy 数据。

### 3.1 导入和数据结构

- `1`: 导入 `dataclass`
- `3`: 导入 `torch`
- `4`: 导入 `TensorDataset`

- `7`: `@dataclass`
- `8`: `class DatasetBundle:`
  - 定义一个统一的数据打包结构。
- `9`: `labeled: TensorDataset`
  - 有标签数据集。
- `10`: `unlabeled: TensorDataset`
  - 无标签数据集。
- `11`: `val: TensorDataset`
  - 验证集。
- `12`: `test: TensorDataset`
  - 测试集。

### 3.2 随机种子

- `15`: `def set_seed(seed: int) -> None:`
  - 设置随机种子。
- `16`
  - 调用 `torch.manual_seed(seed)`。

### 3.3 真实决策函数

- `19`: `def decision_score(x: torch.Tensor) -> torch.Tensor:`
  - 这是“真实世界”的打分函数。
- `20`: `x1 = x[:, 0]`
  - 取第一维特征。
- `21`: `x2 = x[:, 1]`
  - 取第二维特征。
- `22`
  - 真正的打分公式。
  - 它是线性项、正弦项和二次项的组合，所以不是一条简单直线。

这一行的意义是：

- 让问题比线性分类稍复杂
- 给规则留下发挥空间
- 但又不至于复杂到看不懂

### 3.4 标签生成

- `25`: `def make_labels(x: torch.Tensor) -> torch.Tensor:`
  - 根据打分函数转成 0/1 标签。
- `26`
  - `decision_score(x) > 0.0` 为真记为 1，否则为 0。
  - `.long()` 把布尔值转成整型类别标签。

### 3.5 候选池采样

- `29`: `_sample_candidate_pool(...)`
  - 这是内部辅助函数。
- `30`
  - 在 `[-2, 2] x [-2, 2]` 内均匀采样二维点。
- `31`
  - 根据真实函数给这些点打标签。
- `32`
  - 返回候选样本和标签。

### 3.6 平衡采样函数

- `35`: `def make_balanced_dataset(...)`
  - 为了避免类别极不均衡，这个函数会主动采正负类各一半左右。
- `36`: `half = num_points // 2`
  - 目标正类大概一半。
- `37-38`
  - 分别算出还需要多少正类和负类。
- `40-43`
  - 初始化存储容器和计数器。

- `45`
  - 只要正类或负类还没凑够，就继续采样。
- `46`
  - 每次从候选池里采一大批样本，避免反复小采样。
- `48-49`
  - 做正类 mask 和负类 mask。

- `51-56`
  - 如果正类还不够，就从当前候选池里拿一部分正类追加进去。
- `58-63`
  - 对负类做同样的事。

- `65`
  - 把所有临时片段拼接起来。
- `66`
  - 拼接标签。
- `67`
  - 再随机打乱顺序。
- `68`
  - 返回平衡后的数据和标签。

### 3.7 最终数据集构造

- `71`: `def create_datasets(config) -> DatasetBundle:`
  - 用配置对象统一生成四份数据。
- `72`
  - 创建带固定种子的 `torch.Generator`。

- `74-77`
  - 依次生成 labeled、unlabeled、val、test。

- `79-84`
  - 打包成 `DatasetBundle` 返回。

这一整个文件要表达的是：

> 先定义一个真实但不太复杂的二维分类边界，再人为制造“少标注 + 多无标签”的环境。

---

## 4. [model.py](./model.py)

这个文件非常简单，只定义模型。

### 4.1 导入

- `1`: `import torch.nn as nn`
  - 导入 PyTorch 的神经网络模块。

### 4.2 模型类

- `4`: `class TinyMLP(nn.Module):`
  - 定义一个小型多层感知机。
- `5`: `def __init__(self, hidden_dim: int = 32) -> None:`
  - 构造函数，允许外部指定隐藏维度。
- `6`: `super().__init__()`
  - 初始化父类。

- `7-13`
  - 用 `nn.Sequential` 堆出一个两层隐藏层的网络：
  - `8`: 输入 2 维 -> 隐藏层
  - `9`: `Tanh`
  - `10`: 隐藏层 -> 隐藏层
  - `11`: `Tanh`
  - `12`: 隐藏层 -> 2 类输出 logits

- `15`: `def forward(self, x):`
  - 前向传播入口。
- `16`
  - 把输入直接送进 `self.net`。

这里的设计意图是：

- 模型足够小，便于看清规则作用
- 非线性足够表达 toy 数据的边界

---

## 5. [rules.py](./rules.py)

这个文件是项目的关键，因为它把“规则”变成了“teacher 概率分布”。

### 5.1 导入

- `1`: 导入 `asdict` 和 `dataclass`
  - `RuleSpec` 需要 dataclass，后面写 JSON 要用 `asdict`
- `3`: 导入 `torch`
- `4`: 导入 `torch.nn.functional as F`

### 5.2 单条规则的数据结构

- `7`: `@dataclass(frozen=True)`
  - `frozen=True` 表示规则定义后不可修改。
- `8`: `class RuleSpec:`
  - 用来描述一条规则。
- `9`: `name`
  - 规则名。
- `10`: `description`
  - 人类可读描述。
- `11`: `coefficients`
  - 规则线性边界的系数 `(a, b)`。
- `12`: `bias`
  - 偏置项。
- `13`: `positive_class`
  - 满足规则时偏向哪个类别。
- `14`: `weight`
  - 这条规则在多规则聚合中的权重。
- `15`: `temperature_scale`
  - 这条规则自己的温度倍率。

### 5.3 规则集字典

- `18`: `RULE_SETS = {`
  - 整个项目所有规则集都集中写在这里。

#### `single_good`

- `19-27`
  - 一条正确规则：`x1 > x2` 时偏向 class 1。

#### `single_bad`

- `28-36`
  - 同样的几何边界，但故意把偏向类别写反，变成错误规则。

#### `multi_good`

- `37-61`
  - 三条总体正确的规则。
  - `diag_positive`：对角线规则。
  - `x1_positive`：`x1 > 0.15` 倾向 class 1。
  - `x2_small`：`x2 < 0.10` 倾向 class 1。

#### `multi_mixed`

- `62-86`
  - 两条相对合理规则 + 一条故意偏坏的规则。
  - 用来测 mixed rules 的鲁棒性。

#### `multi_bad`

- `87-111`
  - 多条系统性错误规则。
  - 用来测“teacher 会不会被坏知识整体带偏”。

### 5.4 规则集查询函数

- `115-116`
  - 返回所有规则集名称并排序。
- `119-123`
  - 按名称取规则集。
  - 如果名称非法，就抛异常。

### 5.5 规则序列化

- `125-126`
  - 把规则对象列表转成字典列表，方便写进 JSON。

### 5.6 单条规则的 margin

- `129`: `def rule_margin(x, rule):`
  - 计算规则线性边界的 margin。
- `130`
  - 实际就是 `a*x1 + b*x2 + c`。

这个 margin 的意义是：

- 大于 0：更满足规则
- 小于 0：更不满足规则

### 5.7 单条规则的软概率

- `133`: `soft_rule_probability_for_rule(...)`
  - 把 margin 变成 soft probability。
- `134`
  - 当前规则温度 = 全局温度 * 规则私有缩放。
- `135`
  - 用 sigmoid 把线性 margin 转成 `(0, 1)` 概率。
- `136-138`
  - 如果这条规则偏向 class 1，就直接返回 sigmoid。
  - 如果偏向 class 0，就返回 `1 - sigmoid`。

### 5.8 单条规则的分布形式

- `141`: `rule_distribution(...)`
  - 把单个 class1 概率扩成完整二分类分布 `[p(class0), p(class1)]`。
- `142`
  - 先算 class1 概率。
- `143`
  - 用 `torch.stack` 拼成二分类分布。

### 5.9 单条规则的硬预测

- `146-147`
  - 如果 soft probability >= 0.5，就视作预测 class 1，否则 class 0。

### 5.10 多规则聚合

- `150`: `aggregate_rule_distribution(...)`
  - 把多条规则聚合成一个总规则分布。
- `151-152`
  - 没有规则时直接报错。
- `154`
  - 初始化 `log_rule`，形状是 `(batch_size, 2)`。
- `155-157`
  - 对每条规则：
    - 算出这条规则对应的二分类分布
    - 取对数
    - 按规则权重加到总 log 分布上
- `158`
  - 最后再 softmax 回普通概率分布。

这里的设计相当于：

> 多条规则在 log-probability 空间做加权融合。

### 5.11 聚合后的硬预测

- `161-162`
  - 先聚合规则分布，再取 `argmax` 得到多规则最终预测。

### 5.12 teacher 构造

- `165-171`
  - 定义核心函数 `build_teacher_probs(...)`。
- `172`
  - 先把 student logits 转成 student 概率。
- `173`
  - 再算规则聚合后的概率。

- `175-176`
  - 对 student 概率和 rule 概率都取 log。
- `177`
  - `teacher_logits = log_student + rule_strength * log_rule`
  - 这是核心：teacher 不是独立模型，而是当前 student 和规则一起构造出来的。
- `178`
  - 最后再 softmax 得到 teacher 分布。

### 5.13 规则标签简写

- `181-183`
  - 给画图时的规则标签做一个短格式，比如 `diag_positive->c1`。

这个文件的核心思想可以压缩成一句话：

> 规则先变成概率分布，多条规则先聚合，再和当前 student 概率融合，形成 teacher。

---

## 6. [trainer.py](./trainer.py)

这是整个项目最重要的文件，真正实现训练、评估和画图。

### 6.1 导入部分

- `1-5`
  - 导入拷贝、循环器、JSON、数学库、路径工具。
- `7-10`
  - 导入 `matplotlib`、`torch`、损失函数接口和 `DataLoader`。
- `12-18`
  - 导入模型和规则相关工具。
- `20`
  - 指定 `matplotlib` 使用 `Agg` 后端，这样即使没有 GUI 也能存图。
- `21`
  - 再导入 `pyplot`。

### 6.2 手写优化器 `ManualAdamW`

这个类是因为当前环境里 `torch.optim` 受 `sympy` 问题影响，所以这里手写了一个简版 AdamW。

#### 初始化

- `24`: 定义 `ManualAdamW`
- `25-30`
  - 接收参数、学习率、权重衰减、betas、eps。
- `26`
  - 只保留需要梯度的参数。
- `31`
  - 初始化 step 计数。
- `32-38`
  - 为每个参数建立状态字典：
    - 一阶动量 `exp_avg`
    - 二阶动量 `exp_avg_sq`

#### 清梯度

- `40-43`
  - 遍历参数，如果存在梯度就归零。

#### 更新一步

- `45`
  - `step()` 真正执行参数更新。
- `46-48`
  - 更新步数并计算偏置修正项。
- `50-70`
  - 对每个参数执行 AdamW 风格更新：
    - 取当前梯度
    - 更新一阶矩
    - 更新二阶矩
    - 计算分母和步长
    - 加权衰减参数
    - 按 Adam 公式更新参数

### 6.3 小工具函数

- `73-74`
  - `_to_device`：把一个 batch 中的每个张量都搬到目标设备。

- `77-79`
  - `distill_weight_at`：蒸馏权重调度器。
  - 前 `ramp_up_epochs` 内线性增长，之后保持 `max_distill_weight`。

### 6.4 评估函数

- `82`: `evaluate(model, dataset, config, rule_specs)`
  - 用于验证集/测试集评估。
- `83`
  - 切换到 eval 模式。
- `84`
  - 从 `TensorDataset` 里取出整份 `x, y`。
- `85-86`
  - 把张量搬到设备上。
- `87-90`
  - 前向计算 logits、概率和预测类别。

- `92`
  - 交叉熵损失。
- `93`
  - 准确率。

- `94-101`
  - 如果没有规则集，就只返回纯监督指标。
  - baseline 验证时会走这个分支。

- `103`
  - 算聚合规则的硬预测。
- `104`
  - 规则本身在当前数据上的准确率。
- `105`
  - 模型预测和规则预测的一致率。

- `107-118`
  - 逐条规则做单独统计：
    - 规则名
    - 规则描述
    - 规则权重
    - 这条规则在数据上的准确率
    - 模型与这条规则的一致率

- `120-126`
  - 返回完整评估字典。

### 6.5 优化器工厂

- `129-134`
  - `_make_optimizer`：统一返回 `ManualAdamW`。

### 6.6 baseline 训练

- `137`: `train_baseline(...)`
  - 纯监督训练版本。
- `138`
  - 初始化模型。
- `139`
  - 初始化优化器。
- `140`
  - 为 labeled 数据创建 DataLoader。

- `142-144`
  - 保存当前最佳模型状态和训练历史。

- `146-167`
  - 主训练循环：
    - `147` 进入 train 模式
    - `148` 初始化 epoch loss
    - `149-156` 遍历 labeled batch
      - 搬设备
      - 清梯度
      - 前向
      - 监督交叉熵
      - 反向传播
      - 参数更新
    - `158` 算平均训练损失
    - `159` 在验证集上评估
    - `160-161` 记录历史
    - `163-165` 如果验证准确率更好，就保存当前模型

- `167-168`
  - 训练结束后恢复最佳模型并返回。

### 6.7 logic-guided 训练

- `171`: `train_logic_guided(...)`
  - 逻辑蒸馏版本。
- `172-175`
  - 初始化模型、优化器、有标签和无标签 DataLoader。
- `177-179`
  - 初始化最佳模型记录和 history。

- `181-239`
  - 主训练循环。

#### 每轮开头

- `182`
  - 切换 train 模式。
- `183`
  - 计算当前 epoch 的蒸馏权重 `pi_t`。
- `184-185`
  - 用 `itertools.cycle` 把 loader 变成可循环迭代器。
  - 这样 labeled 和 unlabeled 数量不同也能对齐使用。
- `186`
  - 每轮 step 数取两个 loader 长度的最大值。

#### 每个 step

- `193-195`
  - 各取一个 labeled batch 和一个 unlabeled batch。

- `197`
  - 清梯度。

- `199`
  - 对 labeled 样本前向。
- `200`
  - 计算监督损失。

- `202`
  - 把 labeled + unlabeled 拼起来，作为规则蒸馏的输入。
- `203`
  - student 对这些样本做前向。

- `204-211`
  - 在 `torch.no_grad()` 下构造 teacher：
    - `206` 用当前 student logits
    - `207` 用当前 batch 样本
    - `208` 用规则集
    - `209-210` 用规则强度和温度

- `213-217`
  - 计算 student 与 teacher 的 KL 蒸馏损失。

- `219`
  - 总损失：
  - `(1 - pi_t) * supervised_loss + pi_t * distill_loss`

- `220-221`
  - 反向传播并更新参数。

- `223-227`
  - 统计本轮的总损失、监督损失、蒸馏损失。

#### 每轮末尾

- `229`
  - 在验证集上评估。
- `230-233`
  - 记录三类训练损失和验证准确率。
- `235-237`
  - 维护最佳模型。

- `239-240`
  - 训练完成后恢复最佳模型并返回。

### 6.8 指标保存

- `243-245`
  - 把 metrics 写成 `metrics.json`。

### 6.9 训练曲线画图

- `248-266`
  - 左图画 loss，右图画验证准确率。
  - baseline 只有 train loss。
  - logic-guided 额外显示 total / supervised。

### 6.10 规则边界画图

- `269`: `_plot_rule_boundaries(...)`
  - 根据规则系数在平面上画直线边界。
- `270`
  - 预定义几种颜色。
- `271-283`
  - 对每条规则：
    - 取系数 `a, b, c`
    - 如果 `b != 0`，画斜线 `a*x + b*y + c = 0`
    - 如果 `b == 0` 但 `a != 0`，画竖线

### 6.11 决策边界图

- `286`: `plot_decision_boundaries(...)`
  - 画 baseline 和 logic-guided 两张对照图。
- `287-288`
  - 取出 labeled 和 test 数据。
- `290-295`
  - 构造网格点，用来计算整张图上的分类概率。
- `297-298`
  - 创建两个子图。

- `300-312`
  - 对 baseline 和 logic-guided 逐个作图：
    - 预测网格上每个点的 class1 概率
    - 用 `contourf` 画概率热图
    - 用 `contour` 画 0.5 决策边界
    - 用散点画 test 数据和 labeled 数据
    - 叠加规则边界

- `314-316`
  - 加 y 轴标签、colorbar，并保存图片。

这个文件是整个项目的核心，因为它把“论文思想”真正变成了：

- 可训练
- 可评估
- 可视化

---

## 7. [experiment.py](./experiment.py)

这个文件是实验编排层。

### 7.1 导入

- `1-8`
  - 从 `trainer.py` 导入训练、评估、画图、存指标函数。
- `9`
  - 从 `data.py` 导入数据生成和设种子。
- `10`
  - 从 `rules.py` 导入规则查询和序列化工具。

### 7.2 实验总入口

- `13`: `run_experiment(config, save_artifacts=True, save_plots=True)`
  - 这是整个项目的统一总入口。

- `14`
  - 保证结果目录存在。
- `15`
  - 固定随机种子。
- `16`
  - 生成数据。
- `17`
  - 按规则集名称加载规则。

- `19`
  - 训练 baseline。
- `20`
  - 训练 logic-guided model。

- `22-23`
  - 分别在 test 集上评估两个模型。

- `24-33`
  - 组装完整 metrics：
    - 配置
    - 规则明细
    - baseline 指标
    - logic-guided 指标
    - 准确率增量
    - 规则一致率增量

- `35-50`
  - 如果需要保存结果：
    - 写 `metrics.json`
    - 画决策边界图
    - 画训练曲线

- `52`
  - 返回 metrics，给外层 `run.py` 或 `sweep.py` 使用。

这个文件的作用是：

> 把“配置 -> 数据 -> 训练 -> 评估 -> 落盘”串成一条统一流程。

---

## 8. [run.py](./run.py)

这个文件是单次实验入口。

### 8.1 导入

- `1-2`
  - 导入命令行解析和路径工具。
- `4`
  - 导入配置类。
- `5`
  - 导入总实验入口。
- `6`
  - 导入可选规则集列表。

### 8.2 参数解析

- `9`: `def parse_args():`
  - 定义命令行参数。
- `10`
  - 创建 `ArgumentParser`。
- `11`
  - `--seed`
- `12`
  - `--epochs`
- `13`
  - `--num-labeled`
- `14`
  - `--num-unlabeled`
- `15`
  - `--rule-strength`
- `16`
  - `--max-distill-weight`
- `17`
  - `--rule-set`
  - `choices=available_rule_sets()` 限制只能用已知规则集。
- `18`
  - `--experiment-name`
- `19`
  - `--skip-plots`
  - 如果给了这个 flag，就只跑训练和评估，不画图。
- `20`
  - 返回解析后的参数对象。

### 8.3 参数转配置对象

- `23`: `def build_config(args) -> ExperimentConfig:`
  - 把命令行参数变成配置对象。
- `24-34`
  - 把每个命令行参数塞进 `ExperimentConfig`。
- `33`
  - 明确指定本次实验的结果路径。
- `35`
  - 确保输出目录存在。
- `36`
  - 返回配置对象。

### 8.4 主函数

- `39`: `def main():`
  - 单次实验主入口。
- `40`
  - 解析参数。
- `41`
  - 构造配置。
- `42`
  - 真正运行实验。

- `44`
  - 打印结果目录。
- `45`
  - 打印规则集名称。
- `46-48`
  - 打印 baseline、logic-guided 和增量准确率。

### 8.5 脚本入口

- `51-52`
  - 只有直接运行这个文件时才会调用 `main()`。

---

## 9. [sweep.py](./sweep.py)

这个文件负责批量参数扫描。

### 9.1 导入

- `1-5`
  - 命令行解析、CSV、笛卡尔积、JSON、路径工具。
- `7`
  - 配置类。
- `8`
  - 实验总入口。
- `9`
  - 可选规则集列表。

### 9.2 CSV 风格字符串解析

- `12`: `parse_csv_list(raw_value, cast_fn)`
  - 把类似 `"7,13"` 这种字符串拆成列表。
- `13`
  - 去空格后逐个用 `cast_fn` 转型。

### 9.3 sweep 参数解析

- `16-32`
  - 定义 sweep 脚本的命令行参数：
    - `name`
    - `epochs`
    - `seeds`
    - `num-labeled-values`
    - `rule-strengths`
    - `distill-weights`
    - `rule-sets`
    - `num-unlabeled`
    - `save-plots`

### 9.4 规则集合法性检查

- `35-40`
  - 把用户给的规则集名字和系统支持列表对比。
  - 如果有未知名字就抛错。

### 9.5 保存 sweep 汇总

- `43`: `save_summary(rows, output_dir)`
  - 把所有扫描结果保存为 JSON 和 CSV。
- `44-45`
  - 决定输出文件名。
- `47-48`
  - 写 JSON。
- `50-54`
  - 写 CSV。

### 9.6 sweep 主函数

- `57`: `def main():`
  - 批量实验总入口。
- `58`
  - 读取命令行参数。
- `59-63`
  - 把字符串形式的 seeds、标注量、规则强度、蒸馏权重、规则集转成真正的列表。

- `65-66`
  - 创建本次 sweep 的根目录。

- `68`
  - 初始化结果列表。
- `69-71`
  - 用 `itertools.product` 枚举所有超参数组合。

### 9.7 遍历每个实验组合

- `73`
  - 开始循环，每个组合给一个递增编号。
- `74-77`
  - 为每个组合生成独一无二的实验名。

- `78-88`
  - 用当前组合生成一个 `ExperimentConfig`。

- `90`
  - 跑一次单实验。

- `91-107`
  - 把这次实验的核心结果抽取成一行 summary：
    - 哪个规则集
    - 哪个 seed
    - 标注量多少
    - 规则强度多少
    - 蒸馏权重多少
    - baseline / logic accuracy
    - delta_accuracy
    - 规则一致率变化
    - 结果目录

- `108-112`
  - 在终端打印当前进度和 delta。

### 9.8 sweep 收尾

- `114`
  - 按 `delta_accuracy` 从大到小排序。
- `115`
  - 保存汇总 JSON 和 CSV。
- `117-120`
  - 如果有结果，就额外把最优组合写成 `best_run.json`。
- `121`
  - 打印汇总目录。

- `124-125`
  - 脚本入口。

这个文件的意义是：

> 把单次 toy 实验变成了可做 ablation、可比较不同规则集和超参数的实验平台。

---

## 10. 这套代码最值得你记住的 5 个点

1. `data.py` 定义了一个“少标注 + 多无标签”的 toy 场景。
2. `rules.py` 的核心不是规则本身，而是“把规则变成 teacher 概率分布”。
3. `trainer.py` 的 logic-guided 版本本质是监督损失和 KL 蒸馏损失的组合。
4. `experiment.py` 负责把数据、规则、训练和画图串起来。
5. `sweep.py` 让这个 toy 不再只是 demo，而是可以做系统实验。

如果你下一步要改代码，最推荐先改的地方是：

- `RULE_SETS`
- `build_teacher_probs`
- `distill_weight_at`
- `sweep.py` 里的扫描维度
