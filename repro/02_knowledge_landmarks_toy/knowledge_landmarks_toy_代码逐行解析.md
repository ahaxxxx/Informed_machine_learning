# knowledge_landmarks_toy 代码逐行解析

这份文档对应当前目录下的 `knowledge_landmarks_toy` 项目，目标是把“局部数据 + 全局 landmarks + 组合损失”这套思路彻底拆开。

说明：

- 我按文件顺序讲。
- 我尽量细到逐行粒度，但对连续几行同类字段声明会合并解释。
- 这里只解析代码文件，不解析 `README.md` 和 `notes.md`。

---

## 1. 先看整个项目的数据流

这个 toy 的核心执行路径是：

1. `run.py`
   - 解析参数
   - 构造 `ExperimentConfig`
   - 调用 `experiment.py`

2. `experiment.py`
   - 设随机种子
   - 生成局部训练数据和全局验证/测试网格
   - 读取 landmarks
   - 从 landmarks 采 support points
   - 训练 baseline 和 knowledge-guided 两个模型
   - 保存指标和图

3. `trainer.py`
   - baseline：只拟合局部有标签数据
   - knowledge-guided：同时拟合局部数据和 landmarks regularizer

4. `landmarks.py`
   - 定义输入区间和对应输出区间
   - 构造不同质量的 landmarks 集

5. `data.py`
   - 定义真实函数
   - 只在局部窗口采样训练数据
   - 在全域上生成验证/测试曲线

所以这套代码的核心思想是：

> baseline 只看局部窗口，knowledge-guided 还会被全局 landmarks 拉住，因此更可能在窗口外也表现稳定。

---

## 2. [config.py](./config.py)

这个文件定义所有实验超参数。

### 2.1 导入

- `1`: 导入 `dataclass`
- `2`: 导入 `Path`
- `3`: 导入 `Optional`

### 2.2 配置类

- `6`: `@dataclass`
- `7`: `class ExperimentConfig:`
  - 统一存放所有实验配置。

### 2.3 数据规模和随机种子

- `8`: `seed: int = 11`
  - 随机种子。
- `9`: `num_train_local: int = 32`
  - 局部训练样本数量。
- `10`: `num_val_global: int = 301`
  - 全局验证网格点数量。
- `11`: `num_test_global: int = 601`
  - 全局测试网格点数量。

### 2.4 模型与训练

- `12`: `hidden_dim: int = 48`
  - 回归网络的隐藏层宽度。
- `13`: `batch_size: int = 24`
  - 训练 batch 大小。
- `14`: `epochs: int = 220`
  - 训练轮数。
- `15`: `learning_rate: float = 1e-2`
  - 学习率。
- `16`: `weight_decay: float = 1e-4`
  - 权重衰减。

### 2.5 组合损失相关

- `17`: `lambda_data: float = 0.7`
  - 数据项权重。
  - 对应论文里 `lambda * L1 + (1 - lambda) * L2` 的 `lambda`。
- `18`: `center_pull_weight: float = 0.10`
  - 除了区间出界惩罚之外，再给一个“朝区间中心弱拉回”的权重。

### 2.6 数据窗口和噪声

- `19`: `label_noise_std: float = 0.03`
  - 局部训练标签噪声标准差。
- `20`: `local_region_low: float = -0.7`
  - 局部训练窗口左边界。
- `21`: `local_region_high: float = 0.7`
  - 局部训练窗口右边界。

### 2.7 全域范围与 landmarks support

- `22`: `domain_low: float = -3.0`
  - 整个输入域左边界。
- `23`: `domain_high: float = 3.0`
  - 整个输入域右边界。
- `24`: `support_points_per_landmark: int = 40`
  - 每个 landmark 内部采多少个 support 点用于知识损失。
- `25`: `landmark_set_name: str = "good"`
  - 当前使用哪套 landmarks。

### 2.8 运行环境和输出目录

- `26`: `device: str = "cpu"`
- `27`: `experiment_name: str = "default"`
- `28`: `results_root = 当前目录/results`
- `29`: `results_dir` 初始可为空

### 2.9 创建输出目录

- `31-35`
  - `ensure_results_dir()` 保证结果目录存在。
  - 如果没有指定具体目录，就默认用 `results_root / experiment_name`。

### 2.10 转成字典

- `37-60`
  - `to_dict()` 把所有配置展开成普通字典。
  - 这样 `metrics.json` 里会带完整配置，方便复现实验。

---

## 3. [data.py](./data.py)

这个文件定义真实函数和数据生成方式。

### 3.1 导入与数据结构

- `1`: 导入 `dataclass`
- `3`: 导入 `torch`
- `4`: 导入 `TensorDataset`

- `7`: `@dataclass`
- `8`: `class DatasetBundle:`
  - 用一个结构统一打包三份数据：
- `9`: `train_local`
  - 局部训练数据。
- `10`: `val_global`
  - 全局验证数据。
- `11`: `test_global`
  - 全局测试数据。

### 3.2 随机种子

- `14-15`
  - `set_seed` 只做一件事：设置 `torch.manual_seed(seed)`。

### 3.3 真实函数

- `18`: `def true_function(x)`
  - 定义 toy 回归问题的真实目标函数。
- `19`
  - 把输入 reshape 成列向量。
- `20-25`
  - 真正的函数形式：
    - 三次项
    - 线性项
    - 正弦项
    - 余弦项

这个设计的目的：

- 局部看像平滑函数
- 全局看不是简单直线
- baseline 很容易在局部拟合好、全局外推差

### 3.4 局部训练数据采样

- `28`: `sample_local_data(...)`
  - 只在局部窗口中采训练样本。
- `29`
  - 在 `[low, high]` 里均匀采样一维输入。
- `30`
  - 计算无噪声真值。
- `31`
  - 生成高斯噪声。
- `32`
  - 把噪声加到真值上，得到训练标签。
- `33`
  - 返回输入和带噪输出。

### 3.5 全域网格生成

- `36`: `make_global_grid(...)`
  - 在全域上生成规则网格点。
- `37`
  - 用 `torch.linspace` 均匀覆盖整个定义域。
- `38`
  - 直接计算真实函数值。
- `39`
  - 返回全局 `(x, y)`。

### 3.6 数据打包

- `42`: `create_datasets(config)`
  - 根据配置生成三份数据。
- `43`
  - 创建带固定种子的生成器。

- `45-51`
  - 生成局部训练数据，只覆盖局部窗口。
- `52-56`
  - 生成全域验证网格。
- `57-61`
  - 生成更密的全域测试网格。

- `63-67`
  - 打包成 `DatasetBundle` 返回。

这个文件的关键思想是：

> 故意制造“训练数据只在局部可见，但评估要求看全域”的设定。

---

## 4. [landmarks.py](./landmarks.py)

这个文件是整个 toy 的知识核心。

### 4.1 导入

- `1`: 导入 `asdict` 和 `dataclass`
- `3`: 导入 `torch`
- `5`: 从 `data.py` 导入真实函数 `true_function`

这里直接调用 `true_function` 的目的，是用真函数来构造“理想化知识 landmarks”。

### 4.2 Landmark 数据结构

- `8`: `@dataclass(frozen=True)`
- `9`: `class Landmark:`
  - 一条 landmark 记录一个输入区间和一个输出区间。
- `10`: `name`
  - 名称。
- `11-12`: `x_low`, `x_high`
  - 输入 granule，对应一维输入区间。
- `13-14`: `y_low`, `y_high`
  - 输出 granule，对应允许的输出区间。
- `15`: `quality`
  - 这条 landmark 的质量标签，例如 `good`、`shifted`。

### 4.3 从真实函数计算输出区间

- `18`: `_interval_range(x_low, x_high, padding=0.0, shift=0.0)`
  - 给定一个输入区间，计算真实函数在这个区间内的输出范围。
- `19`
  - 在该区间里生成 160 个采样点。
- `20`
  - 计算这些点的真实函数值。
- `21`
  - 返回最小值和最大值，再加上 padding 和 shift。

这里的 `padding` 和 `shift` 非常关键：

- `padding` 控制“知识区间多宽”
- `shift` 控制“知识是否整体偏移”

### 4.4 输入区间划分

- `24`: `_base_intervals()`
  - 返回 6 个基础输入区间：
  - `[-3,-2], [-2,-1], ..., [2,3]`

### 4.5 根据配置批量生成 landmarks

- `35`: `_build_landmarks(intervals, paddings, shifts, quality_labels)`
  - 根据每段输入区间的 padding 和 shift 生成对应的 landmarks。
- `36`
  - 初始化空列表。
- `37-39`
  - 同时遍历输入区间、padding、shift、quality。
- `40`
  - 调 `_interval_range` 算出输出区间。
- `41-49`
  - 生成 `Landmark` 对象。
- `51`
  - 返回 landmarks 列表。

### 4.6 可用 landmark set 名称

- `54-55`
  - 返回所有合法 landmark set。

### 4.7 具体 landmark set 定义

- `58`: `get_landmarks(set_name)`
  - 这是外部加载知识集的入口。
- `59`
  - 先拿到基础输入区间。

#### good

- `60-66`
  - 每个区间都加很小的 padding
  - 没有 shift
  - 代表较准确的知识

#### coarse_good

- `67-73`
  - padding 更大
  - 没有 shift
  - 代表方向对，但更粗、更宽的知识

#### mixed

- `74-80`
  - 前三段没偏移
  - 后三段整体上移 `0.35`
  - 代表一半知识准确，一半有偏移

#### shifted_bad

- `81-87`
  - 所有区间统一上移 `0.55`
  - 代表系统性错误知识

- `88`
  - 如果名字非法，就报错。

### 4.8 序列化

- `91-92`
  - 把 landmarks 转成字典列表，用于写入 JSON。

### 4.9 从 landmark 采 support 点

- `95`: `sample_landmark_support(...)`
  - 从每个 landmark 的输入区间里采 support 点，供知识损失使用。
- `96`
  - 用 `seed + 1000` 建一个独立生成器，避免和训练数据完全重叠。
- `97-100`
  - 初始化容器。

- `102-107`
  - 对每个 landmark：
    - 在它的输入区间里随机采点
    - 为这些点附上对应输出区间下界
    - 附上对应输出区间上界
    - 记录这些点属于哪个 landmark

- `109-114`
  - 把所有 support 数据拼接起来返回。

这个文件的关键思想是：

> landmark 不是孤零零的矩形，而是会被采样成一批 support 点，进入训练损失。

---

## 5. [model.py](./model.py)

这个文件定义回归模型。

### 5.1 导入

- `1`: 导入 `torch.nn as nn`

### 5.2 模型类

- `4`: `class TinyRegressor(nn.Module):`
  - 一个小型回归网络。
- `5`
  - 接收隐藏层维度参数。
- `6`
  - 初始化父类。
- `7-13`
  - 网络结构：
    - `1 -> hidden`
    - `Tanh`
    - `hidden -> hidden`
    - `Tanh`
    - `hidden -> 1`
- `15-16`
  - 前向传播时直接调用 `self.net(x)`。

它的角色非常简单：

> 作为一个低门槛 baseline/kd 共用模型，不把复杂性放到网络结构里，而是放到损失项里。

---

## 6. [trainer.py](./trainer.py)

这个文件实现训练、评估和可视化。

### 6.1 导入

- `1-4`
  - `copy`、`json`、`math`、`Path`
- `6-10`
  - `matplotlib`、`torch`、`F`、画矩形用的 `patches`、`DataLoader`
- `12`
  - 导入模型
- `14`
  - 指定 `Agg` 后端
- `15`
  - 导入 `pyplot`

### 6.2 手写优化器

`ManualAdamW` 和前一个项目思路一样，原因也是一样：避免环境中 `torch.optim` 的问题。

#### 初始化

- `18-32`
  - 保存参数、学习率、weight decay、betas、eps
  - 为每个参数建立一阶矩和二阶矩状态

#### 清梯度

- `34-37`
  - 如果某个参数有梯度，就把它清零

#### 更新一步

- `39-64`
  - 按 AdamW 逻辑更新参数
  - 包括偏置修正、权重衰减、动量更新

### 6.3 优化器工厂

- `67-68`
  - `_make_optimizer` 统一返回 `ManualAdamW`

### 6.4 区间出界惩罚

- `71`: `interval_violation(pred, y_low, y_high)`
  - 定义 prediction 超出区间时的惩罚。
- `72`
  - 如果预测低于下界，就罚 `y_low - pred`
  - 如果预测高于上界，就罚 `pred - y_high`
  - 如果在区间内，就罚 0

### 6.5 知识距离

- `75`: `knowledge_distance(...)`
  - 这是 toy 中 `L2` 的具体实现。
- `81`
  - 先算出界惩罚。
- `82`
  - 计算区间中心。
- `83`
  - 计算半宽度，避免除零用 `clamp_min(1e-6)`。
- `84`
  - 计算 prediction 到区间中心的归一化平方距离。
- `85`
  - 最终知识距离 = 出界惩罚 + 中心拉回项。

这个设计的直觉是：

- 如果只罚出界，模型在大区间里可能乱跑
- 加一个较弱的中心拉回后，good landmarks 的约束信息更强

### 6.6 RMSE 计算

- `88-89`
  - 用 `sqrt(MSE)` 算 RMSE。

### 6.7 评估函数

- `92`: `evaluate(model, x, y, config, support)`
  - 在全域曲线上评估模型。
- `93`
  - 切 eval。
- `94-99`
  - 把测试数据和 support 数据都搬到设备上。

- `101-110`
  - 前向得到：
    - 整条曲线的预测 `pred`
    - support 点上的预测 `support_pred`
    - 出界惩罚 `violation`
    - 总知识惩罚 `knowledge_penalty`

- `112`
  - 定义本地窗口 mask。
- `113`
  - 定义窗口外 mask。

- `115-121`
  - 返回核心指标：
    - 全域 RMSE
    - 本地窗口 RMSE
    - 平均 landmark violation
    - 平均 landmark penalty
    - 在区间内的比例

- `122-125`
  - 如果窗口外有点，再额外计算窗口外 RMSE。

### 6.8 baseline 训练

- `129`: `train_baseline(datasets, support, config)`
  - 纯数据驱动版本。
- `130`
  - 初始化模型。
- `131`
  - 初始化优化器。
- `132`
  - 为本地训练数据建立 DataLoader。

- `134-136`
  - 保存最佳状态和训练历史。

- `138-163`
  - 训练循环：
    - `139` 进入 train
    - `140-141` 初始化累计量
    - `143-153` 遍历训练 batch：
      - 前向
      - MSE 损失
      - 反向
      - 更新
    - `155`
      - 在全域验证集上评估
    - `156-157`
      - 记录训练损失和全域验证 RMSE
    - `159-161`
      - 保存验证集表现最好的模型

- `163-164`
  - 恢复最佳模型并返回。

### 6.9 knowledge-guided 训练

- `167`: `train_knowledge_guided(...)`
  - 知识增强版训练。
- `168-170`
  - 初始化模型、优化器和本地训练 loader。
- `172-175`
  - 把 support 数据提前搬到 device 上。

- `177-179`
  - 初始化最佳模型和 history。

- `181-225`
  - 训练主循环。

#### 每个 batch

- `188-190`
  - 拿一批本地训练数据。
- `191`
  - 清梯度。

- `193`
  - 本地数据前向。
- `194`
  - 数据项损失 `L1 = MSE`。

- `196`
  - 在所有 support 点上做前向。
- `197-202`
  - 知识项损失 `L2 = knowledge_distance(...).mean()`

- `204`
  - 总损失：
  - `lambda_data * data_loss + (1 - lambda_data) * knowledge_loss`

- `205-206`
  - 反向传播和参数更新。

- `208-212`
  - 累计 total/data/knowledge 三类损失。

#### 每轮末尾

- `214`
  - 在全域验证集上评估。
- `215-218`
  - 记录 total/data/knowledge loss 和 val rmse。
- `220-222`
  - 更新最佳模型。

- `224-225`
  - 恢复最佳模型。

### 6.10 保存指标

- `228-230`
  - 把 metrics 写进 JSON。

### 6.11 训练曲线图

- `233-252`
  - 左图：loss
    - baseline
    - kd total
    - kd data
    - kd knowledge
  - 右图：验证 RMSE

### 6.12 画 landmarks

- `255`: `_draw_landmarks(ax, landmarks)`
  - 在图上把每个 landmark 画成一个矩形。
- `257-258`
  - 计算矩形宽高。
- `259-268`
  - 构造矩形：
    - 如果 quality 里包含 `good` 或 `coarse`，就用橙色
    - 否则用红色
    - 边框虚线表示“知识约束区间”
- `269`
  - 把矩形加到图上。

### 6.13 预测曲线图

- `272`: `plot_prediction_curves(...)`
  - 画 baseline 和 knowledge-guided 的全域预测对照图。
- `273-274`
  - 取测试网格和本地训练点。
- `276-279`
  - 分别算 baseline 和 kd 的全域预测。

- `281-282`
  - 创建两个共享坐标轴的子图。

- `284-292`
  - 对每个模型画：
    - 真实函数
    - 预测曲线
    - 本地训练点
    - 局部窗口灰色阴影
    - 所有 landmarks 矩形

- `294-296`
  - 加 y 标签并保存图像。

这个文件的关键思想可以概括为：

> baseline 只有 `L1`，knowledge-guided 有 `L1 + L2`，而 `L2` 来自 landmarks 支持点上的区间惩罚。

---

## 7. [experiment.py](./experiment.py)

这个文件负责组织整个实验流程。

### 7.1 导入

- `1`
  - 导入数据生成和种子设置。
- `2`
  - 导入 landmarks 加载、support 采样和序列化。
- `3-10`
  - 导入训练、评估、画图、落盘函数。

### 7.2 总实验入口

- `13`: `run_experiment(config, save_artifacts=True, save_plots=True)`
  - 整个 toy 的统一实验入口。

- `14`
  - 确保输出目录存在。
- `15`
  - 固定随机种子。
- `16`
  - 生成数据集。
- `17`
  - 读取当前 landmark set。
- `18-22`
  - 从 landmarks 中采 support 点。

- `24`
  - 训练 baseline。
- `25`
  - 训练 knowledge-guided。

- `27-28`
  - 用测试集分别评估两者。

- `30-43`
  - 组装 metrics：
    - 配置
    - landmarks 列表
    - baseline 指标
    - knowledge-guided 指标
    - 全域 RMSE 改善量
    - 窗口外 RMSE 改善量
    - landmark violation 改善量
    - landmark penalty 改善量

- `45-60`
  - 如果需要落盘：
    - 保存 `metrics.json`
    - 画预测曲线图
    - 画训练曲线图

- `62`
  - 返回 metrics，供 `run.py` 打印。

---

## 8. [run.py](./run.py)

这个文件是单次实验入口。

### 8.1 导入

- `1-2`
  - 命令行解析和路径工具。
- `4`
  - 配置类。
- `5`
  - 总实验入口。
- `6`
  - 可用 landmark set 列表。

### 8.2 参数解析

- `9`: `parse_args()`
  - 定义命令行参数。
- `10`
  - 创建 `ArgumentParser`。
- `11`
  - `--seed`
- `12`
  - `--epochs`
- `13`
  - `--num-train-local`
- `14`
  - `--lambda-data`
- `15`
  - `--center-pull-weight`
- `16`
  - `--landmark-set`
  - `choices=available_landmark_sets()` 限制输入合法。
- `17`
  - `--label-noise-std`
- `18`
  - `--experiment-name`
- `19`
  - `--skip-plots`
- `20`
  - 返回参数对象。

### 8.3 参数构造成配置

- `23`: `build_config(args)`
  - 把命令行参数转成 `ExperimentConfig`。
- `24-34`
  - 逐项把参数写入配置。
- `33`
  - 明确指定本次实验的结果目录。
- `35`
  - 创建输出目录。
- `36`
  - 返回配置。

### 8.4 主函数

- `39`: `main()`
  - 脚本主入口。
- `40`
  - 解析参数。
- `41`
  - 生成配置对象。
- `42`
  - 调用 `run_experiment()` 真正运行实验。

- `44`
  - 打印输出目录。
- `45`
  - 打印使用的 landmark set。
- `46`
  - 打印 baseline 全域 RMSE。
- `47`
  - 打印 knowledge-guided 全域 RMSE。
- `48`
  - 打印 RMSE 改善量。

- `51-52`
  - 标准脚本入口。

---

## 9. 这套代码最值得你记住的 5 个点

1. `data.py` 故意制造“训练只在局部窗口、评估看全局”的困难场景。
2. `landmarks.py` 把知识写成输入区间-输出区间对，而不是点标签。
3. `trainer.py` 里的 `knowledge_distance` 就是 toy 版 `L2`。
4. `lambda_data` 决定数据项和知识项谁更强。
5. `good / coarse_good / mixed / shifted_bad` 让你可以系统比较知识质量的影响。

如果你下一步要改这套代码，最推荐优先改：

- `true_function`
- `local_region_low/high`
- `get_landmarks`
- `knowledge_distance`
- `lambda_data` 和 `center_pull_weight`
