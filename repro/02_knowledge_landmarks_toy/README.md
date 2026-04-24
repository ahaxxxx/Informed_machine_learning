# knowledge_landmarks_toy

一个对应 `Informed Machine Learning with Knowledge Landmarks` 的最小可运行 toy reproduction。

这个项目抓住原论文的核心结构：

- 只有局部区域有数值数据
- 全局空间有知识 landmarks
- 模型训练目标由 `data fitting + knowledge regularization` 组成
- 对比 baseline 和 knowledge-guided model 的全局泛化能力

## 这个 toy 在做什么

任务是一维回归。

- 输入：`x`
- 真实函数：一个局部近似线性、全局非线性的连续函数
- 局部观测数据：只来自一个有限窗口
- 知识 landmarks：覆盖整个输入域的若干输入区间-输出区间对

直观上：

- baseline 只能看到局部窗口里的数据
- knowledge-guided model 除了看局部数据，还会被 landmarks 约束
- 目标是让知识版本在全局测试域上更稳

## 当前支持的 landmark sets

- `good`
  - 较准确、较具体的知识 landmarks
- `coarse_good`
  - 方向正确，但输出区间更宽，知识更粗
- `mixed`
  - 一部分 landmarks 正确，一部分偏移
- `shifted_bad`
  - 全局系统性偏移的错误 landmarks

这几组主要用来比较：

- 知识是否有帮助
- 知识越具体是否越有用
- 部分错误知识是否还能容忍
- 全错知识会不会误导模型

## 和原论文的对应关系

这个 toy 保留了原论文最关键的四件事：

1. 本地数据 `D*` 只覆盖局部区域
2. 知识 `K = {(Ai, Bi)}` 覆盖全域
3. 损失由 `lambda * L1 + (1 - lambda) * L2` 构成
4. `L2` 扮演 knowledge regularizer 的角色

简化的地方：

1. 从真实物理问题简化为 1D toy regression
2. 输入 granule 简化为区间
3. 输出 granule 简化为区间
4. regularizer 简化为“预测落在输出区间外的 hinge penalty”

## 目录结构

```text
02_knowledge_landmarks_toy/
  README.md
  notes.md
  config.py
  data.py
  landmarks.py
  model.py
  trainer.py
  experiment.py
  run.py
  results/
```

## 单次实验

默认运行：

```bash
python run.py
```

指定 landmark set：

```bash
python run.py --landmark-set good --experiment-name good_demo
python run.py --landmark-set coarse_good --experiment-name coarse_demo
python run.py --landmark-set mixed --experiment-name mixed_demo
python run.py --landmark-set shifted_bad --experiment-name bad_demo
```

调节数据项权重：

```bash
python run.py --lambda-data 0.7
```

调节知识中心拉回强度：

```bash
python run.py --center-pull-weight 0.08
```

跳过绘图：

```bash
python run.py --skip-plots
```

## 运行后会生成什么

默认会在 `results/<experiment_name>/` 下生成：

- `metrics.json`
- `prediction_curves.png`
- `training_curves.png`

## 推荐先看的对照

1. `good` vs baseline
   - 看知识 landmarks 是否帮助全局泛化
2. `good` vs `coarse_good`
   - 看知识粒度与 regularization 强度如何交互
3. `good` vs `mixed`
   - 看部分错误知识的影响
4. `good` vs `shifted_bad`
   - 看系统性错误知识是否会误导模型
