# semantic_loss_toy

一个对应 `A Semantic Loss Function for Deep Learning with Symbolic Knowledge` 的最小可运行 toy reproduction。

这版不是空目录，而是已经把最核心的结构落成了一个能跑的小实验：
- 任务是二维输入、4 类输出的 toy 分类
- 模型输出不是 `softmax`，而是 4 个独立 `sigmoid` 概率
- 监督部分用 one-hot 标签的 `BCEWithLogits`
- 无标签部分加 semantic loss
- 对照 `baseline` 和 `semantic-guided` 两个版本

## 这个 toy 在做什么？

论文里的关键思想不是“再加一个普通惩罚项”，而是：
> 直接计算模型输出分布落在“满足逻辑约束的赋值集合”上的概率质量。

在这个 toy 里：
- 每个样本有 4 个输出位
- 合法输出应该是 one-hot 结构
- 因此默认约束是 `exactly_one`
- semantic loss 会鼓励模型把概率质量放到合法 one-hot 赋值上

之所以不用 `softmax`，是因为如果直接用 `softmax`，`exactly-one` 约束会变得过于平凡，难以演示 semantic loss 真正在约束什么。

## 当前支持的约束集

- `exactly_one`
  - 正确约束：4 个输出位中恰好 1 个为真
- `at_least_one`
  - 较弱但仍合理的约束：至少 1 个输出位为真
- `exactly_two_bad`
  - 故意错误的约束：恰好 2 个输出位为真

这三组主要用于看：
- 正确约束是否能提高约束满足率，甚至改善泛化
- 弱约束和强约束的效果差异
- 错误约束会不会把模型往错误方向推

## 和论文的对应关系

这个 toy 保留了论文里最重要的几件事：
1. 输出是由多个二值变量组成的结构化对象
2. 约束不是作用在硬标签上，而是作用在模型输出分布上
3. semantic loss 度量的是“满足约束的总概率质量”
4. 无标签数据也可以参与训练

简化掉的地方：
1. 从论文里更一般的符号逻辑公式，缩到一个最容易看懂的 one-hot 约束
2. 从真实任务缩到二维 toy 分类
3. 从复杂结构输出缩到 4 个输出位

## 目录结构

```text
03_semantic_loss_toy/
  README.md
  notes.md
  config.py
  data.py
  constraints.py
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

指定实验名：
```bash
python run.py --experiment-name default_demo
```

切换约束集：
```bash
python run.py --constraint-set exactly_one --experiment-name good_demo
python run.py --constraint-set at_least_one --experiment-name weak_demo
python run.py --constraint-set exactly_two_bad --experiment-name bad_demo
```

调节 semantic loss 权重：
```bash
python run.py --lambda-semantic 0.6
```

跳过画图：
```bash
python run.py --epochs 40 --skip-plots
```

## 运行后会生成什么？

默认会在 `results/<experiment_name>/` 下生成：
- `metrics.json`
- `decision_and_constraint_maps.png`
- `training_curves.png`

## 最值得先看的对照

1. `exactly_one` vs baseline
   - 看正确语义约束是否提升约束满足率，以及是否改善分类
2. `exactly_one` vs `at_least_one`
   - 看弱约束和强约束的差别
3. `exactly_one` vs `exactly_two_bad`
   - 看错误知识如何误导模型
