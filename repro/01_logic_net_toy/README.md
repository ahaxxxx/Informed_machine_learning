# logic_net_toy

一个对应 `Harnessing Deep Neural Networks with Logic Rules` 的扩展 toy reproduction。

当前版本已经不只是最小单次实验，而是支持：

- 单规则和多规则
- 正确规则、错误规则、混合规则对照
- 参数扫描
- 统一保存单次实验和 sweep 结果

## 这个 toy 在做什么

任务是二维二分类。

- 输入：二维点 `(x1, x2)`
- 标签：由一个非线性真实边界生成
- 模型：一个小型 MLP
- 规则：通过软规则构造 rule-aware teacher，再蒸馏给 student

## 当前支持的规则集

- `single_good`
  - `x1 > x2 => class 1`
- `single_bad`
  - `x1 > x2 => class 0`
- `multi_good`
  - `x1 > x2 => class 1`
  - `x1 > 0.15 => class 1`
  - `x2 < 0.10 => class 1`
- `multi_mixed`
  - 两条相对合理的规则
  - 一条故意错误的规则
- `multi_bad`
  - 多条系统性错误规则

这几组规则分别用于：

- 看单条好规则是否有帮助
- 看错误规则会不会拖后腿
- 看多条规则是否能进一步提升
- 看 mixed rules 下模型是否还有一定鲁棒性
- 看全错规则集会如何影响 teacher

## 和原论文的对应关系

这个 toy 保留了原论文最核心的结构：

1. `student -> teacher -> student`
2. teacher 由当前 student 输出和规则联合构造
3. 训练目标同时包含标签监督和 teacher 模仿

简化部分：

1. 任务从文本改成二维分类
2. 规则从一阶逻辑简化成可解释的软规则集合
3. teacher 从完整 posterior regularization 简化成“student 概率 + rule 概率”的对数加权融合

## 目录结构

```text
01_logic_net_toy/
  README.md
  notes.md
  config.py
  data.py
  experiment.py
  model.py
  rules.py
  sweep.py
  trainer.py
  run.py
  results/
```

## 单次实验

默认运行：

```bash
python run.py
```

带规则集和实验名：

```bash
python run.py --rule-set multi_good --experiment-name multi_good_demo
```

错误规则对照：

```bash
python run.py --rule-set single_bad --experiment-name wrong_rule_demo
```

跳过画图：

```bash
python run.py --epochs 80 --rule-set multi_mixed --skip-plots
```

## 参数扫描

最常用的扫描命令：

```bash
python sweep.py --name quick_scan --epochs 40
```

自定义扫描：

```bash
python sweep.py ^
  --name labeled_vs_rules ^
  --epochs 50 ^
  --seeds 7,13 ^
  --num-labeled-values 32,64,128 ^
  --rule-strengths 0.5,1.0,1.5 ^
  --distill-weights 0.35,0.65 ^
  --rule-sets single_good,single_bad,multi_good,multi_mixed,multi_bad
```

扫描结果会输出到：

- `results/sweeps/<name>/summary.json`
- `results/sweeps/<name>/summary.csv`
- `results/sweeps/<name>/best_run.json`

每个组合的单独 `metrics.json` 也会保留。

## 运行后会产生什么

单次实验默认会在 `results/<experiment_name>/` 下生成：

- `metrics.json`
- `decision_boundaries.png`
- `training_curves.png`

## 建议你先看的对照

1. `single_good` vs `single_bad`
   - 看规则方向是否真的重要
2. `single_good` vs `multi_good`
   - 看多规则是否比单规则更稳定
3. `multi_good` vs `multi_mixed`
   - 看加入错误规则后会不会被拖累
4. `multi_good` vs `multi_bad`
   - 看系统性错误规则会不会把 teacher 带偏
5. 小标注量 vs 大标注量
   - 看规则是不是在 low-label 场景更有帮助
