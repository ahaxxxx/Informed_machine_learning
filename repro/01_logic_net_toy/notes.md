# Notes

## 当前版本

- 单规则与多规则都可配置
- 支持错误规则和 mixed rules 对照
- 增加了 sweep 脚本
- 单次实验与批量实验都能落盘

## 下一步建议

1. 增加更系统的 seed 平均与方差统计
2. 扫 `rule_set x labeled_budget x distill_weight` 后画汇总图
3. 加一个“教师规则只用 unlabeled / 同时用 labeled+unlabeled”的对照
4. 尝试把 teacher 构造改得更接近 posterior regularization
