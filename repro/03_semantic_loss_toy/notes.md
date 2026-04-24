# semantic_loss_toy 笔记

## 1. 这份 toy 想帮你真正看清什么？

最核心的是这句：

> 逻辑知识不一定要先变成 teacher，也不一定只是人工写一个启发式 penalty，它可以直接变成“输出分布满足约束的概率质量”。

所以这条路线和 `logic_net_toy` 的区别是：
- `logic_net_toy` 更像规则蒸馏
- `semantic_loss_toy` 更像直接约束输出空间

## 2. 这里为什么用 4 个独立 sigmoid，而不是 softmax？

因为 `softmax` 天生就会把概率和归一到 1。
如果你再去演示 one-hot / exactly-one 约束，很多现象会变得不明显。

这里故意把输出写成 4 个独立的二值变量，是为了保留 semantic loss 最原始的味道：
- 每个输出位都有自己的边际概率
- 合法赋值是若干二进制向量
- semantic loss 就是这些合法向量概率的总和

## 3. 这个 toy 里的 semantic loss 具体是什么？

设模型输出 4 个 logit，对应 4 个二值变量。
对每个变量，用 `sigmoid` 得到它取 1 的概率。

如果约束是 `exactly_one`，那么合法赋值只有 4 个：
- `1000`
- `0100`
- `0010`
- `0001`

semantic loss 做的事就是：
1. 计算这 4 个合法赋值各自的概率
2. 把它们加起来，得到“满足约束的总概率质量”
3. 取负对数，作为损失

所以：
- 如果模型把概率放在合法 one-hot 结构上，loss 会小
- 如果模型经常输出多位同时激活或全部不激活，loss 会大

## 4. 这个 toy 为什么适合作为第三个实现？

因为你前面已经有：
1. `logic_net_toy`
2. `knowledge_landmarks_toy`

接下来补 `semantic_loss_toy`，正好可以把主线推进到：
- 规则蒸馏
- 结构化输出约束
- 后面再过渡到 `DL2`

## 5. 后面最自然的扩展方向

1. 加入 sweep
   - 扫 `lambda_semantic`
   - 扫无标签比例
   - 扫错误约束比例

2. 加更复杂的约束
   - `mutual exclusion`
   - `implication`
   - 组间约束

3. 加代码讲解文档
   - 和前两个 toy 保持一致，写一份逐行解析版
