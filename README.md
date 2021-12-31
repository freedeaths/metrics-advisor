# metrics-advisor
Analyze reshaped metrics from TiDB cluster Prometheus and give some advice about anomalies and correlation.

## Team
- [freedeaths](https://github.com/freedeaths)
- [mashenjun](https://github.com/mashenjun)
- [CescWang1991](https://github.com/CescWang1991)
- [Azure-blog](https://github.com/Azure-blog)

## 背景
当用户利用 metrics 去定位性能问题或这分析集群行为时，如下问题会影响定位和分析的效率。
1. 需要有很充分的背景知识，才能理解 metrics 本身的现实含义以及 metrics 之间的关联关系，进而利用其反映的信息来帮助定位性能问题和理解集群行为。
2. 有了背景知识并理解 metrics 的现实含义后。用户按照自己的思路去观察 metrics 或者对比不同的 metrics 时，在 UI 界面上也还会涉及到很多人工的操作，导致效率不高。

## 项目介绍
metrics 本身都属于时序数据。时序数据本身有一些共有的性质和通用的分析手段，这些分析手段本身不依赖任何背景知识。项目的目标是通过时序数据的分析手段，在不依赖任何背景知识的前提下帮助用户分析出 metrics 中表现出的特性。
我们选择 TiDB Read and Write Performance 面板涉及的 metrics 做为分析的数据，最终回答用户如下两个问题：

1. 其中是否有异常的 metrics？
2. 哪些 metrics 是与异常 metrics 相关的？

## 项目设计
TODO

# 项目进度
- [ ] 准备 workload 和 metrics data
- [ ] 异常点检测
- [ ] change point 检测 
- [ ] 相似性计算
- [ ] 结果可视化

