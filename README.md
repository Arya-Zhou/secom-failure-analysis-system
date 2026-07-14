# SECOM 失效分析系统（重构版）

> 将原 `secom.ipynb` 的探索性分析重构为模块化、可配置、可复现的工程骨架。
> 面向"失效分析 / 良率提升工程师"岗位的求职展示用。重构日期：2026-07-10。

## 这是什么

基于 SECOM 半导体制造过程数据（1567 片晶圆 × 591 维匿名传感器特征，失败率 6.64%），
识别潜在低良率晶圆、定位关键工艺参数候选、以 BER/召回为核心控制漏检风险。

原 notebook 负责"一眼看结果"的可视化展示；本目录负责"可测试、可运行、可扩展"的系统逻辑。

## 目录结构

```
secom_refactor_20260710/
├── config.yaml            # 唯一配置源（参数/阈值/开关/全局 random_state）
├── .env.example           # 密钥模板（复制为 .env 填真值，.env 不入库）
├── .gitignore             # 忽略数据/生成物/密钥
├── requirements.txt       # 锁版本依赖
├── main.py                # 统一入口（支持 --quick）
├── baseline_metrics.json  # notebook 基线指标（回归比对用）
├── feature_map.yaml       # 匿名特征->业务含义映射（扩展点，现为空）
├── src/
│   ├── config.py          # 配置 + 密钥加载
│   ├── data_io.py         # 数据加载与标签转换
│   ├── preprocessing.py   # 删空列->填充->标准化（防泄漏管线）
│   ├── feature_selection.py  # 四方法综合排名（待搬运）
│   ├── modeling.py        # 配置驱动的模型查表（扩展点）
│   ├── evaluation.py      # BER/召回等不平衡指标
│   ├── explain.py         # 可解释性/根因（扩展点占位）
│   └── feature_mapping.py # 匿名特征映射层
└── tests/
    └── test_reproducibility.py  # 带容差的回归测试
```

## 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 配置密钥（基础版可跳过，无外部 API 调用）
cp .env.example .env

# 3. 准备数据：从 UCI 下载 SECOM，把 secom.data / secom_labels.data
#    放到本目录上一级（即 Secom/ 根目录），路径见 config.yaml
#    https://archive.ics.uci.edu/dataset/179/secom

# 4. 运行
python main.py            # 完整流程
python main.py --quick    # 小样本、跳过耗时步骤（调试/演示用）

# 5. 回归测试
pytest tests/
```

## 设计要点（面试可讲）

- **配置分离**：所有会改变结果的数字集中在 `config.yaml`；密钥走 `.env` 不入库。
- **防数据泄漏**：预处理用 sklearn Pipeline，只在训练集 fit、测试集 transform。
- **可复现**：全局单一 `random_state`（config），配合回归测试容差兜住数值抖动。
- **配置驱动的扩展点**：加新模型/异常检测只在 `modeling.py` 注册一行、config 换名字，主流程不动——预留干净扩展点而非空接口。
- **匿名数据的应对**：`feature_mapping` 层现为空，接真实产线数据只需填映射表，流程复用。

## 数据特点与对应处理

| 数据特点 | 处理位置 | 做法 |
| --- | --- | --- |
| 高维稀疏 + 大量缺失 | `preprocessing` | 删全空列 + 中位数填充 + 标准化 |
| 类别极度不平衡(6.64%) | 横切 config/model/eval | 分层抽样 + class_weight + BER/召回主指标 |
| 特征匿名 | `feature_mapping` | 预留映射层，输出表述为"候选"而非"根因" |

## 当前状态

骨架已就绪，`feature_selection` 等逻辑待从 `secom.ipynb` 逐模块搬入。
搬运时以 `baseline_metrics.json` 为准做回归校验（容差 0.01）。

## 后续扩展（面试反馈后按需）

见上级目录 `SECOM_项目增强优化与包装建议.md` 第 5 章的六个增强方向：
可解释性报告、两级异常检测、漏检风险控制、根因候选链、制程漂移监测、失效模式分群。
