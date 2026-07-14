"""SECOM 失效分析系统 —— 源码包。

模块按"数据处理阶段"划分，各阶段职责单一、边界清晰：
    config       配置加载（单一事实来源，含全局 random_state）
    data_io      数据加载与标签转换
    preprocessing 预处理管线（删空列 -> 填充 -> 标准化）
    feature_selection 多方法综合特征选择
    modeling     模型查表构建与训练
    evaluation   不平衡场景评估（BER/召回为主）
    explain      可解释性/根因（扩展点，基础版留占位）
    feature_mapping 匿名特征 -> 业务含义映射层（扩展点）

设计原则：低耦合 + 配置驱动。加新功能时改动局部模块，不牵连主流程。
"""
