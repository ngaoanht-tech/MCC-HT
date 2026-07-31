#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义异常类，用于区分不同类型的错误。

这些异常帮助快速定位问题根源，避免宽泛的 Exception 捕获掩盖真实错误。
"""


class ConfigurationError(Exception):
    """配置文件缺失、格式错误或缺少必需字段。
    
    Examples:
        - config.yaml 文件不存在
        - 配置中缺少 constraints.initial_storage
        - 配置值类型不匹配（如期望数值但给了字符串）
    """
    pass


class DataFileError(Exception):
    """数据文件缺失、格式错误或内容不完整。
    
    Examples:
        - train/2020.csv 文件不存在
        - CSV 列名不符合预期（缺少"入库流量"等关键列）
        - 数据时段数量不足（期望36个时段，实际只有20个）
    """
    pass


class ConstraintError(Exception):
    """约束文件缺失或约束数值不合法。
    
    Examples:
        - shuxing/约束条件.csv 文件不存在
        - q_min > q_max（最小流量大于最大流量）
        - 水位曲线文件缺失
    """
    pass


class ModelArchitectureError(Exception):
    """模型结构配置错误或初始化失败。
    
    Examples:
        - input_dim 与实际特征维度不匹配
        - Transformer nhead 不能被 d_model 整除
        - 模型权重文件损坏
    """
    pass
