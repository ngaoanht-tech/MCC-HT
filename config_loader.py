"""
配置加载器模块
用于加载和管理YAML配置文件
"""

import yaml
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = self._find_config_path(config_path)
        self._last_modified = None
        self.config = self._load_config()
    
    def _find_config_path(self, config_path: str) -> str:
        """
        智能查找配置文件路径
        
        Args:
            config_path: 原始配置文件路径
            
        Returns:
            实际的配置文件路径
        """
        # 如果是绝对路径且存在，直接返回
        if os.path.isabs(config_path) and os.path.exists(config_path):
            return config_path
        
        # 如果当前目录存在配置文件，直接返回
        if os.path.exists(config_path):
            return config_path
        
        # 尝试在脚本所在目录查找
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_config_path = os.path.join(script_dir, config_path)
        if os.path.exists(script_config_path):
            return script_config_path
        
        # 尝试在transformer单模型目录查找
        transformer_dir = os.path.join(os.path.dirname(script_dir), "transformer单模型")
        if os.path.exists(transformer_dir):
            transformer_config_path = os.path.join(transformer_dir, config_path)
            if os.path.exists(transformer_config_path):
                return transformer_config_path
        
        # 尝试向上查找包含config.yaml的目录
        current_dir = os.getcwd()
        while current_dir != os.path.dirname(current_dir):  # 直到根目录
            potential_path = os.path.join(current_dir, config_path)
            if os.path.exists(potential_path):
                return potential_path
            
            # 检查是否有transformer单模型子目录
            transformer_subdir = os.path.join(current_dir, "transformer单模型", config_path)
            if os.path.exists(transformer_subdir):
                return transformer_subdir
                
            current_dir = os.path.dirname(current_dir)
        
        # 如果都找不到，返回原始路径（会在_load_config中报错）
        return config_path
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        # 记录文件修改时间
        self._last_modified = os.path.getmtime(self.config_path)
        
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
        
        for encoding in encodings:
            try:
                with open(self.config_path, 'r', encoding=encoding) as f:
                    config = yaml.safe_load(f)
                    if config is None:
                        config = {}
                    return config
            except UnicodeDecodeError:
                continue
            except yaml.YAMLError as e:
                raise ValueError(f"配置文件 YAML 解析失败 ({encoding}): {e}") from e
            except Exception as e:
                raise RuntimeError(f"配置加载出错 ({encoding}): {e}") from e
                
        raise ValueError(f"无法解码配置文件 {self.config_path}")
    
    def _check_and_reload(self) -> None:
        """检查文件是否被修改，如果是则重新加载"""
        if not os.path.exists(self.config_path):
            return
            
        current_modified = os.path.getmtime(self.config_path)
        if current_modified != self._last_modified:
            print(f"检测到配置文件变化，自动重新加载: {self.config_path}")
            self.config = self._load_config()
    
    def __len__(self) -> int:
        """返回配置项的数量"""
        return len(self.config) if self.config else 0
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套键路径
        
        Args:
            key_path: 配置键路径，如 "model.input_dim"
            default: 默认值
            
        Returns:
            配置值
        """
        # 自动检查并重新加载配置
        self._check_and_reload()
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    
    
    
    
    
    
    
    
    
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.get('training', {})
    
    def get_loss_config(self) -> Dict[str, Any]:
        """获取损失函数配置"""
        return self.get('loss', {})
    
    def get_constraints_config(self) -> Dict[str, Any]:
        """获取约束配置"""
        return self.get('constraints', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return self.get('evaluation', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置"""
        return self.get('output', {})
    
    def get_reservoir_config(self) -> Dict[str, Any]:
        """获取水库配置"""
        return self.get('reservoirs', {})
    
    def get_compute_config(self) -> Dict[str, Any]:
        """获取计算资源配置"""
        return self.get('compute', {})
    
    def get_hyperparameter_search_space(self, quick_search: bool = False) -> Dict[str, List]:
        """
        获取超参数搜索空间
        
        Args:
            quick_search: 是否使用快速搜索
            
        Returns:
            超参数搜索空间
        """
        search_type = 'quick_search' if quick_search else 'full_search'
        return self.get(f'training.hyperparameters.{search_type}', {})
    
    def get_data_years(self) -> Dict[str, List[int]]:
        """获取数据年份配置"""
        return self.get('data.years', {})
    
    def get_train_years(self) -> List[int]:
        """获取训练年份"""
        return self.get('data.years.train', [1959, 1960, 1961, 1962, 1963])
    
    def get_val_years(self) -> List[int]:
        """获取验证年份"""
        return self.get('data.years.validation', [1964, 1965])
    
    def get_test_years(self) -> List[int]:
        """获取测试年份"""
        return self.get('data.years.test', [1966])
    
    def get_reservoir_names(self) -> List[str]:
        """获取水库名称列表"""
        return self.get('reservoirs.names', ["乌东德", "白鹤滩", "溪洛渡", "向家坝", "三峡", "葛洲坝"])

    def get_initial_levels(self) -> List[float]:
        """获取各水库的初始水位配置"""
        levels = self.get('constraints.initial_levels')
        if levels is None:
            raise KeyError("config.yaml 缺少 constraints.initial_levels")
        return [float(v) for v in levels]

    def get_target_levels(self) -> List[float]:
        """获取各水库的末水位配置"""
        levels = self.get('constraints.target_levels')
        if levels is None:
            raise KeyError("config.yaml 缺少 constraints.target_levels")
        return [float(v) for v in levels]

    def get_results_dir(self) -> str:
        """获取结果目录"""
        return self.get('data.results_dir', 'results')
    
    def get_train_data_dir(self) -> str:
        """获取训练数据目录"""
        return self.get('data.train_data_dir', 'train')
    
    def get_constraint_weights(self) -> Dict[str, float]:
        """获取约束权重"""
        return self.get('loss.constraint_weights', {
            'boundary': 1.0,
            'ecological': 0.8,
            'ramp_rate': 0.6,
            'cascade_consistency': 0.9
        })
    
    def get_model_params(self, model_type: str = 'transformer') -> Dict[str, Any]:
        """
        获取模型参数
        
        Args:
            model_type: 模型类型 ('transformer', 'hierarchical', 'seq_transformer')
            
        Returns:
            模型参数字典
        """
        base_params = {
            'input_dim': self.get('model.input_dim', 22),
            'output_dim': self.get('model.output_dim', 6)
        }
        
        if model_type == 'transformer':
            transformer_params = self.get('model.transformer', {})
            base_params.update(transformer_params)
        elif model_type == 'hierarchical':
            hierarchical_params = self.get('model.hierarchical', {})
            base_params.update(hierarchical_params)
        elif model_type == 'seq_transformer':
            seq_transformer_params = self.get('model.seq_transformer', {})
            # 为seq_transformer设置默认参数
            default_seq_params = {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'sequence_length': 168,
                'output_sequence_length': 24
            }
            default_seq_params.update(seq_transformer_params)
            base_params.update(default_seq_params)
        
        return base_params
    
    def get_training_params(self) -> Dict[str, Any]:
        """获取训练参数"""
        return {
            'max_epochs': self.get('training.max_epochs', 100),
            'patience': self.get('training.patience', 15),
            'early_stopping_patience': self.get('training.early_stopping_patience', 8),
            'gradient_clipping_max_norm': self.get('training.gradient_clipping.max_norm', 1.0)
        }
    
    def get_optimizer_params(self) -> Dict[str, Any]:
        """获取优化器参数"""
        return {
            'type': self.get('training.optimizer.type', 'AdamW'),
            'weight_decay': self.get('training.optimizer.weight_decay', 1e-4)
        }
    
    def get_scheduler_params(self) -> Dict[str, Any]:
        """获取学习率调度器参数"""
        return {
            'type': self.get('training.scheduler.type', 'ReduceLROnPlateau'),
            'mode': self.get('training.scheduler.mode', 'min'),
            'factor': self.get('training.scheduler.factor', 0.5),
            'patience': self.get('training.scheduler.patience', 8)
        }
    
    def get_loss_params(self) -> Dict[str, Any]:
        """获取损失函数参数"""
        return {
            'type': self.get('loss.base_loss.type', 'WeightedMSELoss'),
            'high_flow_threshold_percentile': self.get('loss.base_loss.high_flow_threshold_percentile', 75),
            'high_flow_weight': self.get('loss.base_loss.high_flow_weight', 3.0)
        }
    
    def get_power_optimization_config(self) -> Dict[str, Any]:
        """获取发电量优化配置"""
        return {
            'enable': self.get('loss.power_optimization.enable', False),
            'power_weight': self.get('loss.power_optimization.power_weight', 0.1),
            'flow_weight': self.get('loss.power_optimization.flow_weight', 1.0),
            'mode': self.get('loss.power_optimization.mode', 'maximize')
        }
    
    def get_output_filenames(self, model_type: str) -> Dict[str, str]:
        """
        获取输出文件名
        
        Args:
            model_type: 模型类型
            
        Returns:
            文件名字典
        """
        return {
            'model': self.get(f'output.model_files.{model_type}', f'{model_type}_best_model.pth'),
            'hyperparameters': self.get(f'output.hyperparameter_files.{model_type}', f'{model_type}_hyperparameters.json'),
            'prediction_comparison': f'{model_type}{self.get("output.result_files.prediction_comparison", "_prediction_comparison.csv")}',
            'training_results': f'{model_type}{self.get("output.result_files.training_results", "_training_results.png")}',
            'evaluation_report': f'{model_type}{self.get("output.result_files.evaluation_report", "_evaluation_report.json")}'
        }
    
    def update_config(self, key_path: str, value: Any) -> None:
        """
        更新配置值
        
        Args:
            key_path: 配置键路径
            value: 新值
        """
        keys = key_path.split('.')
        config = self.config
        
        # 导航到最后一级的父级
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置最终值
        config[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径，默认为原配置文件路径
        """
        save_path = output_path or self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)


# 全局配置实例
_global_config = None


def get_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    获取最新配置实例。

    为避免返回缓存的旧对象，每次调用都会重新加载配置。
    """
    return reload_config(config_path)


def reload_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    重新加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        新的配置加载器实例
    """
    global _global_config
    _global_config = ConfigLoader(config_path)
    return _global_config


# ---------------------------------------------------------------------------
# Lightweight helpers used across training/inference/generation
# Avoid re-implementing the same config probing logic in many modules.
# ---------------------------------------------------------------------------
def _ensure_loader(cfg: Optional["ConfigLoader"]) -> "ConfigLoader":
    return cfg if isinstance(cfg, ConfigLoader) else get_config()


def cfg_lookup(key: str, default: Any = None, cfg: Optional["ConfigLoader"] = None) -> Any:
    """Lookup a key with multiple common namespaces.

    Resolution order:
      1) generation.{key}
      2) {key}
      3) inference.{key}
      4) schedule_generation.{key}
    """
    c = _ensure_loader(cfg)
    for prefix in ("generation.", "", "inference.", "schedule_generation."):
        path = f"{prefix}{key}" if prefix else key
        val = c.get(path, None)
        if val is not None:
            return val
    return default


def cfg_lookup_bool(key: str, default: bool = False, cfg: Optional["ConfigLoader"] = None) -> bool:
    val = cfg_lookup(key, default, cfg)
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return bool(val)


def parse_window(value: Any, fallback: int, total: int) -> int:
    """Parse a window length supporting keywords like 'full'/'all'."""
    if value is None:
        return fallback
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"full", "all"}:
            return total
        if val.isdigit():
            return int(val)
        try:
            return int(float(val))
        except ValueError:
            return fallback
    if isinstance(value, (int, float)):
        return max(1, min(total, int(value)))
    return fallback


if __name__ == "__main__":
    # 测试配置加载器
    try:
        config = ConfigLoader("config.yaml")
        
        print("=== 配置加载测试 ===")
        print(f"输入维度: {config.get('model.input_dim')}")
        print(f"输出维度: {config.get('model.output_dim')}")
        print(f"训练年份: {config.get_train_years()}")
        print(f"水库名称: {config.get_reservoir_names()}")
        print(f"约束权重: {config.get_constraint_weights()}")
        print(f"发电量优化配置: {config.get_power_optimization_config()}")
        
        print("\n=== 超参数搜索空间 ===")
        quick_space = config.get_hyperparameter_search_space(quick_search=True)
        print(f"快速搜索: {quick_space}")
        
        full_space = config.get_hyperparameter_search_space(quick_search=False)
        print(f"完整搜索: {full_space}")
        
        print("\n=== 模型参数 ===")
        transformer_params = config.get_model_params('transformer')
        print(f"Transformer参数: {transformer_params}")
        
        hierarchical_params = config.get_model_params('hierarchical')
        print(f"分层模型参数: {hierarchical_params}")
        
        print("\n配置加载成功！")
        
    except Exception as e:
        print(f"配置加载失败: {e}")


