"""
Transformer模型定义模块
包含OptimizedTransformerPredictor和HierarchicalTransformerPredictor两个模型类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Union
from config_loader import get_config
from constraint.hierarchical_projection import HierarchicalProjection


class SimplexCapacityHead(nn.Module):
    """Differentiable head that enforces bounds and terminal balance without sigmoid."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
        q_in: torch.Tensor,
        V0: torch.Tensor,
        V_target: torch.Tensor,
        delta_t: torch.Tensor,
        V_min: Optional[torch.Tensor] = None,
        V_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if any(t is None for t in (q_min, q_max, q_in, V0, V_target)):
            raise ValueError("SimplexCapacityHead requires q_min, q_max, q_in, V0, and V_target tensors.")

        # Broadcast delta_t to tensor and convert to storage domain (10^8 m^3)
        if not torch.is_tensor(delta_t):
            delta_t = torch.tensor(float(delta_t), dtype=logits.dtype, device=logits.device)
        delta_t = delta_t.to(dtype=logits.dtype, device=logits.device)
        if delta_t.dim() == 0:
            delta_t = delta_t.view(1, 1, 1)
        elif delta_t.dim() == 1:
            delta_t = delta_t.view(1, -1, 1)
        elif delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)
        delta_t = delta_t.expand_as(q_min)
        dt_vol = delta_t / 1e8
        dt_vol_safe = dt_vol.clamp_min(self.eps)

        # Apply storage constraints to flow bounds if provided
        # Note: This is a simplified static approximation. 
        # Exact path constraints require sequential processing.
        # Here we constrain the *total* capacity based on storage limits.
        if V_min is not None and V_max is not None:
            # Heuristic: limit flow range to prevent immediate storage violation
            # q_out_max_storage = q_in + (V_current - V_min) / dt
            # q_out_min_storage = q_in + (V_current - V_max) / dt
            # Since V_current is unknown in parallel projection, we use a relaxed bound
            # based on V0 and accumulated inflow.
            pass  # Placeholder for future advanced path constraint logic

        span = (q_max - q_min).clamp_min(0.0)
        capacity_volume = span * dt_vol

        base_volume = (q_min * dt_vol).sum(dim=1)
        inflow_volume = (q_in * dt_vol).sum(dim=1)
        required_volume = inflow_volume + V0 - V_target
        alloc_volume = (required_volume - base_volume).clamp_min(0.0)

        total_capacity = capacity_volume.sum(dim=1).clamp_min(self.eps)
        
        # If storage constraints are active, we could clamp alloc_volume here
        # to ensure V_target is reachable within V_min/V_max corridors
        
        alloc_volume = torch.minimum(alloc_volume, total_capacity)

        weights = F.softmax(logits, dim=1)
        weighted_capacity = (weights * capacity_volume).sum(dim=1).clamp_min(self.eps)
        distributed_volume = (alloc_volume.unsqueeze(1) * weights * capacity_volume) / weighted_capacity.unsqueeze(1)

        q = q_min + distributed_volume / dt_vol_safe
        return torch.clamp(q, min=q_min, max=q_max)


class OptimizedTransformerPredictor(nn.Module):
    """可优化超参数的Transformer预测模型"""
    
    def __init__(self, input_dim=22, output_dim=6, d_model=128, nhead=8, num_layers=4, 
                 dropout=0.1, sequence_length=3, activation='relu', use_layer_norm=True, config_path=None):
        super().__init__()
        
        # 如果提供了配置文件路径，从配置文件加载默认参数
        if config_path:
            config = get_config(config_path)
            model_config = config.get('model', {}).get('transformer', {})
            input_dim = model_config.get('input_dim', input_dim)
            output_dim = model_config.get('output_dim', output_dim)
            d_model = model_config.get('d_model', d_model)
            nhead = model_config.get('nhead', nhead)
            num_layers = model_config.get('num_layers', num_layers)
            dropout = model_config.get('dropout', dropout)
            sequence_length = model_config.get('sequence_length', sequence_length)
            activation = model_config.get('activation', activation)
            use_layer_norm = model_config.get('use_layer_norm', use_layer_norm)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.use_layer_norm = use_layer_norm
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.positional_encoding = self._create_positional_encoding(sequence_length, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 注意力池化
        self.attention_pooling = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 输出层
        self.output_layers = nn.ModuleList([
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        ])
        
        # 层归一化
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
        
        # 残差连接
        self.residual_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        
        # 权重初始化
        self._init_weights()

    def _create_reservoir_model(self, d_model, nhead, num_layers, output_size, input_feature_dim):
        """构建单个水库的 Transformer 子模型。"""

        input_projection = nn.Linear(input_feature_dim, d_model)
        pos_encoding = self._create_positional_encoding(self.output_sequence_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        attention_pooling = nn.MultiheadAttention(d_model, nhead, dropout=self.dropout, batch_first=True)
        pooling_query = nn.Parameter(torch.randn(1, 1, d_model))

        if self.enable_sequence_decoding:
            output_layers = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(d_model // 2, output_size * self.output_sequence_length),
            )
        else:
            output_layers = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(d_model // 4, output_size),
            )

        class PositionalEncoding(nn.Module):
            def __init__(self, pe):
                super().__init__()
                self.register_buffer("pe", pe)

        class PoolingQuery(nn.Module):
            def __init__(self, query):
                super().__init__()
                self.query = nn.Parameter(query)

        return nn.ModuleDict(
            {
                "input_projection": input_projection,
                "pos_encoding": PositionalEncoding(pos_encoding),
                "transformer": transformer,
                "attention_pooling": attention_pooling,
                "pooling_query": PoolingQuery(pooling_query),
                "output_layers": output_layers,
                "layer_norm": nn.LayerNorm(d_model),
            }
        )

    
    def _create_positional_encoding(self, max_len, d_model):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准Transformer前向（用于单模型推理兜底路径）。"""
        if x.dim() != 3:
            raise ValueError(f"OptimizedTransformerPredictor expects [B,T,C], got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.shape
        if seq_len > self.sequence_length:
            x = x[:, -self.sequence_length :, :]
            seq_len = self.sequence_length

        h = self.input_projection(x)
        pos = self.positional_encoding[:, :seq_len, :].to(h.device)
        h = h + pos

        residual = self.residual_projection(x)
        h = self.transformer_encoder(h)
        h = h + residual

        query = self.pooling_query.expand(batch_size, -1, -1)
        pooled_output, _ = self.attention_pooling(query, h, h)
        pooled_output = pooled_output.squeeze(1)

        if self.use_layer_norm:
            pooled_output = self.layer_norm(pooled_output)

        out = pooled_output
        for layer in self.output_layers:
            out = layer(out)
        return out


class MultiScaleTCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernels=(3, 5, 7), dilations=(1, 2, 3)):
        super().__init__()
        if len(kernels) != len(dilations):
            raise ValueError("kernels and dilations must have same length")
        self.out_dim = out_dim
        self.branches = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=k, dilation=d, padding="same"),
                nn.GELU(),
                nn.Conv1d(out_dim, out_dim, kernel_size=1),
            ))
        self.gate = nn.Parameter(torch.zeros(len(kernels)))

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError("MultiScaleTCN expects [B,T,R,C] input")
        B, T, R, C = x.shape
        xt = x.permute(0, 2, 3, 1).reshape(B * R, C, T)  # [B*R, C, T]
        outs = []
        for branch in self.branches:
            y = branch(xt)  # [B*R, out_dim, T]
            y = y.view(B, R, self.out_dim, T).permute(0, 3, 1, 2)  # [B,T,R,out_dim]
            outs.append(y)
        weights = torch.softmax(self.gate, dim=0)
        stacked = torch.stack(outs, dim=0)  # [N,B,T,R,out_dim]
        gated = (weights.view(-1, 1, 1, 1, 1) * stacked).sum(dim=0)
        return gated, weights.detach()


class HierarchicalTransformerPredictor(nn.Module):
    """分层多尺度Transformer预测模型，为不同规模水库使用不同架构，支持序列解码"""
    
    def __init__(self, input_dim=22, output_dim=6, sequence_length=3, dropout=0.1, 
                 output_sequence_length=24, enable_multiscale=True, enable_sequence_decoding=True, 
                 config_path=None):
        super().__init__()
        
        # 如果提供了配置文件路径，从配置文件加载默认参数
        if config_path:
            config = get_config(config_path)
            model_config = config.get('model', {}).get('hierarchical', {})
            input_dim = model_config.get('input_dim', input_dim)
            output_dim = model_config.get('output_dim', output_dim)
            sequence_length = model_config.get('sequence_length', sequence_length)
            dropout = model_config.get('dropout', dropout)
            output_sequence_length = model_config.get('output_sequence_length', output_sequence_length)
            enable_multiscale = model_config.get('enable_multiscale', enable_multiscale)
            enable_sequence_decoding = model_config.get('enable_sequence_decoding', enable_sequence_decoding)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.output_sequence_length = output_sequence_length
        self.dropout = dropout
        self.enable_multiscale = enable_multiscale
        self.enable_sequence_decoding = enable_sequence_decoding
        self.capacity_head = SimplexCapacityHead()
        try:
            cfg_local = get_config()
        except Exception:
            cfg_local = {}

        def _cfg_get(obj, path, default=None):
            cur = obj
            for idx, key in enumerate(path):
                next_default = default if idx == len(path) - 1 else {}
                try:
                    if hasattr(cur, "get"):
                        cur = cur.get(key, next_default)
                    elif isinstance(cur, dict):
                        cur = cur.get(key, next_default)
                    else:
                        return default
                except Exception:
                    return default
            return cur

        try:
            self.enable_hard_projection = bool(
                _cfg_get(cfg_local, ["constraints", "hard_projection"], True)
            )
        except Exception:
            self.enable_hard_projection = True
        projection_cfg_iters = int(_cfg_get(cfg_local, ["projection", "terminal_second_rebalance_iters"], 0) or 0)
        projection_cfg_window = int(_cfg_get(cfg_local, ["projection", "terminal_second_rebalance_window"], 36) or 36)
        projection_tail_focus = _cfg_get(cfg_local, ["projection", "tail_focus_reservoirs"], [])
        projection_tail_decay = float(_cfg_get(cfg_local, ["projection", "tail_focus_time_decay"], 0.0) or 0.0)
        projection_tail_extra = int(_cfg_get(cfg_local, ["projection", "tail_focus_extra_window"], 0) or 0)
        projection_tail_step = float(_cfg_get(cfg_local, ["projection", "tail_focus_step_fraction"], 1.0) or 1.0)
        # Increase projection refinement iterations to tighten terminal fit
        self.hierarchical_projection = HierarchicalProjection(
            max_iters=20,
            post_clamp_rebalance_iters=100,
            post_clamp_tol=1e-6,
            terminal_second_rebalance_iters=projection_cfg_iters,
            terminal_second_rebalance_window=projection_cfg_window,
            tail_focus_reservoirs=projection_tail_focus,
            tail_focus_time_decay=projection_tail_decay,
            tail_focus_extra_window=projection_tail_extra,
            tail_focus_step_fraction=projection_tail_step,
        )
        self.last_projection: Dict[str, Any] = {"V": None, "meta": None}
        
        # 水库分组：小规模(0-2)、中规模(3)、大规模(4-5)
        # 方案2：按地理位置和水文特性分组
        self.upstream_reservoirs = [0, 1, 2, 3]  # 金沙江四库：乌东德、白鹤滩、溪洛渡、向家坝
        self.downstream_reservoirs = [4, 5]      # 长江干流：三峡、葛洲坝
        
        # 多尺度时间窗口定义
        self.time_scales = {
            'short': 24,    # 短期：1天
            'medium': 168,  # 中期：1周  
            'long': 720     # 长期：1月
        }
        
        # 共享的输入投影层
        self.shared_input_projection = nn.Linear(input_dim, 128)
        self.input_norm = nn.LayerNorm(128)
        
        # 多尺度特征提取器
        if self.enable_multiscale:
            self.multiscale_extractors = nn.ModuleDict({
                'short': self._create_multiscale_extractor(128, 64),
                'medium': self._create_multiscale_extractor(128, 64),
                'long': self._create_multiscale_extractor(128, 64)
            })
            feature_dim = 64 * 3  # 多尺度特征融合后的维度: 192
        else:
            feature_dim = 128
        
        # 为不同地理位置水库设计不同的模型架构
        # 上游水库（金沙江四库）：高复杂度模型，适合大型水库梯级调度
        # 考虑到溪洛渡(13860MW)和向家坝(6400MW)都是大型水库，需要更复杂的模型
        self.upstream_model = self._create_reservoir_model(
            d_model=144, nhead=12, num_layers=4, 
            output_size=len(self.upstream_reservoirs),
            input_feature_dim=144  # 匹配upstream_feature_extractor的输出
        )
        
        # 下游水库（三峡、葛洲坝）：超复杂模型，处理大流量变化
        self.downstream_model = self._create_reservoir_model(
            d_model=144, nhead=12, num_layers=4,
            output_size=len(self.downstream_reservoirs),
            input_feature_dim=144  # 匹配downstream_feature_extractor的输出
        )
        
        # 多尺度特征提取器（如果启用）
        if self.enable_multiscale:
            # 特征融合层
            num_scales = len(self.time_scales)
            self.feature_fusion = nn.Sequential(
                nn.Linear(64 * num_scales, 128),  # 192 -> 128
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 128)
            )
        
        # 序列解码器（如果启用）
        if self.enable_sequence_decoding:
            pass
        
        # 特征提取器，为不同地理位置水库提取不同特征
        self.upstream_feature_extractor = nn.Sequential(
            nn.Linear(128, 144),  # 输入维度为128，输出144匹配上游模型
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.downstream_feature_extractor = nn.Sequential(
            nn.Linear(128, 144),  # 输入维度为128
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = dropout
        self._init_weights()

    def _create_reservoir_model(self, d_model, nhead, num_layers, output_size, input_feature_dim):
        """构建单个水库的 Transformer 子模型。"""

        input_projection = nn.Linear(input_feature_dim, d_model)
        pos_encoding = self._create_positional_encoding(self.output_sequence_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        attention_pooling = nn.MultiheadAttention(d_model, nhead, dropout=self.dropout, batch_first=True)
        pooling_query = nn.Parameter(torch.randn(1, 1, d_model))

        if self.enable_sequence_decoding:
            output_layers = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(d_model // 2, output_size * self.output_sequence_length),
            )
        else:
            output_layers = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(d_model // 4, output_size),
            )

        class PositionalEncoding(nn.Module):
            def __init__(self, pe):
                super().__init__()
                self.register_buffer("pe", pe)

        class PoolingQuery(nn.Module):
            def __init__(self, query):
                super().__init__()
                self.query = nn.Parameter(query)

        return nn.ModuleDict(
            {
                "input_projection": input_projection,
                "pos_encoding": PositionalEncoding(pos_encoding),
                "transformer": transformer,
                "attention_pooling": attention_pooling,
                "pooling_query": PoolingQuery(pooling_query),
                "output_layers": output_layers,
                "layer_norm": nn.LayerNorm(d_model),
            }
        )
    
    def _create_multiscale_extractor(self, input_dim, output_dim):
        """构建多尺度特征提取器。"""

        class MultiscaleExtractor(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.feature_projection = nn.Linear(input_dim, output_dim)
                self.conv1 = nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm1d(output_dim)
                self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=5, padding=2)
                self.bn2 = nn.BatchNorm1d(output_dim)
                self.pool = nn.AdaptiveAvgPool1d(1)

            def forward(self, x):
                x = self.feature_projection(x)
                x = x.transpose(1, 2)
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = self.pool(x).squeeze(-1)
                return x

        return MultiscaleExtractor(input_dim, output_dim)

    def _create_positional_encoding(self, seq_len, d_model):
        """生成位置编码"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _forward_reservoir_group(self, x, model_dict):
        """为特定水库组执行前向传播"""
        batch_size, seq_len, _ = x.shape
        x = model_dict['input_projection'](x)
        x = x + model_dict['pos_encoding'].pe[:, :seq_len, :].to(x.device)
        x = model_dict['transformer'](x)
        query = model_dict['pooling_query'].query.expand(batch_size, -1, -1)
        pooled_output, _ = model_dict['attention_pooling'](query, x, x)
        pooled_output = pooled_output.squeeze(1)
        pooled_output = model_dict['layer_norm'](pooled_output)
        output = model_dict['output_layers'](pooled_output)
        if self.enable_sequence_decoding:
            output = output.view(batch_size, self.output_sequence_length, -1)
        return output

    def forward(
        self,
        x,
        q_min: Optional[torch.Tensor] = None,
        q_max: Optional[torch.Tensor] = None,
        q_in: Optional[torch.Tensor] = None,
        V0: Optional[torch.Tensor] = None,
        V_target: Optional[torch.Tensor] = None,
        delta_t: Optional[Union[float, torch.Tensor]] = None,
        V_min: Optional[torch.Tensor] = None,
        V_max: Optional[torch.Tensor] = None,
        terminal_reachable_mask: Optional[torch.Tensor] = None,
        terminal_best_effort_target: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ):
        """前向传播"""
        batch_size, seq_len, _ = x.shape

        shared_features = self.shared_input_projection(x)
        shared_features = self.input_norm(shared_features)

        if self.enable_multiscale:
            multiscale_features = []
            for extractor in self.multiscale_extractors.values():
                scale_features = extractor(shared_features)
                scale_features = scale_features.unsqueeze(1).expand(-1, seq_len, -1)
                multiscale_features.append(scale_features)
            fused_features = torch.cat(multiscale_features, dim=-1)
            fused_features = self.feature_fusion(fused_features)
        else:
            fused_features = shared_features

        if self.enable_multiscale:
            upstream_features = self.upstream_feature_extractor(fused_features)
            downstream_features = self.downstream_feature_extractor(fused_features)
        else:
            upstream_features = self.upstream_feature_extractor(shared_features)
            downstream_features = self.downstream_feature_extractor(shared_features)

        upstream_output = self._forward_reservoir_group(upstream_features, self.upstream_model)
        downstream_output = self._forward_reservoir_group(downstream_features, self.downstream_model)

        if self.enable_sequence_decoding:
            logits = torch.zeros(batch_size, self.output_sequence_length, self.output_dim, device=x.device)
            upstream_reservoir_count = len(self.upstream_reservoirs)
            downstream_reservoir_count = len(self.downstream_reservoirs)
            logits[:, :, self.upstream_reservoirs] = upstream_output[:, :, :upstream_reservoir_count]
            logits[:, :, self.downstream_reservoirs] = downstream_output[:, :, :downstream_reservoir_count]
            if hasattr(self, 'sequence_decoder'):
                output_flat = logits.view(batch_size, -1)
                memory = self.sequence_decoder.feature_projection(output_flat).unsqueeze(1)
                target_emb = self.sequence_decoder.target_embedding.expand(batch_size, -1, -1)
                target_emb = target_emb + self.sequence_decoder.positional_encoding[:, :self.output_sequence_length, :]
                decoded = self.sequence_decoder.decoder(target_emb, memory)
                logits = self.sequence_decoder.output_projection(decoded)
        else:
            logits = torch.zeros(batch_size, self.output_dim, device=x.device)
            logits[:, self.upstream_reservoirs] = upstream_output
            logits[:, self.downstream_reservoirs] = downstream_output

        if q_min is None or q_max is None:
            return logits

        if q_in is None or V0 is None or V_target is None:
            raise ValueError("q_in, V0, and V_target must be provided together with q_min/q_max.")

        if delta_t is None:
            delta_t_tensor = torch.full(
                (seq_len,), 864_000.0, dtype=logits.dtype, device=x.device
            )
        else:
            if not torch.is_tensor(delta_t):
                delta_t_tensor = torch.tensor(delta_t, dtype=logits.dtype, device=x.device)
            else:
                delta_t_tensor = delta_t.to(dtype=logits.dtype, device=x.device)
            if delta_t_tensor.dim() == 0:
                delta_t_tensor = delta_t_tensor.view(1).repeat(seq_len)
            else:
                delta_t_tensor = delta_t_tensor.view(-1)
                if delta_t_tensor.numel() != seq_len:
                    raise ValueError(
                        f"delta_t length mismatch: expected {seq_len}, got {delta_t_tensor.numel()}"
                    )

        def _expand_time_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.size(0) == 1 and tensor.size(0) != batch_size:
                tensor = tensor.expand(batch_size, -1, -1)
            return tensor

        q_min = _expand_time_tensor(q_min.to(dtype=logits.dtype, device=x.device))
        q_max = _expand_time_tensor(q_max.to(dtype=logits.dtype, device=x.device))
        q_in = _expand_time_tensor(q_in.to(dtype=logits.dtype, device=x.device))
        V0 = V0.to(dtype=logits.dtype, device=x.device)
        if V0.dim() == 1:
            V0 = V0.unsqueeze(0).expand(batch_size, -1)
        V_target = V_target.to(dtype=logits.dtype, device=x.device)
        if V_target.dim() == 1:
            V_target = V_target.unsqueeze(0).expand(batch_size, -1)

        q_base = self.capacity_head(logits, q_min, q_max, q_in, V0, V_target, delta_t_tensor)

        # Apply hard projection if enabled and constraints are available
        if self.enable_hard_projection and V_min is not None and V_max is not None:
            q_proj, V_proj, stats = self.hierarchical_projection(
                q_raw=q_base,
                q_in=q_in,
                V0=V0,
                V_target=V_target,
                q_min=q_min,
                q_max=q_max,
                Vmin=V_min,
                Vmax=V_max,
                delta_t=delta_t_tensor,
                terminal_reachable_mask=terminal_reachable_mask,
                terminal_best_effort_target=terminal_best_effort_target,
            )
            q_phys = q_proj
            projection_record: Dict[str, Any] = {"V": V_proj, "meta": stats}
        else:
            # Fallback to base capacity head output
            projection_record: Dict[str, Any] = {"V": None, "meta": None}
            q_phys = q_base
            
        self.last_projection = projection_record

        if return_logits:
            return q_phys, logits, q_base
        return q_phys

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)



