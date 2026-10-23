# MCC-HT manuscript-aligned core with retained joint projection

MCC-HT用于乌东德、白鹤滩、溪洛渡、向家坝、三峡和葛洲坝六库梯级的长期调度。本仓库包含模型、ICR协调、梯级更新、评估与计时程序，并保留后续加入的联合投影层及其训练、验证、推理调用。当前版本不是生成投稿结果时的归档版本，尚未用完整数据和原始权重复算论文结果及运行时间。

## 方法与评估

- 初步生成返回式(9)–(11)的原始流量；首次流量限幅由ICR Stage 1完成。
- 每次外层更新包含完整ICR；至少两轮、最多三轮。按重构轨迹的运行约束和0.01m末水位判据决定是否提前结束，不以相邻轮流量变化作为终止条件。
- 平滑项改为式(24)的平方差；末水位MSE取消释放能力归一化，与流量MSE共用lambda_exp。
- 式(23)训练功率代理使用W，最终发电仍使用容量限幅、过机流量和弃水分离。验证选模的功率指标仍以GW缩放，避免把旧选模系数误用到W。
- `paper_evaluate.py`分别计算原始pre-ICR、post-ICR/pre-joint和最终post-joint的指标。FR只统计全年全部水库末水位达标，TPR按水库—年份统计；完整可行率另列。路径违反不增加数值容差。
- `benchmark_paper.py`每年重复五次，先按年平均，再对年份平均；GPU计时前后同步。范围含整个推理入口、ICR、梯级更新、保留的联合投影与阶段导出，不含训练。此入口包含模型/变换加载及文件导出开销，不能直接替代或声称验证论文原始计时。

## 环境

推荐按 `requirements.txt` 配置PyTorch和联合投影依赖。Windows可用本次验证的 `requirements-windows-tested.txt`。在项目目录执行：

```powershell
python -m pip install -r requirements-windows-tested.txt
python -m pytest tests -q
```

Windows测试环境为Python 3.10.10、PyTorch 2.8.0+cu128、NumPy 1.26.4、CVXPY 1.6.5、cvxpylayers 0.1.6、diffcp 1.0.23。安装PyTorch时应选择适合本机的版本。较旧diffcp会给出弃用警告。测试覆盖模型、ICR、梯级更新、联合投影和梯度回传；合成样例测试不代表论文实验复现。

## 配置和数据

默认 `config.yaml` 为论文划分：1959–2008训练、2009–2014验证、2015–2022测试。完整数据、checkpoint、超参数文件和训练变换不在原包内，须自行恢复；缺少年份会报错，不会静默缩减评估集。

`config.example.yaml`只用于包内参考数据：1959、1961、1962训练，1963、1964验证，2022测试。它不能用于复现论文结果。

```powershell
# 示例配置，仅在当前命令行会话生效
$env:MCC_HT_CONFIG = 'config.example.yaml'
python train.py
python infer.py
python paper_evaluate.py --directory diverse_results --years 2022
python benchmark_paper.py --years 2022
```

恢复完整数据并使用论文配置时：

```powershell
$env:MCC_HT_CONFIG = 'config.yaml'
python train.py
python infer.py
python paper_evaluate.py --directory diverse_results --years 2015 2016 2017 2018 2019 2020 2021 2022
python benchmark_paper.py --years 2015 2016 2017 2018 2019 2020 2021 2022
```

训练产物位于本目录下results。推理要求results目录内存在multiscale_best_model.pth、multiscale_hyperparameters.json和annual_transforms.pkl。当前代码不保证与旧checkpoint原先的输出相同。

## 阶段输出

- `pre_icr_schedules_YEAR.csv`：第2.5.1节生成后、首次ICR前。
- `post_icr_schedules_YEAR.csv`：梯级更新结束、额外联合投影之前。
- `diverse_schedules_YEAR.csv`：额外联合投影后的最终交付方案，保持原文件名。
- `paper_stages_YEAR.json`：阶段说明、外层次数和各项停止检查结果。
- `paper_metrics_inputs_YEAR.npz`：scheme_01对应的阶段流量、库容、水位、容量限幅功率、目标与动态界限；供统计脚本读取。

统计同时列出pre→post-ICR与pre→最终（含联合投影）的调整量，不把后者全部归因于ICR。阶段数据缺失、非有限值、训练验证重叠、实验年份缺失都会明确报错。多种方法的比较需分别运行相应模型；本仓库未包含POA/PSO实现。
