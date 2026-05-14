# auto_research_chinese

这是一个面向表格机器学习任务的 autoresearch 示例目录。当前任务基于 F1 比赛逐圈记录，预测赛车是否会在下一圈进站，目标字段为 `PitNextLap`。

项目的核心思路是：围绕 `train.py` 反复进行可复现实验，记录每次模型、特征和参数调整带来的指标变化，并保留真正有效的改动。

## 项目背景

本项目是基于 `autoresearch_local-master/README.md` 的思路改造而来。原始 `autoresearch` 项目的出发点是：给 AI 编程助手一个小而完整的机器学习训练环境，让它像自动研究员一样持续提出实验想法、修改 `train.py`、运行训练、读取指标，并根据结果决定保留还是丢弃改动。人类不再直接逐行调模型，而是通过 Markdown 说明文件定义实验规则、约束和目标，让 AI agent 在这些规则下自主迭代。

原始项目更偏向单 GPU 训练场景，训练代码来自简化后的 nanochat 思路，关注模型结构、优化器、训练循环和固定训练时长内的验证指标表现。它的关键设计包括：

- 只开放 `train.py` 作为主要可编辑文件，减少实验范围和审查成本；
- 用固定时间预算运行每次实验，让不同模型和参数设置具备可比性；
- 用统一指标判断实验是否进步，进步则保留，否则回退；
- 用 `program.md` 描述 agent 的工作协议，让自动化实验可以持续进行。

本目录保留了这套“自动研究”的组织方式，但把目标从原始的语言模型/神经网络训练实验，改造成一个更通用的普通机器学习任务框架。也就是说，它关注的不再是某一个固定的深度学习训练脚本，而是更常见的机器学习工作流：读取结构化数据、做数据清洗和特征工程、选择模型、训练验证、生成预测结果，并持续记录实验表现。当前目录里的 F1 进站预测只是这个通用框架下的一个具体示例。

改造后的重点变化如下：

| 原始 autoresearch | 当前 auto_research_chinese |
| --- | --- |
| 面向单 GPU 训练和语言模型实验 | 面向普通机器学习任务 |
| 主要关注模型结构、优化器和训练循环 | 主要关注数据处理、特征工程、模型选择和参数调优 |
| 通过 `program.md` 约束 agent 行为 | 通过 `program_ml.md` 和本 README 描述实验流程 |
| 指标围绕训练脚本输出的验证结果 | 指标可按任务类型切换，如 AUC、F1、accuracy、RMSE、MAE 等 |
| 输出训练日志和实验记录 | 输出预测文件、评估结果和实验记录 |

因此，这个项目可以理解为一个“普通机器学习版 autoresearch”：它不追求复刻原项目的 GPU 训练环境，而是借用了原项目的实验管理思想，把 AI agent 的自主迭代流程迁移到分类、回归、表格预测等常见机器学习任务中。F1 进站预测只是当前放在该框架下演示和运行的任务。

## 目录结构

```text
auto_research_chinese/
├── data/
│   ├── train.csv              # 训练集，包含标签 PitNextLap
│   ├── test.csv               # 测试集，不包含标签
│   ├── sample_submission.csv  # 提交文件格式示例
│   └── submission.csv         # 已生成的预测结果
├── autoresearch_usage_report.html
├── program_ml.md
├── train.py
└── README.md
```

## 数据说明

训练集共有 439140 行，测试集共有 188165 行。主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `id` | 样本编号 |
| `Driver` | 车手 |
| `Compound` | 轮胎类型 |
| `Race` | 分站赛名称 |
| `Year` | 赛季年份 |
| `PitStop` | 当前是否已经发生进站相关状态 |
| `LapNumber` | 当前圈数 |
| `Stint` | 当前轮胎 stint 编号 |
| `TyreLife` | 当前轮胎已使用圈数 |
| `Position` | 当前名次 |
| `LapTime (s)` | 当前圈速，单位秒 |
| `LapTime_Delta` | 圈速变化 |
| `Cumulative_Degradation` | 累计性能衰减 |
| `RaceProgress` | 比赛进度比例 |
| `Position_Change` | 名次变化 |
| `PitNextLap` | 训练标签，下一圈是否进站 |

提交文件需要包含两列：

```csv
id,PitNextLap
439140,0
439141,0
```

## 运行环境

建议使用 Python 3.10 或更高版本，并安装以下依赖：

```bash
pip install pandas numpy scikit-learn
```

## 运行方式

`train.py` 会从当前工作目录读取 `train.csv` 和 `test.csv`，并在当前工作目录写出 `submission.csv`。由于数据文件放在 `data/` 目录下，推荐从 `data/` 目录启动脚本：

```bash
cd auto_research_chinese/data
python ../train.py
```

Windows PowerShell 中也可以这样运行：

```powershell
Set-Location auto_research_chinese\data
python ..\train.py
```

运行完成后会生成或覆盖：

```text
auto_research_chinese/data/submission.csv
```

## 当前模型

当前 `train.py` 中的 `your_function(train_df, test_df)` 使用了一个基于 scikit-learn 的 `GradientBoostingClassifier`：

- 对 `Driver`、`Compound`、`Race` 使用 `OrdinalEncoder` 编码；
- 使用圈数、轮胎寿命、名次、圈速、比赛进度等数值特征；
- 训练二分类模型预测 `PitNextLap`；
- 输出符合提交格式的 `submission.csv`。

当前特征列包括：

```python
[
    "Driver", "Compound", "Race", "Year", "PitStop",
    "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]
```

## 实验建议

建议只围绕 `train.py` 做实验，保持每次改动目标清晰，并记录实验结果。可尝试的方向包括：

- 更换分类模型，例如随机森林、ExtraTrees、HistGradientBoosting、XGBoost 或 LightGBM；
- 改进类别特征处理方式，例如目标编码、频次编码或 one-hot 编码；
- 构造赛道、车手、轮胎与比赛阶段的交互特征；
- 使用时间或比赛进度相关的验证切分，减少数据泄漏风险；
- 调整类别不平衡策略，例如样本权重或阈值优化；
- 使用 AUC、F1、准确率等指标对比模型表现。

推荐用 `results.tsv` 记录每次实验：

```tsv
commit	metric	score	status	description
```

示例：

```tsv
commit	metric	score	status	description
a1b2c3d	auc	0.8123	keep	baseline GradientBoostingClassifier
b2c3d4e	auc	0.8275	keep	add race progress interaction features
c3d4e5f	auc	0.8010	discard	overfit with deeper trees
```

## 相关文件

- `program_ml.md`：autoresearch 实验流程说明，但当前文件存在编码显示问题。
- `autoresearch_usage_report.html`：对项目、数据样例、实验过程和结果的可视化说明。
- `train.py`：主要可编辑入口，负责训练模型并生成预测结果。
