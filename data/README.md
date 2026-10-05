# data/

可再生数据产物目录。

- `carry_clip_candidates.csv`：由 `tools/pipeline/select_carry_clips.py` 从外部 HOI-Retarget 数据（外部数据工作区，不入仓）筛选生成的候选片段排名表；仓内暂无读取者，仅供选片参考。上游来源：OMOMO 人物交互数据集（BSD-3）经 HOI-Retarget 重定向的统计产物。再生成（先设置环境变量 `HOI_RETARGET_DATA` 指向外部数据目录）：

```bash
export HOI_RETARGET_DATA=<HOI_RETARGET_DATA_DIR>
python3 tools/pipeline/select_carry_clips.py
```
