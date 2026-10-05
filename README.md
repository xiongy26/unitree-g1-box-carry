# unitree-g1-box

MuJoCo WASM + Three.js：机器人走到交接区，双手取箱、搬到货架、在架外降到目标高度，再水平送入槽位，松手后返回。两种料箱、两层货架、每层两格；右下角小地图显示当前规划路线、目标、机器人和料箱。

## 功能特性

- 浏览器端零 npm 依赖运行：MuJoCo WASM 动力学 + Three.js 渲染，本地仅需 Python 3 起静态服务器。
- 空手行走由 RL 策略驱动 MuJoCo 动力学；取放、持箱步态和原地转向是接触约束下的运动学演示；松手后的箱体、地面和层板接触由 MuJoCo 求解。当前实现不是可直接部署到真实 G1 的力控、平衡或抓握控制器。
- 接触轨迹按箱型分别求解：双手从前侧两角靠近侧面握持区域，箱口保持朝上；把箱口翻边计入避让外廓，增大相邻格位间距；进入物理放置前箱子已完成姿态与中心偏移对齐，松手退手后才后退、再起身。
- 原地转向使用小步换脚：身体在摆动阶段改变朝向，骨盆仅做轻微侧移，不再每步反复蹲起。
- 货架层数（1–2 层）、列数（1–2 列）与箱型（largebox / plasticbox）可配置，在下一次开始时生效。
- 运动学持箱使用显示体与物理体双体，取放轨迹通过几何距离检查避免穿插，释放后由物理箱接管；运行时校验轨迹来源签名，拒绝使用失配的数据。

## 快速开始

### 环境要求

现代浏览器即可运行页面；本地服务器需 Python 3。需要 HTTP 加载 ES 模块和 WASM，不能直接打开 HTML 文件；本地服务器只监听 `127.0.0.1`。浏览器端无需安装 npm 依赖。

### 克隆与启动

```bash
cd unitree-g1-box
python3 serve.py
# 打开 http://localhost:8000/
```

## 使用说明

点 **料箱上架** 或按 `3` 开始。`R` 重置，空格暂停；支持速度与相机调整。层数、列数、箱型在下一次开始时生效。完成后保留已上架料箱，重置清场。

小地图默认放大作业区，**全厂 / 作业区** 按钮切换范围；蓝线为本段路线，黄色十字为目标，白色箭头为机器人，粉色十字为当前槽位。空手接近、搬运、返回三段都会更新路线。

### 录制分享

点 **● 录制** 或按 `V` 开始/停止，录制「3D 画面 + 右下角小地图面板（含底板、边框、标题与图例）」的合成视频，不含顶部/侧边控制面板；停止后浏览器自动下载。料箱上架全部完成或失败后自动留 2.5 秒收尾再停止，按 `R` 重置也会先收片。编码优先 H.264 MP4（Chrome 126+），不支持时回退 WebM，可用 ffmpeg 转换：

```bash
ffmpeg -i g1-*.webm -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p g1.mp4
```

视频分辨率为录制开始时的窗口像素（奇数高自动取偶），时长与真实操作一致；录制中调整窗口大小只会等比加黑边。小地图面板即使被隐藏也照常入镜。

## 目录结构

```text
unitree-g1-box/
├── index.html            # 入口页面（importmap 与脚本锚点）
├── serve.py              # 本地静态服务器（以进程 CWD 为根）
├── package.json          # scripts 别名与 engines 声明（零依赖）
├── src/                  # 浏览器端模块（平铺）
│   └── controller/       # RL 策略推理与步行控制（g1Policy、walkController）
├── tools/
│   ├── checks/           # 回归与校验（headless_check、contact_plan_check、recorder_check、browser_smoke、check_manipulation.py）
│   ├── probes/           # 一次性/实验探针（motion_probe、policy_probe、handbox_probe）
│   ├── lib/              # 跨语言共享库（manipulation_reference.mjs）
│   └── pipeline/         # 离线数据管线（analyze_clips、export_motion、export_policy_weights、select_carry_clips、solve_manipulation）
├── data/                 # 可再生数据产物（carry_clip_candidates.csv，见 data/README.md）
├── model/                # MuJoCo 模型（g1.xml、scene.xml、assets/）
├── motions/              # 运动轨迹 JSON（walk_meta、manipulation、carry 片段）；随仓库分发
├── vendor/               # 三方运行时（three、mujoco）与策略权重（policy）
└── docs/                 # 验收记录与探针笔记
```

## 工具与验证

| 文件 | 职责 |
|---|---|
| `src/stackCore.js` | 任务状态、槽位、行走与取放的衔接、物理释放 |
| `src/pivotPlan.js` | 原地转向足端规划、双腿 IK 与平滑采样 |
| `src/manipulationPlan.js` | 验证接触轨迹、平移到列槽位、提供搬运路线 |
| `src/pathPlanner.js` | 障碍膨胀、可见性图、路径搜索 |
| `src/controller/` | RL 策略推理、步行控制、跌倒与到位检测 |
| `src/factory.js` | 货架与设施的统一布局、模型和地图几何 |
| `src/minimapModel.js` / `src/minimap.js` | 地图数据与 Canvas 绘制 |
| `src/recorder.js` | 录制合成（3D 全幅 + 小地图面板）与 MediaRecorder 编码下载 |
| `tools/checks/recorder_check.mjs` | 真实页面录制回归：编码选择、下载落盘、完成后自动收片 |
| `tools/lib/manipulation_reference.mjs` | 用展示端同一套变换导出最终货架任务参考 |
| `tools/pipeline/solve_manipulation.py` | 接触与足端约束求解，生成可回放轨迹 |
| `tools/checks/check_manipulation.py` | 对帧间插值姿态独立检查几何间隙与抓握误差 |
| `motions/manipulation.json` | 四组接触轨迹：两种箱型 × 两层 |

计划单独记录源帧数和释放帧，允许延长送箱与退手时间；原动捕的节奏不强行套到货架任务。源动捕文件保留不变，`manipulation.json` 是明确标注的派生运动。改动源动作、箱型尺寸、货架高度或朝向后须重新求解；运行时校验来源签名，拒绝使用失配的轨迹。

回归验证命令：

```bash
node tools/checks/contact_plan_check.mjs
node tools/checks/headless_check.mjs --unit-only
node tools/checks/headless_check.mjs --layers 1 --cols 1
node tools/checks/headless_check.mjs --layers 2 --cols 2
node tools/checks/headless_check.mjs --box plasticbox --layers 2 --cols 2
node tools/checks/headless_check.mjs --policy-selftest
node tools/checks/headless_check.mjs --fk
```

录制功能用真实页面检查（含下载落盘与 ffprobe 容器校验；调试端口 Chrome 无头即可，启动命令见脚本头注释）：

```bash
node tools/checks/recorder_check.mjs http://localhost:8000 http://localhost:9223
node tools/checks/recorder_check.mjs http://localhost:8000 http://localhost:9223 --full-loop  # 完整循环 + 自动收片
```

网页加载回归检查使用真实页面默认配置，并验证两种箱型的一层／两层启动；避免只验证 Node 装配而漏掉网页加载清单。Chrome 开启本机调试端口 `9223` 后可运行：

```bash
node tools/checks/browser_smoke.mjs http://localhost:8000 http://localhost:9223
```

完整循环检查实际完成、箱体落点、物理放置稳定、脚步交替、足底滑移、位姿连续性、避障与接触距离。实验性 `--carry rl` 保留用于对照，不使用新的离线取放接触规划。

当前状态：2026-10-05 验收，大料箱 2×2 全循环 **45/45**，小料箱 2×2 全循环 **47/47**；四组轨迹均通过 **120 Hz** 帧间几何检查，检查覆盖取箱前、持箱、退手、后退和起身以及第二列与已放料箱，所检查碰撞几何未检测到穿插，最大掌端目标误差 25.9 mm；来源与损坏数据校验、列平移检查和 RL 前向 golden fixture 均通过。

## 重新求解（可选，Python 依赖）

需要 Node、Python、NumPy、SciPy 和 MuJoCo。自建 conda 环境（Python、NumPy、SciPy、MuJoCo）后运行：

```bash
cd unitree-g1-box
conda activate hoi-retarget && python3 tools/pipeline/solve_manipulation.py
conda activate hoi-retarget && python3 tools/checks/check_manipulation.py
```

选片脚本 `tools/pipeline/select_carry_clips.py` 从环境变量 `HOI_RETARGET_DATA` 读取外部数据目录，也可用 `--data <HOI_RETARGET_DATA_DIR>` 显式指定。

求解器只读本项目的源动作和模型，不修改 HOI-Retarget 本地检出目录。输出先写临时文件再整体替换；求解失败保留旧轨迹。`--only largebox:0` 可以单独生成某个箱型、某层的轨迹。

独立检查包括含翻边的保守箱体外廓与机器人/层板的距离、机器人与货架的距离及第二列与已放料箱的距离、脚底高度、双手接触目标误差、关节限位和速度。默认 120 Hz 插值检查；`--hz 500` 可按仿真子步频率加密。另外检查站稳送箱与退手期间骨盆、双腿固定，料箱继续由手臂送入。该检查独立于优化器的成功标志。

## 数据来源与第三方组件

| 组件 | 位置 | 许可与来源 |
|---|---|---|
| three.js r180 | `vendor/three/` | MIT，Copyright 2010-2025 Three.js Authors（文件头 SPDX） |
| G1 velocity/v0 步行策略权重 | `vendor/policy/` | Apache-2.0，unitreerobotics/unitree_rl_lab（commit 4960b84），见 `vendor/policy/LICENSE-NOTE` |
| MuJoCo WASM 运行时 | `vendor/mujoco/` | 来源与许可待补（正式发布前核实补齐） |
| G1 机器人模型 | `model/` | 来源与许可待补（自述来自 Unitree G1 / MuJoCo Menagerie，正式发布前核实补齐） |
| 运动轨迹数据（OMOMO 人物交互数据重定向产物） | `motions/`（随仓库分发） | 上游自述 BSD-3；建议发布前与上游数据集许可对账 |
| 外部 HOI-Retarget 衍生统计表 | `data/carry_clip_candidates.csv` | 上游自述 BSD-3（指标统计表，随仓库分发） |

接触轨迹参考 HOI-Retarget 项目的 `optimization/window.py` 与 `contact/retarget.py`：固定物体坐标系中的接触目标，优化机器人关节，并约束关节限位、速度与时序连续性。本项目的接触求解器独立实现，没有复制上游优化器代码。原始导出流程保留在 `tools/pipeline/export_motion.py`、`tools/pipeline/select_carry_clips.py` 和 `tools/pipeline/analyze_clips.py`。

## 许可

本项目以 [Apache License 2.0](LICENSE) 发布。第三方组件与衍生数据的许可及来源汇总见 [NOTICE](NOTICE)；上表标注「待补」的条目在正式发布前完成核实。

再分发本仓库或其衍生作品时，请保留 LICENSE 与 NOTICE 文件，并按 Apache-2.0 第 4(d) 条传递第三方声明。

## 历史方案文档说明

源码注释中引用的 `docs/box-stacking-plan.md`、`docs/rl-walking-plan.md`、`docs/mocap-carry-walk-plan.md`、`docs/factory-scene-minimap-plan.md`、`docs/walk-avoid-plan.md` 五个方案文档未随本仓发布；注释中形如「方案 X 4.5.2」的引用为历史方案章节编号，仅作设计溯源锚点，不指向仓内现存文件。
