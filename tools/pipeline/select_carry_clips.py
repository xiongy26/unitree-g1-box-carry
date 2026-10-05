#!/usr/bin/env python3
"""MERO-10 P0 选片：从 OMOMO 原始 .pt 筛选"完整弧"抱箱行走片段（方案 4.2 规格）。

数据链路（复跑命令，原样可抄；先在本机配置外部数据目录环境变量）：
  conda activate hoi-retarget && python3 tools/pipeline/select_carry_clips.py \
      --data <HOI_RETARGET_DATA_DIR> \
      --out data/carry_clip_candidates.csv

只读原始 .pt（不入仓），输出排名 CSV + 每个桶 Top-3 摘要。依赖 numpy/torch
（hoi-retarget conda env 自带），无其他第三方依赖。

切片布局（hoi_retarget/datasets/intermimic.py 的 INTERMIMIC_RAW_SLICE，(T,591) @30fps）：
  root_pos[0:3]、dof[9:162]、body_pos[162:318]（L_Ankle=3 / R_Ankle=7 / L_Wrist=17 /
  R_Wrist=36 世界坐标）、obj_pos[318:321]、contact_obj[330:331]。

场景比例尺：原始 .pt 为人体尺度，重定向按 G1 object_scale=0.83 统一缩放（实测校准：
sub10_largebox_051/048/050 的 obj z_release ×0.83 与 motions/*.json 的 z_release 误差
≤1cm，obj 位移同理），故本脚本加载后即把全部位置 ×0.83，所有阈值用场景尺度
（与方案 4.2 的桶区间/槽位中心一致）。

对方案 4.2 的三处最小偏离（理由见各行注释与 MERO-10 交接报告）：
  1. 单回合校验改判"抬升回合数"：回合内 obj z 升越 rest+0.25 的上升沿 >1 才判多回合
     （方案字面为 grasp..release 段 z < rest+0.03 帧数 ≤5）。实测字面口径把自然动作
     全部误杀：grasp..lift_start 是"握住但未抬起"、下放到位后手掌仍贴箱至 contact
     结束（sub16_largebox_010 两段合计 26 帧 z≈rest，z 剖面为干净单回合）。
  2. 释放剖面在"z 回落到 rest+0.15 并保持"之外，接受"稳定在任意高度"的释放
     （release 后 z 不得超过释放帧 +0.15，即不得再拿起）——否则 mid/high（放桌面）
     桶永远为空，4.2 的交叉校验无从谈起；low 桶语义不受影响（真低释放必然回到 rest 带）。
  3. release_complete 的 0.5s 观察窗不足（raw contact 贴到片尾，如 sub10_largebox_048/
     050）时记 n/a 不设门——释放后收尾在管线产物里本就不参与回放判定（回放止于 release）。
  4. 新增硬门"起点抓取"：grasp 帧 obj z ≤ rest+0.10（箱子必须还在地面时被握住）——
     R1 完整弧的必要条件，方案字面未写但"起点抓取"语义隐含；实测 sub10_largebox_014
     即反例（contact 从片头开始、箱已在空中）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    sys.exit("需要 numpy/torch（自建 conda 环境，先 conda activate hoi-retarget 再用 python3 跑）")

ROOT = Path(__file__).resolve().parents[2]

# INTERMIMIC_RAW_SLICE（intermimic.py:17-27）
SL_ROOT = slice(0, 3)
SL_BODY = slice(162, 162 + 52 * 3)
SL_OBJ = slice(318, 321)
COL_CONTACT_OBJ = 330
# body 索引（INTERMIMIC_OMOMO_BODY_NAMES）
IDX_L_ANKLE, IDX_R_ANKLE = 3, 7
IDX_L_WRIST, IDX_R_WRIST = 17, 36
FPS = 30.0
POS_SCALE = 0.83  # G1 object_scale（README"数据复跑"；实测校准见文件头）

# 阈值（场景尺度，方案 4.2）
ROUND_MIN_SEC = 2.0          # 接触回合最短时长
LIFT_ABOVE = 0.25            # 抬起判定：obj z ≥ rest+0.25
RELEASE_BELOW = 0.15         # 释放判定：z 回落至 rest+0.15 内
BELOW_REST_MAX = 5           # 单回合校验：携带中 z < rest+0.03 的帧数上限
CARRY_DISP_RANGE = (0.8, 1.8)    # 主选位移带
CARRY_DISP_PREFER = (0.9, 1.4)   # 首选位移带（≈STACK_DIST 场景尺度）
STRAIGHT_MAX_DEG = 45.0      # 航向总变化上限（≤30° 优先）
FIT_P95_MAX = 0.25           # 腕-箱贴合 p95 上限（重定向后贴合的上界代理）
ROOT_SPEED_MIN = 0.15        # 携带窗 root 水平速度均值下限
STRIDE = 0.5                 # 支撑切换次数 ≥ carry_disp/STRIDE
STANCE_DIST_MIN = 0.3        # 释放站位人-箱水平距离下限
STANCE_ANGLE_MAX_DEG = 45.0  # 站位方向 vs -û（走来的方向）夹角上限
PRELUDE_MAX = 0.8            # |root_xy(0)−root_xy(grasp)| 上限
ENDING_MAX = 1.0             # |root_xy(T−1)−root_xy(release)| 上限
T_MAX = 360                  # K6：T ≤ 12s
POST_RELEASE_MIN = 15        # 释放后至少 0.5s 观察窗（release_complete 用）
ANKLE_STANCE_Z = 0.04        # 踝世界 z 低于此值视为支撑相（场景尺度）
# 槽位中心目标（方案 4.2；origin z = center z − offZ，offZ 取 motions/*.json 实测值）
SLOT_TARGET_CENTER = {
    "largebox": {"low": 0.169, "mid": 0.508, "high": 0.847},
    "plasticbox": {"low": 0.160, "mid": 0.480, "high": 0.800},
}
BOX_OFF_Z = {"largebox": 0.0048, "plasticbox": -0.0023}  # obj 原点→geom 中心偏置（JSON 实测）
BUCKETS = {"low": (0.08, 0.35), "mid": (0.35, 0.65), "high": (0.65, 0.95)}


def wrap_deg(a: float) -> float:
    """角度归一到 (-180,180]。"""
    return (a + 180.0) % 360.0 - 180.0


def heading_total_change(xy: np.ndarray, seg_len: float = 0.3) -> float:
    """携带窗行进航向总变化（deg）：按 ~seg_len 路程分段的航向角差绝对值之和。

    逐帧航向会被步态横向晃动/位置噪声打爆（实测 sub16 逐帧口径 517°，路径实际近直
    线），故按累计路程 ≥seg_len 分段再累计航向变化（经典 path-heading 口径）。
    """
    total = 0.0
    acc = 0.0
    seg_start = xy[0]
    prev_h = None
    for i in range(1, len(xy)):
        acc += math.hypot(xy[i][0] - xy[i - 1][0], xy[i][1] - xy[i - 1][1])
        if acc < seg_len:
            continue
        h = math.atan2(xy[i][1] - seg_start[1], xy[i][0] - seg_start[0])
        if prev_h is not None:
            total += abs(wrap_deg(math.degrees(h - prev_h)))
        prev_h = h
        seg_start = xy[i]
        acc = 0.0
    return total


def support_switches(ankle_z_l: np.ndarray, ankle_z_r: np.ndarray) -> tuple[int, int]:
    """左右踝支撑切换次数（离地上升沿）。

    阈值相对化（实测：OMOMO 踝关节中心离地 ~8-10cm，绝对阈判不出任何离地）——
    取该足回合内 z 的 10% 分位（支撑相基线）+3cm 为离地阈。
    """
    out = []
    for z in (ankle_z_l, ankle_z_r):
        thr = float(np.percentile(z, 10)) + 0.03
        n = 0
        was = z[0] < thr
        for v in z[1:]:
            now = v < thr
            if was and not now:
                n += 1
            was = now
        out.append(n)
    return out[0], out[1]


def analyze_round(a: np.ndarray, g: int, r: int, rest_z: float, box: str) -> dict | None:
    """分析单个接触回合 [g, r]；不构成"完整弧携带"时返回 None（原因记入 reject）。"""
    T = a.shape[0]
    obj = a[:, SL_OBJ]
    root = a[:, SL_ROOT]
    body = a[:, SL_BODY]
    z = obj[:, 2]
    reject = None

    # 抬起帧：z 首次升越 rest+0.25
    lift_start = next((i for i in range(g, r + 1) if z[i] >= rest_z + LIFT_ABOVE), None)
    if lift_start is None:
        return {"reject": "no_lift"}

    # 起点抓取门（R1"完整弧"语义）：grasp 帧箱子必须仍在地面（rest+0.10 内）。
    # 实测驱动：sub10_largebox_014 的 contact 回合从片头开始而箱已在 0.55m 空中
    # （人已持箱行进），回放时箱会从生成位 0.33s 内猛拉到手上（N10/N13 基线双破）。
    if z[g] > rest_z + 0.10:
        return {"reject": "grasp_airborne"}

    # 单回合校验（偏离 1，见文件头）：回合内 z 升越 rest+0.25 的上升沿数 == 1，
    # >1 即"中途放下再拿起"的多回合片段。below_* 列保留作诊断。
    above = (z[g:r + 1] >= rest_z + LIFT_ABOVE)
    lift_episodes = int(np.sum(np.diff(np.concatenate(([0], above.view(np.int8), [0]))) == 1))
    below_seg = z[lift_start:r + 1]
    below_frames = int(np.sum(below_seg < rest_z + 0.03))
    below_frames_all = int(np.sum(z[g:r + 1] < rest_z + 0.03))
    if lift_episodes > 1:
        return {"reject": "multi_round", "lift_episodes": lift_episodes,
                "below_frames": below_frames, "below_frames_all": below_frames_all}

    # 释放观察窗：raw contact 贴到片尾时窗口不足（偏离 3，见文件头）→ release_complete=n/a
    has_post = (T - 1 - r) >= POST_RELEASE_MIN

    # 释放剖面（偏离 2，见文件头）：release 后 z 不得再拿起（超过释放帧 +0.15）
    post = z[r + 1:]
    if len(post) and np.any(post > z[r] + 0.15):
        return {"reject": "post_relift", "lift_episodes": lift_episodes}

    # release_complete：释放后 0.5s 内水平速度 <0.05m/s 且 z 保持（窗口足够时才有判定资格）
    w = min(POST_RELEASE_MIN, T - 1 - r)
    h_speed = np.hypot(np.diff(obj[r:r + w + 1, 0]), np.diff(obj[r:r + w + 1, 1])) * FPS
    z_drift = float(np.max(np.abs(post[:w] - z[r]))) if w > 0 else 1.0
    release_complete = None
    if has_post:
        release_complete = bool(np.max(h_speed) < 0.05 and z_drift < 0.02)

    # 度量（场景尺度）
    carry_disp = float(math.hypot(*(obj[r, 0:2] - obj[g, 0:2])))
    u = obj[r, 0:2] - obj[g, 0:2]
    yaw_u = math.atan2(u[1], u[0]) if carry_disp > 1e-6 else 0.0
    straightness = heading_total_change(obj[lift_start:r + 1, 0:2])

    q = obj[g:r + 1]  # 携带窗
    dL = np.linalg.norm(body[g:r + 1, IDX_L_WRIST * 3:IDX_L_WRIST * 3 + 3] - q, axis=1)
    dR = np.linalg.norm(body[g:r + 1, IDX_R_WRIST * 3:IDX_R_WRIST * 3 + 3] - q, axis=1)
    fit_max = float(max(dL.max(), dR.max()))
    fit_p95 = float(max(np.percentile(dL, 95), np.percentile(dR, 95)))

    root_speed = float(np.mean(np.hypot(np.diff(root[g:r + 1, 0]), np.diff(root[g:r + 1, 1]))) * FPS)
    sw_l, sw_r = support_switches(body[g:r + 1, IDX_L_ANKLE * 3 + 2], body[g:r + 1, IDX_R_ANKLE * 3 + 2])

    stance_d = root[r, 0:2] - obj[r, 0:2]
    stance_dist = float(math.hypot(stance_d[0], stance_d[1]))
    stance_angle = abs(wrap_deg(math.degrees(math.atan2(stance_d[1], stance_d[0]) - math.atan2(-u[1], -u[0])))) \
        if carry_disp > 1e-6 else 180.0

    prelude = float(math.hypot(*(root[0, 0:2] - root[g, 0:2])))
    ending = float(math.hypot(*(root[T - 1, 0:2] - root[r, 0:2])))

    # 硬门（方案 4.2-4/5；位移/贴合的放宽档由 --relaxed 控制）
    disp_lo, disp_hi = CARRY_DISP_RANGE
    if carry_disp < disp_lo or carry_disp > disp_hi:
        reject = "carry_disp"
    elif straightness > STRAIGHT_MAX_DEG:
        reject = "straightness"
    elif fit_p95 > FIT_P95_MAX:
        reject = "fit_p95"
    elif root_speed < ROOT_SPEED_MIN:
        reject = "root_speed"
    elif sw_l + sw_r < carry_disp / STRIDE or min(sw_l, sw_r) < 1 or abs(sw_l - sw_r) > max(2, (sw_l + sw_r) / 3):
        reject = "gait_switches"
    elif stance_dist < STANCE_DIST_MIN or stance_angle > STANCE_ANGLE_MAX_DEG:
        reject = "stance"
    elif prelude > PRELUDE_MAX:
        reject = "prelude"
    elif ending > ENDING_MAX:
        reject = "ending"
    elif not release_complete:
        reject = "release_complete" if release_complete is False else None

    z_rel_center = float(z[r] + BOX_OFF_Z.get(box, 0.0))
    bucket = next((b for b, (lo, hi) in BUCKETS.items() if lo <= z_rel_center <= hi), None)

    return {
        "grasp": g, "release": r, "lift_start": lift_start, "lift_episodes": lift_episodes,
        "carry_disp": carry_disp, "straightness_deg": straightness,
        "release_z": float(z[r]), "release_z_center": z_rel_center, "bucket": bucket,
        "fit_max": fit_max, "fit_p95": fit_p95,
        "root_speed": root_speed, "switches_l": sw_l, "switches_r": sw_r,
        "stance_dist": stance_dist, "stance_angle_deg": stance_angle,
        "prelude": prelude, "ending": ending,
        "below_frames": below_frames, "below_frames_all": below_frames_all,
        "release_complete": release_complete,
        "reject": reject, "rest_z": float(rest_z), "T": T,
    }


def score(res: dict, box: str) -> float:
    """桶内排名分（越小越好）：位移带偏离 + 释放高度差 + 贴合 + 航向 + 速度带。"""
    target = SLOT_TARGET_CENTER[box].get(res["bucket"], res["release_z_center"])
    lo, hi = CARRY_DISP_PREFER
    disp_pen = max(0.0, lo - res["carry_disp"]) + max(0.0, res["carry_disp"] - hi)
    return (disp_pen * 2.0 + abs(res["release_z_center"] - target) * 3.0
            + res["fit_p95"] * 1.0 + res["straightness_deg"] / 100.0
            + max(0.0, 0.7 - res["root_speed"]) * 0.5)


def rest_z_of(a: np.ndarray, g: int, r: int) -> float:
    """rest z：回合前 45 帧 obj z 中位数（回合贴着片头时用释放后帧兜底）。"""
    obj = a[:, SL_OBJ]
    pre = obj[max(0, g - 45):g, 2]
    if len(pre) >= 10:
        return float(np.median(pre))
    post = obj[r + 1:r + 46, 2]
    return float(np.median(post)) if len(post) else float(obj[g, 2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.environ.get("HOI_RETARGET_DATA", ""),
                    help="OMOMO 原始数据目录（默认读环境变量 HOI_RETARGET_DATA）")
    ap.add_argument("--out", default=str(ROOT / "data" / "carry_clip_candidates.csv"))
    ap.add_argument("--box", default="all", help="largebox|plasticbox|all（默认 all）")
    ap.add_argument("--relaxed", action="store_true", help="K1 放宽档：位移下限 0.6、贴合 p95 ≤0.30")
    args = ap.parse_args()

    if not args.data:
        ap.error("未指定外部数据目录：请设置环境变量 HOI_RETARGET_DATA 或传 --data <HOI_RETARGET_DATA_DIR>")

    if args.relaxed:
        global CARRY_DISP_RANGE, FIT_P95_MAX
        CARRY_DISP_RANGE = (0.6, CARRY_DISP_RANGE[1])
        FIT_P95_MAX = 0.30

    box_types = ("largebox", "plasticbox") if args.box == "all" else (args.box,)
    rows: list[dict] = []
    rejects = 0
    for box in box_types:
        paths = sorted(glob.glob(os.path.join(args.data, f"*_{box}_*.pt")))
        print(f"[{box}] {len(paths)} 个候选", flush=True)
        for k, p in enumerate(paths):
            name = os.path.basename(p)
            try:
                d = torch.load(p, map_location="cpu", weights_only=False)
                a = (d.detach().numpy() if torch.is_tensor(d) else np.asarray(d)).astype(np.float64)
            except Exception as exc:
                print(f"  [load-fail] {name}: {type(exc).__name__}: {exc}", flush=True)
                continue
            a *= POS_SCALE
            contact = a[:, COL_CONTACT_OBJ] > 0.5
            # 连续接触回合（时长 ≥2.0s）
            edges = np.diff(np.concatenate(([0], contact.view(np.int8), [0])))
            starts = np.nonzero(edges == 1)[0]
            ends = np.nonzero(edges == -1)[0] - 1
            for g, r in zip(starts, ends):
                if r - g + 1 < ROUND_MIN_SEC * FPS:
                    continue
                res = analyze_round(a, int(g), int(r), rest_z_of(a, int(g), int(r)), box)
                if res is None:
                    continue
                if "reject" in res and res["reject"] not in (None,):
                    rejects += 1
                res.update({"clip": name, "box": box})
                if res.get("reject") is None:
                    res["score"] = round(score(res, box), 4)
                rows.append(res)
            if (k + 1) % 100 == 0:
                print(f"  … {k + 1}/{len(paths)}（pass {sum(1 for x in rows if x.get('reject') is None)}）", flush=True)

    cols = ["clip", "box", "T", "grasp", "lift_start", "release", "bucket", "carry_disp",
            "straightness_deg", "release_z", "release_z_center", "fit_max", "fit_p95",
            "root_speed", "switches_l", "switches_r", "stance_dist", "stance_angle_deg",
            "prelude", "ending", "lift_episodes", "below_frames", "below_frames_all",
            "release_complete", "score", "reject", "rest_z"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: (round(row[c], 4) if isinstance(row.get(c), float) else row.get(c, '')) for c in cols})
    passed = [r for r in rows if r.get("reject") is None]
    print(f"\n共 {len(rows)} 个回合（其中 {rejects} 个被拒），合格 {len(passed)} 个 → {args.out}")

    # 每个箱型 × 桶 Top-3 摘要
    for box in box_types:
        for bucket in ("low", "mid", "high"):
            pool = [r for r in passed if r["box"] == box and r["bucket"] == bucket]
            pool.sort(key=lambda r: r["score"])
            print(f"\n== {box}/{bucket} Top-{min(3, len(pool))}（共 {len(pool)}）==")
            for r in pool[:3]:
                print(f"  {r['clip']} score={r['score']} disp={r['carry_disp']:.2f}m "
                      f"rel_z={r['release_z_center']:.3f} fit_p95={r['fit_p95']:.3f} "
                      f"straight={r['straightness_deg']:.0f}° spd={r['root_speed']:.2f}m/s "
                      f"sw={r['switches_l']}+{r['switches_r']} stance={r['stance_dist']:.2f}m/"
                      f"{r['stance_angle_deg']:.0f}° T={r['T']} g={r['grasp']} r={r['release']}")


if __name__ == "__main__":
    main()
