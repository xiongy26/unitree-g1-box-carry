#!/usr/bin/env python3
"""hoi-retarget stage-pkl -> viewer 轨迹 JSON 转换器（码箱子模式数据链路第 2 段）。

数据链路（复跑命令原样可抄）：
  # 1) 管线段（在外部 HOI-Retarget 数据工作区跑，产出 outputs/<clip>/contact_window.pkl）
  cd <HOI_RETARGET_DIR> && conda activate hoi-retarget
  hoi-retarget --input_file data/InterMimic/OMOMO_new/sub10_largebox_046.pt \
      --out_dir outputs/sub10_largebox_046 --no-record_video

  # 2) 转换段（本脚本；numpy 来自 hoi-retarget conda env，无其他第三方依赖）
  cd unitree-g1-box
  conda activate hoi-retarget && python3 tools/pipeline/export_motion.py \
      --pkl <HOI_RETARGET_DIR>/outputs/sub10_largebox_046/contact_window.pkl \
      --g1 model/g1.xml \
      --obj-obj <HOI_RETARGET_DIR>/assets/objects/objects/largebox/largebox.obj \
      --out motions/carry_low_largebox.json

  # 扫描候选表（选 clip_low / clip_mid 用）：
  conda activate hoi-retarget && python3 tools/pipeline/export_motion.py \
      --scan <HOI_RETARGET_DIR>/outputs

输出 JSON 契约（g1-box-motion/1）见 docs/box-stacking-plan.md 5.1。要点：
- dof 按名映射：pkl 的 dof_names 与展示端 g1.xml 关节名逐一 diff，不按位信任；
- 关节限位按展示端 g1.xml jnt_range 断言（超限帧比例 >2% 报错换 clip，≤2% 截断并记录）；
- grasp/release 帧来自 pkl 接触数据（fixed_contact_points NaN 模式，per_link_contact_flags 校验）；
- 箱体半尺寸/中心偏移从 OBJ 顶点算轴对齐包围盒再乘 object_scale；
- 数值保留 4 位小数（0.1mm），fps 保持 30，单文件 ≤200KB。

许可说明：本脚本是自研转换器（只读 hoi-retarget BSD-3 产物，不复制其代码）；
OMOMO 原始 .pt 数据不入仓，仅转换后的 JSON 进 motions/。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("需要 numpy（用 hoi-retarget conda env 的 python 跑：~/.miniconda3/envs/hoi-retarget/bin/python）")

REQUIRED_KEYS = (
    "fps", "root_pos", "root_rot", "dof_pos", "dof_names",
    "world_body_pos", "object_pos", "object_rot",
    "per_link_contact_flags", "fixed_contact_points_per_frame_in_object_frame",
    "object_scale", "link_body_list",
)

# 接触槽位顺序（pkl contact_link_names 实测为 palms 在前、ankle_roll 在后），
# "持握"判据只看两个 palm 槽位；不硬编码索引，运行时按名字定位。
PALM_SUBSTR = "palm"

# 限位容差：小量超出视为数值噪声
RANGE_EPS = 1e-3
# 超限帧比例阈值（方案 4.1-2 / R2）
RANGE_VIOLATION_MAX_RATIO = 0.02


# ---------------- g1.xml 解析 ----------------
def parse_g1_joints(g1_xml_path: str) -> dict:
    """从展示端 g1.xml 提取 29 个 hinge 关节名与 jnt_range（按 XML 出现顺序）。"""
    import re

    with open(g1_xml_path, "r", encoding="utf-8") as f:
        text = f.read()
    joints = {}
    for m in re.finditer(r'<joint\s+name="([^"]+)"[^>]*?range="([^"]+)"', text):
        name = m.group(1)
        lo, hi = (float(x) for x in m.group(2).split())
        joints[name] = (lo, hi)
    if len(joints) != 29:
        raise SystemExit(f"g1.xml 解析到 {len(joints)} 个带 range 的关节，预期 29")
    return joints


def check_joint_limits(dof_pos, retarget_names, g1_joints):
    """按名 diff + 限位检查。返回 (重排后的 dof_pos, 截断数, 超限帧比例)。"""
    # 名字集合 diff：两边应完全一致（方案 3.3 实测同名同序，这里仍按名映射）
    missing_g1 = [n for n in retarget_names if n not in g1_joints]
    missing_rt = [n for n in g1_joints if n not in retarget_names]
    if missing_g1 or missing_rt:
        raise SystemExit(f"关节名 diff 失败: pkl 有而 g1.xml 无={missing_g1}, g1.xml 有而 pkl 无={missing_rt}")

    order = [retarget_names.index(n) for n in g1_joints]  # g1.xml 顺序 <- pkl 列号
    dof = dof_pos[:, order]

    lo = np.array([g1_joints[n][0] for n in g1_joints])
    hi = np.array([g1_joints[n][1] for n in g1_joints])
    over = (dof < lo - RANGE_EPS) | (dof > hi + RANGE_EPS)
    frame_over = over.any(axis=1)
    ratio = float(frame_over.mean())
    n_clamped = int(over.sum())
    if n_clamped:
        dof = np.clip(dof, lo, hi)
    return dof, n_clamped, ratio


# ---------------- 接触帧提取 ----------------
def extract_grasp_release(md):
    """从 fixed_contact_points 的 NaN 模式求 grasp/release 帧；per_link_contact_flags 交叉校验。

    grasp = 首个任一 palm 槽位非 NaN 帧；release = 最后一个非 NaN 帧。
    """
    pts = np.asarray(md["fixed_contact_points_per_frame_in_object_frame"])  # (T,4,3)
    names = list(md["contact_link_names"])
    palm_idx = [i for i, n in enumerate(names) if PALM_SUBSTR in n]
    if not palm_idx:
        raise SystemExit(f"contact_link_names 里找不到 palm 槽位: {names}")

    palm_pts = pts[:, palm_idx, :]                        # (T, P, 3)
    touched = np.isfinite(palm_pts).all(axis=2).any(axis=1)        # (T,) 任一 palm 非 NaN
    frames = np.nonzero(touched)[0]
    if len(frames) == 0:
        raise SystemExit("palm 接触全为 NaN，提不出 grasp/release 帧")
    grasp, release = int(frames[0]), int(frames[-1])

    # 交叉校验：per_link_contact_flags 的 palm 行应覆盖 [grasp, release]
    flags = np.asarray(md["per_link_contact_flags"])      # (T,4) bool
    palm_flags = flags[:, palm_idx].any(axis=1)
    n_mismatch = int((palm_flags != touched).sum())
    return grasp, release, n_mismatch


# ---------------- OBJ 包围盒 ----------------
def parse_obj_aabb(obj_path: str):
    """OBJ 顶点轴对齐包围盒 -> (half_size, center_offset)，未缩放原始值。"""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                for i, v in enumerate((x, y, z)):
                    lo[i] = min(lo[i], v)
                    hi[i] = max(hi[i], v)
    if lo[0] is math.inf:
        raise SystemExit(f"OBJ 无顶点: {obj_path}")
    half = [(hi[i] - lo[i]) / 2.0 for i in range(3)]
    center = [(hi[i] + lo[i]) / 2.0 for i in range(3)]
    return half, center


def infer_box_type(obj_path: str) -> str:
    stem = os.path.basename(obj_path)
    for t in ("largebox", "plasticbox", "smallbox"):
        if t in stem:
            return t
    return os.path.splitext(stem)[0]


# ---------------- FK 交叉验证（T3，消 A1 假设） ----------------
def fk_check_and_embed(doc, md, g1_xml: str) -> None:
    """把 JSON 轨迹的 3 个采样帧写入展示端 g1.xml 模型做 FK，与 pkl world_body_pos 按身体名对比。

    通过标准（方案 T3）：最大身体位置误差 < 2cm；同时在 JSON 里嵌 fk_ref
    （帧号 + 共有身体名 + FK 位置），供 headless_check.mjs --fk 在 Node 侧复验。
    """
    try:
        import mujoco as mj
    except ImportError:
        print("[WARN] conda env 无 python mujoco，跳过 FK 校验（不嵌入 fk_ref）")
        return

    model = mj.MjModel.from_xml_path(g1_xml)
    data = mj.MjData(model)
    link_names = list(md["link_body_list"])
    wpos = np.asarray(md["world_body_pos"], dtype=np.float64)
    model_bodies = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
    shared = [n for n in link_names if n in model_bodies]

    frames = sorted({0, doc["grasp_frame"], doc["release_frame"]})
    jadr = {}
    for n in doc["joint_names"]:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)
        jadr[n] = int(model.jnt_qposadr[jid])

    max_err = 0.0
    err_by_body = {}
    ref_pos = []
    for f in frames:
        data.qpos[:] = 0
        data.qpos[0:3] = doc["root_pos"][f]
        data.qpos[3:7] = doc["root_quat"][f]
        for i, n in enumerate(doc["joint_names"]):
            data.qpos[jadr[n]] = doc["dof_pos"][f][i]
        mj.mj_forward(model, data)
        row = []
        for n in shared:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)
            ref = wpos[f, link_names.index(n)]
            err = float(np.linalg.norm(data.xpos[bid] - ref))
            max_err = max(max_err, err)
            err_by_body[n] = max(err_by_body.get(n, 0.0), err)
            row.append([round(float(v), 4) for v in data.xpos[bid]])
        ref_pos.append(row)

    doc["fk_ref"] = {"frames": frames, "bodies": shared, "pos": ref_pos,
                     "max_err_m": round(max_err, 5), "note": "JSON 轨迹写入展示端 g1.xml 的 FK 位置（对照 pkl world_body_pos，容差 2cm）"}
    worst = max(err_by_body, key=err_by_body.get)
    print(f"[FK] max_err={max_err * 100:.2f}cm over {len(shared)} bodies × {len(frames)} frames (worst={worst})")
    if max_err >= 0.02:
        raise SystemExit(f"FK 交叉验证失败：max_err={max_err:.4f}m ≥ 2cm（A1/dof 顺序问题，见方案 R3）")



def export(pkl_path: str, g1_xml: str, obj_path: str, out_path: str, mass: float, clip_name: str | None = None,
           fk: bool = True) -> dict:
    with open(pkl_path, "rb") as f:
        md = pickle.load(f)

    meta = md.get("meta", {})
    if meta.get("schema_version") != 2:
        raise SystemExit(f"schema_version={meta.get('schema_version')}，预期 2（不是最终 stage 产物？换 contact_window.pkl）")
    for k in REQUIRED_KEYS:
        if k not in md:
            raise SystemExit(f"pkl 缺键: {k}")

    fps = float(md["fps"])
    root_pos = np.asarray(md["root_pos"], dtype=np.float64)
    root_quat = np.asarray(md["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(md["dof_pos"], dtype=np.float64)
    obj_pos = np.asarray(md["object_pos"], dtype=np.float64)
    obj_quat = np.asarray(md["object_rot"], dtype=np.float64)
    T = root_pos.shape[0]

    dof_names = list(md["dof_names"].keys()) if isinstance(md["dof_names"], dict) else list(md["dof_names"])
    g1_joints = parse_g1_joints(g1_xml)
    dof_pos, n_clamped, ratio = check_joint_limits(dof_pos, dof_names, g1_joints)
    if ratio > RANGE_VIOLATION_MAX_RATIO:
        clip_label = clip_name or os.path.basename(str(meta.get("source_file", pkl_path)))
        raise SystemExit(
            f"关节限位超限帧比例 {ratio:.1%} > 阈值 {RANGE_VIOLATION_MAX_RATIO:.0%}"
            f"（clip={clip_label}）。超限过多，换 clip 重导（方案 R2）"
        )

    grasp, release, n_mismatch = extract_grasp_release(md)
    if n_mismatch:
        print(f"[WARN] NaN 模式与 per_link_contact_flags 有 {n_mismatch} 帧不一致（以 NaN 模式为准）")

    scale = float(md["object_scale"])
    half_raw, center_raw = parse_obj_aabb(obj_path)
    half = [h * scale for h in half_raw]
    center_off = [c * scale for c in center_raw]

    z_release = float(obj_pos[release, 2] + center_off[2])
    z_lift_peak = float(obj_pos[:, 2].max() + center_off[2] - (obj_pos[0, 2] + center_off[2]))
    pick_pos = [float(v) for v in obj_pos[grasp]]

    joint_names = list(g1_joints.keys())
    doc = {
        "format": "g1-box-motion/1",
        "source_clip": clip_name or os.path.basename(str(meta.get("source_file", pkl_path))),
        "pipeline": f"hoi-retarget BSD-3, object_scale={scale:.2f}, stage={meta.get('stage', '?')}",
        "fps": int(round(fps)),
        "T": T,
        "box": {
            "type": infer_box_type(obj_path),
            "half_size": [round(v, 4) for v in half],
            "bbox_center_offset": [round(v, 4) for v in center_off],
            "mass": mass,
        },
        "grasp_frame": grasp,
        "release_frame": release,
        "z_release": round(z_release, 4),
        "z_lift_peak": round(z_lift_peak, 4),
        "pick_pos": [round(v, 4) for v in pick_pos],
        "joint_names": joint_names,
        "root_pos": [[round(v, 4) for v in row] for row in root_pos],
        "root_quat": [[round(v, 5) for v in row] for row in root_quat],
        "dof_pos": [[round(v, 4) for v in row] for row in dof_pos],
        "obj_pos": [[round(v, 4) for v in row] for row in obj_pos],
        "obj_quat": [[round(v, 5) for v in row] for row in obj_quat],
        "contact_targets": {
            "links": [n for n in md["contact_link_names"] if PALM_SUBSTR in n],
            "object_frame_points": [
                [[round(float(v), 5) if np.isfinite(v) else None for v in point] for point in row]
                for row in np.asarray(md["fixed_contact_points_per_frame_in_object_frame"])[:,
                    [i for i, n in enumerate(md["contact_link_names"]) if PALM_SUBSTR in n]]
            ],
        },
        "contact": [int(v) for v in np.asarray(md["per_link_contact_flags"])[:, [i for i, n in enumerate(md["contact_link_names"]) if PALM_SUBSTR in n]].any(axis=1)],
    }

    if fk:
        fk_check_and_embed(doc, md, g1_xml)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    size = os.path.getsize(out_path)

    print(f"[OK] {out_path}  T={T} grasp={grasp} release={release} z_release={z_release:.3f} "
          f"z_lift_peak={z_lift_peak:.3f} clamp={n_clamped}({ratio:.1%}) size={size/1024:.0f}KB")
    if size > 200 * 1024:
        print(f"[WARN] 体积 {size/1024:.0f}KB 超 200KB 预算（R6：降到 15fps）")
    return doc


# ---------------- --scan 候选表 ----------------
def scan(outputs_dir: str, g1_xml: str | None) -> None:
    """扫 outputs/*/{contact,kinematic}_window.pkl，打候选表（clip、箱型、z_lift 峰值、z_release、grasp/release）。"""
    pkl_paths = sorted(glob.glob(os.path.join(outputs_dir, "*", "contact_window.pkl")))
    pkl_paths += sorted(glob.glob(os.path.join(outputs_dir, "*", "kinematic_window.pkl")))
    if not pkl_paths:
        raise SystemExit(f"{outputs_dir} 下没有任何 window pkl")

    print(f"{'clip':34s} {'stage':10s} {'box':10s} {'T':>4s} {'grasp':>5s} {'release':>7s} "
          f"{'z_release':>9s} {'z_lift_pk':>9s} {'pick_z':>6s}")
    for p in pkl_paths:
        clip = os.path.basename(os.path.dirname(p))
        stage = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p, "rb") as f:
                md = pickle.load(f)
            if md.get("meta", {}).get("schema_version") != 2:
                print(f"{clip:34s} {stage:10s}  SKIP schema={md.get('meta', {}).get('schema_version')}")
                continue
            names = list(md["contact_link_names"])
            palm_idx = [i for i, n in enumerate(names) if PALM_SUBSTR in n]
            pts = np.asarray(md["fixed_contact_points_per_frame_in_object_frame"])[:, palm_idx, :]
            touched = ~np.isnan(pts).any(axis=(1, 2))
            frames = np.nonzero(touched)[0]
            if len(frames) == 0:
                print(f"{clip:34s} {stage:10s}  无 palm 接触帧")
                continue
            grasp, release = int(frames[0]), int(frames[-1])
            obj_pos = np.asarray(md["object_pos"], dtype=np.float64)
            scale = float(md["object_scale"])
            obj_path = md.get("object_model_path", "")
            # 箱型：从 object_model_path 推断；z 用包围盒中心偏移修正
            box_type = "unknown"
            off_z = 0.0
            for t in ("largebox", "plasticbox", "smallbox"):
                if t in obj_path:
                    box_type = t
                    break
            obj_stem = os.path.splitext(obj_path)[0]
            candidates = glob.glob(os.path.join(os.path.dirname(os.path.dirname(obj_path)) or "/nonexistent", "objects", box_type, f"{box_type}.obj"))
            if candidates:
                _, center_raw = parse_obj_aabb(candidates[0])
                off_z = center_raw[2] * scale
            z_release = obj_pos[release, 2] + off_z
            z_lift = obj_pos[:, 2].max() + off_z - (obj_pos[0, 2] + off_z)
            print(f"{clip:34s} {stage:10s} {box_type:10s} {obj_pos.shape[0]:4d} {grasp:5d} {release:7d} "
                  f"{z_release:9.3f} {z_lift:9.3f} {obj_pos[grasp, 2] + off_z:6.3f}")
        except Exception as exc:
            print(f"{clip:34s} {stage:10s}  FAIL {type(exc).__name__}: {exc}")

    print("\n选 clip 提示（方案 4.1）：layer1 槽位中心 z=hz(layer1), layer2 z=3*hz；")
    print("  clip_low: z_release ≈ 0.15–0.28；clip_mid: z_release ≈ 0.40–0.60 且 z_lift_pk ≥ 槽位+0.2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", help="stage pkl（首选 contact_window.pkl）")
    ap.add_argument("--g1", default=None, help="展示端 model/g1.xml（限位断言）")
    ap.add_argument("--obj-obj", dest="obj_obj", default=None, help="物体 OBJ（算包围盒）")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--mass", type=float, default=3.0, help="箱体质量 kg（演示值，A4）")
    ap.add_argument("--clip", default=None, help="source_clip 名（默认取 pkl meta）")
    ap.add_argument("--no-fk", dest="fk", action="store_false", help="跳过 FK 交叉验证与 fk_ref 嵌入")
    ap.add_argument("--scan", metavar="DIR", default=None, help="批量扫 DIR/*/*.pkl 打候选表后退出")
    args = ap.parse_args()

    if args.scan:
        scan(args.scan, args.g1)
        return

    if not (args.pkl and args.obj_obj and args.out):
        ap.error("--pkl/--obj-obj/--out 必填（或用 --scan DIR）")
    g1_xml = args.g1 or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "model", "g1.xml")
    export(args.pkl, g1_xml, args.obj_obj, args.out, args.mass, args.clip, fk=args.fk)


if __name__ == "__main__":
    main()
