#!/usr/bin/env python3
# 开发期工具：官方 velocity 策略 ONNX → float32 权重二进制 + manifest.json + golden fixture。
# 运行期零 python 依赖；本脚本只在导出/复跑时使用（方案 docs/rl-walking-plan.md 5.2/8.1）。
#
# 用法：
#   python3 tools/pipeline/export_policy_weights.py --onnx /tmp/g1_velocity_v0.onnx --out vendor/policy/
#   （onnx 样本缺失时按 README「许可」节的 unitree_rl_lab raw 直链重新下载）
#
# 产物：
#   vendor/policy/g1_velocity_v0.weights.bin  各层 W(out×in float32) 与 b 逐层交错拼接
#   vendor/policy/manifest.json               单一事实源（装订参数由 T0 探针冻结）
#   vendor/policy/golden.bin                  64 组随机 obs + 1 组全零固定 obs 的参考前向
#
# 落盘前置校验：numpy 手工前向 vs onnxruntime 参考，max|Δ| < 1e-5（方案 5.2）。
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

POLICY_SEQ_JOINTS = [
    # 策略关节序 = Unitree USD 左右交错序（T0 定论，见 docs/policy-probe-notes.md）
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# T0 源码定论（deploy/robots/g1_29dof/src/State_RLBase.cpp: motor_cmd[map[i]] = action[i]；
# unitree_articulation.h: joint_pos[i] = motor_state[map[i]]）→ policy→mjcf 方向
JOINT_MAP_POLICY_TO_MJCF = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16,
                            23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

# deploy.yaml stiffness/damping：SDK/MJCF 执行器序（State_RLBase.h: motor_cmd[i].kp = stiffness[i]）
GAINS_MJCF = {
    "kp": [100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
           100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
           200.0, 200.0, 200.0] + [40.0] * 14,
    "kd": [2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
           2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
           5.0, 5.0, 5.0] + [10.0] * 14,
}

# deploy.yaml default_joint_pos：策略序（obs joint_pos_rel 按 policy 序相减）
DEFAULT_JOINT_POS_POLICY = [-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3,
                            0.3, 0.3, -0.2, -0.2, 0.25, -0.25, 0.0, 0.0, 0.0, 0.0,
                            0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0]


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


def read_gemm_layers(onnx_path):
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(onnx_path))
    inits = {i.name: i for i in model.graph.initializer}
    layers = []
    for node in model.graph.node:
        if node.op_type == "Gemm":
            w = numpy_helper.to_array(inits[node.input[1]]).astype(np.float32)
            b = numpy_helper.to_array(inits[node.input[2]]).astype(np.float32)
            if node.input[1] not in inits or node.input[2] not in inits:
                raise SystemExit(f"Gemm 节点 {node.name} 的权重不是 initializer（不支持）")
            layers.append((w, b))
    if len(layers) != 4:
        raise SystemExit(f"期望 4 层 Gemm，实际 {len(layers)} 层")
    return layers


def forward_np(layers, obs):
    x = np.asarray(obs, dtype=np.float32)
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1:
            x = elu(x)
    return x.astype(np.float32)


def ort_forward(onnx_path, obs_batch):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = []
    for obs in obs_batch:
        o = sess.run(None, {"obs": obs.reshape(1, -1).astype(np.float32)})[0]
        out.append(o.reshape(-1))
    return np.stack(out)


def sample_obs(rng, n):
    # 按官方 deploy obs 组装逻辑的物理范围采样（term 优先拼接，5 帧历史）。
    # golden 校验的是 MLP 前向数值一致性（N1）；采样分布取各 term 的物理典型域。
    obs = np.zeros((n, 480), dtype=np.float32)
    ranges_per_term = [
        (3, 0.2, (-3.0, 3.0)),      # base_ang_vel: |ω|≤3 rad/s ×0.2
        (3, 1.0, (-1.0, 1.0)),      # projected_gravity
        (3, 1.0, None),             # velocity_commands（专项采样）
        (29, 1.0, (-1.5, 1.5)),     # joint_pos_rel
        (29, 0.05, (-40.0, 40.0)),  # joint_vel_rel: |dq|≤40 rad/s ×0.05
        (29, 1.0, (-1.0, 1.0)),     # last_action（raw，±1 典型域）
    ]
    off = 0
    for dim, scale, rng_spec in ranges_per_term:
        for _ in range(5):  # history_length 5，旧→新
            if dim == 3 and scale == 1.0 and rng_spec is None:
                # velocity_commands: vx∈[-0.5,1.0], vy∈[-0.3,0.3], wz∈[-0.2,0.2]
                obs[:, off + 0] = rng.uniform(-0.5, 1.0, n).astype(np.float32)
                obs[:, off + 1] = rng.uniform(-0.3, 0.3, n).astype(np.float32)
                obs[:, off + 2] = rng.uniform(-0.2, 0.2, n).astype(np.float32)
            else:
                lo, hi = rng_spec if rng_spec else (-1.0, 1.0)
                obs[:, off:off + dim] = rng_uniform(rng, lo, hi, n, dim, scale)
            off += dim
    assert off == 480
    return obs


def rng_uniform(rng, lo, hi, n, dim, scale):
    # 采样乘 scale 后落在 [lo*scale, hi*scale]（与 term 乘法顺序等价，分布一致）
    return (rng.uniform(lo, hi, (n, dim)) * scale).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True, help="官方导出 policy.onnx 路径")
    ap.add_argument("--out", required=True, help="输出目录（vendor/policy/）")
    ap.add_argument("--commit", default="4960b84732b0c2ec593dccbfe963fda1bcd7b1e3",
                    help="unitree_rl_lab 导出 commit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = read_gemm_layers(args.onnx)
    dims = [layers[0][0].shape[1]] + [w.shape[0] for w, _ in layers]
    print(f"层结构: {dims}（{len(layers)} 层 Gemm + ELU）")

    # ---- 落盘前置校验：numpy 手工前向 vs onnxruntime ----
    rng = np.random.default_rng(args.seed)
    obs64 = sample_obs(rng, 64)
    zero_obs = np.zeros((1, 480), dtype=np.float32)
    obs_all = np.concatenate([zero_obs, obs64], axis=0)
    ref = ort_forward(args.onnx, obs_all)
    mine = forward_np(layers, obs_all)
    max_err = float(np.max(np.abs(ref - mine)))
    print(f"numpy vs onnxruntime: max|Δ| = {max_err:.3e}")
    if max_err >= 1e-5:
        raise SystemExit(f"权重抽取校验失败 max|Δ|={max_err:.3e} >= 1e-5，未落盘")

    # ---- weights.bin：逐层 W(out×in) 与 b 交错 ----
    parts = []
    for w, b in layers:
        parts.append(np.ascontiguousarray(w, dtype="<f4").tobytes())
        parts.append(np.ascontiguousarray(b, dtype="<f4").tobytes())
    weights_bin = b"".join(parts)
    weights_path = out_dir / "g1_velocity_v0.weights.bin"
    weights_path.write_bytes(weights_bin)
    print(f"写出 {weights_path}（{len(weights_bin)} 字节）")

    # ---- golden.bin：int32 count + count 组交错 [obs 480][action 29] float32 ----
    actions = ref.astype(np.float32)  # 与 obs_all 同序（[零向量, 64 随机]）
    count = obs_all.shape[0]
    rows = np.concatenate([obs_all, actions], axis=1)  # (count, 509) 按组交错
    golden = struct.pack("<i", count) + np.ascontiguousarray(rows, dtype="<f4").tobytes()
    golden_path = out_dir / "golden.bin"
    golden_path.write_bytes(golden)
    print(f"写出 {golden_path}（{len(golden)} 字节，{count} 组）")

    # ---- manifest.json：单一事实源（装订参数为 T0 源码定论 + 探针冻结） ----
    manifest = {
        "source": "unitreerobotics/unitree_rl_lab",
        "path": "deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx",
        "license": "Apache-2.0",
        "commit": args.commit,
        "files": {"weights": weights_path.name, "golden": golden_path.name},
        "arch": {"layers": dims, "act": "elu",
                 "layout": "per-layer W(out×in, row-major) then b, float32 LE concatenated"},
        "obs": {
            "frame": 96, "history": 5, "order": "term-major-oldest-first",
            "order_note": "deploy ObservationManager 默认 use_gym_history=false：term 外层，term 内 5 帧旧→新",
            "terms": [
                {"name": "base_ang_vel", "dim": 3, "scale": 0.2, "frame": "body"},
                {"name": "projected_gravity", "dim": 3, "scale": 1.0, "frame": "body"},
                {"name": "velocity_commands", "dim": 3, "scale": 1.0},
                {"name": "joint_pos_rel", "dim": 29, "scale": 1.0, "order": "policy"},
                {"name": "joint_vel_rel", "dim": 29, "scale": 0.05, "order": "policy"},
                {"name": "last_action", "dim": 29, "scale": 1.0, "note": "raw 网络输出（未乘 scale/加 offset）"},
            ],
        },
        "action": {
            "scale": 0.25, "default_joint_pos": DEFAULT_JOINT_POS_POLICY,
            "default_joint_pos_order": "policy",
            "clip": "mjcf range", "clip_note": "官方 clip=null；我方按 MJCF 关节 range 限幅（安全冗余）",
        },
        "joint_map": {
            "policy_to_mjcf": JOINT_MAP_POLICY_TO_MJCF,
            "direction": "verified",
            "direction_note": "T0 定论：motor_cmd[map[i]]=action[i] 与 joint_pos[i]=motor_state[map[i]]（deploy 源码）",
            "policy_joint_names": POLICY_SEQ_JOINTS,
        },
        "gains": {"kp": GAINS_MJCF["kp"], "kd": GAINS_MJCF["kd"], "order": "mjcf-actuator"},
        "cmd_limits": {"vx": [-0.5, 1.0], "vy": [-0.3, 0.3], "wz": [-0.2, 0.2],
                       "note": "训练 UniformLevelVelocityCommand limit_ranges；演示限幅须在此域内"},
        "control": {"step_dt": 0.02, "train_sim_dt": 0.005, "train_decimation": 4,
                    "train_armature": 0.01,
                    "note": "本地物理子步 0.002s（500Hz）高于训练 200Hz；armature 与训练同级 0.01，XML 不改"},
        "export": {"weights_sha256": None, "golden_groups": count, "seed": args.seed,
                   "max_abs_err_vs_ort": max_err},
    }
    import hashlib
    manifest["export"]["weights_sha256"] = hashlib.sha256(weights_bin).hexdigest()
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"写出 {manifest_path}")

    total = weights_path.stat().st_size + golden_path.stat().st_size + manifest_path.stat().st_size
    print(f"合计体积: {total} 字节（预算 ≤3MB 含 LICENSE-NOTE）")


if __name__ == "__main__":
    sys.exit(main())
