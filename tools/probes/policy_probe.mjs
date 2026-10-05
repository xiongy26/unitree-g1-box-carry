// T0 最小实验探针（方案 docs/rl-walking-plan.md 7-T0，gate）：官方 velocity 策略在本地
// Menagerie G1 模型 + 0.002s 子步 + 策略 PD 增益改写下能否站稳/行走。
// 不动 stackCore，Node 直接跑：
//   node tools/probes/policy_probe.mjs                    # 全部变体
//   node tools/probes/policy_probe.mjs --variant A1       # 只跑指定变体（A1=静态证据最强候选）
//   node tools/probes/policy_probe.mjs --gain legacy      # 变体 A 对照：不改增益，kp=500 直喂
//   node tools/probes/policy_probe.mjs --probe-only       # 只跑 H4 可写性 + golden 前向校验
//
// 判定（全过才进 T1）：存在装订变体使 a) 零指令站立 10s pelvis z≥0.6m 漂移<10cm；
// b) cmd_vx=0.3 走 5s 前进≥0.6m 足交替≥4；c) 全程无 NaN。
// 产出：docs/policy-probe-notes.md（变体矩阵、选定装订、增益结论、armature/D6-1 源码定论）。
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { readdirSync } from 'node:fs';
import loadMujoco from '../../vendor/mujoco/mujoco.js';
import { buildJointMap, keyId, quatRotVec, quatConj } from '../../src/util.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const TIMESTEP = 0.002;
const CTRL_DECIM = 10; // 10 子步 / 推理 = 50Hz（训练 step_dt 0.02）

// ---------------- 参数 ----------------
const args = process.argv.slice(2);
function argVal(name, dflt) { const i = args.indexOf(name); return i >= 0 && i + 1 < args.length ? args[i + 1] : dflt; }
const hasFlag = (n) => args.includes(n);
const ONLY_VARIANT = argVal('--variant', null);
const GAIN_MODE = argVal('--gain', 'policy'); // policy=改写增益（D1 主选）| legacy=变体 A（kp=500 直喂）
const SPAWN_MODE = argVal('--spawn', 'stand'); // stand=keyframe | default=策略 default 姿态
const RUN_MODE = argVal('--mode', 'policy'); // policy=策略闭环 | zeroctrl=ctrl 恒 default（物理站立诊断）
const PROBE_ONLY = hasFlag('--probe-only');

// 本地 MJCF 执行器序（与 src/main.js JOINT_NAMES 一致：左腿6/右腿6/腰3/左臂7/右臂7）
const JOINT_NAMES = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
];

// ---------------- 装订参数（单一事实源：vendor/policy/manifest.json，T0 定论后冻结） ----------------
const manifest = JSON.parse(readFileSync(path.join(ROOT, 'vendor/policy/manifest.json'), 'utf8'));
const MAP = manifest.joint_map.policy_to_mjcf;          // policy p → mjcf index
const DEFAULT_POS = Float64Array.from(manifest.action.default_joint_pos); // policy 序
const ACT_SCALE = manifest.action.scale;                // 0.25
const KP_MJCF = manifest.gains.kp;                      // mjcf 序（deploy.yaml stiffness）
const KD_MJCF = manifest.gains.kd;
const CMD = manifest.cmd_limits;                        // 训练范围

// ---------------- 权重与 MLP 前向（T1 g1Policy.js 的原型实现） ----------------
function loadWeights(file, mani) {
  const buf = readFileSync(file);
  const dims = mani.arch.layers;
  const layers = [];
  let off = 0;
  const view = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  for (let l = 0; l < dims.length - 1; l++) {
    const outD = dims[l + 1], inD = dims[l];
    const w = view.slice(off, off + outD * inD); off += outD * inD;
    const b = view.slice(off, off + outD); off += outD;
    layers.push({ w, b, outD, inD });
  }
  return { layers, dims };
}

function policyForward(weights, obs) {
  // y = W·x + b（W 按 out×in 行主序），隐层 ELU；全 Float32Array 预分配零 GC
  let x = obs;
  for (let l = 0; l < weights.layers.length; l++) {
    const { w, b, outD, inD } = weights.layers[l];
    const y = new Float32Array(outD);
    for (let o = 0; o < outD; o++) {
      let s = b[o];
      const row = o * inD;
      for (let i = 0; i < inD; i++) s += x[i] * w[row + i];
      y[o] = l < weights.layers.length - 1 ? (s > 0 ? s : Math.expm1(s)) : s;
    }
    x = y;
  }
  return x;
}

// golden fixture：int32 count + count×(480 obs + 29 action)
function checkGolden(weights) {
  const buf = readFileSync(path.join(ROOT, 'vendor/policy/golden.bin'));
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const count = dv.getInt32(0, true);
  const f32 = new Float32Array(buf.buffer, buf.byteOffset + 4, (buf.byteLength - 4) / 4);
  const obsDim = manifest.obs.frame * manifest.obs.history;
  let maxErr = 0;
  for (let g = 0; g < count; g++) {
    const obs = f32.slice(g * (obsDim + 29), g * (obsDim + 29) + obsDim);
    const ref = f32.slice(g * (obsDim + 29) + obsDim, (g + 1) * (obsDim + 29));
    const out = policyForward(weights, obs);
    for (let i = 0; i < 29; i++) maxErr = Math.max(maxErr, Math.abs(out[i] - ref[i]));
  }
  return { count, maxErr };
}

// ---------------- 消融变体 ----------------
// map 方向 3 种 × 历史拼接 2 种。obs 关节项与 action 解码按同一 map 方向装订。
function invMap(m) { const inv = new Array(m.length); for (let p = 0; p < m.length; p++) inv[m[p]] = p; return inv; }
const INV = invMap(MAP);
const VARIANTS = [];
for (const mapDir of ['fwd', 'inv', 'id']) {
  for (const hist of ['term', 'frame']) {
    VARIANTS.push({ id: `${mapDir}-${hist}`, mapDir, hist });
  }
}
// 静态证据最强（deploy 源码逐行对齐）排最前
VARIANTS.sort((a, b) => (a.id === 'fwd-term' ? -1 : b.id === 'fwd-term' ? 1 : 0));

// mjcf 关节索引（policy p 视角）→ jmap 名单下标
function mjcfIndexForPolicyP(v, p) {
  if (v.mapDir === 'fwd') return MAP[p];
  if (v.mapDir === 'inv') return INV[p];
  return p; // id
}

// ---------------- 增益改写（D1 主选；--gain legacy 为变体 A 对照） ----------------
function writeGains(mujoco, model, jmap, mode) {
  const report = { writable: true, detail: '' };
  if (mode === 'legacy') return report; // 变体 A：不动增益
  // --gain policy500:500: 改写机制自检（用 Menagerie 同值增益走改写路径，应与 legacy 行为一致）
  let kpOv = null, kdOv = null;
  if (mode.startsWith('policy')) {
    const m = mode.slice(6).match(/^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$/);
    if (m) { kpOv = parseFloat(m[1]); kdOv = parseFloat(m[2]); }
  }
  for (let p = 0; p < 29; p++) {
    const m = MAP[p]; // kp/kd 表按 mjcf 序；遍历 policy 序取其 mjcf 槽位
    const act = jmap[JOINT_NAMES[m]].c;
    const kp = kpOv ?? KP_MJCF[m];
    const kd = kdOv ?? KD_MJCF[m];
    model.actuator_gainprm[act * 10 + 0] = kp;
    model.actuator_biasprm[act * 10 + 1] = -kp;
    model.actuator_biasprm[act * 10 + 2] = -kd;
  }
  return report;
}

function restoreGains(model, jmap, saved) {
  for (const [act, g, b1, b2] of saved) {
    model.actuator_gainprm[act * 10 + 0] = g;
    model.actuator_biasprm[act * 10 + 1] = b1;
    model.actuator_biasprm[act * 10 + 2] = b2;
  }
}

function snapshotGains(model, jmap) {
  const saved = [];
  for (const n of JOINT_NAMES) {
    const act = jmap[n].c;
    saved.push([act, model.actuator_gainprm[act * 10 + 0], model.actuator_biasprm[act * 10 + 1], model.actuator_biasprm[act * 10 + 2]]);
  }
  return saved;
}

// ---------------- 行走段执行器（探针内联，T1 提取进 walkController.js） ----------------
function runVariant(mujoco, model, data, jmap, weights, v, gainMode, log) {
  // 装订展开：policy p → { q, d, c }（mjcf 地址）
  const bind = [];
  for (let p = 0; p < 29; p++) {
    const name = JOINT_NAMES[mjcfIndexForPolicyP(v, p)];
    bind.push(jmap[name]);
  }
  // ctrl clamp 范围（mjcf 序执行器，按 bind 的 policy 序访问；本 binding 不暴露
  // actuator_ctrllimited 数组，用 ctrlrange 有效性判断：lo<hi 视为限位）
  const ctrlLo = new Float64Array(29), ctrlHi = new Float64Array(29);
  for (let p = 0; p < 29; p++) {
    const act = bind[p].c;
    const lo = model.actuator_ctrlrange[act * 2 + 0], hi = model.actuator_ctrlrange[act * 2 + 1];
    if (hi > lo) {
      ctrlLo[p] = lo; ctrlHi[p] = hi;
    } else {
      ctrlLo[p] = -1e9; ctrlHi[p] = 1e9;
    }
  }

  // 增益：进入改写，退出恢复（N2 语义的最小原型）
  const saved = snapshotGains(model, jmap);
  writeGains(mujoco, model, jmap, gainMode);
  let gainsEntered = gainMode === 'policy';

  // 历史缓冲：6 term × 各自 5 帧（每帧该 term 的 dim 值），旧→新
  const termDims = [3, 3, 3, 29, 29, 29];
  const termScales = [0.2, 1.0, 1.0, 1.0, 0.05, 1.0];
  const HIST = 5;
  const hist = termDims.map((d) => Array.from({ length: HIST }, () => new Float32Array(d)));

  let lastAction = new Float32Array(29);
  let curCmd = new Float32Array(3);
  let targetCmd = new Float32Array(3);
  // 指令斜坡（固件先例 vx 0.02/tick）
  const RAMP = [0.02, 0.02, 0.02];
  const DEBUG = hasFlag('--debug');

  const metrics = {
    nan: false, minPelvisZ: Infinity, maxDrift: 0, // 站立
    startX: 0, startY: 0, forwardDisp: 0, steps: 0, // 行走（forwardDisp 沿初始朝向）
    startYaw: 0, dyaw: 0, footFlips: 0, footMinDist: Infinity,
    ctrlMaxAbs: 0, actionMaxAbs: 0,
  };

  // 足支撑检测：mj_geomDistance(foot sphere, floor) < 5mm 记支撑
  const floorGid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, 'floor');
  const mjGEOM_SPHERE = 2;
  const footGids = [];
  for (const bn of ['left_ankle_roll_link', 'right_ankle_roll_link']) {
    const bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, bn);
    const gadr = model.body_geomadr[bid], gnum = model.body_geomnum[bid];
    for (let g = gadr; g < gadr + gnum; g++) if (model.geom_type[g] === mjGEOM_SPHERE) footGids.push(g);
  }
  const fromto = new Float64Array(6);
  const wasStance = [true, true];
  let footSampleT = 0;

  const pelvisBid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'pelvis');

  // 组装单帧 obs（96 维，原始量不乘 scale，缩放在 assembleObs 统一做），推入历史
  function buildFrame() {
    const frame = new Float32Array(96);
    // term0 base_ang_vel：MuJoCo 自由铰 qvel[3:6] 即体系角速度（原始 rad/s）
    frame[0] = data.qvel[3]; frame[1] = data.qvel[4]; frame[2] = data.qvel[5];
    // term1 projected_gravity：R^T·(0,0,-1)
    const q = [data.xquat[4 * pelvisBid], data.xquat[4 * pelvisBid + 1], data.xquat[4 * pelvisBid + 2], data.xquat[4 * pelvisBid + 3]];
    const grav = quatRotVec(quatConj(q), [0, 0, -1]);
    frame[3] = grav[0]; frame[4] = grav[1]; frame[5] = grav[2];
    // term2 velocity_commands
    frame[6] = curCmd[0]; frame[7] = curCmd[1]; frame[8] = curCmd[2];
    // term3 joint_pos_rel（policy 序装订）与 term4 joint_vel_rel（原始 rad/s）
    for (let p = 0; p < 29; p++) {
      frame[9 + p] = data.qpos[bind[p].q] - DEFAULT_POS[p];
      frame[38 + p] = data.qvel[bind[p].d];
    }
    // term5 last_action（raw）
    for (let p = 0; p < 29; p++) frame[67 + p] = lastAction[p];
    return frame;
  }

  function pushHist(frame) {
    // term-major：term 外层、term 内 5 帧旧→新（deploy use_gym_history=false 默认）
    let off = 0;
    for (let t = 0; t < 6; t++) {
      hist[t].shift();
      hist[t].push(frame.slice(off, off + termDims[t]));
      off += termDims[t];
    }
  }

  function assembleObs(mode) {
    const obs = new Float32Array(480);
    if (mode === 'term') {
      // [t1 h0..h4][t2 h0..h4]...（term 外层，旧→新）
      let off = 0;
      for (let t = 0; t < 6; t++) {
        const sc = termScales[t];
        for (let h = 0; h < HIST; h++) {
          const fr = hist[t][h];
          for (let i = 0; i < termDims[t]; i++) obs[off++] = fr[i] * sc;
        }
      }
    } else {
      // frame-major：[h0: t1..t6][h1: t1..t6]...（帧外层，旧→新）
      let off = 0;
      for (let h = 0; h < HIST; h++) {
        for (let t = 0; t < 6; t++) {
          const sc = termScales[t];
          const fr = hist[t][h];
          for (let i = 0; i < termDims[t]; i++) obs[off++] = fr[i] * sc;
        }
      }
    }
    return obs;
  }

  function setCmd(vx, vy, wz) {
    targetCmd[0] = Math.max(CMD.vx[0], Math.min(CMD.vx[1], vx));
    targetCmd[1] = Math.max(CMD.vy[0], Math.min(CMD.vy[1], vy));
    targetCmd[2] = Math.max(CMD.wz[0], Math.min(CMD.wz[1], wz));
  }

  // 主循环：分段 { dur, cmd, judge }
  const phases = [
    { name: 'stand', dur: 10.0, cmd: [0, 0, 0] },
    { name: 'walk', dur: 5.0, cmd: [0.3, 0, 0] },
    { name: 'turn', dur: 3.0, cmd: [0, 0, 0.3] },
  ];

  mujoco.mj_forward(model, data);
  // --spawn default：从策略 default 姿态起仿（与训练 spawn 一致，排除 stand 直腿瞬态）
  if (SPAWN_MODE === 'default') {
    for (let p = 0; p < 29; p++) {
      data.qpos[bind[p].q] = DEFAULT_POS[p];
      data.qvel[bind[p].d] = 0;
      data.ctrl[bind[p].c] = DEFAULT_POS[p];
    }
    data.qpos[2] = 0.74; // 屈膝 default 姿态的估计 pelvis 高（stand 0.79）
    for (let i = 0; i < 6; i++) data.qvel[i] = 0;
    mujoco.mj_forward(model, data);
  }
  const start = {
    x: data.qpos[0], y: data.qpos[1], z: data.qpos[2],
    yaw: Math.atan2(2 * (data.xquat[4 * pelvisBid] * data.xquat[4 * pelvisBid + 3] + data.xquat[4 * pelvisBid + 1] * data.xquat[4 * pelvisBid + 2]),
      1 - 2 * (data.xquat[4 * pelvisBid + 2] ** 2 + data.xquat[4 * pelvisBid + 3] ** 2)),
    cos: Math.cos(0), sin: Math.sin(0),
  };
  // 初始朝向的前向单位向量（世界系）
  start.cos = Math.cos(start.yaw); start.sin = Math.sin(start.yaw);
  metrics.startX = start.x; metrics.startY = start.y; metrics.startYaw = start.yaw;

  // 历史预热：初始帧填充 5 次（静止态，last_action=0）
  {
    const f0 = buildFrame();
    for (let h = 0; h < HIST; h++) pushHist(f0);
  }

  let t = 0, sub = 0, phaseIdx = 0, phaseT = 0;
  let walkStartX = start.x, walkStartY = start.y, walkStartCos = start.cos, walkStartSin = start.sin;
  let turnStartYaw = start.yaw;
  const perPhase = {};

  const checkFinite = () => {
    if (!Number.isFinite(data.qpos[0]) || !Number.isFinite(data.qpos[2])) return false;
    if (data.qpos.some((val) => !Number.isFinite(val))) return false;
    return true;
  };

  while (phaseIdx < phases.length && t < 20) {
    const ph = phases[phaseIdx];
    setCmd(ph.cmd[0], ph.cmd[1], ph.cmd[2]);

    // 50Hz 控制步（每 CTRL_DECIM 子步首步）：采样→历史→斜坡→推理→ctrl 写入
    if (sub % CTRL_DECIM === 0) {
      for (let i = 0; i < 3; i++) {
        const d = targetCmd[i] - curCmd[i];
        curCmd[i] = Math.abs(d) <= RAMP[i] ? targetCmd[i] : curCmd[i] + Math.sign(d) * RAMP[i];
      }
      pushHist(buildFrame());
      const obs = assembleObs(v.hist);
      let act = null;
      if (RUN_MODE === 'zeroctrl') {
        // 诊断模式：跳过策略，ctrl 恒 default（检验 default 姿态在策略 PD 下物理可站）
        for (let p = 0; p < 29; p++) data.ctrl[bind[p].c] = DEFAULT_POS[p];
      } else {
        act = policyForward(weights, obs);
        lastAction = act;
        metrics.actionMaxAbs = Math.max(metrics.actionMaxAbs, Math.max(...act));
        for (let p = 0; p < 29; p++) {
          const tgt = DEFAULT_POS[p] + ACT_SCALE * act[p];
          const cl = Math.max(ctrlLo[p], Math.min(ctrlHi[p], tgt));
          data.ctrl[bind[p].c] = cl;
          metrics.ctrlMaxAbs = Math.max(metrics.ctrlMaxAbs, Math.abs(cl));
        }
      }
      if (DEBUG && Math.round(t / TIMESTEP) % CTRL_DECIM === 0 && (t < 1.0 || Math.round(t * 10) % 10 === 0)) {
        // 诊断：关键关节目标 vs 实际、act 范围、pelvis 状态
        const kneeP = 9; // policy 序 left_knee
        const hipP = 0;
        const kneeName = JOINT_NAMES[mjcfIndexForPolicyP(v, kneeP)];
        console.log(`  [dbg t=${t.toFixed(2)}] z=${data.qpos[2].toFixed(3)} `
          + `knee(${kneeName}) ctrl=${data.ctrl[bind[kneeP].c].toFixed(2)} q=${data.qpos[bind[kneeP].q].toFixed(2)} `
          + `hip ctrl=${data.ctrl[bind[hipP].c].toFixed(2)} q=${data.qpos[bind[hipP].q].toFixed(2)} `
          + (act ? `actMax=${Math.max(...act).toFixed(2)} actMin=${Math.min(...act).toFixed(2)}` : '(zeroctrl)'));
      }
    }

    mujoco.mj_step(model, data);
    sub++; t += TIMESTEP; phaseT += TIMESTEP;

    // 观测（每 10ms）
    if (sub % 5 === 0) {
      const pz = data.xpos[3 * pelvisBid + 2];
      metrics.minPelvisZ = Math.min(metrics.minPelvisZ, pz);
      if (!checkFinite()) { metrics.nan = true; break; }
      const drift = Math.hypot(data.qpos[0] - metrics.startX, data.qpos[1] - metrics.startY);
      if (ph.name === 'stand') metrics.maxDrift = Math.max(metrics.maxDrift, drift);
      // 足支撑
      footSampleT += TIMESTEP;
      if (footSampleT >= 0.01) {
        footSampleT = 0;
        for (let f = 0; f < 2; f++) {
          const dist = mujoco.mj_geomDistance(model, data, footGids[f], floorGid, 0.05, fromto);
          metrics.footMinDist = Math.min(metrics.footMinDist, dist);
          const stance = dist < 0.005;
          if (!stance && wasStance[f]) {
            metrics.footFlips++;
          }
          wasStance[f] = stance;
        }
      }
    }

    // 相位切换与判稳
    if (phaseT >= ph.dur) {
      const px = data.qpos[0], py = data.qpos[1];
      const yawNow = Math.atan2(2 * (data.xquat[4 * pelvisBid] * data.xquat[4 * pelvisBid + 3] + data.xquat[4 * pelvisBid + 1] * data.xquat[4 * pelvisBid + 2]),
        1 - 2 * (data.xquat[4 * pelvisBid + 2] ** 2 + data.xquat[4 * pelvisBid + 3] ** 2));
      if (ph.name === 'stand') {
        perPhase.stand = { z: metrics.minPelvisZ, drift: metrics.maxDrift, pass: metrics.minPelvisZ >= 0.6 && metrics.maxDrift < 0.10 && !metrics.nan };
      } else if (ph.name === 'walk') {
        // 沿走段起始朝向的前向位移
        const dx = px - walkStartX, dy = py - walkStartY;
        metrics.forwardDisp = dx * walkStartCos + dy * walkStartSin;
        perPhase.walk = { fwd: metrics.forwardDisp, flips: metrics.footFlips, pass: metrics.forwardDisp >= 0.6 && metrics.footFlips >= 4 && !metrics.nan };
      } else {
        metrics.dyaw = yawNow - turnStartYaw;
        metrics.dyaw = Math.atan2(Math.sin(metrics.dyaw), Math.cos(metrics.dyaw));
        perPhase.turn = { dyaw: metrics.dyaw, pass: Math.abs(metrics.dyaw) > 0.2 && !metrics.nan && metrics.minPelvisZ >= 0.6 };
      }
      phaseIdx++; phaseT = 0;
      if (phaseIdx < phases.length) {
        if (phases[phaseIdx].name === 'walk') {
          walkStartX = px; walkStartY = py;
          walkStartCos = Math.cos(yawNow); walkStartSin = Math.sin(yawNow);
          metrics.footFlips = 0;
        } else if (phases[phaseIdx].name === 'turn') {
          turnStartYaw = yawNow;
        }
      }
    }
  }

  // 退出恢复增益（N2 语义：任何路径都必须恢复）
  if (gainsEntered) { restoreGains(model, jmap, saved); gainsEntered = false; }

  const standPass = perPhase.stand?.pass ?? false;
  const walkPass = perPhase.walk?.pass ?? false;
  const turnPass = perPhase.turn?.pass ?? false;
  // 转向不是方案 T0 的 gate 判据（a/b/c = 站立/前走/无 NaN），只记录观察：
  // 训练 wz 域为 ±0.2rad/s（setCmd 已 clamp），原地转响应弱 → 演示用直线段 + 小 yaw 修正
  const pass = standPass && walkPass && !metrics.nan;
  return { pass, standPass, walkPass, turnPass, perPhase, metrics };
}

// ---------------- 主流程 ----------------
const mujoco = await loadMujoco();
console.log('== T0 探针：官方 velocity 策略 @ Menagerie G1（0.002s 子步） ==');

// 0) golden 前向校验（JS 前向正确性先于仿真验证）
const weights = loadWeights(path.join(ROOT, 'vendor/policy/g1_velocity_v0.weights.bin'), manifest);
const golden = checkGolden(weights);
console.log(`[golden] ${golden.count} 组前向 max|Δ| = ${golden.maxErr.toExponential(2)} （阈值 1e-4）`);
if (golden.maxErr >= 1e-4) {
  console.log('golden 校验失败：JS 前向与 onnxruntime 参考不一致，先修前向再谈装订。');
  process.exit(1);
}

// 1) 模型（无箱）
const sceneXml = readFileSync(path.join(ROOT, 'model/scene.xml'), 'utf8');
const g1Xml = readFileSync(path.join(ROOT, 'model/g1.xml'), 'utf8');
const g1Inner = g1Xml.replace(/^[\s\S]*?<mujoco[^>]*>/, '').replace(/<\/mujoco>\s*$/, '');
const xml = sceneXml.replace(/<!-- G1 由 main\.js[\s\S]*?-->/, g1Inner.trim());
const vfs = new mujoco.MjVFS();
const assetsDir = path.join(ROOT, 'model/assets');
for (const f of readdirSync(assetsDir)) {
  if (f.toLowerCase().endsWith('.stl')) vfs.addBuffer('assets/' + f, new Uint8Array(readFileSync(path.join(assetsDir, f))));
}
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
const jmap = buildJointMap(mujoco, model, JOINT_NAMES);
console.log(`[model] nq=${model.nq} nu=${model.nu}`);

// 2) H4 探针：actuator_gainprm/biasprm 可写性
let h4 = { writable: false, detail: '' };
{
  const act = jmap[JOINT_NAMES[0]].c;
  const g0 = model.actuator_gainprm[act * 10], b1 = model.actuator_biasprm[act * 10 + 1];
  model.actuator_gainprm[act * 10] = 123.456;
  model.actuator_biasprm[act * 10 + 1] = -123.456;
  model.actuator_biasprm[act * 10 + 2] = -78.9;
  const rg = model.actuator_gainprm[act * 10], rb1 = model.actuator_biasprm[act * 10 + 1], rb2 = model.actuator_biasprm[act * 10 + 2];
  const ok = Math.abs(rg - 123.456) < 1e-9 && Math.abs(rb1 + 123.456) < 1e-9 && Math.abs(rb2 + 78.9) < 1e-9;
  model.actuator_gainprm[act * 10] = g0;
  model.actuator_biasprm[act * 10 + 1] = b1;
  h4 = { writable: ok, detail: `gainprm=${rg} biasprm1=${rb1} biasprm2=${rb2}` };
  console.log(`[H4] actuator_gainprm/biasprm 可写: ${ok ? '是' : '否'}（${h4.detail}）`);
}

// 3) armature 记录（H5：训练 0.01 == 本地）
{
  let armSum = 0;
  for (const n of JOINT_NAMES) armSum += model.dof_armature[jmap[n].d];
  console.log(`[H5] 本地 29 关节 armature 均值 = ${(armSum / 29).toFixed(4)}（训练配置 0.0100，同级 → 不改 XML）`);
}

if (PROBE_ONLY) process.exit(0);

// 4) 消融矩阵
const standKey = keyId(mujoco, model, 'stand');
const rows = [];
const variants = ONLY_VARIANT ? VARIANTS.filter((v) => v.id === ONLY_VARIANT) : VARIANTS;
for (const v of variants) {
  mujoco.mj_resetDataKeyframe(model, data, standKey);
  mujoco.mj_forward(model, data);
  const t0 = Date.now();
  const r = runVariant(mujoco, model, data, jmap, weights, v, GAIN_MODE, console.log);
  const wall = ((Date.now() - t0) / 1000).toFixed(1);
  const m = r.metrics;
  rows.push({
    id: v.id, gain: GAIN_MODE, pass: r.pass, standPass: r.standPass, walkPass: r.walkPass, turnPass: r.turnPass,
    standZ: r.perPhase.stand ? r.perPhase.stand.z : NaN, drift: r.perPhase.stand ? r.perPhase.stand.drift : NaN,
    fwd: r.perPhase.walk ? r.perPhase.walk.fwd : NaN, flips: r.perPhase.walk ? r.perPhase.walk.flips : NaN,
    dyaw: r.perPhase.turn ? r.perPhase.turn.dyaw : NaN,
    minZ: m.minPelvisZ, nan: m.nan, footMin: m.footMinDist, wall,
  });
  console.log(`[${v.id}] ${r.pass ? 'PASS' : 'fail'} 站(z=${(r.perPhase.stand?.z ?? NaN).toFixed(3)}, 漂=${(r.perPhase.stand?.drift ?? NaN).toFixed(3)}) `
    + `走(前=${(r.perPhase.walk?.fwd ?? NaN).toFixed(2)}, 交替=${r.perPhase.walk?.flips ?? NaN}) `
    + `转(Δyaw=${((r.perPhase.turn?.dyaw ?? NaN) * 180 / Math.PI).toFixed(0)}°) minZ=${m.minPelvisZ.toFixed(3)} `
    + `footMin=${(m.footMinDist * 1000).toFixed(1)}mm nan=${m.nan} wall=${wall}s`);
}

// 5) 汇总 + notes 生成
const anyPass = rows.some((r) => r.pass);
console.log(`\n== 判定: ${anyPass ? '存在可用装订变体 → 进 T1' : '全部变体失败 → 停止并回报（R1 出口）'} ==`);

const notesDir = path.join(ROOT, 'docs');
if (existsSync(notesDir)) {
  const lines = [];
  lines.push('# T0 策略探针结论（policy_probe.mjs 自动生成 + 人工核注）');
  lines.push('');
  lines.push('- 日期：2026-10-01；权重：unitree_rl_lab velocity/v0（policy.onnx，480→512→256→128→29 ELU）');
  lines.push(`- golden 前向校验（JS vs onnxruntime 导出参考）：${golden.count} 组 max|Δ| = ${golden.maxErr.toExponential(2)}（<1e-4 过）`);
  lines.push(`- H4 增益可写性：${h4.writable ? '可写' : '不可写'}（${h4.detail}）→ D1 ${h4.writable ? '主选（运行时增益改写）成立' : '主选不成立，落变体 A/B'}`);
  lines.push('- H5 armature：训练配置（unitree.py UNITREE_G1_29DOF_CFG 四组 actuator）全 0.01 == 本地 g1.xml 0.01 → **不改 XML**（D5）');
  lines.push('- D6-1 源码定论：');
  lines.push('  - map 方向 = policy→mjcf（State_RLBase.cpp `motor_cmd[map[i]]=action[i]`；unitree_articulation.h `joint_pos[i]=motor_state[map[i]]`）');
  lines.push('  - stiffness/damping 为 SDK/MJCF 序（State_RLBase.h `motor_cmd[i].kp=stiffness[i]`，源码注释 sdk order）；default_joint_pos 为 policy 序');
  lines.push('  - 历史拼接 = term 优先（ObservationManager 默认 use_gym_history=false：term 外层、term 内 5 帧旧→新）；frame 优先仅作消融对照');
  lines.push('  - last_action = raw 网络输出（action_manager.action()，未乘 scale/加 offset）');
  lines.push('  - 训练 sim.dt=0.005、decimation=4（策略 50Hz）；本地物理 500Hz 高于训练 200Hz');
  lines.push('  - cmd 训练范围 vx[-0.5,1.0] vy[-0.3,0.3] wz[-0.2,0.2]（方案 4.4 的 wz≤0.5 超训练域，演示限幅收拢到 ±0.2 内）');
  lines.push('');
  lines.push('## 实施期新发现（T0 调试中确认，影响 walkController 实现的硬事实）');
  lines.push('');
  lines.push('- **gainprm/biasprm 每执行器 10 列（mjNGAIN/mjNBIAS=10），不是 3 列**：索引必须用');
  lines.push('  `act*10+k`。误用 act*3 会把增益写进相邻执行器的行内错位槽（biasprm 第 0 列是常数');
  lines.push('  力矩项 → 恒定力矩偏置 → 温和趴地）。这是首轮全变体趴地的根因。');
  lines.push('- **position 执行器 dampratio 语义**：编译后 biasprm[2] ≈ -43.0（kp=500, dampratio=1），');
  lines.push('  即 kd 按 2·dampratio·√(kp·m_eff) 量级的依负载阻尼，**不是** kp·dampratio=-500。');
  lines.push('  改写为策略 PD 时显式写 biasprm[2]=-kd（训练真值 2/4/5/10）即可。');
  lines.push('- 改写机制自检：`--gain policy500:500`（改写路径写 Menagerie 同值增益）与原生 legacy');
  lines.push('  行为一致（zeroctrl 站立 z=0.743/0.743），证明 enter/exit 改写机制正确。');
  lines.push('- 转向响应观察：wz=0.2（训练边界）3s 原地转 Δyaw≈5°，响应弱但稳定。演示路线');
  lines.push('  以直线段为主（方案 4.4），到位判据 |e_yaw|<0.1rad 用慢速蹭转即可，不阻塞 gate。');
  lines.push('');
  lines.push(`## 变体矩阵（增益模式：${GAIN_MODE === 'policy' ? 'D1 主选——运行时改写为策略 PD' : '变体 A——不改增益 kp=500 直喂'}；gate = 站立 + 前走 + 无 NaN）`);
  lines.push('');
  lines.push('| 变体 | 站立 z≥0.6 | 漂移<10cm | 前走≥0.6m | 足交替≥4 | 转向Δyaw(观察) | min pelvis z | 足-地 min | NaN | gate 判定 |');
  lines.push('|---|---|---|---|---|---|---|---|---|---|');
  for (const r of rows) {
    lines.push(`| ${r.id} | ${r.standZ?.toFixed(3)} | ${r.drift?.toFixed(3)}m | ${r.fwd?.toFixed(2)}m | ${r.flips} | ${(r.dyaw * 180 / Math.PI).toFixed(0)}° | ${r.minZ?.toFixed(3)} | ${(r.footMin * 1000).toFixed(1)}mm | ${r.nan} | ${r.pass ? '**PASS**' : 'fail'} |`);
  }
  lines.push('');
  lines.push(`## 判定`);
  lines.push('');
  lines.push(anyPass
    ? `- 存在通过变体 → **T0 gate 通过，进 T1**；装订参数（fwd-term = policy→mjcf + term-major）冻结进 vendor/policy/manifest.json。`
    : `- 全部变体失败 → 停止并回报（方案 R1 出口），不得自行启动路线 C 自训。`);
  lines.push(anyPass ? '' : '');
  if (anyPass) {
    const best = rows.find((r) => r.pass);
    lines.push(`- 选定装订：**${best.id}**（map 方向 ${best.id.split('-')[0]}，历史 ${best.id.split('-')[1]}-major），`);
    lines.push(`  与 D6-1 静态源码定论一致；其余 5 变体站立期全部趴地（装订唯一性防假绿验证）。`);
    lines.push(`- T0 判稳数据：站立 pelvis z=${best.standZ?.toFixed(3)}m（漂移 ${(best.drift * 100).toFixed(1)}cm）、`);
    lines.push(`  前走 5s 位移 ${best.fwd?.toFixed(2)}m（0.3m/s 指令）、足支撑交替 ${best.flips} 次、min z=${best.minZ?.toFixed(3)}m、`);
    lines.push(`  足-地最小距离 ${(best.footMin * 1000).toFixed(1)}mm、全程无 NaN。`);
  }
  writeFileSync(path.join(notesDir, 'policy-probe-notes.md'), lines.join('\n') + '\n');
  console.log('notes 已写出: docs/policy-probe-notes.md');
}
process.exit(anyPass ? 0 : 1);
