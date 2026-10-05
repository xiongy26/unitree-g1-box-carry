// 动作排查实验探针（MERO-5）：不动生产代码，Node 直接跑隔离实验。
//   E1 圆弧行走：恒定 cmd=(vx,0,±wz) 从干净站立起步，测策略能否稳定走弯（去定格转身的基石）。
//   E2 定格转身→行走 接管实验：复现槽 4 跌倒场景（reset 站立 → kinematic 转身 163° → 预热 → 直走），
//      对照 qvel 写法/热启动清零变体，隔离跌倒根因。（干净场景，无箱子）
//   E3 全保真复现：真实场景（双箱型 24 体 + exclude + 潜伏暂持 + qfrc 清零）+ 真实
//      walkController，复刻槽 4 重试路径（reset 站立 yaw=3.05 → TURN_APPROACH 163° →
//      walkTo 3 路点 + 预热），按 flag 逐项剔除场景要素定位跌倒根因。
// 用法：
//   node tools/probes/motion_probe.mjs --e1              # 圆弧行走矩阵
//   node tools/probes/motion_probe.mjs --e2              # 转身接管矩阵
//   node tools/probes/motion_probe.mjs --e2 --ang 105    # 指定转身角（度）
//   node tools/probes/motion_probe.mjs --e3 [--scene full|nox|clean] [--placed 3] [--turn 163] [--warm 0.6]
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import loadMujoco from '../../vendor/mujoco/mujoco.js';
import { loadPolicy, policyForward } from '../../src/controller/g1Policy.js';
import { buildJointMap, keyId, quatSlerp, quatRotVec, yawQuat } from '../../src/util.js';
import { planPathAvoid } from '../../src/pathPlanner.js';
import {
  boxXmlSnippet, injectBoxes, computeStackFrame, stackLayout,
  remapClipForSlot, sampleRemapped, MAX_BOXES, RELEASE_DROP,
} from '../../src/stackCore.js';
import { createWalkController } from '../../src/controller/walkController.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const TIMESTEP = 0.002;

const args = process.argv.slice(2);
const argVal = (n, d) => { const i = args.indexOf(n); return i >= 0 && i + 1 < args.length ? args[i + 1] : d; };
const hasFlag = (n) => args.includes(n);

const JOINT_NAMES = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
];

// ---------------- 场景装配（与 headless_check 同款） ----------------
function loadClips() {
  const files = {
    largebox: { low: 'carry_low_largebox', mid: 'carry_mid_largebox', high: 'carry_high_largebox' },
    plasticbox: { low: 'carry_low_plasticbox' },
  };
  const clips = {};
  for (const [type, tiers] of Object.entries(files)) {
    clips[type] = {};
    for (const [tier, stem] of Object.entries(tiers)) {
      try { clips[type][tier] = JSON.parse(readFileSync(path.join(ROOT, 'motions', `${stem}.json`), 'utf8')); }
      catch { /* 与 headless 同语义：缺文件跳过 */ }
    }
  }
  return clips;
}
function buildMergedXml(mujoco, { withBoxes = false, withExcludes = true, clips } = {}) {
  const sceneXml = readFileSync(path.join(ROOT, 'model/scene.xml'), 'utf8');
  const g1Xml = readFileSync(path.join(ROOT, 'model/g1.xml'), 'utf8');
  const g1Inner = g1Xml.replace(/^[\s\S]*?<mujoco[^>]*>/, '').replace(/<\/mujoco>\s*$/, '');
  const merged = sceneXml.replace(/<!-- G1 由 main\.js[\s\S]*?-->/, g1Inner.trim());
  let xml = merged;
  if (withBoxes) {
    let snippets = '';
    for (const [type, base] of [['largebox', 0], ['plasticbox', 100]]) {
      const box = clips[type]?.low?.box;
      if (!box) continue;
      for (let i = 0; i < MAX_BOXES; i++) snippets += boxXmlSnippet(base + i, box, [0, 0, -5]) + '\n';
    }
    xml = withExcludes ? injectBoxes(merged, snippets)
      : merged.replace('</worldbody>', `${snippets}\n  </worldbody>`);
  }
  const vfs = new mujoco.MjVFS();
  const assetsDir = path.join(ROOT, 'model/assets');
  for (const f of readdirSync(assetsDir)) {
    if (f.toLowerCase().endsWith('.stl')) vfs.addBuffer('assets/' + f, new Uint8Array(readFileSync(path.join(assetsDir, f))));
  }
  return { xml, vfs };
}
// ---------------- 最小 walker（walkController 的控制环复刻，指令外注入；E1/E2 用） ----------------
function makeMiniWalker({ mujoco, model, data, policy, jmap }) {
  const manifest = policy.manifest;
  const MAP = manifest.joint_map.policy_to_mjcf;
  const DEFAULT_POS = Float64Array.from(manifest.action.default_joint_pos);
  const ACT_SCALE = manifest.action.scale;
  const N = 29;
  const bind = new Array(N);
  for (let p = 0; p < N; p++) bind[p] = jmap[JOINT_NAMES[MAP[p]]];
  const pelvisBid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'pelvis');

  const termDims = [3, 3, 3, N, N, N];
  const termScales = [0.2, 1.0, 1.0, 1.0, 0.05, 1.0];
  const HIST = 5;
  const hist = termDims.map((d) => Array.from({ length: HIST }, () => new Float32Array(d)));
  const obsBuf = new Float32Array(manifest.obs.frame * manifest.obs.history);
  let lastAction = new Float32Array(N);
  const curCmd = new Float32Array(3);
  let sub = 0;

  function buildFrame() {
    const frame = new Float32Array(96);
    frame[0] = data.qvel[3]; frame[1] = data.qvel[4]; frame[2] = data.qvel[5];
    const q = [data.xquat[4 * pelvisBid], data.xquat[4 * pelvisBid + 1], data.xquat[4 * pelvisBid + 2], data.xquat[4 * pelvisBid + 3]];
    const grav = rotVec(conj(q), [0, 0, -1]);
    frame[3] = grav[0]; frame[4] = grav[1]; frame[5] = grav[2];
    frame[6] = curCmd[0]; frame[7] = curCmd[1]; frame[8] = curCmd[2];
    for (let p = 0; p < N; p++) {
      frame[9 + p] = data.qpos[bind[p].q] - DEFAULT_POS[p];
      frame[38 + p] = data.qvel[bind[p].d];
    }
    for (let p = 0; p < N; p++) frame[67 + p] = lastAction[p];
    return frame;
  }
  function rotVec(q, v) {
    const [w, x, y, z] = q;
    const [vx, vy, vz] = v;
    return [
      vx * (1 - 2 * (y * y + z * z)) + vy * 2 * (x * y - z * w) + vz * 2 * (x * z + y * w),
      vx * 2 * (x * y + z * w) + vy * (1 - 2 * (x * x + z * z)) + vz * 2 * (y * z - x * w),
      vx * 2 * (x * z - y * w) + vy * 2 * (y * z + x * w) + vz * (1 - 2 * (x * x + y * y)),
    ];
  }
  function conj(q) { return [q[0], -q[1], -q[2], -q[3]]; }
  function pushHist(frame) {
    let off = 0;
    for (let t = 0; t < 6; t++) { hist[t].shift(); hist[t].push(frame.slice(off, off + termDims[t])); off += termDims[t]; }
  }
  function resetHistory() {
    for (let h = 0; h < HIST; h++) pushHist(buildFrame());
    lastAction.fill(0); sub = 0;
  }
  // cmdWant: [vx, vy, wz]，斜坡 0.02/tick（与 walkController 同款）
  function step(dt, cmdWant) {
    if (sub % 10 === 0) {
      for (let i = 0; i < 3; i++) {
        const d = cmdWant[i] - curCmd[i];
        curCmd[i] = Math.abs(d) <= 0.02 ? cmdWant[i] : curCmd[i] + Math.sign(d) * 0.02;
      }
      pushHist(buildFrame());
      let off = 0;
      for (let t = 0; t < 6; t++) {
        const sc = termScales[t];
        for (let h = 0; h < HIST; h++) for (let i = 0; i < termDims[t]; i++) obsBuf[off++] = hist[t][h][i] * sc;
      }
      const act = policyForward(policy.weights, obsBuf);
      lastAction = act;
      for (let p = 0; p < N; p++) {
        data.ctrl[bind[p].c] = DEFAULT_POS[p] + ACT_SCALE * act[p];
      }
    }
    sub++;
  }
  function yaw() {
    const w = data.xquat[4 * pelvisBid], x = data.xquat[4 * pelvisBid + 1];
    const y = data.xquat[4 * pelvisBid + 2], z = data.xquat[4 * pelvisBid + 3];
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }
  function gravZ() { return rotVec(conj([data.xquat[4 * pelvisBid], data.xquat[4 * pelvisBid + 1], data.xquat[4 * pelvisBid + 2], data.xquat[4 * pelvisBid + 3]]), [0, 0, -1])[2]; }
  return { step, resetHistory, yaw, gravZ, get cmd() { return curCmd; } };
}

function fmt(s) { return s.toFixed(2); }
const smooth3 = (x) => { const t = Math.max(0, Math.min(1, x)); return t * t * (3 - 2 * t); };

// ---------------- 场景构建 ----------------
const mujoco = await loadMujoco();
const CLIPS = loadClips(); // E1/E2 不注箱；E3 按场景变体编译
const { xml, vfs } = buildMergedXml(mujoco);

function freshScene() {
  const model = mujoco.MjModel.from_xml_string(xml, vfs);
  const data = new mujoco.MjData(model);
  return { model, data };
}

async function main() {
  const policy = await loadPolicy('vendor/policy', (url) => Promise.resolve({
    ok: true, status: 200,
    json: async () => JSON.parse(readFileSync(path.join(ROOT, url), 'utf8')),
    arrayBuffer: async () => readFileSync(path.join(ROOT, url)).buffer,
  }));
  const KP = policy.manifest.gains.kp, KD = policy.manifest.gains.kd;
  const MAP = policy.manifest.joint_map.policy_to_mjcf;

  // 增益改写（与 walkController.enter 同款）
  function rewriteGains(model, jmap) {
    for (let p = 0; p < 29; p++) {
      const m = MAP[p];
      const act = jmap[JOINT_NAMES[m]].c;
      model.actuator_gainprm[act * 10 + 0] = KP[m];
      model.actuator_biasprm[act * 10 + 1] = -KP[m];
      model.actuator_biasprm[act * 10 + 2] = -KD[m];
    }
  }
  function resetStand(mujoco, model, data, yaw = 0) {
    const standKey = keyId(mujoco, model, 'stand');
    mujoco.mj_resetDataKeyframe(model, data, standKey);
    if (yaw !== 0) {
      const yq = yawQuat(yaw);
      for (let i = 0; i < 4; i++) data.qpos[3 + i] = yq[i];
      for (let i = 0; i < 6; i++) data.qvel[i] = 0;
    }
    mujoco.mj_forward(model, data);
  }

  if (hasFlag('--e1')) {
    // E1：圆弧行走矩阵。vx × wz（wz ≤ 0.2 训练域内），从干净站立出发，12s。
    console.log('== E1 圆弧行走（恒定指令，从干净站立起步，12s） ==');
    const combos = [];
    for (const vx of [0.25, 0.3, 0.4]) for (const wz of [0.1, 0.15, 0.2]) {
      combos.push([vx, wz], [vx, -wz]);
    }
    const shared = freshScene(); // 单模型复用（每次编译新模型会耗尽 wasm 堆）
    const jmap = buildJointMap(mujoco, shared.model, JOINT_NAMES);
    rewriteGains(shared.model, jmap);
    for (const [vx, wz] of combos) {
      const { model, data } = shared;
      resetStand(mujoco, model, data, 0);
      const w = makeMiniWalker({ mujoco, model, data, policy, jmap });
      w.resetHistory();
      const x0 = data.qpos[0], y0 = data.qpos[1], yaw0 = w.yaw();
      let fell = false, minZ = Infinity, tFall = -1;
      const steps = Math.round(12 / TIMESTEP);
      for (let s = 0; s < steps; s++) {
        w.step(TIMESTEP, [vx, 0, wz]);
        mujoco.mj_step(model, data);
        minZ = Math.min(minZ, data.qpos[2]);
        if (!fell && w.gravZ() > -0.6 && data.qpos[2] < 0.5) { fell = true; tFall = s * TIMESTEP; break; }
      }
      const dx = data.qpos[0] - x0, dy = data.qpos[1] - y0;
      const dist = Math.hypot(dx, dy);
      let dyaw = w.yaw() - yaw0;
      dyaw = Math.atan2(Math.sin(dyaw), Math.cos(dyaw));
      const dur = fell ? tFall : 12;
      // 拟合圆弧半径：|dyaw| 转过的弧，弦长 chord=2R sin(dyaw/2)
      const R = Math.abs(Math.sin(dyaw / 2)) > 0.05 ? dist / (2 * Math.abs(Math.sin(dyaw / 2))) : Infinity;
      console.log(`cmd=(${vx.toFixed(2)},0,${wz >= 0 ? '+' : ''}${wz.toFixed(2)})`
        + ` ${fell ? `FELL@${fmt(tFall)}s` : 'ok  '}`
        + ` 移动${fmt(dist)}m Δyaw=${(dyaw * 180 / Math.PI).toFixed(0)}°`
        + ` 平均yaw率=${fmt((dyaw / dur) * 180 / Math.PI)}°/s`
        + ` R≈${R === Infinity ? '∞' : fmt(R) + 'm'} minZ=${fmt(minZ)}`);
    }
  }

  if (hasFlag('--e2')) {
    // E2：定格转身→行走 接管实验（复现槽 4：站立 yaw=3.05 → kinematic 转 163° → 预热 → 直走）
    const ANG = parseFloat(argVal('--ang', '163')) * Math.PI / 180;
    const WALKS = parseFloat(argVal('--walksec', '6'));
    console.log(`== E2 定格转身→行走 接管（角度 ${ANG * 180 / Math.PI}°，时长 max(0.8, |a|/0.8)s，预热 0.6/1.0s） ==`);
    const variants = [
      { id: 'diff-qvel', qvel0: false, warm: 0 },
      { id: 'zero-qvel', qvel0: true, warm: 0 },
      { id: 'diff-qvel+warm0.6', qvel0: false, warm: 0.6 },
      { id: 'zero-qvel+warm0.6', qvel0: true, warm: 0.6 },
      { id: 'diff-qvel+warm1.0', qvel0: false, warm: 1.0 },
      { id: 'no-turn+warm0.6(对照:直接摆位)', qvel0: true, warm: 0.6, instant: true },
    ];
    const shared = freshScene();
    const jmap = buildJointMap(mujoco, shared.model, JOINT_NAMES);
    rewriteGains(shared.model, jmap);
    for (const v of variants) {
      const { model, data } = shared;
      const startYaw = 3.05; // 槽 4 复位摆位
      resetStand(mujoco, model, data, startYaw);
      const q0 = [], dof0 = [];
      for (let i = 0; i < 4; i++) q0.push(data.qpos[3 + i]);
      for (let i = 0; i < 29; i++) dof0.push(data.qpos[jmap[JOINT_NAMES[i]].q]);
      const endYaw = startYaw - ANG;
      const turnDur = v.instant ? 0 : Math.max(0.8, Math.abs(ANG) / 0.8);
      const targetQ = yawQuat(endYaw);

      // kinematic 转身（TURN_APPROACH 同款：blendPose 写 qpos，ctrl 跟踪）
      const smoothstep = (x) => { const t = Math.max(0, Math.min(1, x)); return t * t * (3 - 2 * t); };
      if (v.instant) {
        for (let i = 0; i < 4; i++) data.qpos[3 + i] = targetQ[i];
        mujoco.mj_forward(model, data);
      } else {
        const n = Math.round(turnDur / TIMESTEP);
        let prevQ = [...q0];
        for (let s = 1; s <= n; s++) {
          const a = smoothstep(s / n);
          const q = quatSlerp(q0, targetQ, a);
          for (let i = 0; i < 4; i++) data.qpos[3 + i] = q[i];
          if (v.qvel0) {
            for (let i = 0; i < 6; i++) data.qvel[i] = 0;
          } else {
            // r = conj(prev) ⊗ q，ω ≈ 2·vec(r)/dt（body 系）
            const cq = [prevQ[0], -prevQ[1], -prevQ[2], -prevQ[3]];
            const [pw, px, py, pz] = cq;
            const r = [
              pw * q[0] - px * q[1] - py * q[2] - pz * q[3],
              pw * q[1] + px * q[0] + py * q[3] - pz * q[2],
              pw * q[2] - px * q[3] + py * q[0] + pz * q[1],
              pw * q[3] + px * q[2] - py * q[1] + pz * q[0],
            ];
            data.qvel[3] = 2 * r[1] / TIMESTEP; data.qvel[4] = 2 * r[2] / TIMESTEP; data.qvel[5] = 2 * r[3] / TIMESTEP;
          }
          // ctrl 跟踪（dof 不变 → 恒 stand 目标）
          for (let i = 0; i < 29; i++) data.ctrl[jmap[JOINT_NAMES[i]].c] = dof0[i];
          mujoco.mj_step(model, data);
          prevQ = q;
        }
      }
      const zAfterTurn = data.qpos[2];

      // 预热 + 直走（目标放正前方 2m）
      const w = makeMiniWalker({ mujoco, model, data, policy, jmap });
      w.resetHistory();
      let fell = false, tFall = -1, minZ = Infinity;
      const warmSteps = Math.round(v.warm / TIMESTEP);
      const walkSteps = Math.round(WALKS / TIMESTEP);
      const goalYaw = endYaw;
      const x0 = data.qpos[0], y0 = data.qpos[1];
      for (let s = 0; s < warmSteps + walkSteps; s++) {
        const t = s * TIMESTEP;
        const inWarm = s < warmSteps;
        // 简化追踪：目标在正前方 2m（直走）
        const bearing = goalYaw;
        let eB = Math.atan2(Math.sin(bearing - w.yaw()), Math.cos(bearing - w.yaw()));
        const wz = inWarm ? 0 : Math.max(-0.2, Math.min(0.2, eB));
        w.step(TIMESTEP, inWarm ? [0, 0, 0] : [0.3, 0, wz]);
        mujoco.mj_step(model, data);
        minZ = Math.min(minZ, data.qpos[2]);
        if (!fell && w.gravZ() > -0.6 && data.qpos[2] < 0.5) { fell = true; tFall = t; break; }
      }
      const dist = Math.hypot(data.qpos[0] - x0, data.qpos[1] - y0);
      console.log(`${v.id.padEnd(28)} 转身后z=${fmt(zAfterTurn)}`
        + ` ${fell ? `FELL@t=${fmt(tFall)}s(预热${v.warm}s后)` : '未摔'}`
        + ` 行走位移=${fmt(dist)}m minZ=${fmt(minZ)}`);
    }
  }

  if (hasFlag('--e3')) {
    // E3：全保真复现槽 4 重试路径。真实场景变体 + 真实 createWalkController +
    // stackCore 逐位复刻（writeRobot/暂持箱/zeroQfrc/路点构造），按 flag 剔除场景要素。
    const SCENE = argVal('--scene', 'full');                 // full | nox | clean
    const N_PLACED = parseInt(argVal('--placed', '3'), 10);
    const WARM = parseFloat(argVal('--warm', '0.6'));
    const WITH_BOXES = SCENE !== 'clean';
    console.log(`== E3 槽 4 重试路径全保真复现（scene=${SCENE} placed=${N_PLACED} warm=${WARM}s） ==`);

    const { xml: xml3, vfs: vfs3 } = buildMergedXml(mujoco, {
      withBoxes: WITH_BOXES, withExcludes: SCENE === 'full', clips: CLIPS,
    });
    const model3 = mujoco.MjModel.from_xml_string(xml3, vfs3);
    const data3 = new mujoco.MjData(model3);
    const jmap3 = buildJointMap(mujoco, model3, JOINT_NAMES);
    const walker3 = createWalkController({
      mujoco, model: model3, data: data3, jmap: jmap3, jointNames: JOINT_NAMES,
      policy: { weights: policy.weights, manifest: policy.manifest },
      log: (m) => console.log('  ' + m),
    });

    // ---- box 基础设施（stackCore 逐位复刻） ----
    const mjOBJ3 = mujoco.mjtObj;
    function idsOf(prefix, k) {
      const jid = mujoco.mj_name2id(model3, mjOBJ3.mjOBJ_JOINT.value, `carton${prefix}${k}_joint`);
      return {
        bodyId: mujoco.mj_name2id(model3, mjOBJ3.mjOBJ_BODY.value, `carton${prefix}${k}`),
        jointQadr: model3.jnt_qposadr[jid], jointDadr: model3.jnt_dofadr[jid],
      };
    }
    function writeBox(ids, pos, quat) {
      const q = data3.qpos;
      for (let i = 0; i < 3; i++) q[ids.jointQadr + i] = pos[i];
      for (let i = 0; i < 4; i++) q[ids.jointQadr + 3 + i] = quat[i];
      const d = ids.jointDadr;
      for (let i = 0; i < 6; i++) data3.qvel[d + i] = 0;
    }
    // 2x2 largebox 配置（buildConfig 复刻）
    const boxCfg = CLIPS.largebox.low.box;
    const frame = computeStackFrame(CLIPS.largebox.low);
    const slots = stackLayout(boxCfg, frame.S, frame.psi).filter((sl) => sl.layer < 2);
    const plans = slots.map((sl) => {
      const set = CLIPS.largebox;
      const clip = sl.layer === 0 ? set.low : (set.mid ?? set.low);
      const rel = clip.release_frame;
      const d = [clip.root_pos[rel][0] - clip.obj_pos[rel][0], clip.root_pos[rel][1] - clip.obj_pos[rel][1]];
      const u = [sl.center[0] - frame.S[0], sl.center[1] - frame.S[1]];
      const psi = Math.atan2(u[1], u[0]) - Math.atan2(d[1], d[0]);
      const rm = remapClipForSlot(clip, sl.center, psi, boxCfg.half_size[2], boxCfg.bbox_center_offset[2]);
      const g = clip.grasp_frame;
      const relPose = sampleRemapped(rm, rm.release);
      const chi = frame.psi, c = Math.cos(chi), s = Math.sin(chi);
      const [ox, oy, oz] = boxCfg.bbox_center_offset;
      return {
        slot: sl, rm,
        spawn: {
          pos: [clip.obj_pos[g][0], clip.obj_pos[g][1], boxCfg.half_size[2] - boxCfg.bbox_center_offset[2]],
          quat: [...clip.obj_quat[g]],
        },
        placedQuat: [Math.cos(chi / 2), 0, 0, Math.sin(chi / 2)],
        placedPos: [relPose.objPos[0] - (c * ox - s * oy), relPose.objPos[1] - (s * ox + c * oy), relPose.objPos[2] - oz],
      };
    });
    const STAGING_Y3 = { largebox: 3.0, plasticbox: 4.2 };
    const stagingPos3 = (i, restZ, boxType) => [-3 + (i % MAX_BOXES) * 0.6, STAGING_Y3[boxType] ?? 3.0, restZ];
    const staged3 = new Map();
    if (WITH_BOXES) {
      for (const [type, base] of [['largebox', 0], ['plasticbox', 100]]) {
        const b = CLIPS[type]?.low?.box;
        if (!b) continue;
        const restZ = b.half_size[2] - b.bbox_center_offset[2];
        for (let i = 0; i < MAX_BOXES; i++) {
          const pos = stagingPos3(i, restZ, type);
          staged3.set(`P:${type}:${i}`, { ids: idsOf('', base + i), pos, quat: [1, 0, 0, 0] });
          staged3.set(`G:${type}:${i}`, { ids: idsOf('G', base + i), pos, quat: [1, 0, 0, 0] });
        }
      }
      for (const h of staged3.values()) writeBox(h.ids, h.pos, h.quat);
    }
    // ---- 机器人写入（writeRobot 复刻） ----
    const prevRobot3 = { pos: null, quat: null, dof: null };
    function writeRobot3(pose, dt) {
      const q = data3.qpos, v = data3.qvel;
      q[0] = pose.pos[0]; q[1] = pose.pos[1]; q[2] = pose.pos[2];
      for (let i = 0; i < 4; i++) q[3 + i] = pose.quat[i];
      for (let i = 0; i < JOINT_NAMES.length; i++) q[jmap3[JOINT_NAMES[i]].q] = pose.dof[i];
      if (dt > 0 && prevRobot3.pos) {
        for (let i = 0; i < 3; i++) v[i] = (pose.pos[i] - prevRobot3.pos[i]) / dt;
        const r = quatMulLocal(quatConjLocal(prevRobot3.quat), pose.quat);
        v[3] = 2 * r[1] / dt; v[4] = 2 * r[2] / dt; v[5] = 2 * r[3] / dt;
        for (let i = 0; i < JOINT_NAMES.length; i++) {
          v[jmap3[JOINT_NAMES[i]].d] = (pose.dof[i] - prevRobot3.dof[i]) / dt;
        }
      } else {
        for (let i = 0; i < 6 + JOINT_NAMES.length; i++) v[i] = 0;
      }
      prevRobot3.pos = [...pose.pos]; prevRobot3.quat = [...pose.quat]; prevRobot3.dof = [...pose.dof];
      for (let i = 0; i < JOINT_NAMES.length; i++) data3.ctrl[jmap3[JOINT_NAMES[i]].c] = pose.dof[i];
    }
    function quatMulLocal(a, b) {
      const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
      return [aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw];
    }
    function quatConjLocal(q) { return [q[0], -q[1], -q[2], -q[3]]; }
    function capture3() {
      return {
        pos: [data3.qpos[0], data3.qpos[1], data3.qpos[2]],
        quat: [data3.qpos[3], data3.qpos[4], data3.qpos[5], data3.qpos[6]],
        dof: JOINT_NAMES.map((n) => data3.qpos[jmap3[n].q]),
      };
    }
    const yawOf3 = (q) => Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
    function zeroQfrc3() { for (let i = 0; i < model3.nv; i++) data3.qfrc_applied[i] = 0; }
    const wrap3 = (a) => Math.atan2(Math.sin(a), Math.cos(a));

    // ---- 序列：retryCurrentSlot 复刻 → TURN_APPROACH → startRLWalk('approach') ----
    walker3.enter(); // 真实流程中增益自槽 1 起持续为策略 PD（enter 幂等）
    const standKey3 = keyId(mujoco, model3, 'stand');
    mujoco.mj_resetDataKeyframe(model3, data3, standKey3);
    mujoco.mj_forward(model3, data3);
    const yawHome = Math.atan2(CLIPS.largebox.low.root_pos[0][1], CLIPS.largebox.low.root_pos[0][0]);
    const yq = [Math.cos(yawHome / 2), 0, 0, Math.sin(yawHome / 2)];
    for (let i = 0; i < 4; i++) data3.qpos[3 + i] = yq[i];
    for (let i = 3; i < 6; i++) data3.qvel[i] = 0;
    mujoco.mj_forward(model3, data3);
    if (WITH_BOXES) {
      for (let i = 0; i < N_PLACED && i < plans.length; i++) {
        writeBox(idsOf('', i), plans[i].placedPos, plans[i].placedQuat);
        staged3.delete(`P:largebox:${i}`);
      }
      staged3.delete('G:largebox:3');
    }
    const plan4 = plans[3];
    const goal0 = sampleRemapped(plan4.rm, 0);
    const gx = goal0.rootPos[0], gy = goal0.rootPos[1];

    // TURN_APPROACH（startApproachTurn 复刻）
    const from3 = capture3();
    const bearing = Math.atan2(gy - from3.pos[1], gx - from3.pos[0]);
    const eYaw = wrap3(bearing - yawOf3(from3.quat));
    const turnDur = Math.max(0.8, Math.abs(eYaw) / 0.8);
    console.log(`[e3] 定格转身 ${(eYaw * 180 / Math.PI).toFixed(0)}°（${turnDur.toFixed(2)}s），goal=(${fmt(gx)},${fmt(gy)})`);
    prevRobot3.pos = null;
    const turnSteps = Math.round(turnDur / TIMESTEP);
    for (let s = 1; s <= turnSteps; s++) {
      const a = smooth3(Math.min(1, s / turnSteps));
      const pose = {
        pos: from3.pos,
        quat: quatSlerp(from3.quat, yawQuat(bearing), a),
        dof: from3.dof,
      };
      writeRobot3(pose, TIMESTEP);
      if (WITH_BOXES) writeBox(idsOf('G', 3), plan4.spawn.pos, plan4.spawn.quat);
      for (const h of staged3.values()) writeBox(h.ids, h.pos, h.quat);
      zeroQfrc3();
      mujoco.mj_step(model3, data3);
    }

    // startRLWalk('approach') 复刻（resetObsHistory → makeWaypoints → walkTo → warmUp）
    // --inject <wd.json>：walkTo 前把真实运行的 walkTo 时刻状态整体注入（qpos/qvel/ctrl/
    // qacc_warmstart），判定「状态本身致摔（不稳悬崖）」vs「模型/隐藏状态差异」
    if (argVal('--inject')) {
      const snap = JSON.parse(readFileSync(argVal('--inject'), 'utf8'));
      for (let i = 0; i < data3.qpos.length; i++) data3.qpos[i] = snap.qpos[i];
      for (let i = 0; i < data3.qvel.length; i++) data3.qvel[i] = snap.qvel[i];
      for (let i = 0; i < data3.ctrl.length; i++) data3.ctrl[i] = snap.ctrl[i];
      for (let i = 0; i < data3.qacc_warmstart.length; i++) data3.qacc_warmstart[i] = snap.qacc_warmstart[i];
      console.log(`[e3] 已注入真实 run walkTo 状态（walkSeq=${snap.walkSeq} ncon=${snap.ncon} time=${snap.time.toFixed(3)}）`);
    }
    walker3.resetObsHistory();
    const fromW = { x: data3.qpos[0], y: data3.qpos[1], yaw: yawOf3([data3.qpos[3], data3.qpos[4], data3.qpos[5], data3.qpos[6]]) };
    const toW = { x: gx, y: gy, yaw: Math.atan2(gy - fromW.y, gx - fromW.x) };
    // 路径规划换用 pathPlanner（walk-avoid-plan F3：walker.planPath 已退役删除）。探针障碍集
    // 与生产链 stackCore.collectWalkObstacles 同源口径的简化版：目标箱 spawn（role=target，
    // 收缩膨胀）+ 已放物理箱（qpos 写入位 = placedPos 真值）；geom 中心 = body 位姿 ⊗
    // bbox_center_offset。
    const probeRects = [];
    {
      const bx = CLIPS.largebox.low.box;
      const [pox, poy, poz] = bx.bbox_center_offset;
      const cS = quatRotVec(plan4.spawn.quat, [pox, poy, poz]);
      probeRects.push({
        cx: plan4.spawn.pos[0] + cS[0], cy: plan4.spawn.pos[1] + cS[1], yaw: yawOf3(plan4.spawn.quat),
        hx: bx.half_size[0], hy: bx.half_size[1], role: 'target', key: 'target:slot3',
      });
      for (let i = 0; i < N_PLACED && i < plans.length; i++) {
        const cP = quatRotVec(plans[i].placedQuat, [pox, poy, poz]);
        probeRects.push({
          cx: plans[i].placedPos[0] + cP[0], cy: plans[i].placedPos[1] + cP[1], yaw: yawOf3(plans[i].placedQuat),
          hx: bx.half_size[0], hy: bx.half_size[1], role: 'placed', key: `placed:${i}`,
        });
      }
    }
    let wps = planPathAvoid(fromW, toW, probeRects).points;
    const dense = [];
    for (let i = 0; i < wps.length; i++) {
      const a = i === 0 ? fromW : { x: wps[i - 1].x, y: wps[i - 1].y };
      const b = wps[i];
      const seg = Math.hypot(b.x - a.x, b.y - a.y);
      const n = Math.max(1, Math.ceil(seg / 0.3));
      for (let k = 1; k <= n; k++) {
        const t = k / n;
        const px = a.x + (b.x - a.x) * t, py = a.y + (b.y - a.y) * t;
        const nyaw = k === n ? b.yaw : Math.atan2(wps[i].y - py, wps[i].x - px);
        dense.push({ x: px, y: py, yaw: nyaw });
      }
    }
    let pathLen = 0, px3 = fromW.x, py3 = fromW.y;
    for (const wp of dense) { pathLen += Math.hypot(wp.x - px3, wp.y - py3); px3 = wp.x; py3 = wp.y; }
    walker3.walkTo({
      waypoints: dense, carrying: false, armGoal: null, vxScale: 1.0,
      timeLimitSec: 4.0 + 3.0 * pathLen / 0.3, yawTolerance: Math.PI,
    });
    walker3.warmUp(WARM);
    const x0 = data3.qpos[0], y0 = data3.qpos[1];

    // WALK 循环（onStepRL WALK_APPROACH + 外层 mj_step 复刻）
    let fell = false, tWalk = 0, minZ = Infinity, doneAt = -1;
    const maxWalkSteps = Math.round(12 / TIMESTEP);
    let dbgAcc = 0;
    for (let s = 0; s < maxWalkSteps; s++) {
      const r = walker3.step(TIMESTEP);
      if (!r.done && WITH_BOXES) writeBox(idsOf('G', 3), plan4.spawn.pos, plan4.spawn.quat);
      for (const h of staged3.values()) writeBox(h.ids, h.pos, h.quat);
      zeroQfrc3();
      mujoco.mj_step(model3, data3);
      tWalk += TIMESTEP;
      minZ = Math.min(minZ, data3.qpos[2]);
      dbgAcc += TIMESTEP;
      if (dbgAcc >= 0.1) {
        dbgAcc = 0;
        console.log(`[e3] t=${fmt(tWalk)} xy=(${fmt(data3.qpos[0])},${fmt(data3.qpos[1])}) z=${fmt(data3.qpos[2])}`
          + ` yaw=${(yawOf3([data3.qpos[3], data3.qpos[4], data3.qpos[5], data3.qpos[6]]) * 180 / Math.PI).toFixed(0)}`
          + ` cmd=(${r.cmd[0].toFixed(2)},${r.cmd[1].toFixed(2)},${r.cmd[2].toFixed(2)}) ncon=${data3.ncon}`);
      }
      if (r.fallen) { fell = true; console.log(`[e3] 跌倒检测触发 @t=${fmt(tWalk)}s z=${fmt(data3.qpos[2])}`); break; }
      if (r.done) { doneAt = tWalk; break; }
      if (data3.qpos[2] < 0.3) { fell = true; console.log(`[e3] 塌落（z<0.3，检测未及）@t=${fmt(tWalk)}s`); break; }
    }
    const dist = Math.hypot(data3.qpos[0] - x0, data3.qpos[1] - y0);
    console.log(`[e3] 结果: ${fell ? 'FELL' : doneAt >= 0 ? 'DONE' : 'TIMEOUT(12s)'}`
      + ` 行走位移=${fmt(dist)}m minZ=${fmt(minZ)} 终态z=${fmt(data3.qpos[2])}`);
  }

  if (!hasFlag('--e1') && !hasFlag('--e2') && !hasFlag('--e3')) {
    console.log('用法: node tools/probes/motion_probe.mjs --e1 | --e2 [--ang 163] | --e3 [--scene full|nox|clean] [--placed 3]');
  }
}
main();
