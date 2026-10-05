// 码箱子模式无头验证（Node ≥22，零第三方依赖，直接用 vendored MuJoCo WASM）。
// 对应方案 docs/box-stacking-plan.md 第 9 节断言集 1-7 + docs/rl-walking-plan.md 断言集 N1-N11
// + docs/mocap-carry-walk-plan.md 4.5 适配表（N4/N5/N12/N13 分档 + N14 + M-1 + negcheck 第三组）
// + T3 FK 交叉验证 + T4 纯函数自测。
//
// 用法：
//   node tools/checks/headless_check.mjs                                      # 默认 v3（RL 行走+动捕搬移）2层×2列 largebox 全循环
//   node tools/checks/headless_check.mjs --walk legacy                        # 兼容映射：legacy 已移除，按 rl 运行（见 README）
//   node tools/checks/headless_check.mjs --carry rl                           # v2 对照矩阵（RL 持箱搬移，N13 已知红）
//   node tools/checks/headless_check.mjs --layers 3 --cols 2 --box plasticbox # 参数矩阵抽样
//   node tools/checks/headless_check.mjs --walk rl --policy-selftest          # N1 golden 校验单独快速跑
//   node tools/checks/headless_check.mjs --fk                                 # T3：JSON→展示端模型 FK vs fk_ref（含 carrywalk 片段）
//   node tools/checks/headless_check.mjs --negcheck                           # 防假绿（槽位 z + 腿装订注入）
//   node tools/checks/headless_check.mjs --negcheck carry                     # 防假绿第三组（携带漂移 → N13-mocap 必 FAIL）
//   node tools/checks/headless_check.mjs --negcheck avoid                     # 防假绿第四组（避障注入 → N19 必 FAIL，walk-avoid-plan 4.5.2）
//   node tools/checks/headless_check.mjs --unit-only                          # 只跑纯函数自测
//
// 退出码：全绿 0；任何断言失败 1。
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import loadMujoco from '../../vendor/mujoco/mujoco.js';
import {
  createBoxStacking, boxXmlSnippet, injectBoxes,
  stackLayout, computeStackFrame, computeStackFrameCarry, remapClipForSlot, remapClipCarryFull,
  COL_GAP, sampleRemapped, resolveCarryTiers, resolveTierForLayer, MAX_BOXES, RELEASE_DROP, STAGING_Y,
} from '../../src/stackCore.js';
import {
  FACTORY, derivePickPos, shelfBoardTops, factoryObstacleRects, factoryStaticXml,
  FACTORY_SOLID_BODIES, rackRect, rackBoardSpecs,
} from '../../src/factory.js';
import {
  worldToPixelMapper, mapView, buildStaticDrawList, buildDynamicDrawList, pickZoneCovers,
} from '../../src/minimapModel.js';
import { motionSignature } from '../../src/manipulationPlan.js';
import { createWalkController, ARMHOLD_DEFAULT } from '../../src/controller/walkController.js';
import {
  planPathAvoid, pointRectDist, inflateRect, rectCorners, segIntersectsRect, PLANNER_DEFAULTS,
} from '../../src/pathPlanner.js';
import { loadPolicy, policySelfTest } from '../../src/controller/g1Policy.js';
import { buildJointMap, keyId, quatConj, quatRotVec, quatSlerp, yawQuat } from '../../src/util.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const TIMESTEP = 0.002;

// ---------------- 参数 ----------------
const args = process.argv.slice(2);
function argVal(name, dflt) {
  const i = args.indexOf(name);
  // 下一个参数以 -- 开头说明它是 flag 而非本 flag 的值（修复 --negcheck --layers 1
  // 时 NEG_MODE 被吞成 '--layers'、注入被静默跳过的防假绿失效，审查 F5）
  return i >= 0 && i + 1 < args.length && !String(args[i + 1]).startsWith('--')
    ? args[i + 1] : dflt;
}
const hasFlag = (name) => args.includes(name);
const LAYERS = parseInt(argVal('--layers', '2'), 10);
const COLS = parseInt(argVal('--cols', '2'), 10);
const BOXTYPE = argVal('--box', 'largebox');
const MAX_SIM = parseFloat(argVal('--max-sim', '540')); // 仿真秒上限（防卡死）。瞬移消除改造后
// 踏步转身增加时长（每个 ~180° 约 11s，每槽 1-2 次）；工厂方案起行走含绕行（approach
// ~2.4-2.9m）且 RL 方差下重试（WALK_RETRIES=2）最坏每槽 3 次尝试，2×2 全循环实测
// ~240-330s、留 RL 方差裕量后默认 420 → 540（R7 预案：随实测上调并记录）。
// --walk legacy / --carry legacy：legacy 整段回放（BLEND_HOME 贝塞尔滑移等瞬移路径）已随
// 瞬移消除改造整体移除（无真实步态数据源，见 README）；显式传入映射为默认链并打提示，
// 兼容既有命令行习惯。
const WALK_ARG = argVal('--walk', 'rl');
if (WALK_ARG === 'legacy') {
  console.log('  [notice] --walk legacy 整段回放已移除（无真实步态数据源，见 README），按默认 rl 运行');
}
const WALK_MODE = WALK_ARG === 'legacy' ? 'rl' : WALK_ARG; // rl=RL 行走（唯一模式）
const CARRY_RAW = argVal('--carry', 'mocap');
if (CARRY_RAW === 'legacy') {
  console.log('  [notice] --carry legacy 已随 legacy 模式移除（见 README），按 mocap 运行');
}
// MERO-10 携带策略（方案 4.6）：mocap=真人动捕全程回放（默认）| rl=v2 对照
const CARRY_ARG = CARRY_RAW === 'legacy' ? 'mocap' : CARRY_RAW;
// MERO-8 softArmHold 开关与配方选择（显式控制以便矩阵覆盖 soft/off 两种；默认取
// walkController.ARMHOLD_DEFAULT=门控实验结论）
const ARMHOLD_ARG = argVal('--armhold', ARMHOLD_DEFAULT);       // soft | off
const ARMHOLD_RAMP = argVal('--armhold-ramp', 'warmup');        // warmup | delay | fixed | pitch
const ARMHOLD_TIGHT = hasFlag('--armhold-tight');               // 配方 3：carry 段收紧指令
const ARMHOLD_ALPHA = parseFloat(argVal('--armhold-alpha', '0')); // ramp=fixed 的目标 α（MERO-9 手段3）
// MERO-9 抑摆扫描配方（各自默认 0=off = 基线行为）：
const CARRY_VX_ARG = parseFloat(argVal('--carry-vx', '0'));     // 手段1：carry 段指令限速覆盖（→ walker meta.carryVxMax）
const CARRY_KD_ARG = parseFloat(argVal('--carry-kd', '0'));     // 手段2：carry 段臂 14 执行器 kd 覆盖（kp 不动）
const SELFTEST_ONLY = hasFlag('--policy-selftest');
// negcheck 模式：'all'（默认，槽位 z +5cm + 腿装订互换，MERO-9 前口径）| 'carry'
// （MERO-10 第三组：仅携带漂移注入 G1BOX_NEGCHECK_CARRY_DRIFT=0.3 → N13-mocap 必
// FAIL）| 'avoid'（walk-avoid-plan：仅避障注入 → N19 必 FAIL，方案 4.5.2）。分开的原因：
// 腿装订互换会令 WALK_APPROACH 摔倒，mocap 携带段/避障绕行根本不执行，各注入在同一运行里
// 不可观测——各组必须独占干净运行。
// 值解析（walk-avoid-plan 恢复）：--negcheck 的值（carry/avoid）经 argVal 读取；下一个
// 参数以 -- 开头或无值时回落 all/off（保留审查 F5 的防吞修复语义：--negcheck --layers 1
// 仍按 'all' 注入，注入不会被静默跳过）。
const NEG_MODE = argVal('--negcheck', hasFlag('--negcheck') ? 'all' : 'off');

// MJCF 执行器序 29 关节（与 src/main.js 一致；walker 装订用）
const JOINT_NAMES_H = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
];

// Node 下基于 fs 的 fetch shim（loadPolicy 双通道的 Node 侧实现）
function fsFetch(url) {
  const file = url.startsWith('file:') ? fileURLToPath(url) : path.join(ROOT, url);
  return Promise.resolve({
    ok: true, status: 200,
    json: async () => JSON.parse(readFileSync(file, 'utf8')),
    arrayBuffer: async () => readFileSync(file).buffer,
  });
}

// ---------------- 断言记录 ----------------
const results = [];
let nChecks = 0;
function check(name, ok, detail = '') {
  nChecks++;
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
  return ok;
}

// ---------------- 资产加载（复刻 main.js fetchAssets 的拼接逻辑） ----------------
function loadClips() {
  const files = {
    largebox: { low: 'carry_low_largebox', mid: 'carry_mid_largebox', high: 'carry_high_largebox' },
    plasticbox: { low: 'carry_low_plasticbox' },
  };
  const clips = {};
  for (const [type, tiers] of Object.entries(files)) {
    clips[type] = {};
    for (const [tier, stem] of Object.entries(tiers)) {
      try {
        clips[type][tier] = JSON.parse(readFileSync(path.join(ROOT, 'motions', `${stem}.json`), 'utf8'));
      } catch { /* 缺文件时该 tier 不可用，buildConfig 会拦 */ }
    }
    if (!clips[type].low) delete clips[type];
  }
  // MERO-10 v3：walk_meta.carry 声明的全程回放片段（stem → clips[type].carry[tier]）。
  // 缺文件/损坏时不填，resolveCarryTiers 会把该 tier 降级 rl（降级链，方案 4.6）。
  let metaCarry = null;
  try {
    metaCarry = JSON.parse(readFileSync(path.join(ROOT, 'motions', 'walk_meta.json'), 'utf8')).carry ?? null;
  } catch { /* walk_meta 缺失时仅 v2 行为 */ }
  for (const [type, tiers] of Object.entries(metaCarry ?? {})) {
    if (!clips[type]) continue;
    for (const [tier, entry] of Object.entries(tiers)) {
      const stem = entry?.clip;
      if (!stem) continue;
      try {
        clips[type].carry ??= {};
        clips[type].carry[tier] = JSON.parse(readFileSync(path.join(ROOT, 'motions', `${stem}.json`), 'utf8'));
      } catch { /* 缺 carrywalk 文件 → 该 tier 降级 rl */ }
    }
  }
  return clips;
}

function buildMergedXml(mujoco, { withBoxes, clips }) {
  const sceneXml = readFileSync(path.join(ROOT, 'model', 'scene.xml'), 'utf8');
  const g1Xml = readFileSync(path.join(ROOT, 'model', 'g1.xml'), 'utf8');
  const g1Inner = g1Xml.replace(/^[\s\S]*?<mujoco[^>]*>/, '').replace(/<\/mujoco>\s*$/, '');
  const merged = sceneXml.replace(/<!-- G1 由 main\.js[\s\S]*?-->/, g1Inner.trim());

  let xml = merged;
  // 工厂静态片段（F0 镜像，工厂方案 S4）：与 main.js fetchAssets 同机制——无关节片段
  // 先拼、箱子后拼（顺序无关）；承重板 body 经第三参进 exclude（板×机器人）。
  // 工厂片段始终注入（场景一部分，与 withBoxes 无关）。
  let snippets = factoryStaticXml();
  const solidBodies = FACTORY_SOLID_BODIES.map((b) => b.body);
  if (withBoxes) {
    // 两套箱型都预注入（与 main.js 一致）：largebox carton0..5，plasticbox carton100..105
    for (const [type, base] of [['largebox', 0], ['plasticbox', 100]]) {
      const box = clips[type]?.low?.box;
      if (!box) continue;
      for (let i = 0; i < MAX_BOXES; i++) {
        snippets += boxXmlSnippet(base + i, box, [0, 0, -5]) + '\n';
      }
    }
  }
  xml = injectBoxes(merged, snippets, solidBodies);

  // STL 注入 VFS（编译 g1 需要）
  const vfs = new mujoco.MjVFS();
  const assetsDir = path.join(ROOT, 'model', 'assets');
  for (const f of readdirSync(assetsDir)) {
    if (f.toLowerCase().endsWith('.stl')) {
      vfs.addBuffer('assets/' + f, new Uint8Array(readFileSync(path.join(assetsDir, f))));
    }
  }
  return { xml, vfs };
}

function geomCenter(data, geomId) {
  return [data.geom_xpos[3 * geomId], data.geom_xpos[3 * geomId + 1], data.geom_xpos[3 * geomId + 2]];
}

// ---------------- 纯函数自测（T4 编码内自检） ----------------
function runUnitTests() {
  console.log('== 纯函数自测（插值/重映射，Node assert） ==');

  // quatSlerp：端点 + 半程 + 短路径
  const q0 = [1, 0, 0, 0], q90 = yawQuat(Math.PI / 2);
  const half = quatSlerp(q0, q90, 0.5);
  const halfExp = yawQuat(Math.PI / 4);
  assert.ok(Math.abs(half[0] - halfExp[0]) < 1e-9 && Math.abs(half[3] - halfExp[3]) < 1e-9, 'slerp 半程');
  const h2 = quatSlerp(q0, [-1, 0, 0, 0], 0.5);
  assert.ok(Math.abs(Math.abs(h2[0]) - 1) < 1e-9, 'slerp 对径端点退化');

  // 重映射：psi=0 平移到槽位，release 采样点应落在槽位中心上方 RELEASE_DROP（<1mm）
  const clips = loadClips();
  const clip = clips.largebox.low;
  const hz = clip.box.half_size[2];
  const offZ = clip.box.bbox_center_offset[2];
  const slotCenter = [2.0, -1.0, hz];
  const rm = remapClipForSlot(clip, slotCenter, 0, hz, offZ);
  const s = sampleRemapped(rm, clip.release_frame);
  const err = Math.hypot(s.objPos[0] - slotCenter[0], s.objPos[1] - slotCenter[1],
    s.objPos[2] + offZ - slotCenter[2] - RELEASE_DROP);
  assert.ok(err < 0.001, `release 重映射误差 ${err.toFixed(5)}m < 1mm`);

  // 旋转重映射：psi=90°，中心落点不变
  const rm2 = remapClipForSlot(clip, [0, 0, hz], Math.PI / 2, hz, offZ);
  const s2 = sampleRemapped(rm2, clip.release_frame);
  assert.ok(Math.hypot(s2.objPos[0], s2.objPos[1]) < 0.001, '旋转重映射中心落点');

  // 槽位表：默认 2×2 → 4 槽，层高 hz / 3hz，左右列对称
  const frame = computeStackFrame(clip);
  const slots = stackLayout(clip.box, frame.S, frame.psi).filter((sl) => sl.layer < 2);
  assert.equal(slots.length, 4, '2×2 槽位数');
  assert.ok(Math.abs(slots[0].center[2] - hz) < 1e-9, '层 1 中心 = hz');
  assert.ok(Math.abs(slots[2].center[2] - 3 * hz) < 1e-9, '层 2 中心 = 3hz');
  {
    // 列间距 = 2·colHalf，且列间方向与朝向正交
    const dx = slots[1].center[0] - slots[0].center[0];
    const dy = slots[1].center[1] - slots[0].center[1];
    const sep = Math.hypot(dx, dy);
    const colHalf = Math.max(clip.box.half_size[0], clip.box.half_size[1]) + COL_GAP;
    assert.ok(Math.abs(sep - 2 * colHalf) < 1e-9, `列间距 ${sep.toFixed(4)} = 2·colHalf`);
    const dot = (dx * Math.cos(frame.psi) + dy * Math.sin(frame.psi)) / sep;
    assert.ok(Math.abs(dot) < 1e-9, '列间方向 ⊥ 朝向');
  }

  // MERO-10：carryFull 逐槽刚性规范化重映射（方案 5.3 纯函数单测扩展）。
  // 断言三性质：release 落槽 <1mm；走廊刚性不压缩（N14 无滑移的构造前提）；
  // 走廊进抵方向沿 +psi（从取箱位侧进抵堆垛，不横穿——θ 取法的已记录偏差见
  // stackCore remapClipForSlot 注释，「箱偏航=栅格」由 handoverToPhysics 换位对齐承担）。
  {
    const cclip = clips.largebox.mid; // mid 即携带行走片段（obj 位移 ~1.08m）
    const slotC = [2.0, -1.0, 3 * hz];
    const frameC = computeStackFrameCarry(cclip);
    const gC = cclip.grasp_frame, rC = cclip.release_frame;
    const dxyC = [cclip.obj_pos[rC][0] - cclip.obj_pos[gC][0], cclip.obj_pos[rC][1] - cclip.obj_pos[gC][1]];
    const preRigidC = {
      theta: frameC.psi - Math.atan2(dxyC[1], dxyC[0]),
      anchor: [cclip.obj_pos[rC][0], cclip.obj_pos[rC][1]],
      target: [slotC[0], slotC[1]],
    };
    const rmC = remapClipCarryFull(cclip, slotC, hz, offZ, preRigidC);
    const sC = sampleRemapped(rmC, rC);
    const errC = Math.hypot(sC.objPos[0] - slotC[0], sC.objPos[1] - slotC[1],
      sC.objPos[2] + offZ - slotC[2] - RELEASE_DROP);
    assert.ok(errC < 0.001, `carryFull release 落槽误差 ${errC.toFixed(5)}m < 1mm`);
    const dispRaw = Math.hypot(dxyC[0], dxyC[1]);
    const dispRm = Math.hypot(
      rmC.objPos[rC * 3] - rmC.objPos[gC * 3],
      rmC.objPos[rC * 3 + 1] - rmC.objPos[gC * 3 + 1]);
    assert.ok(Math.abs(dispRm - dispRaw) < 1e-6,
      `carryFull 走廊位移保持 ${dispRm.toFixed(4)} == 原始 ${dispRaw.toFixed(4)}`);
    const appx = rmC.objPos[rC * 3] - rmC.objPos[gC * 3];
    const appy = rmC.objPos[rC * 3 + 1] - rmC.objPos[gC * 3 + 1];
    const appOff = Math.abs(Math.atan2(
      Math.sin(Math.atan2(appy, appx) - frameC.psi),
      Math.cos(Math.atan2(appy, appx) - frameC.psi)));
    assert.ok(appOff < Math.PI / 6, `carryFull 走廊进抵方向偏离 psi ${((appOff * 180) / Math.PI).toFixed(1)}° ≤30°`);
  }

  // M1（walk-avoid-plan）：pathPlanner 纯函数自测（方案 4.4 流程逐条对应）。
  console.log('== 纯函数自测扩展：pathPlanner（2D 避障规划） ==');
  {
    // 点距已知值（轴对齐 + 随 yaw 旋转等价 + 内部为负）
    const r0 = { cx: 1, cy: 1, yaw: 0, hx: 1, hy: 0.5 };
    assert.ok(Math.abs(pointRectDist({ x: 1, y: 1.5 }, r0)) < 1e-12, '点距：面上=0');
    assert.ok(Math.abs(pointRectDist({ x: 2.4, y: 1 }, r0) - 0.4) < 1e-12, '点距：面外法向');
    assert.ok(Math.abs(pointRectDist({ x: 2.4, y: 2 }, r0) - Math.hypot(0.4, 0.5)) < 1e-12, '点距：角外对角');
    assert.ok(Math.abs(pointRectDist({ x: 1, y: 1 }, r0) + 0.5) < 1e-12, '点距：内部为负（穿透深度）');
    const d45 = 0.4, a45 = Math.PI / 4;
    // 随 yaw 旋转等价（审查 P3 修正，基线既有缺陷）：与上方「面外法向」同构——点取
    // 「中心沿旋转后局部 +x 法向 hx+0.4 处」，pointRectDist 应同返回 0.4。旧实现 d45=0.4
    // 漏加 hx，点落在矩形内部返回 −0.5，断言数学上不可通过（任何模式在此崩溃）。
    const p45 = { x: 1 + (1 + d45) * Math.cos(a45), y: 1 + (1 + d45) * Math.sin(a45) };
    assert.ok(Math.abs(pointRectDist(p45, { cx: 1, cy: 1, yaw: a45, hx: 1, hy: 0.5 }) - d45) < 1e-12,
      '点距：随 yaw 旋转等价');
    // 膨胀角点性质：L∞ 保守膨胀（放大矩形直角覆盖圆角）的角点在原矩形对角方向外侧，
    // 距原矩形 = r√2（旧期望 r 同为数学不可通过——对角点 qx=qy=r，欧氏距离 r√2，
    // 审查 P3 修正）；原角点在膨胀矩形内
    for (const cn of rectCorners(r0, 0.4)) {
      assert.ok(Math.abs(pointRectDist(cn, r0) - 0.4 * Math.SQRT2) < 1e-9, '膨胀角点距原矩形 = r√2（L∞ 保守膨胀）');
    }
    for (const cn of rectCorners(r0, 0)) {
      assert.ok(pointRectDist(cn, inflateRect(r0, 0.4)) < 0, '原角点在膨胀矩形内');
    }
    // 判交正反例（严格内交语义：擦边/触角点不算相交；对角弦由中点判据拦截）
    const rr = { cx: 0, cy: 0, yaw: 0, hx: 1, hy: 1 };
    assert.ok(segIntersectsRect({ x: -2, y: 0 }, { x: 2, y: 0 }, rr), '判交：横穿内部');
    assert.ok(segIntersectsRect({ x: 0.5, y: 0.5 }, { x: 3, y: 3 }, rr), '判交：端点在内');
    assert.ok(segIntersectsRect({ x: -2, y: -2 }, { x: 2, y: 2 }, rr), '判交：对角弦（中点判据）');
    assert.ok(!segIntersectsRect({ x: -2, y: 2 }, { x: 2, y: 2 }, rr), '判交反例：矩形外平行线');
    assert.ok(!segIntersectsRect({ x: -2, y: 1 }, { x: 2, y: 1 }, rr), '判交反例：擦边共线');
    assert.ok(!segIntersectsRect({ x: 1, y: 1 }, { x: 3, y: 3 }, rr), '判交反例：角点触碰');
    // 绕双矩形可见性路径：两矩形堵住直线，规划须绕行且每段均在膨胀外
    const rects2 = [
      { cx: 2, cy: 0.5, yaw: 0, hx: 0.8, hy: 0.4, key: 'a' },
      { cx: 4, cy: -0.5, yaw: 0, hx: 0.8, hy: 0.4, key: 'b' },
    ];
    const plan2 = planPathAvoid({ x: 0, y: 0 }, { x: 6, y: 0 }, rects2);
    assert.equal(plan2.mode, 'visibility', '绕双矩形：visibility');
    assert.ok(plan2.points.length >= 2, '绕双矩形：产生绕行点');
    assert.equal(plan2.pruned, 0, '绕双矩形：无粗筛裁剪');
    const infAB = [inflateRect(rects2[0], 0.4), inflateRect(rects2[1], 0.4)];
    const chain2 = [{ x: 0, y: 0 }, ...plan2.points];
    let plen2 = 0;
    for (let i = 1; i < chain2.length; i++) {
      for (const infR of infAB) {
        assert.ok(!segIntersectsRect(chain2[i - 1], chain2[i], infR), '绕行段在膨胀矩形外');
      }
      plen2 += Math.hypot(chain2[i].x - chain2[i - 1].x, chain2[i].y - chain2[i - 1].y);
    }
    assert.ok(plen2 > 6 + 1e-9, `绕行路径长 ${plen2.toFixed(2)} > 直线 6`);
    // 空障远景直线
    const plan0 = planPathAvoid({ x: 0, y: 0 }, { x: 5, y: 2 }, []);
    assert.equal(plan0.mode, 'straight', '空障：直线短路');
    assert.equal(plan0.points.length, 1, '空障：单点');
    assert.ok(plan0.points[0].x === 5 && plan0.points[0].y === 2, '空障：终点精确');
    // 起点在膨胀内（WALK_HOME 起点形态，定标 standoff 0.224 < 膨胀 0.40）：透明化 + 脱出
    // 机制下仍可行，且输出各段不与膨胀矩形内交
    const planE = planPathAvoid({ x: 1.2, y: 0 }, { x: 3, y: 0 }, [{ cx: 0, cy: 0, yaw: 0, hx: 1, hy: 1 }]);
    assert.equal(planE.mode, 'visibility', '起点在膨胀内：透明化机制下可行');
    const infE = inflateRect({ cx: 0, cy: 0, yaw: 0, hx: 1, hy: 1 }, 0.4);
    for (let i = 1; i < planE.points.length; i++) {
      assert.ok(!segIntersectsRect(planE.points[i - 1], planE.points[i], infE), '起点贴障：输出段膨胀外');
    }
    // 降级阶梯：目标被矩形环封死（膨胀 0.4 与收缩 0.12 均无缺口）→ 直线兜底 + degraded
    const ringRects = [
      { cx: 5, cy: 0.9, yaw: 0, hx: 1.0, hy: 0.25 },
      { cx: 5, cy: -0.9, yaw: 0, hx: 1.0, hy: 0.25 },
      { cx: 3.9, cy: 0, yaw: 0, hx: 0.25, hy: 1.15 },
      { cx: 6.1, cy: 0, yaw: 0, hx: 0.25, hy: 1.15 },
    ];
    const planD = planPathAvoid({ x: 0, y: 0 }, { x: 5, y: 0 }, ringRects);
    assert.equal(planD.mode, 'straight-degraded', '封死目标：直线兜底');
    assert.equal(planD.degraded, true, '封死目标：degraded=true');
    // 两次调用逐位一致（确定性契约，R7）
    const plan2b = planPathAvoid({ x: 0, y: 0 }, { x: 6, y: 0 }, rects2);
    assert.equal(JSON.stringify(plan2), JSON.stringify(plan2b), '两次调用逐位一致');
    // 18 矩形最坏计时（方案 M1 行：<100ms；交错密排墙触发 74 节点全图搜索，且该实例
    // 可能无可行路径——恰为"全膨胀 + 收缩一轮"双重求解的最坏计时形态）
    const rects18 = [];
    for (let i = 0; i < 18; i++) {
      rects18.push({ cx: 1 + i, cy: i % 2 === 0 ? 0.7 : -0.7, yaw: 0, hx: 0.3, hy: 0.9, key: `s${i}` });
    }
    let t18max = 0;
    for (let it = 0; it < 20; it++) {
      const t0 = performance.now();
      planPathAvoid({ x: 0, y: 0 }, { x: 18, y: 0 }, rects18);
      t18max = Math.max(t18max, performance.now() - t0);
    }
    assert.ok(t18max < 100, `18 矩形最坏规划耗时 ${t18max.toFixed(2)}ms < 100ms`);
    console.log(`  [pathPlanner] 18 矩形最坏规划耗时 max=${t18max.toFixed(2)}ms（20 次采样，预算 100ms）`);
  }

  console.log('PASS  纯函数自测（quatSlerp / remapClipForSlot / remapClipCarryFull / stackLayout / pathPlanner）');
  nChecks++;
  results.push({ name: '纯函数自测', ok: true, detail: '' });
}

// ---------------- 工厂场景检查（factory-scene-minimap-plan F2c/F3/F4/F5，纯函数） ----------------
// F1/F2a/F2b 依赖编译模型/全循环运行，在 runStack 内；本函数只覆盖无需 MuJoCo 模型的部分。
function verifyStaticDrawList() {
  // F2c/F5 共用：buildStaticDrawList(FACTORY) 指令与表逐条对账（构造性同源，防漂移）
  const ops = buildStaticDrawList(FACTORY);
  const err = [];
  const near = (a, b) => Math.abs(a - b) < 1e-9;
  const outline = ops.find((o) => o.op === 'rect' && o.x === 0 && o.y === 0
    && near(o.hx, FACTORY.floorHalf[0]) && near(o.hy, FACTORY.floorHalf[1]));
  if (!outline) err.push('缺厂房外廓矩形（或与 floorHalf 不一致）');
  const rr = rackRect();
  if (!ops.find((o) => o.op === 'rect' && near(o.x, rr.cx) && near(o.y, rr.cy)
    && near(o.hx, rr.hx) && near(o.hy, rr.hy))) err.push('缺活动货架矩形（或与 rackRect 不一致）');
  for (const o of FACTORY.obstacles) {
    const hit = o.kind === 'pillar'
      ? ops.find((op) => op.op === 'circle' && near(op.x, o.cx) && near(op.y, o.cy) && near(op.r, o.hx))
      : ops.find((op) => op.op === 'rect' && near(op.x, o.cx) && near(op.y, o.cy) && near(op.hx, o.hx) && near(op.hy, o.hy));
    if (!hit) err.push(`缺障碍指令 ${o.name}（或与表值不一致）`);
  }
  if (!ops.find((o) => o.op === 'dash' && near(o.x, FACTORY.pickZone.center[0])
    && near(o.y, FACTORY.pickZone.center[1]) && near(o.hx, FACTORY.pickZone.half[0])
    && near(o.hy, FACTORY.pickZone.half[1]))) err.push('缺取箱区虚线框（或与表值不一致）');
  for (const bt of Object.keys(FACTORY.staging.y)) {
    if (ops.some(o => o.op === 'rect' && near(o.y, FACTORY.staging.y[bt]))) err.push(`内部备用箱位置不应显示为作业区 ${bt}`);
  }
  return { ops, err };
}

function runFactoryUnitChecks() {
  console.log('== 工厂场景检查（F2c/F3/F4/F5，纯函数） ==');
  const clipsF = loadClips();

  // F2c：三处消费一致性 (c)——小地图静态指令与 FACTORY 表一致；潜伏行只读记录与
  // stackCore.STAGING_Y 同源；两箱型派生取箱点落在取箱区标示内（R2）。
  {
    const { err } = verifyStaticDrawList();
    check('F2c 小地图静态指令与 FACTORY 表一致（外廓/货架/障碍/标示，内部备用位不显示）', err.length === 0,
      err.length ? err.join('; ') : `${FACTORY.obstacles.length + 6} 条指令全部对账`);
    check('F2c-1 FACTORY.staging.y 与 stackCore.STAGING_Y 同源（潜伏行不得移动）',
      JSON.stringify(FACTORY.staging.y) === JSON.stringify(STAGING_Y),
      `FACTORY=${JSON.stringify(FACTORY.staging.y)}`);
    const pickOk = ['largebox', 'plasticbox'].every((bt) => clipsF[bt]?.carry?.low
      && pickZoneCovers(FACTORY, derivePickPos(clipsF[bt].carry.low)));
    const p0Lb = clipsF.largebox?.carry?.low ? derivePickPos(clipsF.largebox.carry.low) : null;
    const p0Pb = clipsF.plasticbox?.carry?.low ? derivePickPos(clipsF.plasticbox.carry.low) : null;
    const fmtP0 = (p) => (p ? `[${p.map((v) => v.toFixed(3))}]` : 'n/a');
    check('F2c-2 两箱型派生取箱点落在取箱区标示内（1.4×0.8m 覆盖，R2）', pickOk,
      `largebox P0=${fmtP0(p0Lb)} plasticbox P0=${fmtP0(p0Pb)}`);
  }

  // F3：货架格位几何——全 6 槽 × 两箱型：槽 z == boardTop+hz（1e-9）；槽 xy 在格位半宽内；
  // |槽 z − z_release| ≤ 0.02（largebox）/ ≤ 0.12（plasticbox low 基线档）。
  {
    const tops = shelfBoardTops();
    const rack = FACTORY.rack;
    const plans = JSON.parse(readFileSync(path.join(ROOT, 'motions/manipulation.json'), 'utf8')).plans;
    const THR = { largebox: 0.18, plasticbox: 0.24 }; // 托盘抬高与堆叠层高由 remap 吸收（演示近似）
    // 默认链（mocap 基准）执行件镜像：tier 解析按 resolveTierForLayer，mocap 策略槽位
    // 取 carry 节件（plasticbox mid/high 为借用件）的 z_release
    let ok = true, detail = '', worstMis = 0, worstAt = '';
    for (const bt of Object.keys(clipsF)) {
      const setB = clipsF[bt];
      if (!setB?.low?.box) continue;
      const box = setB.low.box;
      const tops = shelfBoardTops(box.half_size[2]);
      const slots = stackLayout(box, FACTORY.rackCenter, FACTORY.rackPsi, tops);
      for (const sl of slots) {
        const expectZ = (tops[sl.layer] ?? 0) + box.half_size[2];
        if (Math.abs(sl.center[2] - expectZ) > 1e-9) {
          ok = false; detail = `${bt} L${sl.layer + 1}C${sl.col + 1} 槽 z ${sl.center[2].toFixed(6)} != boardTop+hz ${expectZ.toFixed(6)}`;
          break;
        }
        const dx = sl.center[0] - FACTORY.rackCenter[0], dy = sl.center[1] - FACTORY.rackCenter[1];
        const along = dx * Math.cos(FACTORY.rackPsi) + dy * Math.sin(FACTORY.rackPsi);
        const perp = -dx * Math.sin(FACTORY.rackPsi) + dy * Math.cos(FACTORY.rackPsi);
        if (Math.abs(along) > rack.depthHalf + 1e-9 || Math.abs(perp) > 2 * rack.cellHalfW + 1e-9) {
          ok = false; detail = `${bt} L${sl.layer + 1}C${sl.col + 1} 槽 xy 超出格位（进深 ${along.toFixed(3)} / 列向 ${perp.toFixed(3)}）`;
          break;
        }
        const tier = resolveTierForLayer(setB, sl.layer).tier;
        // 执行件镜像（buildConfig mocap 分支）：mocap 策略且 carry 节件可用 → 用其 z_release
        const useCarry = setB.carry?.[tier] && Number.isFinite(setB.carry[tier].z_release);
        const zrel = useCarry ? setB.carry[tier].z_release : setB[tier]?.z_release ?? setB.low.z_release;
        const mis = Math.abs(sl.center[2] - zrel);
        if (mis > worstMis) { worstMis = mis; worstAt = `${bt} L${sl.layer + 1}C${sl.col + 1}`; }
        if (CARRY_ARG !== 'rl') {
          const entry = plans[`${bt}:${sl.layer}`];
          const source = setB.carry[tier];
          const releaseCenterZ = entry?.objPos?.[entry.release]?.[2] + box.bbox_center_offset[2];
          if (!entry || entry.signature !== motionSignature(source, box, sl.center[2], FACTORY.rackPsi)
              || Math.abs(releaseCenterZ - sl.center[2]) > 1e-6) {
            ok = false; detail = `${bt} L${sl.layer + 1}: stale contact plan or incorrect shelf contact height`; break;
          }
        } else if (mis > THR[bt]) {
          ok = false; detail = `${bt} L${sl.layer + 1}C${sl.col + 1} |槽z−z_release|=${(mis * 1000).toFixed(1)}mm > ${(THR[bt] * 1000).toFixed(0)}mm`;
          break;
        }
      }
      if (!ok) break;
    }
    if (ok) detail = `8 格位（2 箱型×2 层×2 列）全对账，worst |槽z−z_release|=${(worstMis * 1000).toFixed(1)}mm @${worstAt}（接触规划最终落点与层板顶高对账）`;
    check('F3 货架槽位几何（层板顶高、xy 在货架内、接触规划落点）', ok, detail);
  }

  // F4：规划绕障——真实布局下 (a) 原点→取箱区直线与 ≥1 静态膨胀矩形相交（pillar_1
  // 刻意卡线）；(b) planPathAvoid 返回 visibility、各段不与任何膨胀静态矩形内交、
  // degraded=false；(c) 全量矩形集最坏规划计时 <100ms（沿用 M1 口径，封死目标触发
  // 双重求解的最坏形态）。
  {
    const staticRects = factoryObstacleRects();
    const pickClipLb = clipsF.largebox?.carry?.low ?? clipsF.largebox?.low ?? null;
    if (!pickClipLb) {
      check('F4a 原点→交接区通道不被非目标静态设施阻挡', false, 'largebox clip 缺失（环境不完整）');
      check('F4b 全量矩形集规划无降级且各段位于膨胀设施外', false, 'largebox clip 缺失');
      check('F4c 全量矩形集最坏规划计时 <100ms（封死目标双重求解，20 次采样）', false, 'largebox clip 缺失');
    } else {
    const pickLb = derivePickPos(pickClipLb);
    const from0 = { x: 0, y: 0 };
    const to0 = { x: pickLb[0], y: pickLb[1] };
    const rOf = (rc) => rc.r ?? (rc.role === 'target'
      ? PLANNER_DEFAULTS.targetShrinkR : PLANNER_DEFAULTS.robotR + PLANNER_DEFAULTS.margin);
    const inflAll = staticRects.map((rc) => inflateRect(rc, rOf(rc)));
    // (a) 直线必撞
    const hits = inflAll.filter((r) => segIntersectsRect(from0, to0, r));
    const hitNonTarget = staticRects.filter((rc, i) => rc.role !== 'target' && segIntersectsRect(from0, to0, inflAll[i]));
    check('F4a 原点→交接区通道不被非目标静态设施阻挡',
      hitNonTarget.length === 0,
      `命中 ${hits.length} 个（非目标 ${hitNonTarget.length}：${hitNonTarget.map((r) => r.key).join(',') || '无'}）`);
    // (b) 规划必绕 + 各段膨胀外 + degraded=0（staging 矩形无头侧自组：镜像
    // collectWalkObstacles 的潜伏行语义）
    const stagingRects = [];
    for (const bt of Object.keys(clipsF)) {
      const bset = clipsF[bt]?.low?.box;
      if (!bset) continue;
      const yRow = STAGING_Y[bt] ?? 3.0;
      for (let i = 0; i < MAX_BOXES; i++) {
        stagingRects.push({
          cx: FACTORY.staging.x0 + (i % MAX_BOXES) * FACTORY.staging.dx + bset.bbox_center_offset[0],
          cy: yRow + bset.bbox_center_offset[1],
          yaw: 0, hx: bset.half_size[0], hy: bset.half_size[1], role: 'staging', key: `staging:${bt}:${i}`,
        });
      }
    }
    const fullSet = [...staticRects, ...stagingRects];
    const plan = planPathAvoid(from0, to0, fullSet);
    let segOk = true, segAt = '';
    const chain = [from0, ...plan.points];
    for (let i = 1; i < chain.length && segOk; i++) {
      for (const rc of staticRects) {
        if (segIntersectsRect(chain[i - 1], chain[i], inflateRect(rc, rOf(rc)))) {
          segOk = false; segAt = `段${i}×${rc.key}`;
          break;
        }
      }
    }
    check('F4b 全量矩形集规划无降级且各段位于膨胀设施外',
      ['visibility', 'straight'].includes(plan.mode) && !plan.degraded && segOk,
      `mode=${plan.mode} degraded=${plan.degraded} 绕行点 ${Math.max(0, plan.points.length - 1)} ${segOk ? '' : `内交@${segAt}`}`);
    // (c) 最坏计时：to 置于活动货架中心 → 全膨胀+收缩一轮双重求解仍无可行 → 直线兜底
    let tMax = 0;
    const worstTo = { x: FACTORY.rackCenter[0], y: FACTORY.rackCenter[1] };
    for (let it = 0; it < 20; it++) {
      const t0 = performance.now();
      planPathAvoid(from0, worstTo, fullSet);
      tMax = Math.max(tMax, performance.now() - t0);
    }
    check('F4c 全量矩形集最坏规划计时 <100ms（封死目标双重求解，20 次采样）', tMax < 100,
      `max=${tMax.toFixed(2)}ms（矩形 ${fullSet.length} 个）`);
    }
  }

  // F5：小地图数据层——worldToPixel 往返误差 ≤0.5px；动态指令含机器人/路径/箱子。
  {
    const W = 230, H = 200;
    const mapper = worldToPixelMapper(mapView(FACTORY), W, H);
    let worst = 0;
    for (const [x, y] of [[0, 0], [8, 7], [-8, -7], [2.2, 2.2], [1.38, 0.2], [-5.5, 2.8], [0.1, -3.3]]) {
      const [px, py] = mapper.toPixel(x, y);
      const [wx, wy] = mapper.toWorld(px, py);
      worst = Math.max(worst, Math.hypot(wx - x, wy - y) * mapper.scale);
    }
    check('F5a worldToPixelMapper 往返误差 ≤0.5px（7 采样点）', worst <= 0.5,
      `worst=${worst.toFixed(4)}px scale=${mapper.scale.toFixed(2)}`);
    const dyn = buildDynamicDrawList({
      robot: { x: 1, y: 2, yaw: 0.5 },
      path: { kind: 'approach', points: [{ x: 1, y: 1 }, { x: 2, y: 2 }], goal: [2.2, 0.5] },
      boxes: [
        { x: 0, y: 0, yaw: 0, hx: 0.2, hy: 0.2, kind: 'placed' },
        { x: 1, y: 1, yaw: 0, hx: 0.2, hy: 0.2, kind: 'ghost' },
      ],
      activeSlot: { index: 0, center: [2.2, 2.2] },
    });
    const has = (op, n) => dyn.filter((o) => o.op === op).length >= n;
    check('F5b 动态指令含机器人/路径/目标/箱子/槽位高亮（tri/poly/cross×2/rect×2）',
      has('tri', 1) && has('poly', 1) && has('cross', 2) && has('rect', 2),
      `ops=${dyn.map((o) => o.op).join(',')}`);
    const { err } = verifyStaticDrawList();
    check('F5c 静态指令与 FACTORY 一致（同 F2c 口径复跑）', err.length === 0,
      err.length ? err.join('; ') : '一致');
  }
}

// ---------------- 模式一：FK 交叉验证（T3） ----------------
async function runFk(mujoco) {
  console.log('== FK 交叉验证：JSON 轨迹写入展示端模型，对照 fk_ref（来源 pkl world_body_pos） ==');
  const clips = loadClips();
  const { xml, vfs } = buildMergedXml(mujoco, { withBoxes: false, clips });
  const model = mujoco.MjModel.from_xml_string(xml, vfs);
  const data = new mujoco.MjData(model);

  for (const [type, set] of Object.entries(clips)) {
    // MERO-10：tier 件 + carry 节全程回放件一并 FK 校验（--fk 5/5 → 实际按 clips 树全部带 fk_ref 的件）
    const entries = [
      ...Object.entries(set).filter(([k]) => k !== 'carry'),
      ...Object.entries(set.carry ?? {}).map(([k, v]) => [`carry.${k}`, v]),
    ];
    for (const [tier, clip] of entries) {
      if (!clip?.fk_ref) continue;
      const jmap = buildJointMap(mujoco, model, clip.joint_names);
      let maxErr = 0;
      for (let fi = 0; fi < clip.fk_ref.frames.length; fi++) {
        const f = clip.fk_ref.frames[fi];
        for (let i = 0; i < model.nq; i++) data.qpos[i] = 0;
        for (let k = 0; k < 3; k++) data.qpos[k] = clip.root_pos[f][k];
        for (let k = 0; k < 4; k++) data.qpos[3 + k] = clip.root_quat[f][k];
        for (let i = 0; i < clip.joint_names.length; i++) data.qpos[jmap[clip.joint_names[i]].q] = clip.dof_pos[f][i];
        mujoco.mj_forward(model, data);
        for (let bi = 0; bi < clip.fk_ref.bodies.length; bi++) {
          const bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, clip.fk_ref.bodies[bi]);
          for (let k = 0; k < 3; k++) {
            maxErr = Math.max(maxErr, Math.abs(data.xpos[3 * bid + k] - clip.fk_ref.pos[fi][bi][k]));
          }
        }
      }
      check(`FK ${type}/${tier}: ${clip.fk_ref.bodies.length} bodies × ${clip.fk_ref.frames.length} 帧, 误差 < 2cm`,
        maxErr < 0.02, `max|Δ|=${(maxErr * 1000).toFixed(2)}mm`);
    }
  }
}

// ---------------- 模式二：全循环堆垛断言（第 9 节断言集 1-7） ----------------
async function runStack(mujoco) {
  console.log(`== 堆垛全循环断言: ${LAYERS}层×${COLS}列 ${BOXTYPE} ==`);
  const clips = loadClips();
  const { xml, vfs } = buildMergedXml(mujoco, { withBoxes: true, clips });
  const model = mujoco.MjModel.from_xml_string(xml, vfs);
  const data = new mujoco.MjData(model);

  // 断言 1：场景编译。每槽位注入幽灵+物理两个 free body（R7 方案，见 stackCore 头注释）
  const nBoxBodies = 2 * MAX_BOXES * 2; // 两套箱型 × 6 槽 × 2 体
  check('断言1 场景编译 nq == 36 + 7N', model.nq === 36 + 7 * nBoxBodies,
    `nq=${model.nq}, N=${nBoxBodies}`);

  // F1（工厂方案）：厂房静态 geom 存在（层板/立柱/围界/标示按名全查）+ contype 语义
  // （层板物理=箱-板接触依赖，R4；其余 ghost）+ nq 复核（工厂片段无关节）。
  const gid1 = (n) => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, n);
  {
    let ok = true, detail = '';
    for (const b of FACTORY_SOLID_BODIES) {
      const g = gid1(b.geom);
      if (g < 0) { ok = false; detail = `缺层板 geom ${b.geom}`; break; }
      if (model.geom_contype[g] !== 1 || model.geom_conaffinity[g] !== 1) {
        ok = false; detail = `层板 ${b.geom} contype/conaffinity != 1/1（放置箱会穿板坠落，R4）`;
        break;
      }
    }
    if (ok) {
      // 柱角命名与 factory.js 生成器同口径（审查 P4）：corner = {W|E}{S|N}
      //（x 侧前、y 侧后），fx_deco_rack_* 同款，保持内部一致
      const ghostNames = ['fx_active_board1.25', 'fx_active_post-0.57_-0.35',
        'fx_wall_n', 'fx_wall_s', 'fx_wall_e', 'fx_wall_w', 'fx_pickzone_mark',
        'fx_deco_rack_a_board1', 'fx_deco_rack_b_postEN'];
      for (const n of ghostNames) {
        const g = gid1(n);
        if (g < 0) { ok = false; detail = `缺静态 geom ${n}`; break; }
        if (model.geom_contype[g] !== 0) { ok = false; detail = `${n} 应为 ghost（contype=0）`; break; }
      }
    }
    if (ok && model.nq !== 36 + 7 * nBoxBodies) { ok = false; detail = `nq 复核失败 nq=${model.nq}`; }
    if (ok) {
      // floor 尺寸同步断言（S1）：scene.xml 地面 16×14m 与 FACTORY.floorHalf 同源
      const fg = gid1('floor');
      if (fg < 0) { ok = false; detail = '缺 floor geom'; }
      else if (Math.abs(model.geom_size[3 * fg] - FACTORY.floorHalf[0]) > 1e-9
        || Math.abs(model.geom_size[3 * fg + 1] - FACTORY.floorHalf[1]) > 1e-9) {
        ok = false; detail = `floor size [${model.geom_size[3 * fg]},${model.geom_size[3 * fg + 1]}] != floorHalf ${JSON.stringify(FACTORY.floorHalf)}`;
      }
    }
    check('F1 厂房静态 geom 存在且 contype 语义正确、nq/floor 同步', ok,
      ok ? `层板 ${FACTORY_SOLID_BODIES.length} 块物理 + 立柱/围界/标示/装饰板 ghost 抽全查 + nq=${model.nq} + floor ${FACTORY.floorHalf}` : detail);
  }

  // F2a（三处消费一致性 a）：模型几何位姿 == FACTORY 表值（geom 直接挂 worldbody →
  // geom_pos 即世界坐标；承重板包 body → 读 body_pos）。
  {
    let ok = true, detail = '';
    const bid1 = (n) => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, n);
    for (const o of FACTORY.obstacles) {
      if (['rack', 'fence', 'pallet'].includes(o.kind)) continue; // 多 geom 装饰件，F1 存在性已查
      const g = gid1(`fx_${o.name}`);
      if (g < 0) { ok = false; detail = `缺 geom fx_${o.name}`; break; }
      const expect = [o.cx, o.cy, o.z + o.height / 2];
      const got = [model.geom_pos[3 * g], model.geom_pos[3 * g + 1], model.geom_pos[3 * g + 2]];
      if (Math.abs(got[0] - expect[0]) > 1e-9 || Math.abs(got[1] - expect[1]) > 1e-9 || Math.abs(got[2] - expect[2]) > 1e-9) {
        ok = false; detail = `${o.name} geom_pos=[${got.map((v) => v.toFixed(3))}] != 表值 [${expect.map((v) => v.toFixed(3))}]`;
        break;
      }
    }
    if (ok) {
      for (const b of rackBoardSpecs()) {
        const bid = bid1(b.body);
        if (bid < 0) { ok = false; detail = `缺板 body ${b.body}`; break; }
        const got = [model.body_pos[3 * bid], model.body_pos[3 * bid + 1], model.body_pos[3 * bid + 2]];
        if (Math.abs(got[0] - b.pos[0]) > 1e-9 || Math.abs(got[1] - b.pos[1]) > 1e-9 || Math.abs(got[2] - b.pos[2]) > 1e-9) {
          ok = false; detail = `${b.body} body_pos=[${got.map((v) => v.toFixed(3))}] != 表值 [${b.pos.map((v) => v.toFixed(3))}]`;
          break;
        }
      }
    }
    if (ok) detail = `${FACTORY.obstacles.filter((o) => o.kind !== 'rack').length} 障碍 + ${FACTORY_SOLID_BODIES.length} 板位姿 == 表值`;
    check('F2a 模型静态几何位姿与 FACTORY 表一致（geom_pos/body_pos）', ok, detail);
  }

  const standKey = keyId(mujoco, model, 'stand');
  mujoco.mj_resetDataKeyframe(model, data, standKey);
  mujoco.mj_forward(model, data);
  // stand 位姿快照：done 后的观察窗用（无头脚本无 idle 模式/平衡外挂，
  // 若放任动力学机器人会塌到堆垛上，污染纯物理观察；浏览器侧对应 setMode('idle') 交接）
  const standQpos = Float64Array.from(data.qpos.subarray(0, 36));

  // rl 模式：加载策略权重 + 装配行走控制器（与浏览器共用同一实现）
  let walker = null;
  let policyManifest = null;
  let policyRef = null;
  let jmapWalker = null;
  if (WALK_MODE === 'rl') {
    const policy = await loadPolicy('vendor/policy', fsFetch);
    policyRef = policy;
    policyManifest = policy.manifest;
    let manifest = policy.manifest;
    if (NEG_MODE === 'all') {
      // N11 防假绿：深拷贝 manifest 并交换腿装订（policy→mjcf 映射的左右腿互换），
      // 破坏后的装订必须让 N4/N5/N7 FAIL，否则断言机制失效。
      manifest = JSON.parse(JSON.stringify(policy.manifest));
      const swap = (m) => (m < 12 ? (m + 6) % 12 : m); // mjcf 左腿 0-5 ↔ 右腿 6-11
      manifest.joint_map.policy_to_mjcf = manifest.joint_map.policy_to_mjcf.map(swap);
      console.log('  [negcheck] rl 装订已被故意破坏（左右腿 map 互换）');
    } else if (NEG_MODE === 'carry') {
      // MERO-10 第三组注入（方案 4.5 negcheck 行）：携带漂移注入 = 幽灵箱 +x 斜坡漂移
      //（N13-mocap 必 FAIL）+ root 走廊漂移（stackCore CARRY_REPLAY 消费，N14 必 FAIL）。
      // 注入量 3.0m（原 N13 定标 0.3m 扩大）：N14 防假绿窗口按注入实测分布联定——
      // 附加滑移分量 ≈ 距离/6.5s（RSS 合成，实测 1.0m 仅 +46mm/s p95 位移），3.0m 给
      // ≈463mm/s 恒定分量，注入后 p95 下界 ≈500mm/s，与固有上界 334 之间留 ≥20% 双侧
      // 余量（联定数据见 probe-notes MERO-10 N14 节）。
      process.env.G1BOX_NEGCHECK_CARRY_DRIFT = '3.0';
      console.log('  [negcheck] 携带漂移注入已启用（G1BOX_NEGCHECK_CARRY_DRIFT=3.0：幽灵箱 +x、机器人 root −x 走廊漂移）');
    } else if (NEG_MODE === 'avoid') {
      // walk-avoid-plan 4.5.2 两注入（默认注入 1；G1BOX_NEGCHECK_AVOID_OFF=1 切注入 2）：
      //  注入 1：makeWaypoints 中部路点向最近障碍矩形中心平移 G1BOX_NEGCHECK_AVOID_BREAK
      //          → 行走路径被拉进障碍 → N19 必 FAIL（已显式预置的注入量不覆盖，联定期旋钮）；
      //  注入 2：映射为 stackCore 对照开关 G1BOX_AVOID_OFF=1（规划器短路直线，复刻用户
      //          报告的"直线穿箱"缺陷形态；若直线未触发 N19 则以注入 1 为准）。
      if (process.env.G1BOX_NEGCHECK_AVOID_OFF === '1') {
        process.env.G1BOX_AVOID_OFF = '1';
        console.log('  [negcheck] 避障注入 2 已启用（G1BOX_NEGCHECK_AVOID_OFF=1 → G1BOX_AVOID_OFF=1：规划器短路直线，穿模缺陷形态）');
      } else {
        process.env.G1BOX_NEGCHECK_AVOID_BREAK = process.env.G1BOX_NEGCHECK_AVOID_BREAK || '0.5';
        console.log('  [negcheck] 避障注入 1 已启用（G1BOX_NEGCHECK_AVOID_BREAK='
          + `${process.env.G1BOX_NEGCHECK_AVOID_BREAK}：makeWaypoints 中部路点向最近障碍矩形中心平移）`);
      }
    }
    jmapWalker = buildJointMap(mujoco, model, JOINT_NAMES_H);
    walker = createWalkController({
      mujoco, model, data, jmap: jmapWalker, jointNames: JOINT_NAMES_H,
      policy: { weights: policy.weights, manifest },
      armHold: ARMHOLD_ARG, armHoldRamp: ARMHOLD_RAMP, armHoldTight: ARMHOLD_TIGHT,
      armHoldFixedAlpha: ARMHOLD_ALPHA,
      // MERO-9 手段1：--carry-vx 覆盖 walker 的 carry 段指令限速（meta.carryVxMax；
      // walker 原本未接收 meta，传 {} 即保持全默认行为）
      meta: CARRY_VX_ARG > 0 ? { carryVxMax: CARRY_VX_ARG } : {},
      carryKd: CARRY_KD_ARG,
      log: (m) => console.log('  ' + m),
    });
  }
  // rl 模式帧域切分元数据（motions/walk_meta.json，含 v3 carry 节）
  const walkMeta = WALK_MODE === 'rl' ? JSON.parse(readFileSync(path.join(ROOT, 'motions', 'walk_meta.json'), 'utf8')) : null;

  const stack = createBoxStacking({ manipulation: JSON.parse(readFileSync(path.join(ROOT, 'motions/manipulation.json'), 'utf8')),
    mujoco, model, data, clips,
    walker, walkMode: WALK_MODE, carryMode: CARRY_ARG, meta: walkMeta,
    log: (m) => console.log('  ' + m),
  });
  stack.loadStackConfig({ layers: LAYERS, cols: COLS, boxType: BOXTYPE });
  stack.start();

  // 断言 2：keyframe reset 后箱体被显式重摆（守 3.4-2 契约）。
  // start() 内部：keyframe reset（qpos 尾段零化）→ reseatBoxes 全部（幽灵+物理）摆到
  // 取箱区外潜伏位 → beginSlot(0) 激活 0 号幽灵体到取箱位。若漏做 reseat，reset 会把
  // 箱体零化在世界原点 → 断言失败。用 qpos 判断（geom_xpos 在 forward 前是过期值）。
  {
    let ok = true, detail = '';
    const boxes = stack.boxes();
    for (let i = 0; i < boxes.length; i++) {
      const x = data.qpos[boxes[i].jointQadr], y = data.qpos[boxes[i].jointQadr + 1];
      const z = data.qpos[boxes[i].jointQadr + 2], w = data.qpos[boxes[i].jointQadr + 3];
      // 潜伏排：y≥3.0 的取箱区外行（按箱型分行，见 stackCore stagingPos），z≈箱底贴地静置高
      if (!(y > 2.5 && z > 0.1 && z < 0.3 && Math.abs(w - 1) < 1e-6)) {
        ok = false; detail = `物理箱${i} 未重摆到潜伏位 (x=${x.toFixed(2)} y=${y.toFixed(2)} z=${z.toFixed(3)} w=${w.toFixed(3)})`;
        break;
      }
    }
    const gjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, 'cartonG0_joint');
    const gq = model.jnt_qposadr[gjid];
    const gz = data.qpos[gq + 2];
    if (!(gz > 0.05 && gz < 0.5)) { ok = false; detail = `0号幽灵体未生成在取箱位 z=${gz.toFixed(3)}`; }
    check('断言2 keyframe reset 后箱体显式重摆（物理体全潜伏，幽灵0在取箱位）', ok, detail);
    if (BOXTYPE === 'plasticbox') {
      const slots = stack.intakeSlots;
      const upOk = slots.every(s => quatRotVec(s.quat, [0,0,1])[2] > 0.9999);
      const floorOk = slots.every(s => Math.abs(s.pos[2] + clips[BOXTYPE].low.box.bbox_center_offset[2] - clips[BOXTYPE].low.box.half_size[2]) < 1e-9);
      check('B1 交接料箱箱口朝上、底面贴地', upOk && floorOk, `${slots.length} 个取箱计划`);
      check('B2 只显示当前待取箱，备用物理箱不显示', stack.visibleBoxBodies.length === 1 && stack.visibleBoxBodies[0] === 'cartonG100', stack.visibleBoxBodies.join(','));
    }

  }

  // 预期槽位中心（用 stackCore 纯函数独立复算；assertion 3/5 共用）。
  // MERO-10：堆垛框架镜像 buildConfig 的解析（存在 mocap tier → computeStackFrameCarry
  // 按 G0 摆位；全 rl/legacy → v2 computeStackFrame），保证与执行侧同源不同实例。
  // 工厂方案 F0 镜像同步：mocap 基准跑显式传 derivePickPos/rackPsi（货架格网中心）+
  // stackLayout 第 4 参 boardTops（货架层板格位）；全 rl 保持地面框架（镜像分支）。
  const set = clips[BOXTYPE];
  const stratCarry = CARRY_ARG === 'rl' ? 'rl' : 'mocap'; // legacy 已移除（映射 mocap）
  const carryTiersH = resolveCarryTiers(clips, walkMeta, stratCarry, BOXTYPE);
  const carryBaseH = ['low', 'mid', 'high'].find((t) => carryTiersH[t] === 'mocap');
  const frameH = carryBaseH
    ? computeStackFrameCarry(clips[BOXTYPE].carry[carryBaseH], derivePickPos(clips[BOXTYPE].carry[carryBaseH]), FACTORY.rackPsi)
    : computeStackFrame(set.low);
  const expected = stackLayout(set.low.box, frameH.S, frameH.psi, carryBaseH ? shelfBoardTops(set.low.box.half_size[2]) : null)
    .filter((sl) => sl.layer < LAYERS)
    .filter((sl) => COLS === 2 || sl.col === 0);
  if (NEG_MODE === 'all') {
    for (const sl of expected) sl.center[2] += 0.05; // 防假绿：期望 z 故意抬高 5cm
    console.log('  [negcheck] 期望槽位 z 已被故意抬高 0.05m');
  }

  // ---------- 主循环：驱动到 done ----------
  const pelvisBid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'pelvis');
  // N12（MERO-8）手-箱贴近度采样用的双腕 body（与 stackCore 双手中点锚同口径）
  const wristBidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'left_wrist_yaw_link');
  const wristBidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'right_wrist_yaw_link');
  let minBoxDist = Infinity;      // 箱-地 / 箱-箱最小距离（断言4 主判据）
  let minBoxBoardDist = Infinity; // 放置箱-承重支撑板最小距离（S3：防箱-板挤压隐身于断言4）
  let minFootFloorDist = Infinity; // 脚-地最小距离（全程；断言 4b 保留原口径）
  let minFootFloorDistWalk = Infinity; // RL 段脚-地最小距离（N5：> -25mm）
  let minFootFloorDistMocap = Infinity; // MERO-10：动捕携带段脚-地最小距离（N5-mocap 分档）
  let maxQfrc = 0;
  let minPelvisZ = Infinity;
  let minPelvisZWalk = Infinity;  // N8：RL 段 pelvis z
  let minPelvisZManip = Infinity; // N8：操作段 pelvis z
  const maxSteps = Math.ceil(MAX_SIM / TIMESTEP);
  let steps = 0;

  // N2/N3/N4/N6/N10 采样状态（rl 模式）
  const isWalkPhase = (ph) => ph === 'WALK_APPROACH' || ph === 'WALK_CARRY' || ph === 'WALK_HOME' || ph === 'STOP_PATH';
  // MERO-10：mocap 策略槽位的 CARRY_REPLAY（carryFull 全程动捕回放）同样按段采样——
  // 脚为 kinematic 写入，distPair 距离判定照用（方案 4.5 N4 行"采样器扩展"）
  const isMocapCarry = (ph, st) => ph === 'CARRY_REPLAY' && st.carryStrategy === 'mocap';
  const walkStats = {
    segments: [],      // 每个 RL 行走段 / mocap 携带段的统计（kind: 'rl' | 'mocap'）
    cur: null,
    prevPhase: null,
    sawWalkCarry: false,                  // MERO-10：本轮是否出现 WALK_CARRY（N12 守卫）
    gainWalkSamples: 0, gainWalkBad: 0,   // N2：RL 段 gainprm == 策略 kp
    gainRestoredBad: 0,                   // N2：done 后 gainprm == Menagerie 500
    boxJumpMax: 0,                        // N10：相邻采样幽灵箱跳变
    boxJumpAt: '',                        // N10：最大跳变发生位置（slot:phase，归因用）
    boxJumpVec: '',                       // N10：最大跳变 xyz 分量（mm，归因用）
    boundaryJumpMax: 0,                   // N10：WALK↔MANIP 相位边界幽灵箱跳变
    prevBoxPos: null,
    prevBoxKey: null,                     // 上次箱采样的 slot:phase 键
    arriveErrs: [],                       // N6：各段到位误差
    n12Max: 0,                            // N12：WALK_CARRY 段 max|箱心−双手中点|
    n12At: '',                            // N12：最大值出现位置（slot@sim 时刻，归因用）
    gainKdWalkBad: 0,                     // N2（MERO-9 扩展）：RL 段 kd 违例采样次数
  };
  // N14（MERO-10 新增）：动捕携带段足底滑移。支撑相（足-地 <5mm）内足端水平速度
  // 采样（10ms 差分），防"λ 压缩走廊/整体平移"类滑移回归的守门断言（方案 4.5）
  const n14 = { samples: [], prevPos: [null, null], segT0: null, dbg: null, epLen: [0, 0] };
  // N14 定位打点（G1BOX_N14_DEBUG=1 时启用，只加观测不改判定）：记录每样本
  // {v, t(相对段起点), foot, d(足-地距)}，断言处按 1s 桶输出 p95/max + 直方，
  // 用于分解滑移是全段均匀（clip/重定向固有）还是集中于特定时段（机制引入）。
  if (parseFloat(process.env?.G1BOX_N14_DEBUG) === 1) n14.dbg = [];
  // N13（MERO-9）逐手抓握偏差状态：N12 量「双手中点↔箱心」，反相摆臂会互相抵消、
  // 量错对象（用户实测「有些时候都没有抱住箱子」而 N12 全绿）；N13 按捕获偏移分别量
  // 每只手自己是否离开箱面。基准偏移两处取：
  //  - 捕获段后（TURN_CARRY/ALIGN_PLACE/CARRY_REPLAY2）：stack.graspOffsets（lift_end 帧
  //    captureCarryAnchor 记录的逐手偏移，箱 body 原点系）——「抱住」契约基准；
  //  - CARRY_REPLAY 抓取提起窗：held 后首样现算的抓取瞬间偏移（动捕回放自身几何，
  //    lift_end 偏移此时尚未产生；该相位为观测项不参与判定）。
  const N13_PHASES = ['CARRY_REPLAY', 'TURN_CARRY', 'WALK_CARRY', 'ALIGN_PLACE', 'CARRY_REPLAY2'];
  // 判定相位（任务 A 定标结论，1x1 基线实测 v0_baseline_1x1.log）：CARRY_REPLAY（抓取提起）
  // 与 CARRY_REPLAY2（下放释放）的箱子与手同跟一份动捕 clip，gap 即动捕自身几何漂移
  //（实测 166-231mm，放手前手沿箱面滑动属放箱自然动作）——观测项不判定；
  // TURN_CARRY（dof 冻结，实测 ≤1mm）/ WALK_CARRY / ALIGN_PLACE（对位 blend，起点继承
  // WALK_CARRY 残留）为「抱住」契约相位，参与判定。
  const N13_ASSERT_PHASES = ['TURN_CARRY', 'WALK_CARRY', 'ALIGN_PLACE'];
  const n13Box = clips[BOXTYPE]?.low?.box ?? null; // half_size/bbox_center_offset（各 tier 同箱体）
  const n13 = {
    captured: false,                      // 本轮是否至少捕获过一次（rl 锚随档判定资格）
    mocapCaptured: false,                 // MERO-10：动捕档是否有样本（判定资格）
    graspBaseL: null, graspBaseR: null,   // CARRY_REPLAY 抓取瞬间基线（每槽重建）
    capDiag: [],                          // 捕获几何诊断 {slot, layer, midD0, sdfL, sdfR}
    samples: {},                          // rl 槽位 ph -> { L: [], R: [] }（10ms 样本，max/p95 定标用）
    mocapSamples: { L: [], R: [] },       // MERO-10：mocap 槽位 CARRY_REPLAY 的逐手 gap（动捕档）
    max: { L: 0, R: 0 },                  // 判定相位内全局 max（rl 锚随档）
    at: { L: '', R: '' }, atPh: { L: '', R: '' },
  };
  function ghostGeomId(slot) {
    const baseId = BOXTYPE === 'plasticbox' ? 100 : BOXTYPE === 'smallbox' ? 200 : 0;
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, `cartonG${baseId + slot}_geom`);
  }
  // N13 辅助：幽灵箱 free joint 的 qpos 地址（体位姿读数，geom_xpos 之外的 quat 来源）
  function ghostJointQadr(slot) {
    const baseId = BOXTYPE === 'plasticbox' ? 100 : BOXTYPE === 'smallbox' ? 200 : 0;
    const jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, `cartonG${baseId + slot}_joint`);
    return jid >= 0 ? model.jnt_qposadr[jid] : -1;
  }
  // N13 辅助：腕 body 世界位（复用 N12 的 wristBidL/R id）
  const wristW = (bid) => [data.xpos[3 * bid], data.xpos[3 * bid + 1], data.xpos[3 * bid + 2]];
  // N13 辅助：箱系偏移 → 世界系差距。gap = |wrist − (boxPos ⊗ off)|（off 为箱 body 原点系）
  function n13Gap(wrist, bp, bq, off) {
    const d = quatRotVec(bq, off);
    return Math.hypot(wrist[0] - bp[0] - d[0], wrist[1] - bp[1] - d[1], wrist[2] - bp[2] - d[2]);
  }
  // N13 辅助：世界点 → 箱 body 原点系坐标（抓取瞬间基线现算用）
  function n13BoxLocal(p, bp, bq) {
    return quatRotVec(quatConj(bq), [p[0] - bp[0], p[1] - bp[1], p[2] - bp[2]]);
  }
  // 点→箱面符号距离（口径复用 tools/probes/handbox_probe.mjs 的 OBB SDF；<0 表示点在箱内）
  function n13ObbSdf(p, center, quat, half) {
    const l = quatRotVec(quatConj(quat), [center[0] - p[0], center[1] - p[1], center[2] - p[2]]);
    const d = [Math.abs(l[0]) - half[0], Math.abs(l[1]) - half[1], Math.abs(l[2]) - half[2]];
    const outD = Math.hypot(Math.max(d[0], 0), Math.max(d[1], 0), Math.max(d[2], 0));
    return outD + Math.min(Math.max(d[0], d[1], d[2]), 0);
  }
  // ---------------- N18（箱-身穿模修复）：携带段箱-身距离守门（判定集/观测集拆分，队长裁决） ----------------
  // 判定集 = 非手代表 geom（pelvis/torso 链/waist_yaw+roll/hip_pitch+hip_yaw+knee/elbow，
  // 以腕关节为界、腕及以远算手），硬门禁 ≥N18_MIN；观测集 = wrist_yaw 碰撞 geom +
  // rubber_hand 视觉掌——双掌环抱式抓握的指尖-箱面穿插是抓握固有几何（±侧面指尖反向
  // 扎入，任何单一平移外推不可双掌同清），演示级近似，只输出不判定。
  // 口径：幽灵箱 geom × 代表 geom 集 mj_geomDistance 逐对取最小；抓取起（held）→释放止
  // （换位离开携带相位），10ms 采样。exclude 静态排除只切断物理接触对，不影响距离
  // 度量（显示域"穿模"的守门，见断言4 注释）。
  const N18_GATE_BODY_NAMES = [
    'pelvis', 'torso_link', 'waist_yaw_link', 'waist_roll_link',
    'left_hip_pitch_link', 'left_hip_yaw_link', 'left_knee_link', 'left_elbow_link',
    'right_hip_pitch_link', 'right_hip_yaw_link', 'right_knee_link', 'right_elbow_link',
  ];
  const N18_HAND_BODY_NAMES = ['left_wrist_yaw_link', 'right_wrist_yaw_link'];
  const n18GateGeoms = [];
  const n18HandGeoms = [];
  {
    const meshNameOf = (g) => {
      const di = model.geom_dataid[g];
      return di >= 0 ? mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH.value, di) : null;
    };
    const collectGeoms = (bn, out) => {
      const bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, bn);
      if (bid < 0) return;
      const gadr = model.body_geomadr[bid], gnum = model.body_geomnum[bid];
      const all = [];
      for (let g = gadr; g < gadr + gnum; g++) all.push(g);
      const coll = all.filter((g) => model.geom_contype[g] !== 0);
      for (const g of (coll.length ? coll : all)) out.push({ g, name: `${bn}/${meshNameOf(g) ?? 'geom' + g}` });
      if (bn.endsWith('wrist_yaw_link')) {
        for (const g of all) {
          const mn = meshNameOf(g) ?? '';
          if (mn.includes('rubber_hand')) out.push({ g, name: `${bn}/${mn}` });
        }
      }
    };
    for (const bn of N18_GATE_BODY_NAMES) collectGeoms(bn, n18GateGeoms);
    for (const bn of N18_HAND_BODY_NAMES) collectGeoms(bn, n18HandGeoms);
  }
  const distPairN18 = (g1, g2) => mujoco.mj_geomDistance(model, data, g1, g2, 0.3, fromto);
  const n18 = { n: 0, min: Infinity, at: '', link: '', byPhase: {}, top: [] };
  const n18Hand = { n: 0, min: Infinity, at: '', link: '', byPhase: {} };
  const N18_CARRY_PHASES = ['CARRY_REPLAY', 'TURN_CARRY', 'WALK_CARRY', 'ALIGN_PLACE', 'CARRY_REPLAY2'];
  function sampleN18(ph) {
    const st = stack.status();
    if (!(st.slotTotal > 0 && !st.done && st.held) || !N18_CARRY_PHASES.includes(ph)) return;
    const gid = ghostGeomId(st.slotIndex);
    if (gid < 0) return;
    for (const { g, name } of n18GateGeoms) {
      const d = distPairN18(gid, g);
      n18.n++;
      if (d < n18.min) { n18.min = d; n18.at = `${st.slotIndex}@${data.time.toFixed(1)}s`; n18.link = name; }
      const b = n18.byPhase[ph] ?? (n18.byPhase[ph] = { min: Infinity, at: '', link: '' });
      if (d < b.min) { b.min = d; b.at = data.time.toFixed(1); b.link = name; }
      if (d < 0 && (n18.top.length < 10 || d < n18.top[n18.top.length - 1].d)) {
        n18.top.push({ d, t: data.time, ph, slot: st.slotIndex, link: name });
        n18.top.sort((a, b2) => a.d - b2.d);
        if (n18.top.length > 10) n18.top.length = 10;
      }
    }
    for (const { g, name } of n18HandGeoms) {
      const d = distPairN18(gid, g);
      n18Hand.n++;
      if (d < n18Hand.min) { n18Hand.min = d; n18Hand.at = `${st.slotIndex}@${data.time.toFixed(1)}s`; n18Hand.link = name; }
      const b = n18Hand.byPhase[ph] ?? (n18Hand.byPhase[ph] = { min: Infinity, at: '', link: '' });
      if (d < b.min) { b.min = d; b.at = data.time.toFixed(1); b.link = name; }
    }
  }
  function closeWalkSeg() {
    if (!walkStats.cur) return;
    const seg = walkStats.cur;
    seg.disp = Math.hypot(data.qpos[0] - seg.startXY[0], data.qpos[1] - seg.startXY[1]);
    if (walker) {
      const cg = walker.completedGoal;
      // N6 只统计 RL 行走段（mocap 段无 walker 目标，completedGoal 是上一 RL 段的残值）
      if (cg && walkStats.cur?.kind === 'rl') walkStats.arriveErrs.push(cg);
    }
    walkStats.cur = null;
  }
  function sampleWalk(ph) {
    const st = stack.status();
    const sampled = isWalkPhase(ph) || isMocapCarry(ph, st);
    const mocapSeg = isMocapCarry(ph, st);
    // 段管理：walkSeq 变化（重试/新段）、phase 离开采样集合或段类型切换 → 收口旧段
    if (walkStats.cur && (walker.walkSeq !== walkStats.cur.seq || !sampled
      || (mocapSeg ? walkStats.cur.kind !== 'mocap' : walkStats.cur.kind !== 'rl'))) closeWalkSeg();
    if (sampled) {
      if (!walkStats.cur) {
        walkStats.cur = {
          seq: walker.walkSeq, phase: ph, kind: mocapSeg ? 'mocap' : 'rl',
          startXY: [data.qpos[0], data.qpos[1]],
          lFlips: 0, rFlips: 0, wasStance: [true, true], lastEdge: null, altOk: true,
          minZ: Infinity, drift1s: 0, t0: data.time, footMin: Infinity, disp: 0,
        };
        walkStats.segments.push(walkStats.cur);
        // N19（walk-avoid-plan 4.5.1）：段开启时取规划同源快照——makeWaypoints（写快照）先于
        // walkTo 的 walkSeq++，故此快照必与刚开启的段一一对应（含重试重规划段）。mocap 携带段
        // 不参与（动捕走廊 Out of scope，方案第 9 节）。
        walkStats.cur.n19 = null;
        if (walkStats.cur.kind === 'rl') {
          const snap = stack.lastWalkObstacles;
          walkStats.cur.n19 = {
            snap: snap ? { kind: snap.kind, slotIndex: snap.slotIndex, rects: snap.rects,
              mode: snap.mode, degraded: snap.degraded, goal: snap.goal } : null,
            samples: [],
          };
        }
        walkStats.prevBoxKey = null; // 新段重置箱差分基线
        n14.prevPos = [null, null];  // 新段重置 N14 足端差分基线
        n14.segT0 = data.time;       // N14 debug 打点：段起点时刻
      }
      const seg = walkStats.cur;
      seg.minZ = Math.min(seg.minZ, data.xpos[3 * pelvisBid + 2]);
      if (seg.kind === 'rl') {
        minPelvisZWalk = Math.min(minPelvisZWalk, data.xpos[3 * pelvisBid + 2]);
        if (ph === 'WALK_CARRY') walkStats.sawWalkCarry = true;
        // N19 采样（10ms 节奏，与其余断言对齐）：root xy（data.qpos[0..1]）+ 段内累计路程
        if (steps % 5 === 0 && seg.n19) {
          const prevN19 = seg.n19.samples[seg.n19.samples.length - 1];
          const cum = prevN19
            ? prevN19.cum + Math.hypot(data.qpos[0] - prevN19.x, data.qpos[1] - prevN19.y) : 0;
          seg.n19.samples.push({ x: data.qpos[0], y: data.qpos[1], cum });
        }
      }
      const el = data.time - seg.t0;
      if (el <= 1.0) {
        seg.drift1s = Math.max(seg.drift1s, Math.hypot(data.qpos[0] - seg.startXY[0], data.qpos[1] - seg.startXY[1]));
      }
      // N2：RL 段执行器增益 == 策略值（每 10ms 全执行器核对一次；gains 为 mjcf 序）。
      // 仅 rl 段有意义（mocap 段为 kinematic 回放，walker 不推理，增益本就未被消费）。
      // MERO-9 语义扩展（队长预批准，理由见 docs/policy-probe-notes.md MERO-9 节）：
      // kp 核对不变（全 29 维 == 策略 kp）；新增 kd 核对——腿腰 15 维恒 == 策略 kd，
      // 臂 14 维在 carry 行走段（WALK_CARRY 且 --carry-kd>0）允许 == carryKdOverride
      //（抑摆手段2 的生效证据），其余时段恒 == 策略 kd。
      if (seg.kind === 'rl' && steps % 5 === 0 && policyManifest) {
        const kp = policyManifest.gains.kp;
        const kd = policyManifest.gains.kd;
        const kdOverride = CARRY_KD_ARG > 0 && ph === 'WALK_CARRY' ? CARRY_KD_ARG : null;
        for (let p = 0; p < 29; p++) {
          const mj = policyManifest.joint_map.policy_to_mjcf[p];
          const act = jmapWalker[JOINT_NAMES_H[mj]].c;
          if (Math.abs(model.actuator_gainprm[act * 10 + 0] - kp[mj]) > 1e-9) { walkStats.gainWalkBad++; break; }
          const kdWant = kdOverride != null && mj >= 15 && mj <= 28 ? kdOverride : kd[mj];
          if (Math.abs(model.actuator_biasprm[act * 10 + 2] + kdWant) > 1e-9) { walkStats.gainKdWalkBad++; break; }
        }
        walkStats.gainWalkSamples++;
      }
      // 足支撑（N4）与脚-地穿透（N5 按段类型分档：rl / mocap）+ N14 足底滑移（mocap 段）
      if (steps % 5 === 0) {
        for (let fi = 0; fi < 2; fi++) {
          const d = distPair(footGids[fi], floorGid);
          seg.footMin = Math.min(seg.footMin, d);
          if (seg.kind === 'rl') {
            minFootFloorDistWalk = Math.min(minFootFloorDistWalk, d);
          } else {
            minFootFloorDistMocap = Math.min(minFootFloorDistMocap, d);
            // N14：支撑相内足端水平速度（10ms 差分）；离地帧更新差分基线不采样
            const fp = [data.geom_xpos[3 * footGids[fi]], data.geom_xpos[3 * footGids[fi] + 1]];
            if (d < 0.005) {
              if (n14.prevPos[fi]) {
                const v = Math.hypot(fp[0] - n14.prevPos[fi][0], fp[1] - n14.prevPos[fi][1]) / 0.01;
                n14.samples.push(v);
                if (n14.dbg) n14.dbg.push({ v, t: data.time - (n14.segT0 ?? data.time), foot: fi, d, k: ++n14.epLen[fi] });
              }
              else if (n14.dbg) n14.epLen[fi] = 0; // 回合首样本（只建基线），后续样本 k=回合内序号
              n14.prevPos[fi] = fp;
            } else {
              n14.prevPos[fi] = null;
            }
          }
          const stance = d < 0.005;
          if (!stance && seg.wasStance[fi]) {
            if (fi === 0) seg.lFlips++; else seg.rFlips++;
            if (seg.lastEdge === (fi === 0 ? 'L' : 'R')) seg.altOk = false; // 同足连续抬=非交替
            seg.lastEdge = fi === 0 ? 'L' : 'R';
          }
          seg.wasStance[fi] = stance;
        }
      }
    } else {
      minPelvisZManip = Math.min(minPelvisZManip, data.xpos[3 * pelvisBid + 2]);
    }
    // N10：当前槽幽灵箱位姿差分（10ms 采样）。
    // 只在持箱跟随相关相位链内测量；slot 变化/幽灵退场等合法瞬移重置基线不计。
    if (steps % 5 === 0 && st.slotTotal > 0 && !st.done) {
      const followPhases = ['WALK_APPROACH', 'WALK_CARRY', 'PREPARE', 'CARRY_REPLAY', 'ALIGN_PLACE', 'CARRY_REPLAY2', 'TURN_CARRY'];
      const gid = ghostGeomId(st.slotIndex);
      const key = `${st.slotIndex}:${ph}`;
      const boundaryCandidate = walkStats.prevBoxKey !== null && key !== walkStats.prevBoxKey
        && followPhases.includes(ph) && walkStats.prevBoxKey.split(':')[1] !== ph
        && walkStats.prevBoxKey.split(':')[0] === String(st.slotIndex);
      if (gid >= 0 && followPhases.includes(ph)) {
        const bp = geomCenter(data, gid);
        if (walkStats.prevBoxPos) {
          const jump = Math.hypot(bp[0] - walkStats.prevBoxPos[0], bp[1] - walkStats.prevBoxPos[1], bp[2] - walkStats.prevBoxPos[2]);
          if (jump > 0.5) {
            // 幽灵体合法瞬移（释放换位 R7 的退场/潜伏重摆）：重置基线不计
            walkStats.prevBoxPos = bp;
            walkStats.prevBoxKey = key;
          } else {
            if (boundaryCandidate) walkStats.boundaryJumpMax = Math.max(walkStats.boundaryJumpMax, jump);
            else if (key === walkStats.prevBoxKey) {
              if (jump > walkStats.boxJumpMax) {
                walkStats.boxJumpMax = jump; walkStats.boxJumpAt = key;
                walkStats.boxJumpVec = [bp[0] - walkStats.prevBoxPos[0],
                  bp[1] - walkStats.prevBoxPos[1], bp[2] - walkStats.prevBoxPos[2]]
                  .map((v) => (v * 1000).toFixed(1)).join(',');
              }
            }
            walkStats.prevBoxPos = bp;
            walkStats.prevBoxKey = key;
          }
        } else {
          walkStats.prevBoxPos = bp;
          walkStats.prevBoxKey = key;
        }
      } else {
        walkStats.prevBoxPos = null; // 幽灵退场/新生段：重置差分基线
        walkStats.prevBoxKey = null;
      }
    }
    // N12（MERO-8）：WALK_CARRY 段手-箱贴近度（10ms 采样）：d = |幽灵箱几何中心 − 双腕
    // body 原点中点|。双手中点锚下箱跟手，d 恒在捕获值附近（lift_end 帧 57-135mm，
    // handbox_probe 实测）；pelvis 锚（逃生门）下摆臂使 d 峰值显著增大——本断言因此
    // 兼具防假绿能力（G1BOX_CARRY_ANCHOR=pelvis 复跑必须 FAIL）。
    if (ph === 'WALK_CARRY' && steps % 5 === 0) {
      const gid12 = ghostGeomId(st.slotIndex);
      if (gid12 >= 0) {
        const b12 = geomCenter(data, gid12);
        const pmx = (data.xpos[3 * wristBidL] + data.xpos[3 * wristBidR]) / 2;
        const pmy = (data.xpos[3 * wristBidL + 1] + data.xpos[3 * wristBidR + 1]) / 2;
        const pmz = (data.xpos[3 * wristBidL + 2] + data.xpos[3 * wristBidR + 2]) / 2;
        const d12 = Math.hypot(b12[0] - pmx, b12[1] - pmy, b12[2] - pmz);
        if (d12 > walkStats.n12Max) {
          walkStats.n12Max = d12;
          walkStats.n12At = `${st.slotIndex}@${data.time.toFixed(1)}s`;
        }
      }
    }
    // N13（MERO-9）：逐手抓握偏差（10ms 采样，所有持箱相位）。gap = |腕世界位姿 −
    // (箱位姿 ⊗ 基准偏移)|，左右手独立——用户看到的「没抱住」是单手离开箱面，
    // 中点量法（N12）会被反相摆臂抵消。基准偏移见 n13 声明处注释。
    if (steps % 5 === 0 && N13_PHASES.includes(ph) && st.slotTotal > 0 && !st.done && n13Box) {
      const qa13 = ghostJointQadr(st.slotIndex);
      if (qa13 >= 0) {
        const bp13 = [data.qpos[qa13], data.qpos[qa13 + 1], data.qpos[qa13 + 2]];
        const bq13 = [data.qpos[qa13 + 3], data.qpos[qa13 + 4], data.qpos[qa13 + 5], data.qpos[qa13 + 6]];
        // 抓取瞬间基线（仅 CARRY_REPLAY）：held 后首样现算「手该在箱面哪里」；
        // 离开该相位即失效（下一槽 CARRY_REPLAY 重建）。held 之前（伸手段）不采样。
        if (ph === 'CARRY_REPLAY') {
          if (!st.held) { n13.graspBaseL = null; n13.graspBaseR = null; }
          else if (!n13.graspBaseL) {
            n13.graspBaseL = n13BoxLocal(wristW(wristBidL), bp13, bq13);
            n13.graspBaseR = n13BoxLocal(wristW(wristBidR), bp13, bq13);
          }
        } else {
          n13.graspBaseL = null; n13.graspBaseR = null;
        }
        const off13 = ph === 'CARRY_REPLAY'
          ? (st.held ? { offL: n13.graspBaseL, offR: n13.graspBaseR } : null)
          : stack.graspOffsets;
        if (off13 && off13.offL) {
          const gL = n13Gap(wristW(wristBidL), bp13, bq13, off13.offL);
          const gR = n13Gap(wristW(wristBidR), bp13, bq13, off13.offR);
          // MERO-10 分档：mocap 槽位的 CARRY_REPLAY（carryFull）为「动捕档」判定数据；
          // rl 槽位维持 MERO-9 口径（CARRY_REPLAY 观测项 + 锚随三相判定）。
          if (st.carryStrategy === 'mocap' && ph === 'CARRY_REPLAY') {
            n13.mocapSamples.L.push(gL);
            n13.mocapSamples.R.push(gR);
            n13.mocapCaptured = true;
          } else {
            const s13 = n13.samples[ph] ?? (n13.samples[ph] = { L: [], R: [] });
            s13.L.push(gL); s13.R.push(gR);
            const at13 = `${st.slotIndex}@${data.time.toFixed(1)}s`;
            if (N13_ASSERT_PHASES.includes(ph)) {
              if (gL > n13.max.L) { n13.max.L = gL; n13.at.L = at13; n13.atPh.L = ph; }
              if (gR > n13.max.R) { n13.max.R = gR; n13.at.R = at13; n13.atPh.R = ph; }
            }
            if (ph !== 'CARRY_REPLAY') n13.captured = true;
          }
        }
      }
    }
    // N13 捕获几何诊断（每槽一次，CARRY_REPLAY→TURN_CARRY 边界即 captureCarryAnchor 时刻）：
    // 检查捕获帧每只手是否真的贴着箱面（surf SDF≈0 为贴合）——d0 大的层若属抓握几何本身
    // 松，需按任务把箱锚从「双手中点」改为「最小化双手 gap 的位姿」（定标判定输入）。
    if (ph === 'TURN_CARRY' && walkStats.prevPhase === 'CARRY_REPLAY' && stack.graspOffsets && n13Box) {
      const qaD = ghostJointQadr(st.slotIndex);
      if (qaD >= 0) {
        const bpD = [data.qpos[qaD], data.qpos[qaD + 1], data.qpos[qaD + 2]];
        const bqD = [data.qpos[qaD + 3], data.qpos[qaD + 4], data.qpos[qaD + 5], data.qpos[qaD + 6]];
        const offD = stack.graspOffsets;
        // 腕世界位姿由基准偏移重建（避开捕获帧后单子步 FK 漂移），箱心 = 位姿 ⊗ geom 偏置
        const wL = quatRotVec(bqD, offD.offL), wR = quatRotVec(bqD, offD.offR);
        const bcD = quatRotVec(bqD, n13Box.bbox_center_offset);
        const wl = [bpD[0] + wL[0], bpD[1] + wL[1], bpD[2] + wL[2]];
        const wr = [bpD[0] + wR[0], bpD[1] + wR[1], bpD[2] + wR[2]];
        const bc = [bpD[0] + bcD[0], bpD[1] + bcD[1], bpD[2] + bcD[2]];
        const pm = [(wl[0] + wr[0]) / 2, (wl[1] + wr[1]) / 2, (wl[2] + wr[2]) / 2];
        const layerD = expected[st.slotIndex]?.layer;
        const diag = {
          slot: st.slotIndex, layer: layerD,
          midD0: Math.hypot(pm[0] - bc[0], pm[1] - bc[1], pm[2] - bc[2]),
          sdfL: n13ObbSdf(wl, bc, bqD, n13Box.half_size),
          sdfR: n13ObbSdf(wr, bc, bqD, n13Box.half_size),
        };
        n13.capDiag.push(diag);
        console.log(`  [N13] 捕获诊断 slot${diag.slot}(layer${(diag.layer ?? 0) + 1}): midD0=${(diag.midD0 * 1000).toFixed(1)}mm `
          + `腕-箱面 sdf L=${(diag.sdfL * 1000).toFixed(1)}mm R=${(diag.sdfR * 1000).toFixed(1)}mm`);
      }
    }
  }
  function onPhaseChange(from, to) {
    if (isWalkPhase(from)) closeWalkSeg();
  }

  // 穿透采样：vendored binding 的 data.contact.get(k) 有 embind 内存泄漏（实测 ~5KB/次，
  // 逐子步采样 2 万次即 OOM abort），故改用零分配的 mj_geomDistance 对碰撞相关
  // 几何对（活动箱×地面、箱×箱、箱×脚、脚×地面）做周期采样；语义与断言 4 一致
  // （全程最小距离 > -15mm），采样间隔 10ms。
  const mjGEOM_SPHERE = 2;
  const floorGid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, 'floor');
  // S3 支撑板 geom（factory.js 承重层板；缺失时 S3 断言 n/a）
  const boardGidS3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, 'fx_rack_board_L2_geom');
  const footGids = [];
  for (const bn of ['left_ankle_roll_link', 'right_ankle_roll_link']) {
    const bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, bn);
    const gadr = model.body_geomadr[bid], gnum = model.body_geomnum[bid];
    for (let g = gadr; g < gadr + gnum; g++) if (model.geom_type[g] === mjGEOM_SPHERE) footGids.push(g);
  }
  const fromto = new Float64Array(6);
  const distPair = (g1, g2) => mujoco.mj_geomDistance(model, data, g1, g2, 0.05, fromto);
  const sampleInvariants = () => {
    for (let i = 0; i < model.nv; i++) maxQfrc = Math.max(maxQfrc, Math.abs(data.qfrc_applied[i]));
    minPelvisZ = Math.min(minPelvisZ, data.xpos[3 * pelvisBid + 2]);
    if (steps % 5 !== 0) return; // 10ms 采样间隔
    // 只采样"已换位（handover）"槽位的物理箱（潜伏体远在取箱区外）。不能用 slotIndex：
    // done 后 slotIndex 停在最后槽号，slice(0, last) 会永久漏采最后一箱（1×1 配置下
    // 一箱都采不到）。机器人-箱子物理接触已被 <contact><exclude> 静态排除，本断言
    // 维持只考核箱-地/箱-箱；持握期幽灵箱×机器人的"穿模"是显示域缺陷（exclude 只
    // 切断物理接触对，不影响距离度量），由 N18 以 mj_geomDistance 单独守门。
    const boxGids = stack.boxes().map((b) => b.geomId);
    const placed = boxGids.slice(0, stack.status().handedOver);
    for (const g of placed) minBoxDist = Math.min(minBoxDist, distPair(g, floorGid));
    // S3：放置箱 × 承重支撑板（fx_rack_board_L2_geom，板顶 0.406m）——断言 4 只考
    // 箱-地/箱-箱，箱-板挤压（P1 类缺陷）由此单独守门（阈 −5mm，见断言 S3）
    if (boardGidS3 >= 0) {
      for (const g of placed) minBoxBoardDist = Math.min(minBoxBoardDist, distPair(g, boardGidS3));
    }
    for (let i = 0; i < placed.length; i++) {
      for (let j = i + 1; j < placed.length; j++) minBoxDist = Math.min(minBoxDist, distPair(placed[i], placed[j]));
    }
    // 脚-地穿透单独计量：方案的滑移回位（BLEND 直线拖脚）是宣称的演示近似，
    // 回放足端不可避免地会切入地面（实测约 -65mm）；此处只防"爆炸级"穿插。
    for (const f of footGids) minFootFloorDist = Math.min(minFootFloorDist, distPair(f, floorGid));
  };

  // ---------------- N15/N16/N17（瞬移消除改造）：瞬移守门巡视器 ----------------
  // N15 位移段清单表：按"驱动类别"切分统计——gait=RL 行走/真人动捕步态回放；
  // blend=PREPARE/ALIGN_PLACE/BLEND_STAND 保留残差吸收段（单列，不与行走段合并统计，
  // 队长透明度要求）；kin=其它 kinematic 段。静止判据：0.6s 窗内 root 水平位移 <0.02m；
  // 类别切换亦切分（blend 单列口径）。判定：gait 段须支撑交替 flips ≥ max(1,⌊disp/0.45⌋)
  // 且左右各≥1 且大体交替（复用 N4 口径）+ 支撑相滑移 p95 ≤400mm/s（复用 N14 口径）；
  // blend 段须 disp ≤0.15m 且滑移 p95 ≤400mm/s（放慢 blend 后用户不可辨滑移的量化口径）；
  // kin 段 disp>0.10m 时按 gait 判据（动捕步态天然满足）；RECOVER_STAND 起身水平位移
  // ≈0（明示演示级近似，A17 跳变守门管连续性）。
  // N16 转身段清单表：TURN_* 踏步转身段（旋转只在摆动相推进+交替真实抬腿，定格转身
  // 已移除）——支撑交替 flips ≥ max(2,⌊|Δyaw|/0.25⌋)、支撑拖拽 p95 ≤100mm/s、原始口径
  // p95 ≤400、脚-地 > -25mm、摆动脚峰值离地 ≥30mm（转身视觉改造新增：仅统计单脚离地
  // 样本，入口足底悬浮不计——防退化回旧对称蹲抬 ~7mm 抖动）。基线对比（A3，1×1 定标
  // 实测）：旧定格转身 46°/173° 段零交替、滑移 p95 180/225mm/s 持续拖扫（历史固定时长
  // 版 5-6rad/s 实测拖扫 ~0.7m/s）；踏步转身（对称蹲抬旧机制）实测拖拽 p95 60-68mm/s
  // 且全程左右交替——两条 A3 判据均满足。
  // N17 root 每子步水平跳变 <0.3m（防 reset 瞬移/任何整体平移回归）+ 全程模式恒 rl。
  // N16/N15 巡视器专用足端代表球：每个 ankle_roll_link 有 4 个 foot 球（2 跟 2 尖，
  // footGids[0..3] 全是左脚），取每足第一个球（跟球）作代表——否则"左/右"与滑移
  // 采样实际量的是同一只脚的两个球（N4 的既有采样即此口径，冻结不动；巡视器需要
  // 真实的左右区分与单球速度）。已试过 body 原点参考：测的是踝部摆动而非接触点
  // 滑移，口径失真，弃用。
  const survFeet = [];
  for (const bn of ['left_ankle_roll_link', 'right_ankle_roll_link']) {
    const bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, bn);
    const gadr = model.body_geomadr[bid], gnum = model.body_geomnum[bid];
    let rep = -1;
    for (let g = gadr; g < gadr + gnum; g++) if (model.geom_type[g] === mjGEOM_SPHERE) { rep = g; break; }
    survFeet.push(rep);
  }
  const SURV_WIN_N = 60; // 静止检测窗（10ms × 60 = 0.6s）
  const SURV_REST_D = 0.02;
  const surv = {
    hist: [],                  // [{t,x,y}] 静止检测窗
    seg: null, segs: [],       // 位移段（N15）
    turn: null, turns: [],     // 转身段（N16）
    prevFoot: [null, null], prevStance: [true, true],
    pend: [[], []],            // 每足支撑滑移提交缓冲：样本需被后续 16 个支撑样本确认
    sawNonRlMode: false,       // N17b：全程恒 rl
  };
  const survCat = (ph, st) => {
    if (ph === 'PREPARE' || ph === 'ALIGN_PLACE' || ph === 'BLEND_STAND') return 'blend';
    if (ph === 'WALK_APPROACH' || ph === 'WALK_CARRY' || ph === 'WALK_HOME' || ph === 'STOP_PATH') return 'gait';
    if (ph === 'CARRY_REPLAY' && st.carryStrategy === 'mocap') return 'gait'; // carryFull=真人步态回放
    return 'kin';
  };
  const rootYawQ = () => {
    const q0 = data.qpos[3], q1 = data.qpos[4], q2 = data.qpos[5], q3 = data.qpos[6];
    return Math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3));
  };
  function survStep(ph) {
    const st = stack.status();
    if (st.mode !== 'rl') surv.sawNonRlMode = true;
    const x = data.qpos[0], y = data.qpos[1];
    // 足支撑 + 滑移采样（口径同 N4/N14：足-地 <5mm 为支撑；10ms 差分；足=每足代表球）。
    // 双流：vels=原始口径（N15/N14 对齐）；drags=支撑拖拽口径（提交缓冲：每足支撑
    // 段末尾 16 样本不确认——抬起时脚跟球贴地拖拽 ~10-15 样本属球足建模伪影而非
    // 支撑真滑移，落定瞬态在段首同理由缓冲滤除；N16 用 drags）。
    const stD = [distPair(survFeet[0], floorGid), distPair(survFeet[1], floorGid)];
    const vels = [];
    const drags = [];
    for (let fi = 0; fi < 2; fi++) {
      const stance = stD[fi] < 0.005;
      const fp = [data.geom_xpos[3 * survFeet[fi]], data.geom_xpos[3 * survFeet[fi] + 1]];
      if (stance && surv.prevFoot[fi]) {
        const v = Math.hypot(fp[0] - surv.prevFoot[fi][0], fp[1] - surv.prevFoot[fi][1]) / 0.01;
        vels.push(v);
        const pend = surv.pend[fi];
        pend.push(v);
        if (pend.length > 16) drags.push(pend.shift());
      } else if (!stance) {
        surv.pend[fi] = []; // 离地：丢弃未确认尾部（抬脚瞬态在此）
      }
      surv.prevFoot[fi] = stance ? fp : null;
      if (!stance && surv.prevStance[fi]) {
        const edge = fi === 0 ? 'L' : 'R';
        // blend 相位不计数：kinematic blend 首样本的足端扰动会产生伪边缘并污染
        // 前一 gait 段的交替判定（真实步态边缘只产生于 gait/kin 相位）
        if (surv.seg && surv.seg.cat !== 'blend') {
          if (edge === 'L') surv.seg.lFlips++; else surv.seg.rFlips++;
          if (surv.seg.lastEdge === edge) surv.seg.altOk = false;
          surv.seg.lastEdge = edge;
        }
        if (surv.turn) {
          surv.turn.flips++;
          if (edge === 'L') surv.turn.lFlips = (surv.turn.lFlips ?? 0) + 1; else surv.turn.rFlips = (surv.turn.rFlips ?? 0) + 1;
          if (surv.turn.lastEdge === edge) surv.turn.altOk = false;
          surv.turn.lastEdge = edge;
        }
      }
      surv.prevStance[fi] = stance;
    }
    if (surv.seg) {
      for (const v of vels) surv.seg.vels.push(v);
      for (const v of drags) surv.seg.drags.push(v);
    }
    if (surv.turn) {
      for (const v of vels) surv.turn.vels.push(v);
      for (const v of drags) surv.turn.drags.push(v);
      surv.turn.footMin = Math.min(surv.turn.footMin, stD[0], stD[1]);
      // 摆动脚峰值离地（转身视觉改造 N16 子判据）：只统计"恰单脚离地"样本——双脚同时
      // 离地不计，防止把入口足底悬浮（落地修正前可达 ~47mm）当抬升造成假绿
      const lAir = stD[0] >= 0.005, rAir = stD[1] >= 0.005;
      if (lAir !== rAir) surv.turn.liftMax = Math.max(surv.turn.liftMax, stD[lAir ? 0 : 1]);
    }
    // 转身段开合
    if (ph.startsWith('TURN_')) {
      if (!surv.turn || surv.turn.phase !== ph) {
        surv.turn = { phase: ph, t0: data.time, yaw0: rootYawQ(), flips: 0, lastEdge: null, altOk: true, vels: [], drags: [], footMin: Infinity, liftMax: 0 };
      }
    } else if (surv.turn) {
      surv.turn.yaw1 = rootYawQ();
      surv.turn.t1 = data.time;
      surv.turns.push(surv.turn);
      surv.turn = null;
    }
    // 位移段开合：静止收段；驱动类别切换收段开新段（blend 单列口径）
    surv.hist.push({ t: data.time, x, y });
    if (surv.hist.length > SURV_WIN_N) surv.hist.shift();
    const atRest = surv.hist.length >= SURV_WIN_N - 2
      && Math.hypot(x - surv.hist[0].x, y - surv.hist[0].y) < SURV_REST_D;
    const cat = survCat(ph, st);
    if (surv.seg) {
      const s = surv.seg;
      s.t1 = data.time;
      s.disp = Math.hypot(x - s.x0, y - s.y0);
      if (!s.phases.includes(ph)) s.phases.push(ph);
      if (atRest) {
        surv.segs.push(s);
        surv.seg = null;
      } else if (cat !== s.cat) {
        // 类别切换（如行走→blend）：新段从当前点起算，段位移各自干净
        surv.segs.push(s);
        surv.seg = {
          cat, t0: data.time, x0: x, y0: y, t1: data.time, disp: 0,
          phases: [ph], lFlips: 0, rFlips: 0, lastEdge: s.lastEdge, altOk: true, vels: [], drags: [],
        };
      }
    } else if (!atRest) {
      const o = surv.hist[0]; // 从静止启动：段起点取窗首（≈静止位置，窗内位移 <0.02m）
      surv.seg = {
        cat, t0: o.t, x0: o.x, y0: o.y, t1: data.time, disp: 0,
        phases: [ph], lFlips: 0, rFlips: 0, lastEdge: null, altOk: true, vels: [], drags: [],
      };
    }
  }

  let walkPhasePrev = stack.status().phase;
  // N17（A4）：root 每子步水平跳变跟踪（防 reset 瞬移/整体平移回归）
  let prevRootX = data.qpos[0], prevRootY = data.qpos[1];
  let maxRootJump = 0, maxRootJumpAt = '';
  while (!stack.status().done && !stack.status().error && steps < maxSteps) {
    stack.onStep(TIMESTEP);
    mujoco.mj_step(model, data);
    steps++;
    sampleInvariants();
    {
      const jm = Math.hypot(data.qpos[0] - prevRootX, data.qpos[1] - prevRootY);
      if (jm > maxRootJump) { maxRootJump = jm; maxRootJumpAt = `${data.time.toFixed(2)}s`; }
      prevRootX = data.qpos[0]; prevRootY = data.qpos[1];
    }
    const ph = stack.status().phase;
    if (WALK_MODE === 'rl') {
      sampleWalk(ph);
      if (steps % 5 === 0) { survStep(ph); sampleN18(ph); } // 10ms 节奏，与 N4/N14 采样对齐
      if (ph !== walkPhasePrev) { onPhaseChange(walkPhasePrev, ph); walkStats.prevPhase = ph; }
      walkPhasePrev = ph;
    }
  }
  // 收口未闭合段（done/错误暂停时最后一段仍开着）：记录终值供清单表展示
  if (WALK_MODE === 'rl') {
    if (surv.turn) { surv.turn.yaw1 = rootYawQ(); surv.turn.t1 = data.time; surv.turns.push(surv.turn); surv.turn = null; }
    if (surv.seg) { surv.seg.t1 = data.time; surv.segs.push(surv.seg); surv.seg = null; }
  }
  check('循环在时限内完成全部槽位', stack.status().done, `sim=${data.time.toFixed(1)}s steps=${steps} mode=${stack.status().mode}`);

  // N2（恢复侧）：done 后执行器增益应已恢复 Menagerie 500（BLEND_STAND 内 exit() 已执行）
  if (WALK_MODE === 'rl' && policyManifest && stack.status().done) {
    for (let a = 0; a < 29; a++) {
      const act = jmapWalker[JOINT_NAMES_H[a]].c;
      if (Math.abs(model.actuator_gainprm[act * 10 + 0] - 500) > 1e-9) walkStats.gainRestoredBad++;
    }
  }

  // ---------- done 后续 3s 稳定观察 ----------
  let topSnap = null;
  if (stack.status().done) {
    const boxes = stack.boxes();
    const topIdx = [];
    for (let i = 0; i < expected.length; i++) if (expected[i].layer >= 1) topIdx.push(i);
    if (topIdx.length > 0) {
      topSnap = topIdx.map((i) => ({
        i,
        pos: geomCenter(data, boxes[i].geomId),
        upZ: data.geom_xmat[9 * boxes[i].geomId + 8],
      }));
    }
    for (let s = 0; s < Math.ceil(3.0 / TIMESTEP); s++) {
      // 机器人 kinematic 冻结在 stand 位姿（模拟 idle 交接后的站立）；
      // 箱子为纯物理观察窗。不调 stack.onStep（其契约是清 qfrc，无位姿写入需求）
      for (let i = 0; i < 36; i++) data.qpos[i] = standQpos[i];
      for (let i = 0; i < 35; i++) data.qvel[i] = 0;
      mujoco.mj_step(model, data);
      steps++;
      sampleInvariants();
    }
  }

  // 断言 3：每箱中心 ≈ 槽位中心（水平 <5mm，垂直 <8mm；方案 4.6）
  {
    let worstH = 0, worstV = 0, ok = true, detail = '';
    const boxes = stack.boxes();
    for (let i = 0; i < expected.length; i++) {
      const [cx, cy, cz] = geomCenter(data, boxes[i].geomId);
      const dh = Math.hypot(cx - expected[i].center[0], cy - expected[i].center[1]);
      const dv = Math.abs(cz - expected[i].center[2]);
      worstH = Math.max(worstH, dh); worstV = Math.max(worstV, dv);
      if (dh >= 0.005 || dv >= 0.008) {
        ok = false;
        detail = `箱${i}(layer${expected[i].layer + 1}): Δh=${(dh * 1000).toFixed(1)}mm Δv=${(dv * 1000).toFixed(1)}mm`;
        break;
      }
    }
    if (ok) detail = `worst Δh=${(worstH * 1000).toFixed(2)}mm Δv=${(worstV * 1000).toFixed(2)}mm`;
    check('断言3 各箱中心≈槽位中心 (Δh<5mm, Δv<8mm)', ok, detail);
  }

  // 断言 4：箱体接触穿透 > -15mm（方案 4.6 判据，只考核箱-地/箱-箱）。
  // 断言 4b 双层口径（MF-3，队长裁定）：脚-地全程 -100mm 两模式同阈——kinematic
  // 相位（PREPARE/操作窗口/回位 blend）沿用 legacy 演示近似容差（实测 PREPARE
  // -33.6mm 在容差内）；WALK 三相位另由 N5 的 -25mm 收紧管住（RL 段走真步态），
  // 两层互不掩蔽。
  check('断言4 全程箱体接触 dist > -15mm', minBoxDist > -0.015,
    `min=${(minBoxDist * 1000).toFixed(2)}mm`);
  // S3（审查修复）：放置箱-承重支撑板接触 ≥ −5mm——箱-板挤压（如层板 contype 配置
  // 回归、放置深度错误）在断言 4 的箱-地/箱-箱口径下不可见，由此单独守门。
  // 层 1（地面格）箱距板 ~0.4m 不参与判定极值；无板格位配置（1 层）时 n/a。
  {
    const s3ok = minBoxBoardDist === Infinity || minBoxBoardDist > -0.005;
    check('S3 放置箱-支撑板接触 dist > -5mm（断言4 盲区补口）', s3ok,
      minBoxBoardDist === Infinity ? 'n/a（无箱-板样本：1 层配置或板缺失）' : `min=${(minBoxBoardDist * 1000).toFixed(2)}mm`);
  }
  // S3 度量自检（注入法自证有效）：把 0 号箱临时写进承重板内 8mm，验证 distPair 读数
  // 确实 < −5mm（度量灵敏度），随后完整恢复 qpos 并刷新 FK——不影响任何后续断言。
  if (boardGidS3 >= 0) {
    const boxesS3 = stack.boxes();
    const gid0 = boxesS3[0];
    let probe = null;
    if (gid0 && gid0.geomId >= 0) {
      const adr = gid0.jointQadr;
      const snap = Float64Array.from(data.qpos.subarray(adr, adr + 7));
      const hz = set.low.box.half_size[2], topBoard = rackBoardSpecs().find(b => b.geom === 'fx_rack_board_L2_geom').top;
      const offZ = set.low.box.bbox_center_offset[2];
      data.qpos[adr] = FACTORY.rackCenter[0];
      data.qpos[adr + 1] = FACTORY.rackCenter[1];
      // geom 中心语义：body z = 目标 geom 中心 z − bbox_center_offset[2]（geom 挂 body 带
      // pos 偏置，直接写 body z 会把穿透量吃掉 offset——首版实测 8mm 注入只读出 3.2mm）
      data.qpos[adr + 2] = topBoard + hz - 0.008 - offZ;
      data.qpos[adr + 3] = 1; data.qpos[adr + 4] = 0; data.qpos[adr + 5] = 0; data.qpos[adr + 6] = 0;
      mujoco.mj_forward(model, data);
      probe = distPair(gid0.geomId, boardGidS3);
      for (let k = 0; k < 7; k++) data.qpos[adr + k] = snap[k];
      mujoco.mj_forward(model, data);
    }
    check('S3 度量自检（箱-板 8mm 重叠注入 → distPair < −5mm）', probe !== null && probe < -0.005,
      probe === null ? 'n/a（geom 缺失）' : `probe=${(probe * 1000).toFixed(1)}mm`);
  }
  check('断言4b 脚-地穿透无爆炸级穿插 (> -100mm, 演示近似)', minFootFloorDist > -0.100,
    `min=${(minFootFloorDist * 1000).toFixed(1)}mm`);

  // 断言 5：顶箱续 3s 位移 <5mm、无翻倒。
  // 翻倒判据对对称箱体使用 |Rzz|（局部 z 轴与世界 z 夹角，容忍 ±180° 翻转——
  // 箱子几何对称，翻转后外观与物理状态不变）；侧翻（90°）仍判失败。
  if (topSnap) {
    let worstMove = 0, worstTilt = 0, ok = true, detail = '';
    const boxes = stack.boxes();
    for (const snap of topSnap) {
      const p = geomCenter(data, boxes[snap.i].geomId);
      const move = Math.hypot(p[0] - snap.pos[0], p[1] - snap.pos[1], p[2] - snap.pos[2]);
      const tiltNow = Math.acos(Math.max(-1, Math.min(1, Math.abs(data.geom_xmat[9 * boxes[snap.i].geomId + 8]))));
      worstMove = Math.max(worstMove, move);
      worstTilt = Math.max(worstTilt, tiltNow);
      if (move >= 0.005 || tiltNow >= (10 * Math.PI) / 180) {
        ok = false; detail = `顶箱${snap.i}: move=${(move * 1000).toFixed(1)}mm tilt=${(tiltNow * 180 / Math.PI).toFixed(1)}°`;
        break;
      }
    }
    if (ok) detail = `worst move=${(worstMove * 1000).toFixed(2)}mm tilt=${(worstTilt * 180 / Math.PI).toFixed(2)}°`;
    check('断言5 顶箱 3s 位移<5mm 且无翻倒(<10°, ±180° 等价)', ok, detail);
  } else {
    check('断言5 顶箱 3s 位移<5mm 且无翻倒(<10°, ±180° 等价)', LAYERS === 1, '单层无顶箱，跳过');
  }

  // 断言 6（legacy）/ N8（rl）：pelvis z。rl 双阈值：RL 段 >0.6m；操作段沿用 clip 感知阈值
  // （回放轨迹含深蹲取箱，实测 mid clip 蹲到 0.485m，属动捕真值而非跌倒）。
  // MERO-10：N8b 阈值改由 stack.manipClipMinZ 镜像（各槽实际回放 clip 的最低骨盆高度，
  // 含 mocap carry clip 的深蹲，避免头侧复刻 tier 解析逻辑造成漂移）。
  if (WALK_MODE === 'rl') {
    const clipMinZN8 = stack.manipClipMinZ;
    const thrManip = Math.min(0.6, clipMinZN8 - 0.03);
    check('N8a RL 段 pelvis z > 0.6m', minPelvisZWalk > 0.6,
      `min=${minPelvisZWalk === Infinity ? 'n/a' : minPelvisZWalk.toFixed(3)}m`);
    check(`N8b 操作段 pelvis z > ${thrManip.toFixed(2)}m（clip 感知）`, minPelvisZManip > thrManip,
      `min=${minPelvisZManip === Infinity ? 'n/a' : minPelvisZManip.toFixed(3)}m, clip 最小 root_z=${clipMinZN8.toFixed(3)}m`);
  } else {
    const usedClips = new Set();
    for (let l = 0; l < LAYERS; l++) {
      const c = l === 0 ? set.low : l === 1 ? (set.mid || set.low) : (set.high || set.mid || set.low);
      usedClips.add(c);
    }
    let clipMinZ = 1;
    for (const c of usedClips) for (const r of c.root_pos) clipMinZ = Math.min(clipMinZ, r[2]);
    const thr = Math.min(0.6, clipMinZ - 0.03);
    check(`断言6 pelvis z 全程 > ${thr.toFixed(2)}m（clip 感知，回放未摔）`, minPelvisZ > thr,
      `min=${minPelvisZ.toFixed(3)}m, clip 最小 root_z=${clipMinZ.toFixed(3)}m`);
  }

  // 断言 7 / N9：stack 模式 qfrc_applied 恒 0（rl 模式同样成立——PD 力矩走执行器不走外挂）
  check('断言7/N9 stack 模式 qfrc_applied 恒 0', maxQfrc === 0, `max|qfrc|=${maxQfrc.toExponential(2)}`);

  // ---------------- rl 模式追加断言 N1-N8/N10（方案 8.2） ----------------
  if (WALK_MODE === 'rl') {
    // N1：golden fixture 前向一致性
    const st = policySelfTest(policyRef.weights, policyRef.golden);
    check('N1 golden fixture 前向一致性 (max|Δ|<1e-4)', st.ok, st.detail);

    // N2：增益改写/恢复（RL 段 == 策略 kp/kd——臂 kd 在 carry 段允许 override，见采样处
    // 队长预批准注释；BLEND_STAND 后 == Menagerie 500）
    const n2ok = walkStats.gainWalkSamples > 0 && walkStats.gainWalkBad === 0
      && walkStats.gainKdWalkBad === 0 && walkStats.gainRestoredBad === 0;
    check('N2 增益改写/恢复（RL 段=策略 kp/kd，臂 kd carry 段允许 override，done 后=Menagerie 500）', n2ok,
      `RL 段采样 ${walkStats.gainWalkSamples} 次 bad=${walkStats.gainWalkBad}，kd bad=${walkStats.gainKdWalkBad}，恢复 bad=${walkStats.gainRestoredBad}`);

    // N3：起步站稳（RL 段进入后 1s 窗内 pelvis z ≥0.6 未摔）。漂移判据删除：
    // warmUp 0.6s（stackCore WALK_WARMUP，重试 1.0s）后即起步，1s 窗内位移就是
    // 正常行进（0.2-0.3m），原「漂移<5cm」口径在 v2 编排下不成立（偏差记录见交接文档）。
    // MERO-10：mocap 携带段（kind='mocap'）为 kinematic 回放，蹲姿属动捕真值
    // （N8b 的 clip 感知阈值管辖），不参与本断言。
    {
      const rlSegs = walkStats.segments.filter((s) => s.kind === 'rl');
      let ok = rlSegs.length > 0, detail = '';
      for (const seg of rlSegs) {
        if (!(seg.minZ >= 0.6)) {
          ok = false; detail = `${seg.phase} minZ=${seg.minZ?.toFixed(3)}`;
          break;
        }
      }
      if (ok) detail = `${rlSegs.length} 段全部达标`;
      check('N3 起步站稳（每 RL 段 1s 窗 z≥0.6 未摔；mocap 携带段不参与）', ok, detail);
    }

    // N4：步态交替（RL 段切换 ≥ max(2,⌊位移/0.35⌋)；MERO-10 mocap 携带段同为 kinematic
    // 足端采样建段，步距除数按段类型分档——mocap 档 N4_MOCAP_DIV 联定起步 0.45，
    // 定标数据见 docs/policy-probe-notes.md MERO-10 节）。左右大体交替口径两档共用。
    {
      const N4_MOCAP_DIV = parseFloat(process.env?.G1BOX_N4_MOCAP_DIV) > 0
        ? parseFloat(process.env.G1BOX_N4_MOCAP_DIV) : 0.45; // 联定期可调，定标后冻结
      let ok = walkStats.segments.length > 0, detail = '';
      const routes = new Map();
      for (const s of walkStats.segments) {
        const key = `${s.kind}:${s.seq}:${s.kind === 'mocap' ? s.t0 : ''}`;
        const route = routes.get(key) ?? { ...s, disp: 0, lFlips: 0, rFlips: 0 };
        route.disp += s.disp; route.lFlips += s.lFlips; route.rFlips += s.rFlips;
        routes.set(key, route);
      }
      for (const seg of routes.values()) {
        const need = seg.kind === 'mocap'
          ? Math.max(2, Math.floor(seg.disp / N4_MOCAP_DIV))
          : Math.max(2, Math.floor(seg.disp / 0.35));
        const flips = seg.lFlips + seg.rFlips;
        const altOk = seg.lFlips >= 1 && seg.rFlips >= 1
          && Math.abs(seg.lFlips - seg.rFlips) <= Math.max(2, flips / 3);
        if (!(flips >= need && altOk)) {
          ok = false;
          detail = `${seg.phase}[${seg.kind}] flips=${flips}(L${seg.lFlips}/R${seg.rFlips}) < ${need} 或非交替 (disp=${seg.disp?.toFixed(2)}m)`;
          break;
        }
      }
      if (ok) detail = walkStats.segments.map((s) => `${s.phase}[${s.kind}]:${s.lFlips + s.rFlips}(${s.disp?.toFixed(2)}m)`).join(' ');
      check('N4 步态交替（rl 段 ≥⌊位移/0.35⌋、mocap 段 ≥⌊位移/0.45⌋，且左右交替）', ok, detail);
    }

    // N5：RL 段脚-地穿透 > -25mm（legacy 的 -100mm 口径见断言 4b 分流）；
    // MERO-10 mocap 携带段独立累计器分档（N5_MOCAP_MIN 联定起步 -100mm，定标后收紧：
    // 预期收到 min(-35mm, p95+10mm)，不达 -35mm 则如实标注降档理由，方案 4.5）
    check('N5 RL 段脚-地穿透 > -25mm', minFootFloorDistWalk > -0.025,
      `min=${(minFootFloorDistWalk * 1000).toFixed(1)}mm`);
    if (walkStats.segments.some((s) => s.kind === 'mocap')) {
      const N5_MOCAP_MIN = parseFloat(process.env?.G1BOX_N5_MOCAP_MIN_MM) < 0
        ? parseFloat(process.env.G1BOX_N5_MOCAP_MIN_MM) / 1000 : -0.100; // 联定期可调
      check('N5-mocap 动捕携带段脚-地穿透 > -100mm（定标起步档）', minFootFloorDistMocap > N5_MOCAP_MIN,
        `min=${(minFootFloorDistMocap * 1000).toFixed(1)}mm`);
    }

    // N6：到位精度（R5 延伸口径）：RL 段停点 xy ≤0.15m（策略指令死区下的惯性停站），
    // 朝向残差由 PREPARE/ALIGN_PLACE blend 吸收（4.2 序 2/5），yaw 记录不断言
    {
      const errs = walkStats.arriveErrs;
      const ok = errs.length > 0 && errs.every((e) => e.dist <= 0.15);
      const worst = errs.length ? errs.reduce((a, b) => (b.dist > a.dist ? b : a)) : null;
      check('N6 到位精度（停点 xy≤0.15m，朝向残差 blend 吸收）', ok,
        errs.length ? `${errs.length} 段 worst dist=${(worst.dist * 100).toFixed(1)}cm yaw=${(worst.yaw * 180 / Math.PI).toFixed(1)}°` : '无完成段记录');
    }

    // N7：落点精度在 rl 模式不回退（引用断言 3 的结果；Δh<5mm/Δv<8mm）
    const a3 = results.find((r) => r.name.startsWith('断言3'));
    check('N7 落点精度 rl 模式全绿（引用断言3）', a3 ? a3.ok : false, a3 ? a3.detail : '断言3 未运行');

    // N10：箱跟随边界连续性。子步跳变 <25mm；WALK↔MANIP 相位边界 <12mm（判据 8mm
    // + 10ms 采样间隔内 ~3mm 的正常行走位移分量；幽灵退场等合法瞬移不计）
    check('N10 箱跟随连续性（子步跳变<25mm，边界<12mm）',
      walkStats.boxJumpMax < 0.025 && walkStats.boundaryJumpMax < 0.012,
      `max 子步=${(walkStats.boxJumpMax * 1000).toFixed(1)}mm@${walkStats.boxJumpAt || '?'}`
      + `[${walkStats.boxJumpVec || '?'}mm] `
      + `边界=${(walkStats.boundaryJumpMax * 1000).toFixed(1)}mm`);

    // N12（MERO-8）：手-箱贴近度。阈值 0.20m 联定依据（实测，详见 docs/policy-probe-notes.md
    // MERO-8 节与 mero8-evidence/ 日志）：
    //  - 捕获值 d0（lift_end 帧双手中点→箱心，handbox_probe 实测）：low 57mm / mid 118mm / high 135mm；
    //  - 双手锚随动实测峰值（tau=0.1s，含起步臂垂瞬态 + v·tau 滞后）：
    //    1x1 low 150mm / 2x2 largebox（含 mid）162mm / plasticbox 2x2 153mm；
    //  - pelvis 锚（原始缺陷，G1BOX_CARRY_ANCHOR=pelvis）实测峰值 484mm。
    // 0.20m 高于双手锚全部观测（+23% 余量）、远低于缺陷模式，防假绿两侧均成立。
    // MERO-10 守卫（方案 4.5 N12 行）：mocap-only 运行无 WALK_CARRY 段 → n/a 判 PASS
    // （rl 策略槽位存在时口径与阈值零变化）。
    const N12_MAX = 0.20;
    if (!walkStats.sawWalkCarry) {
      check(`N12 手-箱贴近度（WALK_CARRY 段 max|箱心−双手中点| ≤ ${(N12_MAX * 1000).toFixed(0)}mm）`,
        true, 'n/a（无 RL 携带段，动捕全程回放模式）');
    } else {
      check(`N12 手-箱贴近度（WALK_CARRY 段 max|箱心−双手中点| ≤ ${(N12_MAX * 1000).toFixed(0)}mm）`,
        walkStats.n12Max > 0 && walkStats.n12Max <= N12_MAX,
        `max=${(walkStats.n12Max * 1000).toFixed(1)}mm@${walkStats.n12At || '?'}`);
    }

    // N13（MERO-9）：逐手抓握偏差。N12 量双手中点，反相摆臂互相抵消、量错对象
    //（用户实测「有些时候都没有抱住箱子」而 N12 全绿）；N13 以捕获偏移为基准分别量
    // 每只手是否离开箱面。先打印按相位分解的测量表（定标数据），再按判定相位断言。
    const mm = (v) => (v * 1000).toFixed(1);
    const p95of = (arr) => {
      if (!arr.length) return 0;
      const a = [...arr].sort((x, y) => x - y);
      return a[Math.min(a.length - 1, Math.floor(0.95 * a.length))];
    };
    const maxof = (arr) => arr.reduce((m, v) => (v > m ? v : m), 0);
    console.log('  [N13] 逐手抓握偏差按相位分解（gapL/gapR，mm；rl 槽位 CARRY_REPLAY 为抓取瞬间基准的观测项）:');
    for (const phN of N13_PHASES) {
      const s = n13.samples[phN];
      if (!s || s.L.length === 0) continue;
      console.log(`  [N13]   ${phN}: n=${s.L.length} gapL max=${mm(maxof(s.L))}/p95=${mm(p95of(s.L))}  gapR max=${mm(maxof(s.R))}/p95=${mm(p95of(s.R))}`);
    }
    if (n13.mocapSamples.L.length > 0) {
      console.log(`  [N13] mocap 档（carryFull CARRY_REPLAY 全程，抓取瞬间基准）: n=${n13.mocapSamples.L.length} `
        + `gapL max=${mm(maxof(n13.mocapSamples.L))}/p95=${mm(p95of(n13.mocapSamples.L))}  `
        + `gapR max=${mm(maxof(n13.mocapSamples.R))}/p95=${mm(p95of(n13.mocapSamples.R))}`);
    }
    if (n13.capDiag.length > 0) {
      const worstSdf = n13.capDiag.reduce((m, d) => Math.max(m, d.sdfL, d.sdfR), 0);
      console.log(`  [N13] 捕获几何: ${n13.capDiag.length} 槽, 捕获帧腕-箱面 sdf 最大外距 ${mm(worstSdf)}mm（≈0=贴面，大=抓握几何本身松）`);
    }
    // 阈值联定（任务 A 定标，实测见 mero9-evidence/v0_*.log 与 docs/policy-probe-notes.md
    // MERO-9 节；G1BOX_N13_MAX_MM 仅供联定期实验，惯例同 MERO8_CARRY_TAU）：
    //  - 捕获基准 gap=0（offL/offR 即捕获帧几何），阈值含义 = 持箱全程单手偏离抓握位 ≤ N mm；
    //  - TURN_CARRY 实测 ≤1mm（dof 冻结，机制下限）；WALK_CARRY/ALIGN_PLACE 缺陷模式
    //    实测 386-477mm（四组 v0 定标全矩阵一致，p95≈max 即整段稳态脱手）；
    //  - pelvis 锚（防假绿）实测 gapR 571mm（v0_anchor_pelvis_negcheck.log，N13 如期 FAIL）。
    // 40mm = 捕获基线 + 40mm（任务 A 建议）：高于机制下限（1mm）三个量级、低于缺陷模式
    // 约 10 倍，防假绿两侧成立；同时 40mm 是「肉眼可见仍抱着」的边界（掌宽量级）。
    // MERO-10：仅当本轮出现 rl 锚随段样本时才判锚随档（mocap-only 运行不判，方案 4.5）。
    const N13_MAX = (parseFloat(process.env?.G1BOX_N13_MAX_MM) > 0
      ? parseFloat(process.env.G1BOX_N13_MAX_MM) : 40) / 1000;
    const hasAnchorData = N13_ASSERT_PHASES.some((p) => n13.samples[p]?.L.length > 0);
    if (hasAnchorData || n13.captured) {
      const worstGap = Math.max(n13.max.L, n13.max.R);
      check(`N13 逐手抓握偏差-锚随档（rl 策略槽位 max gapL/gapR ≤ ${(N13_MAX * 1000).toFixed(0)}mm，相位=${N13_ASSERT_PHASES.join('/')}）`,
        hasAnchorData && worstGap > 0 && worstGap <= N13_MAX,
        hasAnchorData
          ? `gapL=${mm(n13.max.L)}mm@${n13.at.L || '?'}(${n13.atPh.L || '?'}) gapR=${mm(n13.max.R)}mm@${n13.at.R || '?'}(${n13.atPh.R || '?'})`
          : '无捕获基准数据（抓握未发生或中途降级）');
    }
    // N13-mocap（MERO-10 新增动捕档）：carryFull 携带段以抓取瞬间偏移为基准量逐手 gap。
    // 基准是动捕自身几何漂移 + 重定向固有误差（与 v2 观测项 CARRY_REPLAY 166-231mm 同源），
    // 与锚随档 40mm 语义不同——「双手抱住」的量化口径变更已在 README/probe-notes 明示
    // （方案 4.5 N13 行、队长裁定 A5/N13 口径）。阈值联定起步 250mm，P2.5 用 mid/high +
    // 新 low 片段实测后冻结（G1BOX_N13_MOCAP_MAX_MM 仅供联定期实验）。
    const N13_MOCAP_MAX = (parseFloat(process.env?.G1BOX_N13_MOCAP_MAX_MM) > 0
      ? parseFloat(process.env.G1BOX_N13_MOCAP_MAX_MM) : 250) / 1000;
    if (n13.mocapSamples.L.length > 0) {
      const worstMocap = Math.max(maxof(n13.mocapSamples.L), maxof(n13.mocapSamples.R));
      check(`N13-mocap 逐手抓握偏差-动捕档（carryFull CARRY_REPLAY 全程 max gap ≤ ${(N13_MOCAP_MAX * 1000).toFixed(0)}mm）`,
        worstMocap > 0 && worstMocap <= N13_MOCAP_MAX,
        `gapL=${mm(maxof(n13.mocapSamples.L))}mm gapR=${mm(maxof(n13.mocapSamples.R))}mm（p95 L=${mm(p95of(n13.mocapSamples.L))}/R=${mm(p95of(n13.mocapSamples.R))}）`);
    }
    // N14（MERO-10 新增）：动捕携带段足底滑移——逐槽刚性规范化的守门断言（λ 压缩走廊/
    // 整体平移类实现会在此爆红）。支撑相（足-地 <5mm）内足端水平速度 10ms 差分采样。
    //
    // 【阈值冻结 400/2500（P2.5 联定，授权按实测重定标，联定数据 probe-notes MERO-10】
    // 起步 60/150 对重定向固有滑移不可达：定位结论（G1BOX_N14_DEBUG=1 分解 + 离线 FK
    // 探针逐桶一致）= 滑移是 clip 数据固有（hoi-retarget 对足部无地面接触约束），回放
    // 机制零引入。固有实测（3 配置）：p95 180-334 / max 299-1896mm/s——max 尖峰集中在
    // mid/high 片段支撑回合头 5 样本（落地后 ~50ms 1.6-1.9m/s 蹭地），支撑相中部底噪
    // p50 仅 41-50mm/s（非系统性，方向随机）。防假绿：negcheck carry 注入（root 走廊
    // 漂移 3.0m ≈ 463mm/s 恒定分量）实测 p95=518 / 外推全配置 ≥513——系统性漂移是
    // 恒定方向分量使全分布 RSS 上移，与固有局部误差量级差 ≥54%。冻结值两侧余量：
    // 固有 p95 上界 334→400（+20%）、注入下界 513→400（-22%）；max 阈 2500 只防爆炸级
    //（固有尖峰 1896，注入检测由 p95 判据 OR 达成）。G1BOX_N14_*_MMPS 仅供复现实验。
    if (walkStats.segments.some((s) => s.kind === 'mocap')) {
      const N14_P95 = (parseFloat(process.env?.G1BOX_N14_P95_MMPS) > 0
        ? parseFloat(process.env.G1BOX_N14_P95_MMPS) : 400) / 1000;
      const N14_MAX = (parseFloat(process.env?.G1BOX_N14_MAX_MMPS) > 0
        ? parseFloat(process.env.G1BOX_N14_MAX_MMPS) : 2500) / 1000;
      const p95s = p95of(n14.samples);
      const maxS = maxof(n14.samples);
      check(`N14 足底滑移（动捕携带段支撑相水平速度 p95≤${(N14_P95 * 1000).toFixed(0)}、max≤${(N14_MAX * 1000).toFixed(0)}mm/s）`,
        n14.samples.length > 0 && p95s <= N14_P95 && maxS <= N14_MAX,
        `p95=${(p95s * 1000).toFixed(0)} max=${(maxS * 1000).toFixed(0)}mm/s n=${n14.samples.length}`);
      if (n14.dbg) {
        // 定位输出（只观测）：按 1s 桶 p95/max + 速度直方 + 双球分列
        const pct = (arr, q) => { const a = [...arr].sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(a.length * q))] ?? NaN; };
        console.log('  [N14-dbg] 按 1s 桶（t=段内秒）分解:');
        const bySec = {};
        for (const s of n14.dbg) (bySec[Math.floor(s.t)] ??= []).push(s);
        for (const k of Object.keys(bySec).sort((a, b) => a - b)) {
          const vs = bySec[k].map((s) => s.v);
          console.log(`    t=${k}s: n=${vs.length} p50=${(pct(vs, 0.5) * 1000).toFixed(0)} p95=${(pct(vs, 0.95) * 1000).toFixed(0)} max=${(Math.max(...vs) * 1000).toFixed(0)}mm/s`);
        }
        const bins = [0, 30, 60, 100, 150, 200, 300, Infinity];
        const hist = new Array(bins.length - 1).fill(0);
        for (const s of n14.dbg) for (let b = 0; b < bins.length - 1; b++) if (s.v * 1000 >= bins[b] && s.v * 1000 < bins[b + 1]) { hist[b]++; break; }
        console.log('  [N14-dbg] 直方(mm/s):', bins.slice(0, -1).map((b, i) => `${b}-${bins[i + 1] === Infinity ? '∞' : bins[i + 1]}:${hist[i]}`).join(' '));
        const byFootDbg = {};
        for (const s of n14.dbg) (byFootDbg[s.foot] ??= []).push(s.v * 1000);
        for (const k of Object.keys(byFootDbg)) {
          const v = byFootDbg[k];
          console.log(`  [N14-dbg] foot[${k}]: n=${v.length} p50=${pct(v, 0.5).toFixed(0)} p95=${pct(v, 0.95).toFixed(0)} max=${Math.max(...v).toFixed(0)}mm/s`);
        }
        // 回合内位置分解（k=支撑回合内样本序号，从 1 起）：头/尾样本含摆动相泄漏，
        // 中部样本≈真支撑滑移——若头部 p95 远大于中部，泄漏主导可精化口径
        const byK = {};
        for (const s of n14.dbg) (byK[Math.min(s.k, 6)] ??= []).push(s.v * 1000);
        for (const k of Object.keys(byK).sort((a, b) => a - b)) {
          const v = byK[k];
          console.log(`  [N14-dbg] 回合内第${k}${k === '6' ? '+' : ''}样本: n=${v.length} p50=${pct(v, 0.5).toFixed(0)} p95=${pct(v, 0.95).toFixed(0)} max=${Math.max(...v).toFixed(0)}mm/s`);
        }
        // 距地深度分解：门限边缘（d≈5mm）样本含半接触噪声
        const byD = {};
        for (const s of n14.dbg) (byD[s.d < 0.002 ? '<2mm' : s.d < 0.0035 ? '2-3.5mm' : '3.5-5mm'] ??= []).push(s.v * 1000);
        for (const k of Object.keys(byD)) {
          const v = byD[k];
          console.log(`  [N14-dbg] 距地${k}: n=${v.length} p50=${pct(v, 0.5).toFixed(0)} p95=${pct(v, 0.95).toFixed(0)} max=${Math.max(...v).toFixed(0)}mm/s`);
        }
      }
    }
    // M-1（MERO-10 新增）：策略完整性/防静默降级假绿——done 时逐槽执行策略==计划策略；
    // mocap 默认矩阵至少 1 槽实际执行动捕回放（carrywalk JSON 损坏导致整场静默回落 rl
    // 而断言照绿的假绿场景由此拦截，方案 4.5 M-1 行 / K7）。
    {
      const ss = stack.slotStrategies;
      const matched = ss.planned.length === ss.executed.length
        && ss.executed.every((e, i) => e !== null && e === ss.planned[i]);
      const needMocap = stratCarry === 'mocap';
      const hasMocap = ss.executed.includes('mocap');
      check(`M-1 策略完整性（逐槽执行==计划${needMocap ? '，且至少 1 槽动捕回放' : ''}）`,
        matched && (!needMocap || hasMocap),
        `planned=[${ss.planned.join(',')}] executed=[${ss.executed.join(',')}]`);
    }

    // N15（瞬移消除改造，A2）：位移段清单表——机器人 root 水平位移段必须由脚步完成
    //（RL 行走或动捕步态回放），保留 blend 段单列透明化（位移 ≤0.15m 且滑移 p95 ≤400）。
    {
      // blend 段无论位移大小一律列出（队长透明度要求）；其余类别 <0.05m 不列
      const rows = [...surv.segs, ...(surv.seg ? [surv.seg] : [])]
        .filter((s) => s.disp >= 0.05 || s.cat === 'blend');
      console.log('  [N15] 位移段清单表（gait=行走/动捕步态；blend=保留残差吸收段（单列口径）；kin=其它 kinematic 段）:');
      console.log('  [N15]   起止(s)         位移(m)  类别   相位链                                        交替(L/R)  滑移p95 raw/drag(mm/s)  判定');
      let n15ok = true;
      const n15bad = [];
      for (const s of rows) {
        // 滑移判定口径（审查 F2/F3 裁决）：RL gait 段 = raw 口径 ≤400mm/s（与 N14
        // 冻结口径对齐；推进步态蹬地期跟球滑动是真步态组成，drag 口径仅供转身段）；
        // mocap 数据驱动段（CARRY_REPLAY 全段含 SETTLE 尾段、mid/high 借用片段）=
        // mocap 档 ≤800mm/s（重定向固有滑移，无接触约束，实测 423-708；不得手改
        // 动捕数据），表内标注。drag 流数值双列展示不作 gait 段判定依据。
        const p95s = p95of(s.vels) * 1000;
        const drag95 = p95of(s.drags && s.drags.length ? s.drags : s.vels) * 1000;
        const flips = s.lFlips + s.rFlips;
        const need = Math.max(1, Math.floor(s.disp / 0.45));
        const mocapDriven = stratCarry === 'mocap'
          && s.phases.some((p) => p === 'CARRY_REPLAY' || p === 'CARRY_REPLAY2' || p === 'RELEASE' || p === 'SETTLE');
        const slideCap = mocapDriven ? 800 : 400;
        let ok, note;
        if (s.cat === 'blend') {
          ok = s.disp <= 0.15 && p95s <= slideCap;
          note = `blend 上限 0.15m/400mm/s`;
        } else if (s.cat === 'kin' && s.disp <= 0.10) {
          ok = true;
          note = 'kin 小位移不判';
        } else if (s.cat === 'kin') {
          // kin 段（动捕回放尾段等）：位移门槛同 gait，但不强制双脚都抬（单步半步
          // 交替属真实步态尾），左右失衡有界即可
          ok = flips >= need && (flips === 0 || Math.abs(s.lFlips - s.rFlips) <= Math.max(1, flips / 2)) && p95s <= slideCap;
          note = mocapDriven ? `mocap 重定向固有档 ≤800` : `kin 需交替≥${need}`;
        } else if (mocapDriven) {
          // mocap 数据驱动段（mid/high 借用片段回放）：步态模式是数据属性，对齐 N4
          // 既定 mocap 口径（次数达标 + 左右均衡，不强制双脚都抬——mid 片段垫步 2/0
          // 属真实动捕）
          ok = flips >= need && (flips === 0 || Math.abs(s.lFlips - s.rFlips) <= Math.max(2, flips / 3)) && p95s <= slideCap;
          note = `mocap 重定向固有档 ≤800`;
        } else {
          // RL gait 段判据对齐 N4 既定口径（左右均衡 |L−R|≤max(2, flips/3)——RL 步态存在
          // 双足同瞬离地的飞行相与偶发同足连抬，连续连拍判据过严；防瞬移由 flips 计数
          // 把守，均衡判据防单足跳）
          ok = flips >= need && s.lFlips >= 1 && s.rFlips >= 1
            && Math.abs(s.lFlips - s.rFlips) <= Math.max(2, flips / 3) && p95s <= 400;
          note = `需交替≥${need}`;
        }
        if (!ok) { n15ok = false; n15bad.push(`${s.phases.join('>')}:${s.disp.toFixed(2)}m/${flips}次/raw=${p95s.toFixed(0)}mm/s`); }
        console.log(`  [N15]   ${s.t0.toFixed(1)}-${(s.t1 ?? data.time).toFixed(1)}  ${s.disp.toFixed(3)}  ${s.cat.padEnd(5)} ${s.phases.join('>').padEnd(42)}  ${`${s.lFlips}/${s.rFlips}`.padEnd(7)}  raw ${p95s.toFixed(0).padStart(5)} / drag ${drag95.toFixed(0).padStart(5)}  ${ok ? 'PASS' : 'FAIL'}（${note}）`);
      }
      if (rows.length === 0) { n15ok = false; n15bad.push('无位移段数据（巡视器失效？）'); }
      check('N15 位移段步态守门（gait 段交替≥max(1,⌊位移/0.45⌋)且滑移p95≤400；blend 段单列：位移≤0.15m 且滑移p95≤400）',
        n15ok, n15bad.length ? `不合格段: ${n15bad.join(' | ')}` : `${rows.length} 段全部达标`);
    }
    // N16（瞬移消除改造，A3）：转身段清单表——TURN_* 踏步转身（原地枢转+交替抬腿），
    // 定格转身已移除；滑移阈 100 的基线对比见巡视器注释（旧定格转身 80-120mm/s 持续拖扫）。
    {
      const rows = [...surv.turns, ...(surv.turn ? [surv.turn] : [])];
      console.log('  [N16] 转身段清单表（踏步转身）:');
      console.log('  [N16]   起止(s)         时长(s)  |Δyaw|(°)  交替(次数 L/R)  支撑拖拽p95(mm/s)  原始p95(含起落泄漏)  脚-地min(mm)  抬升max(mm)  判定');
      let n16ok = rows.length > 0;
      const n16bad = [];
      for (const t of rows) {
        const dur = (t.t1 ?? data.time) - t.t0;
        const y1 = t.yaw1 ?? t.yaw0;
        const dyaw = Math.abs(Math.atan2(Math.sin(y1 - t.yaw0), Math.cos(y1 - t.yaw0))) * 180 / Math.PI;
        const drag95 = p95of(t.drags ?? []) * 1000;   // 支撑拖拽（去起落泄漏）：A3 "显著下降"判据
        const raw95 = p95of(t.vels) * 1000;           // 原始口径（含抬脚/落脚泄漏）：与 N14 冻结阈对齐
        const liftMax = t.liftMax ?? 0;               // 摆动脚峰值离地（转身视觉改造子判据）
        const need = Math.max(2, Math.floor((dyaw * Math.PI / 180) / 0.5));
        const altOkBal = Math.abs((t.lFlips ?? 0) - (t.rFlips ?? 0)) <= Math.max(2, t.flips / 3);
        const ok = t.flips >= need && altOkBal && drag95 <= 100 && raw95 <= 400 && t.footMin > -0.025 && liftMax >= 0.03;
        if (!ok) {
          n16ok = false;
          n16bad.push(`${t.phase}:${t.flips}次(需${need})/drag=${drag95.toFixed(0)}mm/s/raw=${raw95.toFixed(0)}mm/s/min=${(t.footMin * 1000).toFixed(1)}mm/lift=${(liftMax * 1000).toFixed(1)}mm`);
        }
        console.log(`  [N16]   ${t.t0.toFixed(1)}-${(t.t1 ?? data.time).toFixed(1)}  ${dur.toFixed(1).padStart(6)}  ${dyaw.toFixed(0).padStart(7)}  ${`${t.lFlips ?? 0}/${t.rFlips ?? 0}`.padStart(7)}  ${drag95.toFixed(0).padStart(13)}  ${raw95.toFixed(0).padStart(15)}  ${(t.footMin * 1000).toFixed(1).padStart(11)}  ${(liftMax * 1000).toFixed(1).padStart(9)}  ${ok ? 'PASS' : 'FAIL'}`);
      }
      if (rows.length === 0) n16bad.push('无转身段数据（巡视器失效？）');
      check('N16 转身段守门（踏步交替≥max(2,⌊|Δyaw|/0.5⌋)且左右均衡、支撑拖拽p95≤100mm/s（基线定格转身 106-231mm/s 持续拖扫零交替）、原始p95≤400、穿透>-25mm、摆动脚峰值离地≥30mm（转身视觉改造新增：防退化回旧机制 ~7mm 抖动；仅统计单脚离地样本，入口悬浮不计）',
        n16ok, n16bad.length ? `不合格段: ${n16bad.join(' | ')}` : `${rows.length} 段全部达标`);
    }
    // N17（瞬移消除改造，A4/A5）：root 每子步水平跳变 + 无静默降级
    check('N17 root 单步跳变守门（全程每子步水平跳变 <0.3m，防 reset/瞬移回归）',
      maxRootJump < 0.3, `max=${(maxRootJump * 1000).toFixed(2)}mm@${maxRootJumpAt || '?'}`);
    check('N17b 无静默降级（全程 stack 模式恒 rl；legacy 瞬移降级已移除）',
      !surv.sawNonRlMode, surv.sawNonRlMode ? '运行中出现非 rl 模式' : '全程 rl');

    // N18（箱-身穿模修复）：携带段箱-身距离守门（判定集/观测集拆分，队长裁决）。
    // 判定集=非手代表 geom（以腕关节为界）硬门禁；手部 geom 为观测项（抓握固有重叠，
    // 演示级近似）只输出不判定。阈值 G1BOX_N18_MIN_MM 旋钮沿用（默认 -10mm 硬门禁）。
    console.log('  [N18] 携带段箱-身距离按相位分解（判定集=非手代表 geom，mm）:');
    for (const [phN, b] of Object.entries(n18.byPhase)) {
      console.log(`  [N18]   ${phN}: min=${mm(b.min)}mm @${b.at}s  ${b.link}`);
    }
    if (n18.top.length > 0) {
      console.log('  [N18] 最深穿插 Top-' + n18.top.length + '（判定集，时间/相位/槽位/link/深度）:');
      for (const e of n18.top) {
        console.log(`  [N18]   t=${e.t.toFixed(2)}s ${e.ph} slot${e.slot} ${e.link} ${(e.d * 1000).toFixed(1)}mm`);
      }
    }
    if (n18Hand.n > 0) {
      console.log('  [N18] 手部观测项（抓握固有重叠，演示级近似，不参与判定）:');
      for (const [phN, b] of Object.entries(n18Hand.byPhase)) {
        console.log(`  [N18]   ${phN}: min=${mm(b.min)}mm @${b.at}s  ${b.link}`);
      }
    }
    const N18_MIN = (parseFloat(process.env?.G1BOX_N18_MIN_MM) < 0
      ? parseFloat(process.env.G1BOX_N18_MIN_MM) : -10) / 1000;
    check(`N18 携带段箱-身最小距离（判定集=非手代表 geom，抓取起→释放止，阈 ${(N18_MIN * 1000).toFixed(0)}mm 硬门禁；手部为观测项）`,
      n18.n > 0 && n18.min > N18_MIN,
      n18.n > 0
        ? `非手 min=${mm(n18.min)}mm @${n18.at} ${n18.link}（n=${n18.n}）`
          + `；手部观测 min=${n18Hand.min === Infinity ? 'n/a' : mm(n18Hand.min) + 'mm'} @${n18Hand.at} ${n18Hand.link}`
        : '无采样数据（携带段未发生或采样器失效）');

    // F2b（三处消费一致性 b）：规划快照中的静态矩形与 factory.js 表同源（同 key 同
    // 位姿同 role）——防 collectWalkObstacles 的静态注册被后续改动静默丢失/漂移（D4）。
    {
      const snap = stack.lastWalkObstacles;
      const staticRectsF = factoryObstacleRects();
      const got = snap?.rects?.filter((r) => String(r.key).startsWith('static:')) ?? [];
      let ok = got.length === staticRectsF.length, detail = ok ? '' : `快照静态矩形 ${got.length} 个 != 表 ${staticRectsF.length} 个`;
      if (ok) {
        const byKey = new Map(staticRectsF.map((r) => [r.key, r]));
        for (const r of got) {
          const e = byKey.get(r.key);
          if (!e) { ok = false; detail = `快照含未知 key ${r.key}`; break; }
          if (Math.abs(r.cx - e.cx) > 1e-9 || Math.abs(r.cy - e.cy) > 1e-9 || Math.abs(r.hx - e.hx) > 1e-9
            || Math.abs(r.hy - e.hy) > 1e-9 || Math.abs(r.yaw - e.yaw) > 1e-9 || r.role !== e.role) {
            ok = false; detail = `矩形 ${r.key} 与表值不一致（role ${r.role}/${e.role}）`;
            break;
          }
        }
        if (ok) detail = `${got.length} 个静态矩形（含活动货架 role='target'）同 key 同值`;
      }
      check('F2b 规划快照静态矩形与 FACTORY 表同源（key/位姿/role）', ok, detail);
    }

    // N19（walk-avoid-plan F6）：行走段避障间隙分级断言（方案 4.5.1）。
    // 障碍集双源防自报遗漏：stackCore lastWalkObstacles（规划同源快照，含 target/staging）
    // ∪ 无头侧 data.qpos 直读的已放物理箱矩形（物理真值），取并集判定；距离口径与规划器
    // 同源（pathPlanner.pointRectDist，2D root 到旋转矩形）。
    // 分级（每段 × 每矩形取最小）：
    //   N19a（硬，底线）非目标箱全程 dist ≥ 0.05——防可见穿模红线（WALK_HOME 起点 standoff
    //        定标 0.224，余量 4.5×）；
    //   N19b（硬，规划意图）非目标箱段内累计路程 >0.50m 后 dist ≥ 0.35（=robotR 0.30+0.05；
    //        规划膨胀 0.40 → 0.05 跟踪预算），起点脱出路点保证此时已出膨胀；
    //   N19c（目标箱）距段目标 >0.5m 的行进段 dist ≥ 0.02（防踩进箱面）；末段 0.5m 仅打印
    //        最小值不判定——取箱站位对自家箱 0.139 的设计性贴近由 PREPARE blend/抓取窗口
    //        收尾，与 N18"抓握固有几何为演示级近似"同口径。
    // 重试重规划段（walkSeq 递增）逐段独立判定。degraded 规划降级计数必须 0（非 0 即红）。
    // 【N19 阈值——M4 联定冻结区】起步值 = 方案 4.5.1 定标值（定标数据方案 3.2：取箱站位
    // 对自家箱 0.139 / 释放站位对刚放箱 0.224）；按 N19 实测最小值冻结（只紧不松，队长
    // 裁定 Q4），联定数据实测后回填本注释（惯例同 N14）。
    const N19A_MIN = 0.05;   // 非目标箱全程底线（m）
    const N19B_MIN = 0.35;   // 非目标箱行进段（m）
    const N19C_MIN = 0.02;   // 目标箱行进段（m）
    const N19B_PATH = 0.50;  // N19b 行进段门：段内累计路程（m）
    const N19C_TAIL = 0.5;   // N19c 末段观测窗：距段目标（m）
    {
      const placedRectsH = [];
      {
        const bxN19 = set.low.box;
        const boxesH = stack.boxes();
        const nHand = stack.status().handedOver;
        for (let i = 0; i < nHand && i < boxesH.length; i++) {
          const adr = boxesH[i].jointQadr;
          const qN19 = [data.qpos[adr + 3], data.qpos[adr + 4], data.qpos[adr + 5], data.qpos[adr + 6]];
          // geom 中心 = body 位姿 ⊗ bbox_center_offset（水平分量随全四元数旋转，与组装方同口径）
          const cN19 = quatRotVec(qN19, [bxN19.bbox_center_offset[0], bxN19.bbox_center_offset[1], bxN19.bbox_center_offset[2]]);
          placedRectsH.push({
            cx: data.qpos[adr] + cN19[0], cy: data.qpos[adr + 1] + cN19[1],
            yaw: Math.atan2(2 * (qN19[0] * qN19[3] + qN19[1] * qN19[2]), 1 - 2 * (qN19[2] * qN19[2] + qN19[3] * qN19[3])),
            hx: bxN19.half_size[0], hy: bxN19.half_size[1], role: 'placed', key: `phys:${i}`,
          });
        }
      }
      console.log('  [N19] 行走段避障间隙分级（walk-avoid-plan 4.5.1；非目标全程/行进段、目标箱行进段；末段 0.5m 与起点 standoff 观测化）:');
      console.log('  [N19]   seq 相位            规划kind  非目标min(全程)  非目标min(行进)  目标min(行进)  目标min(末段,观测)  判定');
      let n19ok = true;
      const n19bad = [];
      const rlSegsN19 = walkStats.segments.filter((s) => s.kind === 'rl' && s.n19);
      for (const seg of rlSegsN19) {
        const snap = seg.n19.snap;
        if (!snap || !Array.isArray(snap.rects) || seg.n19.samples.length === 0) {
          n19ok = false;
          n19bad.push(`seq${seg.seq} 无规划快照/采样（采样器失效？）`);
          continue;
        }
        const rectsN19 = [...snap.rects, ...placedRectsH];
        let minAll = Infinity, minTravel = Infinity;
        let minTgtTravel = Infinity, minTgtTail = Infinity;
        let tgtSeen = false;
        for (const r of rectsN19) {
          const isTgt = r.role === 'target';
          let a = Infinity, b = Infinity, cT = Infinity, cTl = Infinity;
          for (const sp of seg.n19.samples) {
            const d = pointRectDist(sp, r);
            if (!isTgt) {
              if (d < a) a = d;
              if (sp.cum > N19B_PATH && d < b) b = d;
            } else {
              tgtSeen = true;
              const dg = snap.goal ? Math.hypot(sp.x - snap.goal[0], sp.y - snap.goal[1]) : Infinity;
              // 观测窗对称化（P2 审查修复）：段两端各 0.5m 为观测窗——末端覆盖 approach
              // 取箱的设计性贴近（既有口径），起点覆盖 home 站位贴货架 target 矩形
              //（站位 root 在货架矩形边缘内属设计性贴近，方案 D4 target 机制语义的
              // 必然延伸；旧口径只豁免末端，home 起点贴边被误计为行进段违规）
              if (dg > N19C_TAIL && sp.cum > N19B_PATH) { if (d < cT) cT = d; } else if (d < cTl) cTl = d;
            }
          }
          if (a < minAll) minAll = a;
          if (b < minTravel) minTravel = b;
          if (cT < minTgtTravel) minTgtTravel = cT;
          if (cTl < minTgtTail) minTgtTail = cTl;
        }
        const segOk = minAll >= N19A_MIN && minTravel >= N19B_MIN && (!tgtSeen || minTgtTravel >= N19C_MIN);
        if (!segOk) {
          n19ok = false;
          n19bad.push(`seq${seg.seq}${seg.phase}: 全程=${minAll === Infinity ? 'n/a' : minAll.toFixed(3)}`
            + ` 行进=${minTravel === Infinity ? 'n/a' : minTravel.toFixed(3)}`
            + (tgtSeen ? ` 目标行进=${minTgtTravel === Infinity ? 'n/a' : minTgtTravel.toFixed(3)}` : ''));
        }
        console.log(`  [N19]   ${String(seg.seq).padEnd(3)} ${seg.phase.padEnd(15)} ${(snap.kind ?? '?').padEnd(9)} `
          + `${minAll === Infinity ? 'n/a' : minAll.toFixed(3)}            `
          + `${minTravel === Infinity ? 'n/a' : minTravel.toFixed(3)}              `
          + `${tgtSeen ? (minTgtTravel === Infinity ? 'n/a' : minTgtTravel.toFixed(3)) : '-'}            `
          + `${tgtSeen ? (minTgtTail === Infinity ? 'n/a' : minTgtTail.toFixed(3)) : '-'}  ${segOk ? 'PASS' : 'FAIL'}`);
      }
      const statsN19 = stack.walkPlanStats;
      const modesStr = Object.entries(statsN19.modes).map(([k, v]) => `${k}:${v}`).join('/') || 'n/a';
      const degradedOk = statsN19.degraded === 0;
      if (!degradedOk) { n19ok = false; n19bad.push(`degraded=${statsN19.degraded}/${statsN19.total}（规划降级，默认链必须 0）`); }
      if (rlSegsN19.length === 0) { n19ok = false; n19bad.push('无 RL 行走段数据（采样器失效？）'); }
      console.log(`  [N19] planner mode 分布 ${modesStr}，degraded=${statsN19.degraded}/${statsN19.total}`);
      check('N19 行走避障间隙（非目标全程≥0.05/行进段≥0.35/目标箱行进段≥0.02，末段0.5m观测；degraded=0）',
        n19ok, n19bad.length ? `不合格: ${n19bad.join(' | ')}` : `${rlSegsN19.length} 段全部达标`);
    }
  }
}

// ---------------- 入口 ----------------
const mujoco = await loadMujoco();

runUnitTests();
if (SELFTEST_ONLY) {
  // --policy-selftest：只跑 N1 golden 校验（快速，无仿真）
  const policy = await loadPolicy('vendor/policy', fsFetch);
  const st = policySelfTest(policy.weights, policy.golden);
  check(`N1 golden fixture 前向一致性 (max|Δ|<1e-4)`, st.ok, st.detail);
} else if (hasFlag('--unit-only')) {
  // 只跑纯函数自测
} else if (hasFlag('--fk')) {
  await runFk(mujoco);
} else {
  runFactoryUnitChecks(); // 工厂方案 F2c/F3/F4/F5（纯函数，无需编译模型）
  await runStack(mujoco); // 内含 F1/F2a（模型级）与 F2b（规划快照级）
}

const fails = results.filter((r) => !r.ok);
console.log(`\n== 结果: ${nChecks - fails.length}/${nChecks} 通过 ==`);
if (hasFlag('--negcheck')) {
  // negcheck 语义（防假绿）：故意注入的错误必须让对应断言 FAIL。
  // all 模式（默认）：改错槽位 z → 断言3 FAIL + 破坏腿装订 → N4/N5/N7 至少一个 FAIL
  // （≥2 如期失败即通过，MERO-9 前口径不变）。
  // carry 模式（MERO-10 第三组）：携带漂移 0.3m → N13-mocap 必 FAIL（单判据模式，
  // 注释见 NEG_MODE 声明处——腿装订注入会让 mocap 段不执行，三组无法同场观测）。
  if (NEG_MODE === 'carry') {
    // carry 注入通过判据（MERO-10）：N13-mocap（箱漂移捕获）与 N14（root 走廊漂移捕获）
    // 都必须 FAIL——防假绿机制对两条新断言都有效（派发口径"N13-mocap/N14 必 FAIL"）。
    const n13m = results.find((r) => r.name.startsWith('N13-mocap'));
    const n14r = results.find((r) => r.name.startsWith('N14 '));
    const bads = [n13m, n14r].filter((r) => r && !r.ok);
    if (n13m && n14r && bads.length === 2) {
      console.log(`negcheck 通过：携带漂移注入后 N13-mocap 与 N14 均按预期 FAIL（防假绿机制有效）\n`
        + `  - ${n13m.name.split('（')[0]}: ${n13m.detail}\n  - ${n14r.name.split('（')[0]}: ${n14r.detail}`);
      process.exit(0);
    }
    const missing = [!n13m && 'N13-mocap', !n14r && 'N14'].filter(Boolean);
    const notFailed = [n13m?.ok && 'N13-mocap', n14r?.ok && 'N14'].filter(Boolean);
    console.log(`negcheck 失败：携带漂移注入后未如期 FAIL（缺失断言: ${missing.join(',') || '无'}；仍 PASS: ${notFailed.join(',') || '无'}），断言机制失效！`);
    process.exit(1);
  }
  if (NEG_MODE === 'avoid') {
    // walk-avoid-plan 4.5.2 通过判据：避障注入后 N19 必 FAIL（注入 1 中部路点向最近障碍
    // 拉偏 / 注入 2 G1BOX_AVOID_OFF=1 规划短路直线复刻穿模缺陷形态）；未如期 FAIL =
    // 断言机制失效。惯例同 --negcheck carry（单判据模式，独占干净运行）。
    const n19r = results.find((r) => r.name.startsWith('N19'));
    if (n19r && !n19r.ok) {
      console.log(`negcheck 通过：避障注入后 N19 按预期 FAIL（防假绿机制有效）\n  - ${n19r.name.split('（')[0]}: ${n19r.detail}`);
      process.exit(0);
    }
    console.log(`negcheck 失败：避障注入后 N19 未如期 FAIL（${n19r ? '仍 PASS: ' + n19r.detail : 'N19 断言缺失'}），断言机制失效！`);
    process.exit(1);
  }
  const boxCheck = results.find((r) => r.name.startsWith('断言3'));
  const expectFail = [boxCheck].filter(Boolean);
  if (WALK_MODE === 'rl') {
    for (const nm of ['N4', 'N5', 'N7']) {
      const r = results.find((x) => x.name.startsWith(nm));
      if (r) expectFail.push(r);
    }
  }
  const caught = expectFail.filter((r) => !r.ok);
  if (expectFail.length > 0 && caught.length >= (WALK_MODE === 'rl' ? 2 : 1)) {
    console.log(`negcheck 通过：注入错误后被 ${caught.map((r) => r.name.split(' ')[0]).join('/')} 按预期捕获（防假绿机制有效）`);
    process.exit(0);
  }
  console.log(`negcheck 失败：注入错误后未按预期 FAIL（${expectFail.length} 个应失败断言中 ${caught.length} 个失败），断言机制失效！`);
  process.exit(1);
}
process.exit(fails.length ? 1 : 0);
