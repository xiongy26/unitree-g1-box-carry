// RL 行走控制器：obs 组装 / 50Hz 推理分频 / 动作解码（softArmHold 渐进混合）/ 指令 P 控制 /
// 增益改写与恢复 / 跌倒检测（方案 docs/rl-walking-plan.md 4.4-4.6、5.1、D1/D2/D7）。
// 纯 ESM、无 DOM/three 依赖，Node 可直接 import（headless 与浏览器共用同一实现）。
// 路径规划已移至 src/pathPlanner.js（walk-avoid-plan）：本模块只负责路点跟踪（walkTo/step/
// 停站/超时机制），不再持有任何规划逻辑（单一事实源）。
//
// 硬契约（方案 5.3）：
// - RL 段不写机器人 qpos/qvel，只写 ctrl 与（enter/exit 时）执行器增益；
// - PD 力矩走 position 执行器 gainprm/biasprm 改写（act*10+k——每执行器 10 列，
//   mjNGAIN/mjNBIAS=10，T0 实测；qfrc_applied 恒 0 契约不受影响）；
// - 增益改写只发生在 enter()/exit() 两时机，exit() 幂等，任何异常路径由调用方保证走到 exit()。
//
// 【MERO-8 softArmHold（渐进动作空间混合）】臂部干预两代失败方案的第三条路线：
//  - D2（ctrl 级覆盖 + last_action=raw）：三通道不一致正反馈 → 摔倒（T3 实测）；
//  - A1（MERO-7，kinematic 定格 qpos/qvel/ctrl）：即使三通道自洽，臂部反作用力不再
//    经关节动力学累积 → 出训练分布 → 摔倒（mero7-evidence/v1_A1_default.log，16/19）；
//  - softArmHold（本实现）：策略 29 维动作全量计算后、下发前，臂 14 维做
//    act_send = (1−α)·act + α·holdAct（holdAct = 持箱姿态反解到 policy 动作域），
//    obs last_action 喂 act_send（闭环自洽）；ctrl = DEFAULT + 0.25·act_send 照常下发，
//    臂部仍由 PD 伺服真实执行（反作用力经动力学）——不写任何 qpos/qvel、不动增益。
//    α 按 armHoldRamp 调度渐入（默认 warmup 0.6s smoothstep）。是否默认开启由门控
//    实验裁定（见 docs/policy-probe-notes.md MERO-8 节），URL ?armhold=off /
//    headless --armhold off 可关。

import { policyForward } from './g1Policy.js';
import { quatRotVec, quatConj, smoothInto } from '../util.js';

// softArmHold 默认开关（门控实验结论回填处；'soft'=默认开启 | 'off'=默认关闭）
export const ARMHOLD_DEFAULT = 'off';

const CTRL_DECIM = 10; // 500Hz 子步 / 10 = 50Hz 推理（训练 step_dt 0.02）
const HIST = 5;        // obs 历史帧数（manifest.obs.history）

// 指令斜坡（固件先例 vx 0.02/tick @50Hz；umw cmd_ramp 记录，T0 探针同参数验证稳定）
const CMD_RAMP = [0.02, 0.02, 0.02];

// 跌倒检测阈值（4.6）：重力投影 z（机体系，直立 -1）|z|<0.6 即倾斜 >53°，
// 且 pelvis z<0.5m，双条件同持 100ms 去抖。口径解读见 docs/policy-probe-notes.md。
const FALL_GRAV_Z = -0.6;
const FALL_PELVIS_Z = 0.5;
const FALL_DEBOUNCE_S = 0.1;

// 到位判据（4.4 + R5 门限放宽延伸）：dist<0.15m 持续 0.5s。T0 实测策略指令死区 ≈0.25
// （|cmd|≲0.25 时站立吸引子占优），精确停靠不可达——采用「惯性停站」：距目标 STOP_BAND
// 内直接零指令，靠步态惯性+站立吸引子停住，剩余 ≤0.15m 位置/朝向残差由 PREPARE /
// ALIGN_PLACE 的 kinematic blend 吸收（4.2 序 2/5 本义）。偏差记录见交接文档。
const ARRIVE_DIST = 0.15;
const ARRIVE_YAW = 0.1;
const ARRIVE_HOLD = 0.5;
// 停止带：距目标 <0.08m 直接零指令（实测停点距目标 0.10-0.12m，落在 ARRIVE_DIST 内）
const STOP_BAND = 0.08;

export function createWalkController({
  mujoco, model, data, jmap, jointNames, policy, meta = {},
  armHold = ARMHOLD_DEFAULT,      // 'off' | 'soft'（MERO-8 softArmHold 总开关）
  armHoldRamp = 'warmup',         // 'warmup' | 'delay' | 'fixed'（MERO-9 手段3：固定α）| 'pitch'（手段4：仅锁肩俯仰）
  armHoldTight = false,           // 配方 3：softArmHold 生效的 carry 段收紧指令（vx≤0.15、wz≤0.1）
  armHoldFixedAlpha = 0,          // ramp='fixed' 的目标 α（0=未配置；渐入节奏同 warmup 0.6s smoothstep）
  carryKd = 0,                    // MERO-9 手段2：carry 行走段臂 14 执行器 kd 覆盖（0=off；仅 biasprm[2]，kp 不动）
  log = () => {},
}) {
  const manifest = policy.manifest;
  const MAP = manifest.joint_map.policy_to_mjcf;               // policy p → mjcf 关节索引
  const DEFAULT_POS = Float64Array.from(manifest.action.default_joint_pos); // policy 序
  const ACT_SCALE = manifest.action.scale;
  const CMD_LIM = manifest.cmd_limits;                         // 训练范围（硬边界）
  const KP_MJCF = manifest.gains.kp;                           // mjcf 执行器序
  const KD_MJCF = manifest.gains.kd;

  // 演示限幅（D7 + T0 能力边界）：指令温和化。T0 实测官方 velocity 策略只有
  // 「前进 + 弱转向」组合稳定（vx=0.3 纯前进 1.24m/5s；vx+vy+wz 全饱和会漂移摔倒），
  // 因此 vx≤0.3、vy/wz 只保留小幅度微调，行进方向对齐交给定格转身 + blend。
  const lim = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const VX_MAX = lim(meta.vxMax ?? 0.3, CMD_LIM.vx[0], CMD_LIM.vx[1]);
  // MERO-9 手段1：carry 段指令限速（纯指令层，零 OOD 风险）。默认 0.3（基线）；
  // 扫描配方经 headless --carry-vx（→ meta.carryVxMax）实验，合格值回填此默认。
  const CARRY_VX_MAX = lim(meta.carryVxMax ?? 0.3, CMD_LIM.vx[0], CMD_LIM.vx[1]);
  const VY_MAX = lim(meta.vyMax ?? 0.15, CMD_LIM.vy[0], CMD_LIM.vy[1]);
  const WZ_MAX = lim(meta.wzMax ?? 0.2, CMD_LIM.wz[0], CMD_LIM.wz[1]);
  const SLOW_BAND = meta.slowBand ?? 0;      // 减速带已废弃：指令死区 ≈0.25 使中途减速即停摆，
  const SLOW_VX = lim(meta.slowVx ?? 0.3, CMD_LIM.vx[0], CMD_LIM.vx[1]); // 停站改用惯性停站（STOP_BAND）
  // MERO-8 配方 3（armHoldTight）：softArmHold 生效的 carry 段专用指令限幅。注意
  // vx=0.15 低于实测指令死区 ~0.25——步态可能不激发致停摆/超时，属门控实验配方，
  // 全红即放弃（死区数据见 docs/policy-probe-notes.md）。
  const CARRY_SOFT_VX_MAX = lim(meta.carrySoftVxMax ?? 0.15, CMD_LIM.vx[0], CMD_LIM.vx[1]);
  const CARRY_SOFT_WZ_MAX = lim(meta.carrySoftWzMax ?? 0.1, CMD_LIM.wz[0], CMD_LIM.wz[1]);
  // KP_LIN=2.0：T0 实测步态激发需要 vx≳0.25（死区 ~0.2）——斜向目标会把 vx 稀释进死区
  // 造成停摆，增益抬高使斜向目标下 vx 仍保持满速激发，侧向由 vy/wz 微调
  const KP_LIN = meta.kpLin ?? 2.0;
  const KP_YAW = meta.kpYaw ?? 1.0;

  // ---------------- 装订展开（policy 序视角） ----------------
  const N = 29;
  const bind = new Array(N);
  for (let p = 0; p < N; p++) bind[p] = jmap[jointNames[MAP[p]]];
  const ctrlLo = new Float64Array(N), ctrlHi = new Float64Array(N);
  for (let p = 0; p < N; p++) {
    const act = bind[p].c;
    const lo = model.actuator_ctrlrange[act * 2 + 0], hi = model.actuator_ctrlrange[act * 2 + 1];
    ctrlLo[p] = hi > lo ? lo : -1e9; // 本 binding 不暴露 ctrllimited，用 range 有效性判断（T0 同款）
    ctrlHi[p] = hi > lo ? hi : 1e9;
  }
  // 腰与臂的 policy 槽位（D2：腿 12 + 腰 3 采纳策略输出；臂 14 由动捕边界姿态驱动）
  const isArm = new Uint8Array(N);
  for (let p = 0; p < N; p++) {
    const mj = MAP[p];
    isArm[p] = mj >= 15 && mj <= 28 ? 1 : 0; // mjcf 左臂 15-21 / 右臂 22-28
  }

  const pelvisBid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'pelvis');

  // ---------------- 增益改写（enter/exit 两时机，exit 幂等） ----------------
  let entered = false;
  let savedGains = null;
  function writeGains() {
    for (let p = 0; p < N; p++) {
      const m = MAP[p]; // kp/kd 表按 mjcf 序
      const act = jmap[jointNames[m]].c;
      model.actuator_gainprm[act * 10 + 0] = KP_MJCF[m];
      model.actuator_biasprm[act * 10 + 1] = -KP_MJCF[m];
      model.actuator_biasprm[act * 10 + 2] = -KD_MJCF[m];
    }
  }
  function enter() {
    if (entered) return;
    // 可写性自证：写读回一个探针值，embind 只读绑定会在此抛出（调用方 catch 后降级）
    const act = bind[0].c;
    const g0 = model.actuator_gainprm[act * 10 + 0];
    model.actuator_gainprm[act * 10 + 0] = g0 + 0.5;
    const ok = Math.abs(model.actuator_gainprm[act * 10 + 0] - (g0 + 0.5)) < 1e-9;
    model.actuator_gainprm[act * 10 + 0] = g0;
    if (!ok) throw new Error('actuator_gainprm 不可写（H4 不成立），走降级路径');
    savedGains = [];
    for (let p = 0; p < N; p++) {
      const act2 = bind[p].c;
      savedGains.push([act2,
        model.actuator_gainprm[act2 * 10 + 0],
        model.actuator_biasprm[act2 * 10 + 1],
        model.actuator_biasprm[act2 * 10 + 2]]);
    }
    writeGains();
    entered = true;
    log('[walker] enter：执行器增益已改写为策略 PD');
  }
  function exit() {
    if (!entered) return; // 幂等
    for (const [act, g, b1, b2] of savedGains) {
      model.actuator_gainprm[act * 10 + 0] = g;
      model.actuator_biasprm[act * 10 + 1] = b1;
      model.actuator_biasprm[act * 10 + 2] = b2;
    }
    savedGains = null;
    entered = false;
    carryKdActive = false;
    log('[walker] exit：Menagerie 增益已恢复');
  }

  // ---------------- MERO-9 手段2：carry 行走段臂部阻尼覆盖（部分/被动抑摆） ----------------
  // 仅在 carry 行走段（walkTo(carry=true) → 段结束的下一个 walkTo(carry=false)）把臂 14
  // 执行器的 biasprm[2]（kd）从策略值抬到 CARRY_KD；kp、腿腰、动作通道全部不动——摆臂
  // 幅度经物理阻尼衰减而非通道接管，属「部分/温和」干预（硬约束：完全接管必摔）。
  // 进出场时机与 writeGains 同址（walkTo / exit），exit() 由 savedGains 全量恢复幂等兜底。
  // 未 enter() 时禁止写（savedGains 尚未建立，会把覆盖值误存为"原始值"破坏恢复语义）。
  let carryKdActive = false;
  const CARRY_KD = carryKd > 0 ? carryKd : 0;
  function applyCarryKd(on) {
    if (!(CARRY_KD > 0) || !entered) return;
    const changed = carryKdActive !== on;
    for (let p = 0; p < N; p++) {
      if (!isArm[p]) continue;
      const act = bind[p].c;
      model.actuator_biasprm[act * 10 + 2] = -(on ? CARRY_KD : KD_MJCF[MAP[p]]);
    }
    carryKdActive = on;
    if (changed) log(`[walker] carry 臂部 kd 覆盖 ${on ? `生效（kd=${CARRY_KD}，kp 不动）` : '恢复策略值'}`);
  }

  // ---------------- obs 历史与动作 ----------------
  const termDims = [3, 3, 3, N, N, N];
  const termScales = [0.2, 1.0, 1.0, 1.0, 0.05, 1.0];
  const hist = termDims.map((d) => Array.from({ length: HIST }, () => new Float32Array(d)));
  const obsBuf = new Float32Array(manifest.obs.frame * manifest.obs.history); // 480
  let lastAction = new Float32Array(N);   // raw（喂 obs 的 last_action）
  let lastActionOut = new Float32Array(N);

  function buildFrame() {
    const frame = new Float32Array(96);
    frame[0] = data.qvel[3]; frame[1] = data.qvel[4]; frame[2] = data.qvel[5]; // 体系角速度（自由铰 qvel[3:6]）
    const q = [data.xquat[4 * pelvisBid], data.xquat[4 * pelvisBid + 1], data.xquat[4 * pelvisBid + 2], data.xquat[4 * pelvisBid + 3]];
    const grav = quatRotVec(quatConj(q), [0, 0, -1]);
    frame[3] = grav[0]; frame[4] = grav[1]; frame[5] = grav[2];
    frame[6] = curCmd[0]; frame[7] = curCmd[1]; frame[8] = curCmd[2];
    for (let p = 0; p < N; p++) {
      frame[9 + p] = data.qpos[bind[p].q] - DEFAULT_POS[p];
      frame[38 + p] = data.qvel[bind[p].d];
    }
    for (let p = 0; p < N; p++) frame[67 + p] = lastAction[p];
    return frame;
  }
  function pushHist(frame) {
    let off = 0;
    for (let t = 0; t < 6; t++) {
      hist[t].shift();
      hist[t].push(frame.slice(off, off + termDims[t]));
      off += termDims[t];
    }
  }
  function assembleObs() {
    // term-major（deploy use_gym_history=false 默认，T0 消融唯一存活装订）：term 外层、旧→新
    let off = 0;
    for (let t = 0; t < 6; t++) {
      const sc = termScales[t];
      for (let h = 0; h < HIST; h++) {
        const fr = hist[t][h];
        for (let i = 0; i < termDims[t]; i++) obsBuf[off++] = fr[i] * sc;
      }
    }
    return obsBuf;
  }

  // ---------------- 指令与行走段状态 ----------------
  let curCmd = new Float32Array(3);
  let targetCmd = new Float32Array(3);
  let waypoints = null;      // [{x, y, yaw}]
  let wpIdx = 0;
  let carrying = false;
  let armGoal = null;        // Float64Array(29)（mjcf 序关节目标；臂 14 生效，指数平滑）
  let armGoalSmooth = null;
  // MERO-8 softArmHold：holdPose = 持箱姿态（mjcf 序 29 维，walkTo 传入）；holdAct =
  // 其反解到 policy 动作域的臂 14 维定值 (pose−DEFAULT_POS)/ACT_SCALE。仅 armHold='soft'
  // 且 carrying 段传入 holdPose 时生效；α 渐入期间 act_send = (1−α)·act + α·holdAct。
  // MERO-9：holdMask 为每维 α 乘子——'pitch' 模式只锁左右 shoulder_pitch 2 维（手段4），
  // 其余模式臂 14 维全 1（手段3 的 fixed-α 走 α 调度封顶，不在 mask 上区分）。
  let holdPose = null;
  const holdAct = new Float32Array(N);
  const holdMask = new Float32Array(N);
  let armHoldAlphaCur = 0;   // 当前 α（诊断视图用）
  let timeLimit = Infinity;
  let yawTol = ARRIVE_YAW;
  let vxScaleCur = 1.0;     // 本段指令速度缩放（重试差异化：确定性仿真下同起点重试必同结果）
  let segMinDist = Infinity; // 本段内距最后路点的最近距离（擦过即达判据）
  let phaseT = 0;
  let warmupT = 0;
  let arrivedT = 0;
  let sub = 0;
  let fallen = false;
  let fallTimer = 0;
  let doneFlag = false;
  let overtime = false;
  let dist = 0;
  let walkSeq = 0;          // 行走段序号（walkTo 递增；采样/断言按此识别段边界，含重试）
  let completedGoal = null; // 最近一个正常完成段到位时刻的误差快照（N6 观测）
  const lastCmdView = new Float32Array(3);

  let turnBrakeT = 0;
  let turnYaw = null;
  let turnReady = null;
  let legStart = null;
  let posture = { x: 0, y: 0, yaw: 0 };
  function readPosture() {
    posture.x = data.qpos[0];
    posture.y = data.qpos[1];
    const w = data.xquat[4 * pelvisBid], x = data.xquat[4 * pelvisBid + 1];
    const y = data.xquat[4 * pelvisBid + 2], z = data.xquat[4 * pelvisBid + 3];
    posture.yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }

  // 行走段设定。waypoints: [{x,y,yaw}]；carrying: 持箱限速；timeLimitSec: 容差超时（降级判定）；
  // armGoal: 29 维关节目标（mjcf 序，臂 14 生效，D2 遗留备查通道，现恒 null）；
  // holdPose: 29 维持箱姿态（mjcf 序，MERO-8 softArmHold，仅 carry 段传入）。
  // 返回 { dist }（起点到终点路径长）。
  function walkTo({ waypoints: wps, carrying: carry = false, timeLimitSec = Infinity, armGoal: arm = null, holdPose: hold = null, yawTolerance = ARRIVE_YAW, vxScale = 1.0 }) {
    waypoints = wps;
    turnBrakeT = 0; turnYaw = null; turnReady = null;
    legStart = { x: data.qpos[0], y: data.qpos[1] };
    wpIdx = 0;
    vxScaleCur = vxScale;
    walkSeq++;
    carrying = carry;
    armGoal = arm ? Float64Array.from(arm) : null;
    armGoalSmooth = arm ? Float64Array.from(arm) : null;
    holdPose = armHold === 'soft' && carry && hold ? Float64Array.from(hold) : null;
    holdMask.fill(0);
    if (holdPose) {
      for (let p = 0; p < N; p++) {
        if (isArm[p]) {
          holdAct[p] = (holdPose[MAP[p]] - DEFAULT_POS[p]) / ACT_SCALE;
          // 'pitch'（手段4）：仅锁左右 shoulder_pitch（mjcf 关节 15/22），其余 12 臂维照常策略输出
          holdMask[p] = armHoldRamp === 'pitch' ? ((MAP[p] === 15 || MAP[p] === 22) ? 1 : 0) : 1;
        }
      }
    }
    armHoldAlphaCur = 0;
    applyCarryKd(carry); // MERO-9 手段2：carry 段臂 kd 覆盖进出场与段配置同址
    timeLimit = timeLimitSec;
    yawTol = yawTolerance;
    phaseT = 0;
    arrivedT = 0;
    segMinDist = Infinity;
    doneFlag = false;
    overtime = false;
    readPosture();
    let pathLen = 0;
    let px = posture.x, py = posture.y;
    for (const wp of wps) { pathLen += Math.hypot(wp.x - px, wp.y - py); px = wp.x; py = wp.y; }
    dist = pathLen;
    log(`[walker] walkTo：${wps.length} 路点 路径 ${pathLen.toFixed(2)}m carry=${carry} `
      + `起点 z=${data.qpos[2].toFixed(2)} xy=(${posture.x.toFixed(2)},${posture.y.toFixed(2)}) yaw=${posture.yaw.toFixed(2)}`);
    if (holdPose) log(`[walker] softArmHold 生效（ramp=${armHoldRamp}${armHoldRamp === 'fixed' ? ` α=${armHoldFixedAlpha}` : ''}${armHoldTight ? '+tight' : ''}）：carry 段臂部动作向持箱姿态渐入（mask=${armHoldRamp === 'pitch' ? '仅肩俯仰2维' : '臂14维'}）`);
    return { dist: pathLen };
  }

  // softArmHold 的 α 调度（MERO-8/MERO-9）：armHoldRamp 决定渐入节奏。
  //  - warmup（默认）：段首 0.6s smoothstep 0→1——与零指令预热窗重合，双臂在起步
  //    下垂瞬态发生前就被保持住（off 模式实测该瞬态 ~0.5s 内垂 0.3m，见 N12 证据）；
  //  - delay：起步 0.8s 后再 1.5s 渐入（配方 2，给策略自由起步窗口）；
  //  - fixed（MERO-9 手段3）：0.6s smoothstep 渐入到恒定 armHoldFixedAlpha（不 ramp 到 1，
  //    部分混合降低 OOD 风险；闭环自洽性与 softArmHold 相同）；
  //  - pitch（手段4）：warmup 同款渐入，但 per-dim α 再乘 holdMask（只锁肩俯仰 2 维）。
  // 返回的是标量调度值，有效 per-dim α = 标量 × holdMask[p]（见控制步混合循环）。
  function armHoldAlpha() {
    if (!holdPose) return 0;
    if (armHoldRamp === 'fixed') {
      return armHoldFixedAlpha > 0 ? armHoldFixedAlpha * smoothstep(Math.min(1, phaseT / 0.6)) : 0;
    }
    if (armHoldRamp === 'delay') {
      return phaseT <= 0.8 ? 0 : smoothstep(Math.min(1, (phaseT - 0.8) / 1.5));
    }
    return smoothstep(Math.min(1, phaseT / 0.6)); // warmup / pitch
  }
  function smoothstep(x) {
    const t = Math.max(0, Math.min(1, x));
    return t * t * (3 - 2 * t);
  }

  // 零指令预热窗口：期间指令恒 0（walkTo 的路点指令延迟注入），策略在闭环内站稳
  function warmUp(sec) { warmupT = Math.max(warmupT, sec); }

  // ---------------- 每子步推进 ----------------
  function step(dt) {
    if (fallen) return { done: doneFlag, fallen, dist, overtime, cmd: lastCmdView, turnYaw: turnReady, braking: turnYaw !== null };
    phaseT += dt;
    if (warmupT > 0) warmupT -= dt;
    readPosture();

    // 跌倒检测（仅 RL 段，双条件 + 去抖）
    const gravZ = assembleGravityZ();
    if (gravZ > FALL_GRAV_Z && data.qpos[2] < FALL_PELVIS_Z) {
      fallTimer += dt;
      if (fallTimer >= FALL_DEBOUNCE_S) {
        fallen = true;
        log(`[walker] 跌倒检测触发 gravZ=${gravZ.toFixed(2)} pelvisZ=${data.qpos[2].toFixed(2)} `
          + `xy=(${posture.x.toFixed(2)},${posture.y.toFixed(2)}) yaw=${posture.yaw.toFixed(2)} `
          + `cmd=(${lastCmdView[0].toFixed(2)},${lastCmdView[1].toFixed(2)},${lastCmdView[2].toFixed(2)}) `
          + `@t=${phaseT.toFixed(1)}s (warmup 剩 ${Math.max(0, warmupT).toFixed(1)}s)`);
      }
    } else {
      fallTimer = 0;
    }

    turnReady = null;
    // 已经过中间路点且横向误差很小：推进路点，避免追逐身后的目标。
    if (waypoints && warmupT <= 0 && wpIdx < waypoints.length - 1) {
      const wp = waypoints[wpIdx], a = legStart;
      const dx = wp.x-a.x, dy = wp.y-a.y, n = Math.hypot(dx,dy);
      const ex = posture.x-wp.x, ey = posture.y-wp.y;
      const passed = n > 1e-6 && (ex*dx+ey*dy) >= 0 && Math.abs(ex*dy-ey*dx)/n < 0.10;
      if (passed || Math.hypot(ex,ey) < 0.10) { legStart = wp; wpIdx++; }
    }
    // 指令生成（每子步更新目标，斜坡在控制步内做）
    let cmdWant = [0, 0, 0];
    if (waypoints && wpIdx < waypoints.length && warmupT <= 0) {
      const wp = waypoints[wpIdx];
      const ex = wp.x - posture.x, ey = wp.y - posture.y;
      const segDist = Math.hypot(ex, ey);
      const eYaw = wrapAngle(wp.yaw - posture.yaw);
      // 汇报到终点的剩余路径长（当前段 + 后续段）
      dist = 0;
      for (let i = wpIdx, px2 = posture.x, py2 = posture.y; i < waypoints.length; i++) {
        dist += Math.hypot(waypoints[i].x - px2, waypoints[i].y - py2);
        px2 = waypoints[i].x; py2 = waypoints[i].y;
      }
      // 到位判据（只对最后路点）：「擦过即达」——追踪 0.3m/s 下机器人会以 <0.08m 掠过
      // 目标（策略指令死区无法精确停靠），记录段内最近距离，一旦 <STOP_BAND 即判到位；
      // 到位后的最终位姿残差（~0.1-0.2m）由 PREPARE/ALIGN_PLACE/BLEND_STAND blend 吸收
      const isLast = wpIdx === waypoints.length - 1;
      if (isLast) segMinDist = Math.min(segMinDist, segDist);
      if (isLast && segMinDist < STOP_BAND) {
        if (!doneFlag) completedGoal = { dist: segMinDist, yaw: Math.abs(eYaw), overtime: false };
        doneFlag = true;
        if (phaseT > timeLimit) overtime = true;
      } else if (phaseT > timeLimit) {
        overtime = true;
        doneFlag = true; // 超时按段失败结束（调用方据 overtime 决定降级）
        log(`[walker] 行走段超时（limit=${timeLimit.toFixed(1)}s dist=${segDist.toFixed(2)}m eYaw=${eYaw.toFixed(2)}）`);
      }
      if (!doneFlag) {
        // 停止带：足够近时直接零指令，靠策略站立吸引子停稳（死区特性）
        if (isLast && segDist < STOP_BAND) {
          cmdWant = [0, 0, 0];
        } else {
          // 纯追踪（pure pursuit）指令（T0 能力边界）：策略只在「前进 + 弱转向」下稳定——
          // vx 恒满速（保持步态激发，KP_LIN 距离投影会在斜向/近距时跌入死区停摆）、
          // vy 恒 0（侧移通道死区且引入不稳定）、wz 追踪目标方位角（行进中转向）。
          // 指令形式的取舍依据见 docs/policy-probe-notes.md 实验记录。
          // MERO-8 配方 3（armHoldTight）：softArmHold 生效的 carry 段用收紧限幅
          // （vx≤0.15 / wz≤0.1；低于指令死区 ~0.25 有停摆风险，属门控实验配方）
          const softCarry = carrying && holdPose != null && armHoldTight;
          const vxMax = (carrying ? (softCarry ? CARRY_SOFT_VX_MAX : CARRY_VX_MAX) : VX_MAX) * vxScaleCur;
          const wzLim = softCarry ? CARRY_SOFT_WZ_MAX : WZ_MAX;
          const bearing = Math.atan2(wp.y - posture.y, wp.x - posture.x);
          const eBearing = wrapAngle(bearing - posture.yaw);
          // 超过策略可控航向范围时停止前进，停稳后交给上层踏步转身。
          if (turnYaw !== null || Math.abs(eBearing) > 0.32) {
            turnYaw = bearing;
            turnBrakeT += dt;
            if (turnBrakeT >= 0.5 && Math.hypot(data.qvel[0], data.qvel[1]) < 0.06) turnReady = bearing;
          } else {
            turnBrakeT = 0;
            cmdWant = [clamp(vxMax, -vxMax, vxMax), 0, clamp(KP_YAW * eBearing, -wzLim, wzLim)];
          }
        }
        // 到达中间路点即切换
        if (!isLast && segDist < Math.max(ARRIVE_DIST, 0.15)) {
          legStart = wp;
          wpIdx++;
          arrivedT = 0;
        }
      } else {
        cmdWant = [0, 0, 0];
      }
    } else if (doneFlag) {
      cmdWant = [0, 0, 0];
    }

    // 50Hz 控制步：斜坡 → 推理 → 解码 → ctrl
    if (sub % CTRL_DECIM === 0) {
      for (let i = 0; i < 3; i++) {
        targetCmd[i] = cmdWant[i];
        const d = targetCmd[i] - curCmd[i];
        curCmd[i] = Math.abs(d) <= CMD_RAMP[i] ? targetCmd[i] : curCmd[i] + Math.sign(d) * CMD_RAMP[i];
        lastCmdView[i] = curCmd[i];
      }
      pushHist(buildFrame());
      const act = policyForward(policy.weights, assembleObs());
      // MERO-8 softArmHold（渐进动作空间混合）：臂 14 维动作下发前混入持箱姿态定值
      // act_send = (1−α)·act + α·holdAct；obs last_action 喂 act_send（闭环自洽，
      // 与 D2 的 last_action=raw 区别）；臂部仍由 PD 真实执行，不写 qpos/qvel、
      // 不动增益（与 A1 kinematic 定格区别）。α=0（armhold=off 或渐入未开始）时
      // act_send=act，与基线逐位一致。
      armHoldAlphaCur = armHoldAlpha();
      if (armHoldAlphaCur > 0) {
        for (let p = 0; p < N; p++) {
          const a = armHoldAlphaCur * holdMask[p]; // per-dim 有效 α（'pitch' 只锁肩俯仰）
          if (a > 0) act[p] = (1 - a) * act[p] + a * holdAct[p];
        }
      }
      lastAction = act; // obs last_action 喂 act_send（α=0 即 raw，deploy 实机语义不变）
      lastActionOut.set(act);
      // 动作解码：全量 29 维计算（闭环完整性），腿+腰采纳策略；臂部 ctrl 目标 =
      // DEFAULT + 0.25·act_send（range 裁剪照旧）。armGoalSmooth 为 D2 遗留备查通道
      // （stackCore 现恒传 armGoal=null），softArmHold 生效期两者互斥、优先 softArmHold。
      for (let p = 0; p < N; p++) {
        let tgt = DEFAULT_POS[p] + ACT_SCALE * act[p];
        if (isArm[p] && armGoalSmooth && !holdPose) {
          tgt = armGoalSmooth[MAP[p]]; // armGoal 按 mjcf 序
        }
        data.ctrl[bind[p].c] = Math.max(ctrlLo[p], Math.min(ctrlHi[p], tgt));
      }
      if (armGoalSmooth && !holdPose) smoothInto(armGoalSmooth, armGoal, CTRL_DECIM * dt, 0.2); // 臂目标指数平滑
    }
    sub++;
    return { done: doneFlag, fallen, dist, overtime, cmd: lastCmdView, turnYaw: turnReady, braking: turnYaw !== null };
  }

  function assembleGravityZ() {
    const q = [data.xquat[4 * pelvisBid], data.xquat[4 * pelvisBid + 1], data.xquat[4 * pelvisBid + 2], data.xquat[4 * pelvisBid + 3]];
    return quatRotVec(quatConj(q), [0, 0, -1])[2];
  }

  // 首次使用前预热历史（复位/重试后调用）：以当前状态填充 5 帧，避免冷启动瞬态。
  // 必须先清零 curCmd/lastAction 再填历史——buildFrame 会把两者烤进 obs 帧
  // （cmd 项 + last_action 项），若带着上一段（尤其是摔倒段）的残值填充，
  // 策略会收到「站立位姿 × 疯狂上段动作」的自相矛盾 obs，经 last_action
  // 闭环正反馈放大直至失控（MERO-5：槽 4 连摔两轮的根因）。
  function resetObsHistory() {
    curCmd.fill(0); targetCmd.fill(0); lastCmdView.fill(0);
    lastAction.fill(0);
    for (let h = 0; h < HIST; h++) pushHist(buildFrame());
    sub = 0; fallen = false; fallTimer = 0;
    doneFlag = false; overtime = false; arrivedT = 0; phaseT = 0;
    turnBrakeT = 0; turnYaw = null; turnReady = null;
  }

  function resumeAfterTurn(seconds) {
    const elapsed = phaseT + seconds;
    resetObsHistory();
    phaseT = elapsed; // 转向不重置本段超时预算，防止无限停转循环。
    warmUp(0.3);
  }

  function wrapAngle(a) {
    return Math.atan2(Math.sin(a), Math.cos(a));
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  resetObsHistory();

  return {
    enter, exit, walkTo, step, warmUp, resetObsHistory, resumeAfterTurn,
    get entered() { return entered; },
    get phase() { return waypoints ? (doneFlag ? 'done' : 'walking') : 'idle'; },
    get lastCmd() { return lastCmdView; },
    get lastAction() { return lastActionOut; },
    get dist() { return dist; },
    get fallen() { return fallen; },
    get completedGoal() { return completedGoal; }, // 最近完成段到位误差 {dist, yaw, overtime}
    get walkSeq() { return walkSeq; },
    get armHoldState() { return { mode: armHold, ramp: armHoldRamp, tight: armHoldTight, alpha: armHoldAlphaCur, active: holdPose != null, carryKd: carryKdActive ? CARRY_KD : 0 }; },
  };
}
