import { buildPivotPlan, samplePivotPlan } from './pivotPlan.js';
// Warehouse task coordinator: approach → grasp → carry → insert → release → return.
// DOM-free and shared by the browser and the MuJoCo WASM integration checks.
// Empty-hand locomotion uses the RL controller; manipulation replays an offline
// contact retargeting solve from manipulationPlan.js. The planner includes bin lips,
// shelf clearance and fixed stance anchors. Released bins use MuJoCo dynamics.
// Manipulation is kinematic, not a deployable whole-body force controller.
// The paired ghost/physical bodies avoid changing compiled collision masks at runtime.
import {
  buildJointMap, keyId, quatMul, quatConj, quatSlerp, quatRotVec, yawQuat,
} from './util.js';
import { applyContactPlan, manipulationRoute } from './manipulationPlan.js';
import { planPathAvoid, pointRectDist } from './pathPlanner.js';
// 工厂布局（方案 docs/factory-scene-minimap-plan.md D2/D3/D4）：mocap 走廊显式摆位到
// 货架格网（derivePickPos/rackPsi）、槽位层高（shelfBoardTops）、静态避障矩形
// （factoryObstacleRects）。另取 binShellGeomXml（2026-10 料箱工厂改造切片 A：开口
// 料箱视觉壳，与 factory.js 装饰料箱共用同一生成器）。factory.js 不反向依赖本模块。
import { FACTORY, derivePickPos, shelfBoardTops, factoryObstacleRects, binShellGeomXml } from './factory.js';

// P1 裁定（两层可用格位）：MAX_LAYERS 3→2——动捕固有释放高度间隔 0.22m < 箱高
// 0.32-0.34m，三层干净格位不可行（动捕素材约束）；层 1→low、层 2→mid，high clip
// 在货架模式下不再被引用（动作文件保留）。货架保留多层外观（装饰顶板，factory.js）。
export const MAX_LAYERS = 2;
export const MAX_COLS = 2;
export const MAX_BOXES = MAX_LAYERS * MAX_COLS;

// 堆垛世界位姿（方案 11：取箱位前方偏 45° / 1.2m，编码期定值）
const STACK_DIST = 1.2;
const STACK_YAW_OFF = Math.PI / 4;
// 列缝 8mm（方案 4.3）
export const COL_GAP = FACTORY.rack.slotGap; // Includes the 18mm bin lips, with visible clearance.

// 各阶段时长（秒）。
// v2（方案 4.7）：PREPARE 0.8→0.4（RL 已走到位，blend 只吸收残差）、BLEND_STAND 0.8→0.5；
// 新增 ALIGN_PLACE 0.4s 与箱随动 blend 0.25s（4.2 序 5）。
// 瞬移消除（本次改造）：BLEND_HOME 贝塞尔滑移随 legacy 模式一并移除；新增
// STAND_UP（原地起身）与 RECOVER_STAND（失败原地恢复）。
const DUR = { PREPARE: 0.4, SETTLE: 1.0, BLEND_STAND: 0.5, ALIGN_PLACE: 0.4, BOX_BLEND: 0.25, STAND_UP: 0.6, RECOVER: 1.2 };
// TURN_APPROACH 触发阈：朝向偏差 >0.3rad 先踏步转身再交给 RL；0.3 以下直接开走
//（更小偏差由 makeWaypoints 密集校准路点在行进中吸收，到达朝向残差由 blend 吸收）
const TURN_APPROACH_MIN = 0.3;
// 小步转向由 pivotPlan 求解：支撑足固定在世界坐标，摆动足抬起后改变落脚朝向。
const TURN_RATE = 0.5;
const TURN_RATE_CARRY = 0.35;
// blend 段对 yaw 残差的自适应加时（秒/rad）：RL 段到位的朝向残差由 kinematic blend
// 吸收（4.2 序 2/5）。瞬移消除改造起取 2.0（旋转速率 0.5rad/s，smoothstep 峰值 0.75）：
// 支撑足滑移 ≈ 足距心 0.1m × 0.75 ≈ 75mm/s，满足 N15 blend 行滑移 p95 ≤400mm/s 且
// 低于 N16 转身阈——大角度旋转一律交给踏步转身（TURN_*），blend 只吸收小残差
//（旧值 1.2/(2π)≈0.19 → 5.2rad/s，小残差下尚可，大残差时脚底拖扫超 N15 阈）
const BLEND_YAW_RATE = 2.0;
// blend 段位置残差吸收速率（m/s）：blend 时长按实测停点残差自适应，峰值速度
//（smoothstep 1.5×均值）≤0.23m/s，远低于 N15 的 400mm/s 滑移阈——用户不可辨滑移
const BLEND_POS_RATE = 0.15;
// 行走目标前移量（m）＝惯性停站的一个停止带：RL 行走"擦过即达"（距路点 <STOP_BAND
// 零指令）后惯性滑行 ~0.02-0.04m（N6 实测停点距目标 0.10-0.12m）。目标前移一个
// 停止带后，触发点恰落在真实目标上，停点残差压到 ~0.03m，blend 段位移随之为
// ≤~0.06m（N15 blend 行单列口径，≤0.15m 上限）。N6 量测对路点距离，口径不变。
const WALK_STOP_LEAD = 0.08;
// WALK 段进入前的零指令预热（4.2：改写增益后 0.3s 站稳再出发）
const WALK_WARMUP = 0.6; // 转身后延长零指令预热，让接触态/姿态在开走前稳定（0.3s 实测不足）
// MF-2（审查实测，MERO-5 修订为全程限速）：TURN_CARRY→WALK_CARRY 动力学接管瞬间
// 接触冲量使基座 ~20ms 内突进 ~3m/s，箱锚刚体随动在单采样窗越 N10 25mm 阈——对箱
// 位姿输出限速 0.5m/s（> 行走 0.3m/s 不伤正常跟随，只削冲量尖峰）。限速必须持续
// 整段而非起步窗：窗口边界一刀切回直写会把窗内未追平的滞后在单个采样窗内整段释放
// （plasticbox 2x2 实测 29.8mm 超 N10 阈）；全程限速下稳态锚速 < 限速，尖峰滞后
// 随即追平，段末无残余滞后。
const CARRY_BOX_VCAP = 0.5; // 箱位姿输出限速（m/s）
// 双手锚模式的箱位姿输出限速（MERO-8，实测联定）：双手中点在 WALK_CARRY 起步 ~0.5s 内
// 会被策略自然下垂 ~0.3m（z 0.97→0.68，峰值 ~1m/s，off 模式无臂部保持时必有该瞬态），
// 0.5m/s 帽追不上会使箱-手误差累积至 ~0.3m（N12 实测 296-303mm）；放宽到 1.2m/s 后
// 瞬态被平滑滞后吸收（≈v·tau），而对 TURN_CARRY→WALK_CARRY 接管冲量（基座 ~3m/s、
// ~20ms）仍削峰至 ≤12mm/10ms 采样窗（< N10 25mm 阈）。pelvis 逃生门维持 0.5（MF-2 原值）。
const CARRY_BOX_VCAP_HANDS = 1.2;
// MERO-8 持箱锚模式：默认 hands=双手中点系锚（箱跟手，消除「箱悬空」缺陷）；
// G1BOX_CARRY_ANCHOR=pelvis 强制回旧 pelvis 系锚（对照实验/防假绿验证逃生门）。
// 浏览器端无 process 对象，读环境变量前须 typeof 守卫。
const CARRY_ANCHOR_HANDS = !(typeof process !== 'undefined' && process?.env?.G1BOX_CARRY_ANCHOR === 'pelvis');
// 双手锚随动目标位姿的指数平滑时间常数（秒）：抑制摆臂微抖经锚直传箱体；代价是
// 稳态跟踪滞后 ≈ v·tau。取下限 0.1s：WALK_CARRY 起步臂部下垂瞬态（~1m/s，off 模式
// 实测）下 tau=0.2 的滞后使 N12 峰值 303mm，tau=0.1 降为 150mm（证据 dev_tau010_vcap12
// / v1a_dev_1x1 日志），稳态 d≈捕获值+30mm；MERO8_CARRY_TAU 环境变量仅供联定期实验。
const CARRY_SMOOTH_TAU = (typeof process !== 'undefined' && process?.env?.MERO8_CARRY_TAU)
  ? Math.max(0.01, parseFloat(process.env.MERO8_CARRY_TAU)) : 0.1;
// WALK 段容差超时：起步阶段（站立→行走加速）占时显著（T0 实测平均 ~0.1-0.25m/s），
// 预算 = 3×理论时长 + 4s 裕量
const WALK_TIME_SLACK = 4.0;
const WALK_TIME_MULT = 3.0;
// RL 行走降级前每槽重试次数（4.6）：工厂方案起 1→2——绕行走廊（pillar_1 卡线强制绕行）
// 相比基线直线行走 RL 方差显著增大（同一规划路径、起点微差经策略混沌放大，实测 2×2
// 同构槽位一成一败），单次差异化预热重试不足；2 次重试提高演示稳健性（重试起点取
// 失败后当前位置 + 预热 1.0s 差异化，输入必不同，方案 4.6 重试语义）。
const WALK_RETRIES = 2;
// ---------------- 行走段 2D 避障（walk-avoid-plan，方案 docs/walk-avoid-plan.md） ----------------
// 持箱行走（WALK_CARRY，?carry=rl 链）的机器人等效半径（方案 4.2/4.3.2，队长裁定 Q2）：
// 持箱跟手机体、本身不是障碍，双手持箱前伸突出量级并入机器人膨胀（0.30 + 前伸 ~0.15）。
const AVOID_ROBOT_R_CARRY = 0.45;

// ---------------- MERO-10：持箱搬移段改用真人"抱箱行走"动捕回放（状态机 v3） ----------------
// 取箱位与走廊进抵方向（方案 4.4-1 编码期常量，与现行场景取箱位/相机取景习惯一致）：
// G0 全局相似把 carry clip 的 grasp 锚 A→P0、携带行进方向 û→dir0，堆垛中心
// S = P0 + carryDisp·dir0（堆垛中心随片段位移落在 1.0-1.7m 半径，处于相机视场内）。
export const CARRY_PICK_POS = [-0.8, -0.3];
export const CARRY_PICK_DIR = Math.PI / 4;
// MERO-10 negcheck 第三组注入（G1BOX_NEGCHECK_CARRY_DRIFT，仅无头防假绿消费；
// >0 时 carryFull 携带段幽灵箱位置按斜坡漂移至该距离 → N13-mocap 必 FAIL）。
// 惰性读取（非 import 期常量）：headless 侧在运行时注入 env 也要能生效；
// 浏览器端无 process 对象，读环境变量前须 typeof 守卫（惯例同 G1BOX_CARRY_ANCHOR）。
function negCarryDrift() {
  const v = typeof process !== 'undefined' ? parseFloat(process?.env?.G1BOX_NEGCHECK_CARRY_DRIFT) : 0;
  return Number.isFinite(v) && v > 0 ? v : 0;
}

// 对照开关（方案 4.3.3）：G1BOX_AVOID_OFF=1 全局短路行走避障为直线（对照实验/紧急逃生门，
// negcheck avoid 注入 2 基于它——复刻用户报告的"直线穿箱"缺陷形态）。启动期明示日志见
// createBoxStacking；惰性读取（非 import 期常量）：headless 侧运行时注入 env 也要能生效；
// 浏览器端无 process 对象，读环境变量前须 typeof 守卫（惯例同 G1BOX_CARRY_ANCHOR）。
function avoidOff() {
  return typeof process !== 'undefined' && process?.env?.G1BOX_AVOID_OFF === '1';
}

// negcheck avoid 注入 1（G1BOX_NEGCHECK_AVOID_BREAK，仅无头防假绿消费；方案 4.5.2）：
// >0 时 makeWaypoints 在规划/密化后把中部路点向最近障碍矩形中心平移该量 → 行走路径被拉进
// 障碍 → N19 必 FAIL。注入物理上就是"坏实现"，不要求路径自洽（语义同 negCarryDrift）。
// 惰性 env 读取 + typeof 守卫（惯例同 negCarryDrift）。
function negAvoidBreak() {
  const v = typeof process !== 'undefined' ? parseFloat(process?.env?.G1BOX_NEGCHECK_AVOID_BREAK) : 0;
  return Number.isFinite(v) && v > 0 ? v : 0;
}

function wrapPi(a) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

// 四元数（wxyz）偏航角
function quatYaw(q) {
  return Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
}

// 携带走廊的堆垛框架（方案 4.4-1，MERO-10 新增）：取 carry clip 固有量
// A=grasp 锚（obj 水平位置）、d=grasp→release 的 obj 水平位移、û=行进方向，
// 按全局相似 G0: A→P0、û→dir0 摆位后 S = P0 + d·dir0；psi=dir0（栅格朝向）。
// 与 computeStackFrame（v2：grasp 锚 + 45°/1.2m）的区别：堆垛中心由片段真实
// 携带位移决定（走廊零压缩零拉伸，N14 足底滑移由构造消除）。
export function computeStackFrameCarry(clip, P0 = CARRY_PICK_POS, dir0 = CARRY_PICK_DIR) {
  const g = clip.grasp_frame, rel = clip.release_frame;
  const A = [clip.obj_pos[g][0], clip.obj_pos[g][1]];
  const dxy = [clip.obj_pos[rel][0] - A[0], clip.obj_pos[rel][1] - A[1]];
  const carryDisp = Math.hypot(dxy[0], dxy[1]);
  const S = [P0[0] + carryDisp * Math.cos(dir0), P0[1] + carryDisp * Math.sin(dir0)];
  return { A, yaw: quatYaw(clip.obj_quat[g]), S, psi: dir0, carryDisp };
}

// 逐槽刚性规范化（方案 4.4-2，MERO-10 核心裁定）：对全部帧（含 t<grasp）先施加
// 刚性预变换 p' = R_θ(p−anchor)+target（仅 xy；root/obj 四元数前乘 q(θ)；z/dof 不动），
// 再走既有 λ 走廊逻辑。carryFull 调用方（remapClipCarryFull）以 ψ=0、λ 平移目标=
// release 锚自身调用，λ 旋转/平移严格为 0 → 重映射退化为恒等 + zShift，
// 步态零压缩零附加平移（N14 由构造满足），release 锚精确落槽位（N7 <1mm）。
//
// 【θ 取法的已记录偏差】方案 4.4-2 原式 θ_k = yaw(u_slot)−yaw(d)+δ_yaw，其 δ_yaw
// 闭式代入后 θ_total ≡ stackPsi − raw_release_yaw（箱偏航=栅格）。实测现有/可复用
// 片段的行进方向 û 与释放帧箱偏航相对角达 65-148°（carry_low/mid/high 实测
// 135.3°/147.6°/-65.2°），该 θ 会把搬运走廊甩到横穿堆垛的方向（违反 R3/N14）。
// 故本实现取 G0 语义：θ = stackPsi − yaw(û)（行进方向对齐走廊进抵方向），站位
// 相对行进的方向角为片段固有量、由选片门限 angle(stance,−û)≤45° 保证落在堆垛
// 外侧半球（方案 4.4-3 的 û≈−d̂ 性质由此成立）；"箱偏航=栅格"交由既有
// handoverToPhysics 换位对齐承担（方案 4.3 表内"换位合法瞬移"，v2 已接受口径）。
// 逐槽 θ 相同、逐槽仅平移目标不同（C_k），取箱点随槽位横移 ~2·colHalf（A4 已接受）。
export function remapClipForSlot(clip, slotCenter, psi, boxHalfZ, boxOffZ, preRigid = null) {
  const T = clip.T, g = clip.grasp_frame, rel = clip.release_frame;
  const K = Math.max(1, rel - g);
  const Br = [clip.obj_pos[rel][0], clip.obj_pos[rel][1]]; // 释放点水平锚

  // 贴地补偿：grasp 帧箱子几何中心沉入地面的深度
  const rawCenterG = clip.obj_pos[g][2] + boxOffZ;
  const sink = Math.max(0, boxHalfZ - rawCenterG);
  // 释放高度 = 槽位中心上方 RELEASE_DROP
  const zShift = (slotCenter[2] + RELEASE_DROP) - (clip.obj_pos[rel][2] + boxOffZ + sink);

  // 逐槽刚性规范化预变换系数（preRigid 缺省 = 现行为逐位不变）。
  // 变换式 p' = R_θ(p−anchor) + target：绕 anchor 转 θ 后平移到 target，
  // prTx/prTy 即 target 本身（不能再预减 R·anchor，否则 anchor 被减两次）。
  let prC = 1, prS = 0, prAx = 0, prAy = 0, prTx = 0, prTy = 0;
  let qTheta = null;
  if (preRigid) {
    prC = Math.cos(preRigid.theta);
    prS = Math.sin(preRigid.theta);
    prAx = preRigid.anchor[0];
    prAy = preRigid.anchor[1];
    prTx = preRigid.target[0];
    prTy = preRigid.target[1];
    qTheta = [Math.cos(preRigid.theta / 2), 0, 0, Math.sin(preRigid.theta / 2)];
  }

  const rootPos = new Float64Array(T * 3);
  const rootQuat = new Float64Array(T * 4);
  const dof = new Float64Array(T * clip.joint_names.length);
  const objPos = new Float64Array(T * 3);
  const objQuat = new Float64Array(T * 4);

  for (let t = 0; t < T; t++) {
    let rp = clip.root_pos[t], op = clip.obj_pos[t];
    let rq = clip.root_quat[t], oq = clip.obj_quat[t];
    if (preRigid) {
      // 刚性预变换：位置绕 anchor 转 θ 再平移（全部帧，含 t<grasp）；四元数前乘
      const rxy = [prC * (rp[0] - prAx) - prS * (rp[1] - prAy) + prTx,
        prS * (rp[0] - prAx) + prC * (rp[1] - prAy) + prTy];
      const oxy = [prC * (op[0] - prAx) - prS * (op[1] - prAy) + prTx,
        prS * (op[0] - prAx) + prC * (op[1] - prAy) + prTy];
      rp = [rxy[0], rxy[1], rp[2]];
      op = [oxy[0], oxy[1], op[2]];
      rq = quatMul(qTheta, rq);
      oq = quatMul(qTheta, oq);
    }
    const remap = t >= g;
    const lam = remap ? smoothstep(Math.min(1, (t - g) / K)) : 0;
    const c = Math.cos(lam * psi), s = Math.sin(lam * psi);
    const bx = Br[0] + lam * (slotCenter[0] - Br[0]);
    const by = Br[1] + lam * (slotCenter[1] - Br[1]);
    const lift = remap ? sink * Math.min(1, (t - g) / SINK_RAMP_FRAMES) : 0;

    if (remap) {
      rootPos[t * 3] = c * (rp[0] - Br[0]) - s * (rp[1] - Br[1]) + bx;
      rootPos[t * 3 + 1] = s * (rp[0] - Br[0]) + c * (rp[1] - Br[1]) + by;
      objPos[t * 3] = c * (op[0] - Br[0]) - s * (op[1] - Br[1]) + bx;
      objPos[t * 3 + 1] = s * (op[0] - Br[0]) + c * (op[1] - Br[1]) + by;
      objPos[t * 3 + 2] = op[2] + lift + lam * zShift;
    } else {
      rootPos[t * 3] = rp[0]; rootPos[t * 3 + 1] = rp[1];
      objPos[t * 3] = op[0]; objPos[t * 3 + 1] = op[1];
      objPos[t * 3 + 2] = op[2];
    }
    rootPos[t * 3 + 2] = rp[2];                      // root z 不动（真人步行骨盆起伏属动捕真值）

    const qpsi = [Math.cos(lam * psi / 2), 0, 0, Math.sin(lam * psi / 2)];
    const rq2 = remap ? quatMul(qpsi, rq) : rq;
    const oq2 = remap ? quatMul(qpsi, oq) : oq;
    for (let i = 0; i < 4; i++) {
      rootQuat[t * 4 + i] = rq2[i];
      objQuat[t * 4 + i] = oq2[i];
    }
    for (let i = 0; i < clip.joint_names.length; i++) dof[t * clip.joint_names.length + i] = clip.dof_pos[t][i];
  }
  return { T, fps: clip.fps, grasp: g, release: rel, rootPos, rootQuat, dof, objPos, objQuat, nDof: clip.joint_names.length };
}

// carryFull 专用重映射（MERO-10）：逐槽刚性规范化（preRigid）已把 release 锚放到
// 槽位，λ 走廊逻辑必须严格恒等——slotCenter.xy 取 clip 释放锚自身（旋转 0/平移 0），
// 仅 z 参与 zShift（释放高度差由既有渐入机制吸收，与 v2 数值同源）。
export function remapClipCarryFull(clip, slotCenter, boxHalfZ, boxOffZ, preRigid) {
  const rel = clip.release_frame;
  const xyNeutral = [clip.obj_pos[rel][0], clip.obj_pos[rel][1], slotCenter[2]];
  return remapClipForSlot(clip, xyNeutral, 0, boxHalfZ, boxOffZ, preRigid);
}

// 层 → tier 名解析（纯函数，MERO-10 扩展后供工厂 clipForLayer 与无头断言共用）：
// 层 1→low，层 2→mid(缺则 low)，层 3→high(缺则 mid/low)。
// 【MERO-10】carry 节可用性同样参与 tier 名解析（plasticbox 的 mid/high 只以借用件
// 存在于 carry 节，方案 5.1）；clip 可为 null（rl/legacy 路径回落 low，mocap 路径用
// set.carry[tier]），回落逻辑见 buildConfig。
export function resolveTierForLayer(set, layer) {
  const hasV2 = (t) => !!set[t];
  const hasCarry = (t) => isCarryClipUsable(set.carry?.[t]);
  if (layer === 0) return { clip: set.low, tier: 'low' };
  if (layer === 1) {
    if (hasV2('mid') || hasCarry('mid')) return { clip: set.mid ?? null, tier: 'mid' };
    return { clip: set.low, tier: 'low' };
  }
  if (hasV2('high') || hasCarry('high')) return { clip: set.high ?? null, tier: 'high' };
  if (hasV2('mid') || hasCarry('mid')) return { clip: set.mid ?? null, tier: 'mid' };
  return { clip: set.low, tier: 'low' };
}

// carry clip JSON 可用性守卫（逐槽降级链的判定输入，方案 4.6）。
// 缺文件由加载层兜底为 undefined；这里拦"解析成功但契约字段残缺"的损坏件。
function isCarryClipUsable(c) {
  return !!(c && Number.isFinite(c.grasp_frame) && Number.isFinite(c.release_frame)
    && c.T > 0 && Array.isArray(c.obj_pos) && c.obj_pos.length === c.T
    && Array.isArray(c.root_pos) && Array.isArray(c.dof_pos) && Array.isArray(c.joint_names));
}

// 逐 tier 携带策略解析（方案 4.6 降级链，buildConfig 与无头断言共用同一实现）：
//   carryMode='mocap' → walk_meta.carry 声明 mocap 且 carry clip 可用 → 'mocap'；
//   否则 'rl'（需 v2 帧域条目，仍缺则该槽 invalid 执行时明确跳过——legacy 降级已移除）。
// carryMode='rl' → 全部 'rl'（v2 对照）。
export function resolveCarryTiers(clips, meta, carryMode, boxType) {
  const tiers = { low: carryMode, mid: carryMode, high: carryMode };
  if (carryMode !== 'mocap') return tiers;
  const carry = meta?.carry?.[boxType];
  for (const t of ['low', 'mid', 'high']) {
    const entry = carry?.[t];
    tiers[t] = (entry?.strategy === 'mocap' && isCarryClipUsable(clips[boxType]?.carry?.[t]))
      ? 'mocap' : 'rl';
  }
  return tiers;
}

const PHASE_LABEL = {
  idle: '空闲', PREPARE: '取箱', CARRY_REPLAY: '搬运', RELEASE: '放置',
  SETTLE: '稳定', BLEND_STAND: '完成',
  WALK_APPROACH: '走向取箱位', WALK_CARRY: '搬运行走', WALK_HOME: '回位行走',
  ALIGN_PLACE: '对位', CARRY_REPLAY2: '放置', TURN_HOME: '转身', TURN_CARRY: '转身',
  STOP_PATH: '停稳等待转向', TURN_PATH: '调整行走方向', TURN_PICK: '对准料箱', TURN_APPROACH: '转身', TURN_STAND: '转身', STAND_UP: '起身', RECOVER_STAND: '恢复站立',
};

// 各箱型注入体号基址：largebox carton0..5，plasticbox carton100..105（同一场景预注入两套）
export function boxTypeBase(boxType) {
  return boxType === 'plasticbox' ? 100 : boxType === 'smallbox' ? 200 : 0;
}

// 潜伏行 y（按箱型分行，原 createBoxStacking 内常量上移模块级以便导出）：
// 与 factory.js FACTORY.staging.y 互为同源（headless F2c 校验一致），不得移动。
export const STAGING_Y = { largebox: 3.0, plasticbox: 4.2, smallbox: 5.4 };

// 料箱配色（2026-10 料箱工厂改造，切片 A）：三箱型统一深蓝硬质塑料（参考用户照片
// 的料箱色，#1D4E9E 邻域）。smallbox 无 motions 注入（boxTypeBase 200 预留），跟随
// 统一深蓝（交接说明已记录：可保留绿系或一并改蓝，取一并改蓝与照片一致）。
// 该色作为开口料箱视觉壳的主色传入 binShellGeomXml（壁/翻边/内底明暗由生成器派生）。
const BOX_RGBA = {
  largebox: '0.11 0.30 0.62 1',
  plasticbox: '0.11 0.30 0.62 1',
  smallbox: '0.11 0.30 0.62 1',
};

// ---------------- XML 片段（单一事实源：fetchAssets 与无头脚本共用） ----------------
// 把箱子 XML 拼进 merged XML 的第一个 </worldbody> 前（g1 内层 worldbody 末尾 →
// qpos 布局保持"机器人优先"，keyframe 36 维正好覆盖机器人段，方案 3.4-3 契约）。
// 同时按注入的 body 名 × merged XML 里的全部机器人 body 生成 <contact><exclude>：
// 实测 g1 内层 merge 后机器人碰撞网格 geom contype=1（覆盖 scene 顶层默认 0），
// kinematic 回放的机器人会与箱子生成接触并把已放箱犁飞；且本 binding 的 exclude
// 只按 body 对过滤（最小化实测不覆盖子树，官方文档语义为子树——已实测修正），
// 所以必须逐个机器人 body 排除。exclude 是 XML 期静态排除，彻底切断机器人-箱子
// 接触对，箱-地、箱-箱接触保持完整。
//
// extraExcludeBodies（工厂方案，可选第三参）：工厂承重板等静态 body 名单（factory.js
// FACTORY_SOLID_BODIES），同机制生成 板×机器人 body exclude——机器人放置站位与层板
// 立面必然重叠（实测 root y≈1.76-1.89 vs 板前缘 1.85），不排除会引入 D4 明示要避免的
// 机器人-静态物新摔倒形态；箱-板接触不经此表（板不在 robotBodies、carton 不在名单），
// 完整保留。
export function injectBoxes(mergedXml, snippetsXml, extraExcludeBodies = []) {
  const cartonNames = [...snippetsXml.matchAll(/<body name="(cartonG?\d+)"/g)].map((m) => m[1]);
  const excludeNames = [...cartonNames, ...extraExcludeBodies];
  const robotBodies = [...mergedXml.matchAll(/<body name="([^"]+)"/g)]
    .map((m) => m[1])
    .filter((n) => !n.startsWith('carton'));
  const excludes = [];
  for (const c of excludeNames) for (const b of robotBodies) excludes.push(`    <exclude body1="${c}" body2="${b}"/>`);
  const contactSection = excludes.length ? `<contact>\n${excludes.join('\n')}\n  </contact>\n` : '';
  return mergedXml
    .replace('</worldbody>', `${snippetsXml}\n  </worldbody>`)
    .replace(/<\/mujoco>\s*$/, `  ${contactSection}</mujoco>`);
}

// 幽灵/物理双体的用意见文件头 R7 注释。
// boxXmlSnippet 一次输出成对的两 bodies（幽灵 + 物理），保持方案 5.2 的单一事实源签名。
// 【2026-10 料箱工厂改造，切片 A：开口料箱造型】
//   - 物理外接盒 geom（carton{G}{k}_geom）名字/尺寸/质量/摩擦/contype 全部不变，只加
//     group="3"：本工程渲染器（visualizer.js）按既有惯例跳过 group 3/4/5（g1 碰撞
//     凸包同款），geom_group 是纯渲染过滤字段、不参与物理。开口料箱的顶面开口无法由
//     实体 box geom 表达（顶面恒实心），故把外接盒从渲染中隐藏、由视觉壳承担外观，
//     物理行为零变化（rgba alpha 隐藏不可用——visualizer 忽略 alpha 会渲染成黑色）。
//   - 视觉壳 = factory.js binShellGeomXml（底板+4 面薄壁+箱口翻边+下腹横向肋条一圈，
//     壳外廓与外接盒重合），ghost/物理两体同款（持握时观众看到的就是幽灵体）；壳件
//     位置加 bbox_center_offset 与主 geom 同心，全部 XML 期声明 contype=0
//     conaffinity=0 mass=0——碰撞与 free 体总质量/惯量严格不变（仍全部来自主 geom）。
//   - 潜伏位幽灵/物理双体共点、壳件逐面共面：与既有同色双箱共点的 z-fight 状况相同
//    （同色不可辨），非本次引入。
export function boxXmlSnippet(k, box, pos) {
  const [hx, hy, hz] = box.half_size;
  const [ox, oy, oz] = box.bbox_center_offset;
  const off = (ox || oy || oz) ? ` pos="${ox} ${oy} ${oz}"` : '';
  const rgba = box.rgba || BOX_RGBA[box.type] || BOX_RGBA.largebox;
  const size = `type="box" size="${hx} ${hy} ${hz}"${off}`;
  const fric = `friction="1.2 0.005 0.0001" mass="${box.mass ?? 3.0}"`;
  const shellXml = (prefix) => binShellGeomXml(prefix, hx, hy, hz, { rgba, rib: true }, [ox, oy, oz]);
  return [
    // 幽灵体：渲染跟随，永不碰撞
    `  <body name="cartonG${k}" pos="${pos[0]} ${pos[1]} ${pos[2]}">`,
    `    <freejoint name="cartonG${k}_joint"/>`,
    `    <geom name="cartonG${k}_geom" ${size} rgba="${rgba}" contype="0" conaffinity="0" mass="0.001" group="3"/>`,
    shellXml(`cartonG${k}`),
    `  </body>`,
    // 物理体：释放后接管
    `  <body name="carton${k}" pos="${pos[0]} ${pos[1]} ${pos[2]}">`,
    `    <freejoint name="carton${k}_joint"/>`,
    `    <geom name="carton${k}_geom" ${size} rgba="${rgba}" contype="1" conaffinity="1" ${fric} group="3"/>`,
    shellXml(`carton${k}`),
    `  </body>`,
  ].join('\n');
}

// ---------------- 纯函数：几何 / 重映射 / 采样 ----------------
function lerp3(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function smoothstep(x) {
  const t = Math.max(0, Math.min(1, x));
  return t * t * (3 - 2 * t);
}

// 堆垛槽位表：层优先（第 1 层左→右，再上一层），槽位中心指 geom 中心。
// stackCenter 为堆垛地面投影中心 [x,y]，yaw 为朝向（局部 +x 为纵深、+y 为左）。
// boardTops（工厂方案 D2，可选第 4 参，缺省 null 行为逐位不变）：传入时层 z =
// boardTops[layer] + hz（货架层板格位，boardTops[i]===null 表示该层落地即 0+hz，
// 与地面堆垛层 1 一致）；不传保持纯堆叠 (2·layer+1)·hz。无头纯函数自测的 2 参
// 调用原样绿（?carry=rl 对照链的地面堆垛语义零改动依赖此缺省）。
export function stackLayout(box, stackCenter, yaw, boardTops = null) {
  const [hx, hy, hz] = box.half_size;
  const colHalf = Math.max(hx, hy) + COL_GAP; // 半列距
  const slots = [];
  for (let layer = 0; layer < MAX_LAYERS; layer++) {
    for (let col = 0; col < MAX_COLS; col++) {
      const ly = col === 0 ? colHalf : -colHalf; // col0 = 左
      const lift = boardTops ? (boardTops[layer] ?? 0) : 2 * layer * hz;
      const lz = lift + hz;                      // 层 1 中心 hz，层 2 中心 3hz（缺省）
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      slots.push({
        layer, col,
        center: [
          stackCenter[0] + cy * 0 - sy * ly,
          stackCenter[1] + sy * 0 + cy * ly,
          lz,
        ],
      });
    }
  }
  return slots;
}

// 从 clip 求堆垛世界位姿：A=grasp 帧物体水平锚点，yaw=grasp 帧 root 偏航，
// S=A 前方偏 45° 1.2m，psi=atan2(S−A)（重映射朝向角）。
export function computeStackFrame(clip) {
  const g = clip.grasp_frame;
  const A = [clip.obj_pos[g][0], clip.obj_pos[g][1]];
  const q = clip.root_quat[g];
  const yaw = Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
  const dir = yaw + STACK_YAW_OFF;
  const S = [A[0] + STACK_DIST * Math.cos(dir), A[1] + STACK_DIST * Math.sin(dir)];
  const psi = Math.atan2(S[1] - A[1], S[0] - A[0]);
  return { A, yaw, S, psi };
}

// MimicGen 式开环重映射（方案 4.3，实现取"渐入走廊"变体，见下）。
// 锚点取 clip 释放点 Br = obj_pos[release].xy（以放置位姿为锚），目标 C = 槽位中心：
// 方案原式以 grasp 锚 A 做刚性变换，存在两处硬伤（实测 051）：
//   1) grasp 帧机器人从取箱位瞬移到堆垛位（|C−A|≈1.2m 视觉跳变）；
//   2) 释放点 = C + Rψ·D（D 为 clip 自身携带水平位移，实测 0.23m），
//      违背方案 4.6 的 5mm 放置精度判据。
// 因此这里把"恒等 → T(p)=Rψ(p−Br)+C_xy"的变换按携带段进度 λ(t)=smoothstep((t−grasp)/(release−grasp))
// 渐入：
//   p' = R(λψ)(p−Br) + Br + λ(C_xy−Br)（xy）；root z 不动（保脚-地）
//   quat' = q(λψ) ⊗ quat；dof 不变；t<grasp 恒等。
// λ(grasp)=0 → grasp 帧连续；λ(release)=1 → 释放点精确落在槽位中心；
// t>release 段 λ 恒 1（纯刚性走廊，机器人自然走开）。
//
// 【R7 落地】垂直方向带"贴地补偿 + 释放抬升"：
//   OMOMO 物体原点比包围盒几何中心低 ~12mm（箱子在原始轨迹里略微沉入地面），
//   而本场景箱子 contype=1 常开（binding 的 geom_contype 运行时写入无效，实测写 0
//   读回 0 但接触照常生成），持握段若按原始高度 kinematic 写入会与地面持续爆穿透。
//   因此箱子 z 抬升 Δsink（grasp 后 10 帧内 ramp 进来），释放点只做 RELEASE_DROP 补偿：
//   实测裁定 RELEASE_DROP=0（零高度放置：释放时箱底与槽位底面重合，不引入自由落体），
//   落稳冲击由释放后 PLACE_HOLD=0.4s 的压持窗口吸收（接触在零相对速度下建立），
//   持握全程不与地面/已放箱接触。
// 返回扁平 Float64Array（便于逐子步插值采样）。
// 【MERO-10】函数本体上移至常量区后的"状态机 v3"段（新增第 6 参 preRigid 逐槽刚性
// 规范化与 remapClipCarryFull / computeStackFrameCarry / resolveCarryTiers）；
// λ 数学与 zShift/sink 语义与本节注释一致，未变。
export const RELEASE_DROP = 0.0;

// 释放后压持时长（秒）：物理体以槽位位姿 kinematic 保持，模拟机器人确认放稳再松爪。
// 实测换位首步的接触建立冲击会让箱体产生 5mm 级确定性滑移（plasticbox），压持窗口
// 让接触在零相对速度下建立，之后交还原 solve（箱-地/箱-箱接触全程物理真实）。
const PLACE_HOLD = 0.4;
const SINK_RAMP_FRAMES = 10;



// 按浮点帧号采样重映射轨迹（帧内线性/slerp）
export function sampleRemapped(rm, f) {
  const T = rm.T;
  const fi = Math.max(0, Math.min(T - 1, f));
  const i0 = Math.min(T - 2, Math.floor(fi));
  const a = fi - i0;
  const lerpA = (arr, n) => {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = arr[i0 * n + i] + (arr[(i0 + 1) * n + i] - arr[i0 * n + i]) * a;
    return out;
  };
  const rootPos = lerpA(rm.rootPos, 3);
  const objPos = lerpA(rm.objPos, 3);
  const dof = lerpA(rm.dof, rm.nDof);
  const rootQuat = quatSlerp(
    [...rm.rootQuat.slice(i0 * 4, i0 * 4 + 4)],
    [...rm.rootQuat.slice((i0 + 1) * 4, (i0 + 1) * 4 + 4)], a);
  const objQuat = quatSlerp(
    [...rm.objQuat.slice(i0 * 4, i0 * 4 + 4)],
    [...rm.objQuat.slice((i0 + 1) * 4, (i0 + 1) * 4 + 4)], a);
  return { rootPos, rootQuat, dof, objPos, objQuat, frame: fi };
}

// ---------------- 状态机工厂 ----------------
// 契约（方案 5.2，v2 / MERO-10 v3）：
//   createBoxStacking({ mujoco, model, data, clips, walker, walkMode='rl', carryMode='mocap', meta, log })
//     .loadStackConfig({ layers, cols, boxType })  // 校验箱型 JSON、生成槽位表；进行中则下一轮生效
//     .start() / .stop() / .reseatBoxes() / .onStep(substepDt)
//     .status() -> { phase, label, slotIndex, handedOver, slotTotal, done, mode, held, carryStrategy }
//     .slotStrategies -> { planned[], executed[] }（M-1 策略完整性断言输入）
//     .boxes()  -> [{ bodyId, geomId, jointQadr }]
// clips 形如 { largebox: {low, mid, high, carry:{low?,mid?,high?}}, ... }，tier 值为 5.1 契约
// JSON；carry.{tier} 为 MERO-10 全程回放片段（可缺，缺则该 tier 降级 rl，方案 4.6）。
// walker: createWalkController 实例（rl 链必需，缺失即抛错——legacy 降级已移除）。
// meta:   motions/walk_meta.json 内容（v2 lift_end/lower_start + v3 carry 节，rl 必需）。
// carryMode: 'mocap'（默认，持箱搬移=真人动捕全程回放）| 'rl'（v2 对照）；'legacy' 映射 mocap。
export function createBoxStacking({ mujoco, model, data, clips, walker = null, walkMode = 'rl', carryMode = 'mocap', meta = null, manipulation = null, log = () => {} }) {
  if (!clips || typeof clips !== 'object') throw new Error('createBoxStacking: clips 必填');

  // 瞬移消除改造：legacy 整段回放（含 BLEND_HOME 贝塞尔滑移、root 直写位移）整体移除
  //（无真实步态数据源，队长裁定，README 记录变更）。显式 ?walk=legacy / --walk legacy
  // / ?carry=legacy 均映射到默认链并打日志，不再提供任何整体平移路径。
  if (walkMode === 'legacy') {
    log('[stack] legacy 整段回放模式已移除（无真实步态数据源，见 README），按默认 rl 链运行');
    walkMode = 'rl';
  }
  if (carryMode === 'legacy') {
    log('[stack] carry=legacy 已随 legacy 模式移除（见 README），按 mocap 运行');
    carryMode = 'mocap';
  }

  // walkMode 开关（方案 5.1）：rl 链需要 walker 与 meta；缺失时明确报错（不再静默降级
  // legacy 整体平移，方案裁定 A5）——调用方（main.js）捕获后禁用码箱子并页面提示。
  if (walkMode !== 'rl') throw new Error(`[stack] 未支持的 walkMode=${walkMode}（legacy 已移除，见 README）`);
  if (!walker) throw new Error('[stack] rl 链缺少 walker（RL 行走装配失败？）；legacy 瞬移降级已移除，见 README');
  if (!meta) throw new Error('[stack] rl 链缺少 walk_meta（lift_end/lower_start/carry 帧域）；legacy 瞬移降级已移除，见 README');
  const mode = 'rl';
  // 对照开关启动期明示日志（方案 4.3.3：G1BOX_AVOID_OFF=1 全局短路行走避障为直线）
  if (avoidOff()) log('[stack] G1BOX_AVOID_OFF=1：行走避障规划全局关闭（对照/逃生门），行走按直线');

  // 逐 tier 计划策略的解析基准：rl 模式区分 mocap/rl
  //（M-1 策略完整性的 planned 侧口径：与 executed 同语义可比，方案 4.5 M-1 行）。
  const stratCarryMode = carryMode === 'rl' ? 'rl' : 'mocap';

  const mjOBJ = mujoco.mjtObj;
  const name2id = (type, name) => mujoco.mj_name2id(model, type.value ?? type, name);

  // 机器人关节映射（按名，不按位）与 stand 关键帧
  const jointNames = clips.largebox?.low?.joint_names;
  if (!jointNames) throw new Error('clips.largebox.low 缺失（机器人关节名表来源）');
  const jmap = buildJointMap(mujoco, model, jointNames);
  const standKey = keyId(mujoco, model, 'stand');
  // 双手中点锚（MERO-8）的腕 body：mjcf 左/右 wrist_yaw_link（rubber_hand 视觉件的挂点刚体）
  const wristBidL = name2id(mjOBJ.mjOBJ_BODY, 'left_wrist_yaw_link');
  const wristBidR = name2id(mjOBJ.mjOBJ_BODY, 'right_wrist_yaw_link');
  // 存在性守卫：id<0 时后续 palmMidPose 读 data.xpos 负下标静默得 NaN（模型改名场景），
  // 启动期即抛——工厂初始化一次校验即可覆盖捕锚/取用全部路径，无逐帧开销。
  if (wristBidL < 0 || wristBidR < 0) {
    throw new Error(`g1 模型缺少腕 body（left_wrist_yaw_link / right_wrist_yaw_link，id=${wristBidL}/${wristBidR}），双手中点锚（MERO-8）不可用，模型命名疑似已变更`);
  }
  const nqRobot = 36; // pelvis free 7 + 29 hinge（keyframe 覆盖段，见方案 3.4-2）

  // 当前生效配置与待生效配置（进行中不打断，下一轮 start() 生效）
  let cfg = null;        // { layers, cols, boxType, slots, plans }
  let pendingCfg = null; // { layers, cols, boxType }

  let phase = 'idle';
  let phaseT = 0;        // 阶段本地时间
  let slotIndex = 0;
  let done = false;
  let carryT = 0;        // CARRY/RELEASE 共用的回放时钟
  let held = false;      // 当前槽位箱子是否已被"抓起"（进入跟随段）
  let boxReleased = false;
  let handedOver = 0;    // 已换位（交物理）箱数：单调递增，done 后等于槽位总数（含最后一箱）

  let startPose = null;  // 本轮 blend 起点（机器人）
  let standPose = null;  // stand 关键帧位姿（每次 start() 重取）
  let cur = null;        // 最近一次写入的机器人位姿（SETTLE 冻结 / blend 起点用）

  // v2（rl 模式）状态：持箱锚、RL 段目标、重试计数
  let boxAnchor = null;      // { pos[3], quat[4] } pelvis 系箱锚（CARRY_REPLAY 段末 live 捕获；pelvis 逃生门模式用）
  let handsAnchor = null;    // { pos[3], quat[4] } 双手中点系箱锚（MERO-8 默认；同点 live 捕获）
  // MERO-9：lift_end 捕获帧两腕相对箱体（body 原点系）的抓握偏移 { offL[3], offR[3] }。
  // N13 逐手抓握偏差断言的基准（无头侧按「箱位姿 ⊗ offX」重算每只手该在的位置）；
  // pelvis 逃生门模式同样记录，保证防假绿复跑下 N13 有数据可判。若定标发现捕获几何
  // 本身松（腕远离箱面），此偏移也是「锚改为最小化双手 gap 位姿」的直接输入。
  let graspOff = null;
  let carryBoxLast = null;   // { pos[3], quat[4] } WALK_CARRY 箱位姿限速的上一写值（MF-2，整段持续限速；MERO-8 平滑亦从此出发）
  let resumeWalkPhase = null;
  let pickAlignPose = null;
  let walkGoal = null;       // RL 段注入 walker 的目标（调试用）
  // 行走避障规划快照与统计（walk-avoid-plan 4.5.1/4.6）：lastWalkObstacles 在每次
  // makeWaypoints 规划时写入（与 walkSeq 一一对应：makeWaypoints 先于 walkTo 的 walkSeq++），
  // 是 N19 断言的双源之一（同源快照）；walkPlanStats 的 degraded 计数默认链必须 0（非 0 即红）。
  let lastWalkObstacles = null; // { kind, slotIndex, rects, mode, degraded, goal:[x,y] }
  // 最近一段规划的原始角点折线（工厂方案 D5 小地图输入；lastWalkObstacles 的 N19
  // 契约形状不动，另立字段——points 为规划器输出原始角点，非密化路点）
  let lastWalkPath = null;
  const walkPlanStats = { total: 0, degraded: 0, modes: {} };
  let walkRetries = 0;       // 当前槽位已重试次数（WALK_RETRIES 上限后明确跳槽，不再降级 legacy）
  // approach 段预规划路点（P2 修复：startApproachTurn 先规划后转身，转身段末的
  // startRLWalk 原样复用——避免二次规划使 walkPlanStats 双计；消费后置空）
  let pendingApproachWps = null;
  let prevPhysPose = null;   // ALIGN_PLACE 起点（RL 段结束的物理位姿）
  let phaseDur = 0;          // PREPARE/blend 段自适应时长（位移/yaw 残差加时）
  let turnHomeStart = null;  // 踏步转身起点物理位姿（TURN_* 共用）
  let turnHomeYaw = 0;       // 踏步转身目标朝向
  let alignBoxFrom = null;   // ALIGN_PLACE 箱 blend 起点（进入时一次性捕获的锚随动位姿）
  let turnDurCur = 0;        // 当前踏步转身时长（半步周期整数倍，见 beginPivotTurn）
  let turnPlan = null;
  let alignBoxDur = 0;       // 箱 blend 自适应时长（按箱-目标距离，恒速 ≤0.8m/s）
  // 瞬移消除改造新增状态：
  let errorMsg = null;       // 非空 = 演示暂停（明确报错态：不做瞬移降级，onStep 只保持箱体）
  let recoverNext = null;    // RECOVER_STAND 结束后的续接动作（重试重走 / 跳槽 / 回位）
  let failedKind = null;     // 本次行走失败的段类别 'approach'|'carry'|'home'
  let consecutiveHomeFails = 0; // 回位段连续失败计数（≥2 跳过回位直接下一槽，防循环）

  // 机器人 qpos 差分缓存（qvel 一致性写入，R5）
  const prevRobot = { pos: null, quat: null, dof: null };

  function captureRobotPose() {
    return {
      pos: [data.qpos[0], data.qpos[1], data.qpos[2]],
      quat: [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]],
      dof: jointNames.map((n) => data.qpos[jmap[n].q]),
    };
  }

  const footBodies = ['left_ankle_roll_link', 'right_ankle_roll_link'].map(n => name2id(mjOBJ.mjOBJ_BODY, n));
  const footSpheres = footBodies.flatMap(b => Array.from({ length: model.body_geomnum[b] }, (_, i) => model.body_geomadr[b] + i))
    .filter(g => model.geom_type[g] === 2);
  const floorId = name2id(mjOBJ.mjOBJ_GEOM, 'floor');
  function writeRobot(pose, dt) {
    const q = data.qpos, v = data.qvel;
    q[0] = pose.pos[0]; q[1] = pose.pos[1]; q[2] = pose.pos[2];
    q[3] = pose.quat[0]; q[4] = pose.quat[1]; q[5] = pose.quat[2]; q[6] = pose.quat[3];
    for (let i = 0; i < jointNames.length; i++) q[jmap[jointNames[i]].q] = pose.dof[i];

    // Keep the complete sole above the floor, including during pose blends.
    mujoco.mj_kinematics(model, data);
    const floorZ = floorId >= 0 ? data.geom_xpos[floorId * 3 + 2] : 0;
    const lowest = Math.min(...footSpheres.map(g => data.geom_xpos[g * 3 + 2] - model.geom_size[g * 3]));
    if (lowest < floorZ + 0.001) {
      q[2] += floorZ + 0.001 - lowest;
      pose = { ...pose, pos: [pose.pos[0], pose.pos[1], q[2]] };
    }

    if (dt > 0 && prevRobot.pos) {
      v[0] = (pose.pos[0] - prevRobot.pos[0]) / dt;
      v[1] = (pose.pos[1] - prevRobot.pos[1]) / dt;
      v[2] = (pose.pos[2] - prevRobot.pos[2]) / dt;
      // 自由铰角速度取 body 系：r = conj(q_prev) ⊗ q_cur，ω ≈ 2·vec(r)/dt
      const r = quatMul(quatConj(prevRobot.quat), pose.quat);
      v[3] = 2 * r[1] / dt; v[4] = 2 * r[2] / dt; v[5] = 2 * r[3] / dt;
      for (let i = 0; i < jointNames.length; i++) {
        v[jmap[jointNames[i]].d] = (pose.dof[i] - prevRobot.dof[i]) / dt;
      }
    } else {
      for (let i = 0; i < 6 + jointNames.length; i++) v[i] = 0;
    }
    prevRobot.pos = [...pose.pos];
    prevRobot.quat = [...pose.quat];
    prevRobot.dof = [...pose.dof];

    // ctrl 跟踪回放值：position 执行器目标=当前位姿 → 伺服力矩≈0，避免内应力（方案 4.2）
    for (let i = 0; i < jointNames.length; i++) data.ctrl[jmap[jointNames[i]].c] = pose.dof[i];
    cur = pose;
  }

  // ---------------- 箱体（幽灵/物理双体，见文件头 R7 注释） ----------------
  function idsOf(prefix, k) {
    const jid = name2id(mjOBJ.mjOBJ_JOINT, `carton${prefix}${k}_joint`);
    return {
      bodyId: name2id(mjOBJ.mjOBJ_BODY, `carton${prefix}${k}`),
      geomId: name2id(mjOBJ.mjOBJ_GEOM, `carton${prefix}${k}_geom`),
      jointQadr: model.jnt_qposadr[jid],
      jointDadr: model.jnt_dofadr[jid],
    };
  }
  // 物理体（boxes() 契约、无头断言用）；幽灵体 G 前缀
  const boxIds = (k) => idsOf('', k);
  const ghostIds = (k) => idsOf('G', k);

  function writeBox(ids, pos, quat) {
    const q = data.qpos;
    q[ids.jointQadr] = pos[0]; q[ids.jointQadr + 1] = pos[1]; q[ids.jointQadr + 2] = pos[2];
    q[ids.jointQadr + 3] = quat[0]; q[ids.jointQadr + 4] = quat[1];
    q[ids.jointQadr + 5] = quat[2]; q[ids.jointQadr + 6] = quat[3];
    const d = ids.jointDadr;
    for (let i = 0; i < 6; i++) data.qvel[d + i] = 0; // kinematic 写入段：qvel=0
  }

  // 潜伏位：取箱区外的地板上整排静置（方案 5.3"取箱区外隐藏位"）。
  // 【勿放地板下】MuJoCo 平面 geom 的碰撞无限延伸，箱体在 z<0 仍会生成大穿透接触
  // （实测 dist=-5.17m、ncon≈276、潜伏堆互相爆炸），污染求解与断言。
  // 幽灵体 contype=0 与任何东西都无接触，可与物理体同行同点静置。
  // y 按箱型分行（largebox y=3.0 / plasticbox y=4.2）：两套箱型的物理体都 contype=1，
  // 若同槽号共点静置，页面加载后 idle 态的背景潜伏箱会互推（实测 0.61m）。
  // STAGING_Y 模块级化并导出（工厂方案 F2c）：factory.js FACTORY.staging.y 的同源校验
  // 基准（潜伏行 y 不得移动——无头断言 2 依赖 y>2.5）。
  const STAGING_QUAT = [1, 0, 0, 0];
  function stagingPos(i, restZ, boxType) {
    return [-3 + (i % MAX_BOXES) * 0.6, STAGING_Y[boxType] ?? 3.0, restZ];
  }
  const stagedHolds = new Map(); // holdKey -> {ids, pos, quat}
  function allBoxKeys() {
    const out = [];
    for (const boxType of Object.keys(clips)) {
      if (!clips[boxType]?.low?.box) continue;
      const box = clips[boxType].low.box;
      const restZ = box.half_size[2] - box.bbox_center_offset[2];
      const base = boxTypeBase(boxType);
      for (let i = 0; i < MAX_BOXES; i++) {
        const pos = stagingPos(i, restZ, boxType);
        out.push([`P:${boxType}:${i}`, boxIds(base + i), pos]);
        out.push([`G:${boxType}:${i}`, ghostIds(base + i), pos]);
      }
    }
    return out;
  }
  function reseatBoxes() {
    stagedHolds.clear();
    for (const [key, ids, pos] of allBoxKeys()) {
      if (ids.bodyId < 0) continue;
      writeBox(ids, pos, STAGING_QUAT);
      stagedHolds.set(key, { ids, pos, quat: STAGING_QUAT });
    }
    held = false;
    boxReleased = false;
    handedOver = 0;
  }

  // ---------------- 配置 ----------------
  function clipForLayer(set, layer) {
    return resolveTierForLayer(set, layer);
  }

  function buildConfig({ layers, cols, boxType }) {
    const set = clips[boxType];
    if (!set || !set.low) throw new Error(`箱型 ${boxType} 缺少动作 JSON（motions/ 预取失败？）`);
    if (!(Number.isInteger(layers) && layers >= 1 && layers <= MAX_LAYERS)) throw new Error(`层数须在 1-${MAX_LAYERS}`);
    if (!(Number.isInteger(cols) && cols >= 1 && cols <= MAX_COLS)) throw new Error(`列数须在 1-${MAX_COLS}`);

    const box = set.low.box;
    // 逐 tier 携带策略解析（MERO-10 方案 4.6；buildConfig 与无头断言共用 resolveCarryTiers）
    const carryTiers = resolveCarryTiers(clips, meta, stratCarryMode, boxType);
    // 堆垛框架：存在 mocap tier 时按 G0（computeStackFrameCarry）由最低 mocap tier 的
    // carry clip 摆位——工厂方案 D2 起取箱点/进抵方向显式传派生值（P0 = 货架格网中心
    // 反推、dir0 = FACTORY.rackPsi），使 frame.S 精确等于货架格网中心（函数本体零改动）；
    // 全 rl 时沿用 v2 框架（computeStackFrame：grasp 锚 + 45°/1.2m 地面堆垛），逐位回退
    // 兼容（?carry=rl 对照链地面堆垛语义零改动，AC5）。
    const carryBaseTier = ['low', 'mid', 'high'].find((t) => carryTiers[t] === 'mocap');
    const frame = carryBaseTier
      ? computeStackFrameCarry(set.carry[carryBaseTier], derivePickPos(set.carry[carryBaseTier]), FACTORY.rackPsi)
      : computeStackFrame(set.low);
    // 货架层板格位（D3）：mocap 基准（货架框架）时层 z = boardTop + hz；全 rl 地面
    // 框架不传（stackLayout 缺省纯堆叠）。混合降级（mocap 基准存在但个别 tier 回落 rl）
    // 时槽位 z 统一按货架框架——层高放置本就是摆位近似（D3），v2 链 zShift 机制吸收高度差。
    const boardTops = carryBaseTier ? shelfBoardTops(box.half_size[2]) : null;
    for (const t of ['low', 'mid', 'high']) {
      if (carryTiers[t] === 'mocap' && t !== carryBaseTier && set.carry[t] !== set.carry[carryBaseTier]) {
        log(`[stack] ${boxType}/${t} 动捕携带片段与堆垛基准（${carryBaseTier}）不同源，按各自走廊规范化落位`);
      }
    }
    const all = stackLayout(box, frame.S, frame.psi, boardTops);
    const slots = all.filter((sl) => sl.layer < layers && (cols === 1 ? sl.col === 0 : true));

    const plans = slots.map((sl) => {
      const expectedTier = sl.layer === 0 ? 'low' : 'mid';
      const requiredClip = meta?.carry?.[boxType]?.[expectedTier]?.clip;
      if (stratCarryMode === 'mocap' && requiredClip && !isCarryClipUsable(set.carry?.[expectedTier])) {
        throw new Error(`箱型 ${boxType}/${expectedTier} 搬运动作 ${requiredClip} 未加载，请检查动作文件或刷新页面`);
      }
      let { clip, tier } = clipForLayer(set, sl.layer);
      let strategy = carryTiers[tier];
      // 双保险：mocap 策略要求 carry clip 可用（resolveCarryTiers 已按可用性解析）
      if (strategy === 'mocap' && !isCarryClipUsable(set.carry?.[tier])) strategy = 'rl';
      // rl 槽位的 tier clip 缺失（如 plasticbox mid 借用件只存在于 carry 节、
      // --carry rl 不消费 carry 节）→ 回落 low 桶（与现行 clipForLayer 回落语义一致）
      if (strategy !== 'mocap' && !clip) {
        log(`[stack] ${boxType}/${tier} 缺 v2 动作 JSON，本层回落 low 桶`);
        tier = 'low';
        clip = set.low;
        strategy = carryTiers[tier];
        if (strategy === 'mocap' && !isCarryClipUsable(set.carry?.low)) strategy = 'rl';
      }
      // v2：帧域切分参数（motions/walk_meta.json，方案 5.2）。rl 策略槽位必需；
      // 缺失时该槽标记 invalid（beginSlot 明确跳过并报错——legacy 降级已移除）。
      const metaEntry = meta?.[boxType]?.[tier] ?? null;
      const liftEnd = metaEntry?.lift_end ?? null;
      const lowerStart = metaEntry?.lower_start ?? null;
      if (stratCarryMode === 'mocap' && strategy !== 'mocap') throw new Error(`箱型 ${boxType}/${tier} 缺少搬运动作，不自动降级`);
      const slotInvalid = strategy !== 'mocap' && (liftEnd == null || lowerStart == null);
      if (slotInvalid) log(`[stack] ${boxType}/${tier} 缺 lift_end/lower_start，该槽标记无效（执行时明确跳过）`);

      if (strategy === 'mocap') {
        // ---- MERO-10 carryFull：持箱搬移=单段全程动捕回放（方案 4.3/4.4） ----
        const cClip = set.carry[tier];
        const rel = cClip.release_frame;
        const g = cClip.grasp_frame;
        // 逐槽规范化旋转：行进方向 û 对齐走廊进抵方向 frame.psi（G0 语义；θ 取法的
        // 已记录偏差见 remapClipForSlot 注释——方案 4.4-2 的 δ_yaw 闭式在实测
        // û−boxyaw 65-148° 下会横穿堆垛，故弃用，站位外侧性由选片门限保证）
        const dxy = [cClip.obj_pos[rel][0] - cClip.obj_pos[g][0], cClip.obj_pos[rel][1] - cClip.obj_pos[g][1]];
        const theta = wrapPi(frame.psi - Math.atan2(dxy[1], dxy[0]));
        const preRigid = {
          theta,
          anchor: [cClip.obj_pos[rel][0], cClip.obj_pos[rel][1]],
          target: [sl.center[0], sl.center[1]],
        };
        const rm = remapClipCarryFull(cClip, sl.center, box.half_size[2], box.bbox_center_offset[2], preRigid);
        // Re-solve contacts after shelf remapping, including height and final bin orientation.
        // The plan also supplies grounded leg IK and a retreat before standing up.
        const contactPlan = applyContactPlan(rm, cClip, box, sl, frame.psi, manipulation);
        const spawn = {
          pos: Array.from(rm.objPos.slice(g * 3, g * 3 + 3)),
          quat: Array.from(rm.objQuat.slice(g * 4, g * 4 + 4)),
        };
        return {
          clip: cClip, rm, spawn, slot: sl, tier, strategy, carryFull: true, contactPlan,
          mismatch: Math.abs(sl.center[2] - cClip.z_release),
          liftEnd: null, lowerStart: null, executedStrategy: null,
          stagePos: stagingPos(slots.indexOf(sl), box.half_size[2] - box.bbox_center_offset[2], boxType),
        };
      }

      // ---- v2 rl 策略槽位（TURN_CARRY/WALK_CARRY/ALIGN_PLACE/CARRY_REPLAY2 全链保留） ----
      // 每槽重映射角：让"机器人释放站位相对箱子的偏移方向"对齐该槽的径向外侧，
      // 即机器人永远站在堆垛外侧——否则释放站位会与相邻已放箱重叠，
      // kinematic 脚（priority=1）会把已放箱犁走（首轮实测 box0 被推离 1.1m）。
      const rel = clip.release_frame;
      const d = [clip.root_pos[rel][0] - clip.obj_pos[rel][0], clip.root_pos[rel][1] - clip.obj_pos[rel][1]];
      const u = [sl.center[0] - frame.S[0], sl.center[1] - frame.S[1]];
      const psi = Math.atan2(u[1], u[0]) - Math.atan2(d[1], d[0]); // 只取方向，不缩放
      const rm = remapClipForSlot(clip, sl.center, psi, box.half_size[2], box.bbox_center_offset[2]);
      // 取箱生成位：箱底贴地（geom 中心 z = hz），朝向取 grasp 帧物体朝向
      const g = clip.grasp_frame;
      const spawn = {
        pos: [clip.obj_pos[g][0], clip.obj_pos[g][1], box.half_size[2] - box.bbox_center_offset[2]],
        quat: [...clip.obj_quat[g]],
      };
      return {
        clip, rm, spawn, slot: sl, mismatch: Math.abs(sl.center[2] - clip.z_release),
        tier, strategy, carryFull: false, liftEnd, lowerStart, executedStrategy: null,
        invalid: slotInvalid,
        stagePos: stagingPos(slots.indexOf(sl), box.half_size[2] - box.bbox_center_offset[2], boxType), // 幽灵体回潜伏位
      };
    });
    return { layers, cols, boxType, slots, plans, box, stackFrame: frame, carryTiers };
  }

  function loadStackConfig(params) {
    const built = buildConfig(params); // 先构建校验，失败即抛
    if (phase === 'idle') {
      cfg = built;
      reseatBoxes();
      log(`[stack] 配置生效: ${params.layers}层×${params.cols}列 ${params.boxType}，共 ${built.slots.length} 槽`);
    } else {
      pendingCfg = params;
      log('[stack] 堆垛进行中，参数将在下一轮生效');
    }
  }

  // ---------------- 阶段推进 ----------------
  // 幽灵体退场回潜伏位（跳槽/未取箱收尾用）：R7 的"合法退场"机制，只动箱不动机器人。
  function retireGhost(plan) {
    writeBox(plan.gids, plan.stagePos, STAGING_QUAT);
    stagedHolds.set(`G:${cfg.boxType}:${slotIndex}`, { ids: plan.gids, pos: plan.stagePos, quat: STAGING_QUAT });
  }

  // 跳过当前槽（重试耗尽/配置缺陷）：明确报错 + 幽灵箱退场，机器人从当前位置继续
  // 下一槽或收尾——不重置、不瞬移、不静默降级（瞬移消除改造，方案裁决 A5）。
  function skipCurrentSlot(reason) {
    const plan = cfg.plans[slotIndex];
    retireGhost(plan);
    held = false;
    boxReleased = false;
    log(`[stack] 槽位 ${slotIndex + 1}/${cfg.slots.length} 跳过（${reason}）；幽灵箱退场，继续`);
    if (slotIndex + 1 < cfg.slots.length) {
              const nextSlot = slotIndex + 1;
              // Settle the empty-hand stance before turning into the next task.
              failedKind = null;
              beginRecover(() => beginSlot(nextSlot));
            }
    else finishAll();
  }

  function beginSlot(i) {
    const isRetry = i === slotIndex && phase !== 'idle'; // 同槽重入 = 失败重试，保留重试计数
    slotIndex = i;
    const plan = cfg.plans[i];
    const base = boxTypeBase(cfg.boxType) + i;
    plan.ids = boxIds(base);     // 物理体：潜伏至释放
    plan.gids = ghostIds(base);  // 幽灵体：本轮取箱+持握跟随
    if (plan.ids.bodyId < 0 || plan.gids.bodyId < 0) {
      throw new Error(`carton XML 未注入（carton${base}/cartonG${base}），fetchAssets 预注入缺失`);
    }
    if (plan.invalid) {
      // rl 策略槽位缺帧域元数据：明确跳过（legacy 降级已移除，瞬移消除改造）
      log(`[stack] 槽位 ${i + 1} 配置缺陷（${cfg.boxType}/${plan.tier} 缺 lift_end/lower_start），明确跳过该槽`);
      if (i + 1 < cfg.slots.length) beginSlot(i + 1);
      else finishAll();
      return;
    }
    // 激活幽灵体（物理体继续潜伏）；生成位 kinematic 保持
    stagedHolds.delete(`G:${cfg.boxType}:${i}`);
    writeBox(plan.gids, plan.spawn.pos, plan.spawn.quat);
    held = false;
    boxReleased = false;
    boxAnchor = null;
    handsAnchor = null;
    graspOff = null;
    carryBoxLast = null;
    alignBoxFrom = null;
    if (!isRetry) walkRetries = 0;

    // MERO-10 carryFull：持箱搬移=单段全程动捕回放（方案 4.3），无需 lift_end/lower_start；
    // WALK_APPROACH/WALK_HOME 仍为 RL 行走，段失败的重试/跳槽机制原样覆盖这两段。
    // v2 rl 策略槽位（?carry=rl）：TURN_CARRY/WALK_CARRY/ALIGN_PLACE/CARRY_REPLAY2 全链保留。
    try {
      walker.enter(); // 幂等；增益改写只发生在 enter/exit 两时机（方案 5.3）
    } catch (e) {
      log(`[stack] walker.enter() 失败（${e?.message ?? e}）——槽位 ${i + 1} 明确跳过（不做瞬移降级）`);
      retireGhost(plan);
      if (i + 1 < cfg.slots.length) beginSlot(i + 1);
      else finishAll();
      return;
    }
    startApproachTurn(plan);
    log(`[stack] 槽位 ${i + 1}/${cfg.slots.length} (layer${plan.slot.layer + 1} col${plan.slot.col + 1}) `
      + `clip=${plan.clip.source_clip} 搬移=${plan.strategy === 'mocap' ? '接触约束重定向(carryFull)' : 'RL 持箱(v2)'} |cz−z_r|=${plan.mismatch.toFixed(3)}m`);
  }

  function blendPose(a, b, alpha) {
    return {
      pos: lerp3(a.pos, b.pos, alpha),
      quat: quatSlerp(a.quat, b.quat, alpha),
      dof: a.dof.map((v, i) => v + (b.dof[i] - v) * alpha),
    };
  }

  function clipPoseAt(rm, f) {
    const s = sampleRemapped(rm, f);
    return { pos: s.rootPos, quat: s.rootQuat, dof: s.dof, objPos: s.objPos, objQuat: s.objQuat, frame: s.frame };
  }

  // ---------------- v2（rl 模式）：RL 行走段编排（方案 4.2/4.4/4.5/4.6） ----------------
  function quatToYaw(q) {
    return Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
  }

  // Solve footfalls once; runtime only interpolates the continuous IK trajectory.
  function beginPivotTurn(from, targetYaw, phaseName, why, rate = TURN_RATE) {
    turnHomeStart = from;
    turnHomeYaw = targetYaw;
    turnPlan = buildPivotPlan({ mujoco, model, data, jmap, jointNames, from, targetYaw, rate });
    turnDurCur = turnPlan.duration;
    startPose = from; cur = from;
    phase = phaseName; phaseT = 0; prevRobot.pos = null;
    log(`[stack] 小步转向 ${why}：${turnPlan.steps} 步 / ${turnDurCur.toFixed(1)}s，足端误差 ${(turnPlan.worstPosition * 1000).toFixed(1)}mm`);
  }
  function pivotTurnPose() { return samplePivotPlan(turnPlan, phaseT); }

  // ---------------- 行走段 2D 避障（walk-avoid-plan F2）：障碍集组装 ----------------
  // 每次 makeWaypoints（即每次 startRLWalk，含重试重规划）调用一次，组装"当前视觉在场
  // 箱子"的 2D 矩形列表（确定性：同一时刻状态同一列表；方案 4.2 口径）：
  //   - 目标箱（仅 approach）：当前槽幽灵 spawn 位姿（取箱生成位），role='target'——规划器
  //     按 targetShrinkR 收缩膨胀，起终点透明化/脱出路点机制覆盖设计性贴近（定标 0.139）；
  //   - 已放槽位：物理体实际 data.qpos 直读（覆盖 ≤5mm 物理滑移与换位对齐跳变，margin 0.10）；
  //     跳过的槽位未换位，物理体仍在潜伏位，由潜伏行矩形覆盖（qpos 真值口径自洽）；
  //   - 两箱型潜伏行：stagingPos 常量位（每箱型 ≤6 个，yaw=0）。非当前槽幽灵与物理体共点
  //     潜伏（stagedHolds），矩形去重并入本行；已换位槽位的原潜伏位为空场幽灵矩形（保守
  //     冗余，planner 走廊 AABB+0.6m 粗筛会裁掉，工作区行走段实际全部被裁，保留是正确性保底）；
  //   - kind='carry'（?carry=rl 的 WALK_CARRY）：持箱跟手机体、本身不是障碍，并入机器人
  //     膨胀 r_carry（AVOID_ROBOT_R_CARRY），障碍集不含 target 矩形（队长裁定 Q2）。
  function collectWalkObstacles(kind) {
    const rects = [];
    const bx = cfg.box;
    const [ox, oy, oz] = bx.bbox_center_offset;
    if (kind === 'approach') {
      const plan = cfg.plans[slotIndex];
      // geom 中心 = body 位姿 ⊗ bbox_center_offset（水平分量随全四元数旋转后取 xy）
      const c = quatRotVec(plan.spawn.quat, [ox, oy, oz]);
      rects.push({
        cx: plan.spawn.pos[0] + c[0], cy: plan.spawn.pos[1] + c[1],
        yaw: quatYaw(plan.spawn.quat),
        hx: bx.half_size[0], hy: bx.half_size[1],
        role: 'target', key: `target:slot${slotIndex}`,
      });
    }
    // 已放槽位：物理体 qpos 真值（handoverToPhysics 后单调递增，含当前槽在 home 段的刚放箱）
    const base = boxTypeBase(cfg.boxType);
    for (let i = 0; i < handedOver; i++) {
      const ids = boxIds(base + i);
      if (ids.bodyId < 0) continue;
      const adr = ids.jointQadr;
      const q = [data.qpos[adr + 3], data.qpos[adr + 4], data.qpos[adr + 5], data.qpos[adr + 6]];
      const c = quatRotVec(q, [ox, oy, oz]);
      rects.push({
        cx: data.qpos[adr] + c[0], cy: data.qpos[adr + 1] + c[1],
        yaw: quatYaw(q),
        hx: bx.half_size[0], hy: bx.half_size[1],
        role: 'placed', key: `placed:${i}`,
      });
    }
    // 两箱型潜伏行（按箱型泛化，A5：smallbox 无 motions 即无行；矩形按 (箱型,i) 唯一，
    // 与共点幽灵去重）
    // 未激活箱体为内部物理缓冲，不显示，也不占用作业通道。
    // 工厂静态障碍（工厂方案 D4）：活动货架（role='target' 复用贴近机制）+ 立柱/围栏/
    // 托盘/道具货架/围界（全膨胀）。key='static:<name>'，三处消费同一张 FACTORY 表
    // （headless F2b 校验规划快照与表同源）。矩形总数最坏 ~32（静态 14 + staging 12 +
    // placed 6 + target 1）→ 节点 ~130，pathPlanner maxNodes 已扩至 256（D4）。
    rects.push(...factoryObstacleRects());
    return rects;
  }

  // 位姿 → 行走路点（walk-avoid-plan F3）：路径规划换用 src/pathPlanner.js 的 2D 避障
  // 规划器（解析膨胀 + 可见性图 + Dijkstra + 角点过渡，方案 4.4；旧"堆垛中心单圆垂向
  // 偏置"机制 walker.planPath 已退役删除，单一事实源）。
  // 路点 yaw = 行进方向（策略只有「前进+弱转向」能力，T0）；操作朝向由 blend 段吸收。
  // 停站机制：距目标 STOP_BAND(0.08) 内零指令、惯性停住（策略指令死区 ≈0.25），
  // 停点误差 ≤0.15m 由 blend 吸收——路点不再内移。
  // 注：2x2 槽 3 回位段失稳经第 3 轮锥形降速试验否定"刹停瞬态"假说（摔倒发生在
  // 满速行走中段的方位失稳盘旋，wz 饱和签名），已回滚，详见 docs/policy-probe-notes.md。
  function makeWaypoints(fromPose, goalPos, kind) {
    const from = { x: fromPose.pos[0], y: fromPose.pos[1], yaw: quatToYaw(fromPose.quat) };
    const to = { x: goalPos[0], y: goalPos[1], yaw: Math.atan2(goalPos[1] - from.y, goalPos[0] - from.x) };
    const rects = collectWalkObstacles(kind);
    // 对照开关（4.3.3）：AVOID_OFF 传空障碍集走同一规划器出口（mode='straight'），统计口径统一
    const plan = planPathAvoid(from, to, avoidOff() ? [] : rects,
      kind === 'carry' ? { robotR: AVOID_ROBOT_R_CARRY } : {});
    // 快照与统计（N19 双源之一与 degraded 计数输入，方案 4.5.1/4.6）
    lastWalkObstacles = { kind, slotIndex, rects, mode: plan.mode, degraded: plan.degraded, goal: [to.x, to.y] };
    // 小地图路径输入（工厂方案 D5）：规划器原始角点折线（非密化路点）+ 最终目标
    lastWalkPath = { kind, points: plan.points, goal: [to.x, to.y] };
    walkPlanStats.total++;
    if (plan.degraded) walkPlanStats.degraded++;
    walkPlanStats.modes[plan.mode] = (walkPlanStats.modes[plan.mode] ?? 0) + 1;
    // 失败语义三级阶梯的明示日志（4.3.3：绝不静默；shrunk=收缩一轮后可行，不计降级但显性化）
    if (plan.degraded) log('[stack] 避障规划失败（无可行路径），明示直线降级'
      + `（from=(${from.x.toFixed(2)},${from.y.toFixed(2)}) to=(${to.x.toFixed(2)},${to.y.toFixed(2)}) `
      + `kept=[${plan.usedRects.map((r) => r.key).join(',')}]）`);
    else if (plan.shrunk) log(`[stack] 避障规划全膨胀无可行路径，收缩一轮后可行（安全余量降为 r_min，障碍 ${rects.length} 个）`);
    else log(`[stack] 行走避障规划（${kind}）：mode=${plan.mode} 障碍 ${rects.length} 个（粗筛裁掉 ${plan.pruned}）绕行点 ${Math.max(0, plan.points.length - 1)} 个`);
    // 航向校准路点：行进转向有效速率仅 ~0.03rad/s（T0），长直线上方位误差会积累成
    // 大弧绕行——按 ~0.3m 步长插密校准点，使方位误差在到达前被路点切换持续重置
    const wps = plan.points;
    const dense = [];
    for (let i = 0; i < wps.length; i++) {
      const a = i === 0 ? from : { x: wps[i - 1].x, y: wps[i - 1].y };
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
    // negcheck avoid 注入 1（G1BOX_NEGCHECK_AVOID_BREAK>0，仅无头防假绿消费，4.5.2）：
    // 规划完成后把中部路点（不含首末）向最近障碍矩形中心平移注入量 → 行走路径被拉进
    // 障碍，N19 必 FAIL。目标矩形取全量组装集（含被粗筛裁掉的——注入要能打到远处障碍）。
    const brk = negAvoidBreak();
    if (brk > 0 && dense.length > 2 && rects.length > 0) {
      let nShift = 0;
      for (let i = 1; i < dense.length - 1; i++) {
        let best = -1, bestD = Infinity;
        for (let r = 0; r < rects.length; r++) {
          const d = pointRectDist(dense[i], rects[r]);
          if (d < bestD) { bestD = d; best = r; }
        }
        if (best < 0) continue;
        const vx = rects[best].cx - dense[i].x, vy = rects[best].cy - dense[i].y;
        const vl = Math.hypot(vx, vy) || 1e-9;
        dense[i] = { x: dense[i].x + vx / vl * brk, y: dense[i].y + vy / vl * brk, yaw: dense[i].yaw };
        nShift++;
      }
      log(`[negcheck] 避障注入生效：${nShift} 个中部路点向最近障碍矩形中心平移 ${brk}m（N19 应 FAIL）`);
    }
    return dense;
  }

  // 启动一段 RL 行走。kind: approach（去取箱位）/ carry（持箱去放置位）/ home（回 stand 原点）。
  // 路点 yaw = 行进方向（策略只有前进能力，T0）；操作朝向残差由 blend 段吸收。
  // WALK_APPROACH 前的踏步转身（TURN_CARRY/TURN_HOME/TURN_STAND 同款机制）：
  // WALK_HOME 结束时机器人朝向来路方向（背对下次取箱位），官方 velocity 策略只有
  // 「前进+弱转向」能力（T0：行进转向有效速率 ~0.03rad/s，原地转 3s≈5°），大角度
  // 偏差下纯追踪会绕点盘旋、中间路点永不捕获直至超时（2x2 槽 2 实测 eYaw=-2.16rad
  // dist 越走越远）——偏差超阈先踏步转身面向取箱位再交给 RL。slot 0 由 start() 摆位
  // 面向取箱位，偏差≈0 不触发转身。
  // 【工厂方案 P2 修复】approach 的转身目标 = 实际第一路点方向（先规划后转身，
  // startApproachTurn 预规划的路点经 preWps 复用，避免 walkPlanStats 双计）：绕行走廊下
  // 第一路点方向与 goal 直线方向差可达 ~15°（mid tier 实测），起步即带航向差会让
  // pure pursuit 一正一反的航向摆动在策略慢响应下发散成盘旋（槽 4 三连败签名：
  // eYaw=-2.89 碎步 3.7m）；对准第一路点后起步零航向差。
  function startRLWalk(kind) {
    const plan = cfg.plans[slotIndex];
    // 规划起点现取物理位姿（walk-avoid-plan 4.3.4 加固）：消除 cur 残留的理论风险——
    // 全部调用点（TURN_* 段末 / RECOVER_STAND 段末 / startApproachTurn）的 data.qpos
    // 都是刚写入或物理推进的现值，captureRobotPose() 与 cur 逐位一致，行为零改变。
    const from = captureRobotPose();
    let goalPos, goalQuat, armDof, carry;
    if (kind === 'approach') {
      const g = clipPoseAt(plan.rm, 0);
      goalPos = [g.pos[0], g.pos[1]];
      goalQuat = g.quat;
      armDof = plan.clip.dof_pos[0];              // 取接近段末的动捕臂姿（关节空间，天然基座相对）
      carry = false;
    } else if (kind === 'carry') {
      const g = clipPoseAt(plan.rm, plan.lowerStart);
      goalPos = [g.pos[0], g.pos[1]];
      goalQuat = g.quat;
      armDof = plan.clip.dof_pos[plan.liftEnd];   // 持箱臂姿 = 提起完成帧
      carry = true;
    } else {
      goalPos = [standPose.pos[0], standPose.pos[1]];
      goalQuat = standPose.quat;
      armDof = standPose.dof;
      carry = false;
    }
    // WALK_STOP_LEAD：目标沿进抵方向前移一个停止带（常量处注释）——惯性停站"擦过即达"
    // 的触发点恰落在真实目标上，停点残差从实测 0.10-0.12m 压到 ~0.03m，后续 blend 段
    // 位移随之为 ≤~0.06m（N15 blend 行单列口径，≤0.15m 上限）。N6 量测对路点距离，口径不变。
    {
      const dgx = goalPos[0] - from.pos[0], dgy = goalPos[1] - from.pos[1];
      const dn = Math.hypot(dgx, dgy);
      if (dn > 1e-6) {
        goalPos = [goalPos[0] + dgx / dn * WALK_STOP_LEAD, goalPos[1] + dgy / dn * WALK_STOP_LEAD];
      }
    }
    const wps = (kind === 'approach' && pendingApproachWps) ? pendingApproachWps : makeWaypoints(from, goalPos, kind);
    if (kind === 'approach') pendingApproachWps = null; // 预规划路点一次性消费（startApproachTurn 复用，防统计双计）
    let pathLen = 0, px = from.pos[0], py = from.pos[1];
    for (const wp of wps) { pathLen += Math.hypot(wp.x - px, wp.y - py); px = wp.x; py = wp.y; }
    const vmax = carry ? 0.3 : 0.3;
    // 臂部干预的实验结论（详见 docs/policy-probe-notes.md）：
    //  - D2（ctrl 级覆盖 + last_action=raw）：obs 三通道不一致正反馈 → 行走失控摔倒（T3 实测）；
    //  - A1（MERO-7 kinematic 定格 qpos/qvel/ctrl）：即使三通道自洽，臂部反作用力不再经
    //    关节动力学累积 → 出训练分布 → 摔倒（mero7-evidence/v1_A1_default.log，16/19）；
    //  - MERO-8 定案：机器人侧不做任何「接管式」臂部定格——持箱观感由「箱跟手」解决
    //    （carryAnchorPose 双手中点锚，箱随手臂走）；可选 softArmHold（渐进动作空间混合：
    //    臂 14 维动作向持箱姿态渐入，last_action 喂混合值闭环自洽、不写 qpos/qvel）在
    //    walker 内部按 armHold 模式生效，是否默认开启由门控实验裁定（?armhold=off 可关）。
    //    armGoal 通道（D2 遗留）保持恒 null。
    const armGoal = null;
    const vxScale = 1.0;
    walker.walkTo({
      waypoints: wps, carrying: carry, armGoal,
      holdPose: carry ? Float64Array.from(armDof) : null, // 持箱姿态=lift_end 帧；walker 侧按 armHold 模式门控
      vxScale,
      // 朝向到位不硬判：路点 yaw 已是行进方向，到位朝向 = 来路方向，由后续 blend 吸收
      timeLimitSec: WALK_TIME_SLACK + WALK_TIME_MULT * pathLen / (vmax * vxScale), yawTolerance: Math.PI,
    });
    // 复位 obs 历史放在 walkTo 之后：段配置（含 holdPose）先就位再清历史，避免把
    // 上一段残值或自相矛盾组合填进 obs（MERO-5 残值机理的同款预防）。
    walker.resetObsHistory();
    // 重试差异化：确定性仿真中相同初始条件必复现同一失败——重试段延长预热（1.0s）
    // 改变起步动态打破失败循环（方案 4.6 重试语义的必要补充）。速度不得缩放：
    // T0 实测指令死区 |cmd|≲0.25，vxMax=0.3 时 vxScale≤0.83 即落入死区步态停摆
    // （首版 2x2 验证中重试恒超时的根因：0.3×0.75=0.225，13s 仅挪 0.59m）。
    walker.warmUp(walkRetries > 0 ? 1.0 : WALK_WARMUP);
    walkGoal = { kind, pathLen, wps };
    phase = kind === 'approach' ? 'WALK_APPROACH' : kind === 'carry' ? 'WALK_CARRY' : 'WALK_HOME';
    phaseT = 0;
    prevRobot.pos = null;
    if (wps.length) {
      const bearing = Math.atan2(wps[0].y-from.pos[1], wps[0].x-from.pos[0]);
      if (Math.abs(wrapPi(bearing-quatToYaw(from.quat))) > 0.32) {
        resumeWalkPhase = phase;
        walker.exit();
        beginPivotTurn(from, bearing, 'TURN_PATH', '对准路径起点', carry ? TURN_RATE_CARRY : TURN_RATE);
      }
    }

  }

  // approach 段的进入编排（P2 修复：先规划后转身）：makeWaypoints 的路点与
  // startRLWalk('approach') 必须同输入（起点=当前物理位姿、终点=goal0+WALK_STOP_LEAD
  // 同款前移），预规划的路点经 preWps 原样复用；转身目标取第一路点方向（起步零航向差，
  // 见 startRLWalk 注释）。TURN_APPROACH_MIN 检查同样对第一路点方向。
  function startApproachTurn(plan) {
    const from = captureRobotPose();
    const g0 = clipPoseAt(plan.rm, 0);
    let goalPos = [g0.pos[0], g0.pos[1]];
    {
      const dgx = goalPos[0] - from.pos[0], dgy = goalPos[1] - from.pos[1];
      const dn = Math.hypot(dgx, dgy);
      if (dn > 1e-6) {
        goalPos = [goalPos[0] + dgx / dn * WALK_STOP_LEAD, goalPos[1] + dgy / dn * WALK_STOP_LEAD];
      }
    }
    const wps = makeWaypoints(from, goalPos, 'approach');
    pendingApproachWps = wps;
    // 第一路点方向（密化路点首点即第一段行进方向；空路点防御性回落 goal 方向）
    const firstDir = wps.length > 0
      ? Math.atan2(wps[0].y - from.pos[1], wps[0].x - from.pos[0])
      : Math.atan2(goalPos[1] - from.pos[1], goalPos[0] - from.pos[0]);
    const eYaw = Math.atan2(Math.sin(firstDir - quatToYaw(from.quat)),
      Math.cos(firstDir - quatToYaw(from.quat)));
    if (Math.abs(eYaw) <= TURN_APPROACH_MIN) {
      startRLWalk('approach');
      return;
    }
    beginPivotTurn(from, firstDir, 'TURN_APPROACH', '朝取箱位');
  }

  // 双手中点位姿（MERO-8）：pos = 两腕 body 原点中点（行走摆臂前后反相，中点晃动远小于
  // 单手——选「中点」而非单手锚的原因），quat = 短路径 slerp 中点。读 data.xpos/xquat
  // （forward 后有效：动力学段来自上一 mj_step 的 forward，滞后一个子步 ≈2ms，远小于
  // 平滑时间常数；kinematic 捕获帧须先显式 forward 刷新，见 captureCarryAnchor）。
  function palmMidPose() {
    const pl = [data.xpos[3 * wristBidL], data.xpos[3 * wristBidL + 1], data.xpos[3 * wristBidL + 2]];
    const pr = [data.xpos[3 * wristBidR], data.xpos[3 * wristBidR + 1], data.xpos[3 * wristBidR + 2]];
    const ql = [data.xquat[4 * wristBidL], data.xquat[4 * wristBidL + 1], data.xquat[4 * wristBidL + 2], data.xquat[4 * wristBidL + 3]];
    const qr = [data.xquat[4 * wristBidR], data.xquat[4 * wristBidR + 1], data.xquat[4 * wristBidR + 2], data.xquat[4 * wristBidR + 3]];
    return {
      pos: [(pl[0] + pr[0]) / 2, (pl[1] + pr[1]) / 2, (pl[2] + pr[2]) / 2],
      quat: quatSlerp(ql, qr, 0.5),
    };
  }

  // 持箱锚随动位姿（4.5 / MERO-8）：obj = 锚系位姿 ⊗ 锚。锚系默认双手中点（箱跟手），
  // G1BOX_CARRY_ANCHOR=pelvis 时为基座（旧 4.5 行为，对照/防假绿逃生门）。
  // 锚在 CARRY_REPLAY 段末 live 捕获（捕获帧手正扶箱），进入跟随段零跳变。
  function carryAnchorPose() {
    if (CARRY_ANCHOR_HANDS && handsAnchor) {
      const pm = palmMidPose();
      const off = quatRotVec(pm.quat, handsAnchor.pos);
      return { pos: [pm.pos[0] + off[0], pm.pos[1] + off[1], pm.pos[2] + off[2]], quat: quatMul(pm.quat, handsAnchor.quat) };
    }
    const p = [data.qpos[0], data.qpos[1], data.qpos[2]];
    const q = [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]];
    const off = quatRotVec(q, boxAnchor.pos);
    return { pos: [p[0] + off[0], p[1] + off[1], p[2] + off[2]], quat: quatMul(q, boxAnchor.quat) };
  }

  // live 捕锚（CARRY_REPLAY 段末 lift_end 帧：臂部 dof 冻结在持箱姿态、手正扶箱）：
  // 锚 = inv(锚系位姿) ⊗ 幽灵箱当前位姿。pelvis 系读 qpos 即 live（kinematic 段直写）；
  // 双手系读腕 body 位姿前须先 mj_forward 刷新——此刻 writeRobot 刚写入 lift_end qpos，
  // data.xpos 还是上一子步的 FK，不刷新会把捕获误差（单子步回放位移）烤进锚里。
  // 两分支统一先刷新 FK 并记录逐手抓握偏移 graspOff（MERO-9 N13 基准；pelvis 逃生门
  // 也记录，使防假绿复跑下 N13 有数据可判「gap 爆炸」而非「无数据」）。
  function captureCarryAnchor(boxPos, boxQuat) {
    mujoco.mj_forward(model, data);
    const wl = [data.xpos[3 * wristBidL], data.xpos[3 * wristBidL + 1], data.xpos[3 * wristBidL + 2]];
    const wr = [data.xpos[3 * wristBidR], data.xpos[3 * wristBidR + 1], data.xpos[3 * wristBidR + 2]];
    const iqB = quatConj(boxQuat);
    graspOff = {
      offL: quatRotVec(iqB, [wl[0] - boxPos[0], wl[1] - boxPos[1], wl[2] - boxPos[2]]),
      offR: quatRotVec(iqB, [wr[0] - boxPos[0], wr[1] - boxPos[1], wr[2] - boxPos[2]]),
    };
    if (!CARRY_ANCHOR_HANDS) {
      const p = [data.qpos[0], data.qpos[1], data.qpos[2]];
      const q = [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]];
      const iq = quatConj(q);
      const rel = [boxPos[0] - p[0], boxPos[1] - p[1], boxPos[2] - p[2]];
      boxAnchor = { pos: quatRotVec(iq, rel), quat: quatMul(iq, boxQuat) };
      return;
    }
    const pm = palmMidPose();
    const iq = quatConj(pm.quat);
    const rel = [boxPos[0] - pm.pos[0], boxPos[1] - pm.pos[1], boxPos[2] - pm.pos[2]];
    handsAnchor = { pos: quatRotVec(iq, rel), quat: quatMul(iq, boxQuat) };
    log(`[stack] 持箱锚捕获：双手中点系（tau=${CARRY_SMOOTH_TAU}s；G1BOX_CARRY_ANCHOR=pelvis 可回退）`);
  }

  // 持箱跟随（4.5 / MERO-8）：默认双手中点系锚随动——目标 = palmMidNow ⊗ 锚，
  // 先从上一写值出发做指数平滑（CARRY_SMOOTH_TAU，抑制摆臂微抖直传箱体；
  // 平滑值从当前位姿出发 → 逐子步连续，N10 不受破坏），再过 MF-2 全程限速帽
  // 写幽灵体。pelvis 逃生门模式保持旧行为（无平滑）。
  // 首子步 carryBoxLast 为空：以当时锚位姿直写初始化基准（与上一段末写连续）。
  // WALK_CARRY 行走段与 RECOVER_STAND 恢复段（持箱失败原地起身，箱子跟手回到
  // 站立持握位）共用。
  function followCarryBox(dt) {
    const plan = cfg.plans[slotIndex];
    const a = carryAnchorPose();
    let bp = a.pos, bq = a.quat;
    if (carryBoxLast) {
      let sp = a.pos, sq = a.quat;
      if (CARRY_ANCHOR_HANDS) {
        const k = Math.min(1, dt / CARRY_SMOOTH_TAU);
        sp = [
          carryBoxLast.pos[0] + (a.pos[0] - carryBoxLast.pos[0]) * k,
          carryBoxLast.pos[1] + (a.pos[1] - carryBoxLast.pos[1]) * k,
          carryBoxLast.pos[2] + (a.pos[2] - carryBoxLast.pos[2]) * k,
        ];
        sq = quatSlerp(carryBoxLast.quat, a.quat, k);
      }
      const maxStep = (CARRY_ANCHOR_HANDS ? CARRY_BOX_VCAP_HANDS : CARRY_BOX_VCAP) * dt;
      const d = [sp[0] - carryBoxLast.pos[0],
        sp[1] - carryBoxLast.pos[1], sp[2] - carryBoxLast.pos[2]];
      const len = Math.hypot(d[0], d[1], d[2]);
      const q = len > maxStep ? maxStep / len : 1;
      bp = [carryBoxLast.pos[0] + d[0] * q,
        carryBoxLast.pos[1] + d[1] * q, carryBoxLast.pos[2] + d[2] * q];
      bq = quatSlerp(carryBoxLast.quat, sq, q);
    }
    carryBoxLast = { pos: bp, quat: bq };
    writeBox(plan.gids, bp, bq);
  }

  // WALK 段失败（跌倒/超时）：无瞬移恢复链（瞬移消除改造，方案裁决 A5/A4）——
  //   1) RECOVER_STAND：原地起身/站稳（kinematic blend，root 水平位置不变——无起身高
  //      保真数据源，属明示的演示级近似；连续性由无头 N17 逐子步跳变断言守门）；
  //   2) 从当前姿态重新规划行走（startApproachTurn/startRLWalk 每次启动现取物理位姿并
  //      重新规划——pathPlanner 避障规划天然支持任意起点，重新走回取箱点/放置点/stand 原点，
  //      不再 reset 瞬移回原点）；
  //   3) 重试耗尽（WALK_RETRIES）：明确报错 + 跳过该槽（幽灵箱退场），不再静默降级
  //      legacy 整体平移（该路径已整体移除，见 README）；
  //   4) 回位段（home）连续失败 ≥2 次：跳过回位行走，从当前位置直接进入下一槽接近段
  //      （取箱行走可从任意位置出发），防失败循环；收尾残差守卫见 finishAll。
  function handleWalkFailure(reason) {
    const failedPhase = phase === 'STOP_PATH' ? resumeWalkPhase : phase;
    failedKind = failedPhase === 'WALK_APPROACH' ? 'approach' : failedPhase === 'WALK_CARRY' ? 'carry' : 'home';
    if (failedKind === 'home') consecutiveHomeFails++;
    if (walkRetries < WALK_RETRIES) {
      walkRetries++;
      log(`[stack] RL 行走${reason}，原地恢复后重新规划本段（第 ${walkRetries}/${WALK_RETRIES} 次，${failedKind} 段）`);
      beginRecover(() => {
        if (failedKind === 'approach') startApproachTurn(cfg.plans[slotIndex]);
        else if (failedKind === 'carry') startRLWalk('carry');
        else startRLWalk('home');
      });
      return;
    }
    errorMsg = `槽位 ${slotIndex + 1}/${cfg.slots.length} · ${PHASE_LABEL[phase]}失败：${reason}。已暂停，请重置后重试`;
    log('[stack] ' + errorMsg);
    // 停在原地，保留任务和已放料箱，不自动跳槽或继续下一任务。
    walker.exit();
    cur = captureRobotPose();
  }

  // 原地恢复站立（RECOVER_STAND 进入）：从当前姿态（摔倒/失稳后物理位姿）blend 到
  // "在当前水平位置站立"——root xy 不变、z 升至 stand 骨架高度、dof→stand、朝向不变。
  // 水平位移 ≈0，非任务位移（明示演示级近似，见 handleWalkFailure 注释）。
  function beginRecover(next) {
    recoverNext = next;
    startPose = captureRobotPose();
    cur = startPose;
    phase = 'RECOVER_STAND';
    phaseT = 0;
    prevRobot.pos = null; // blend 起步首步 qvel=0
  }

  function finishAll() {
    // 结尾回站立重排（瞬移消除改造）：先 blend 位置/关节到 stand（与 WALK_HOME 到位
    // 连续——RL 停站残差经 WALK_STOP_LEAD 已压到 ~0.03m，blend 慢速吸收），朝向保持
    // 来路方向不动；随后若朝向残差 >TURN_APPROACH_MIN 再踏步转身对齐（TURN_STAND，
    // 原地 ~0 位移）。位置残差过大（回位反复失败的病态场景）明确报错暂停，不做
    // 长距离 blend 瞬移。
    if (walker.entered) walker.exit(); // N2：增益恢复 Menagerie（后续均为 kinematic，4.2 序 9）
    startPose = cur;
    if (startPose) {
      const posDisp = Math.hypot(startPose.pos[0] - standPose.pos[0], startPose.pos[1] - standPose.pos[1]);
      const yawGap = Math.abs(Math.atan2(Math.sin(quatToYaw(standPose.quat) - quatToYaw(startPose.quat)),
        Math.cos(quatToYaw(standPose.quat) - quatToYaw(startPose.quat))));
      if (posDisp > 0.30) {
        errorMsg = `结尾回位水平残差 ${posDisp.toFixed(2)}m 过大（回位行走反复失败），演示暂停于当前位置（不做瞬移回位）`;
        log('[stack] ' + errorMsg);
        phase = 'BLEND_STAND'; phaseT = 0; // 相位仅用于 UI 显示；onStep 检测 errorMsg 不再推进
        return;
      }
      phaseDur = DUR.BLEND_STAND + posDisp / BLEND_POS_RATE + yawGap * BLEND_YAW_RATE;
    } else {
      phaseDur = DUR.BLEND_STAND;
    }
    phase = 'BLEND_STAND'; phaseT = 0;
    log('[stack] 全部槽位完成，回站立');
  }

  // stack 模式契约：平衡外挂完全停用（每子步清零，方案 4.2）
  function zeroQfrc() {
    const nv = model.nv;
    for (let i = 0; i < nv; i++) data.qfrc_applied[i] = 0;
  }

  // 释放瞬间：幽灵体与物理体同子步换位（视觉无缝），物理体交还原 solve。
  // 姿态处理（两点，均实测驱动）：
  //  1) 压平：动捕释放帧箱子略带俯仰/滚转，直接沿用会让箱角插入地面数 cm 触发弹跳
  //     （实测释放后水平漂移 17mm）；人放箱子的终态本来就是平放。
  //  2) 偏航对齐栅格：第 2 层起若沿用轨迹偏航（随站位规则可达 ±131°），旋转后箱体
  //     投影超过槽位栅格间距（0.272 > 0.204），落放角部与邻箱侧面相交互推 20mm。
  //     统一对齐栅格朝向后箱子行列整齐（码垛常态）；箱体近方形（0.39×0.38），
  //     换位瞬间的偏航跳变只有亚厘米级剪影变化。
  function handoverToPhysics(plan, pos) {
    const chi = cfg.stackFrame.psi;
    plan.placedQuat = [Math.cos(chi / 2), 0, 0, Math.sin(chi / 2)];
    // 体原点 = 槽位中心 − 旋转后的 bbox_center_offset：
    // 槽位语义是"geom（几何）中心"，而 boxXmlSnippet 把 geom 以 off 偏置挂在体上，
    // URDF 原点一般不在包围盒中心（plasticbox xy 偏 5.4mm → 实测 Δh 5.5mm 超差）。
    const c = Math.cos(chi), s = Math.sin(chi);
    const [ox, oy, oz] = cfg.box.bbox_center_offset;
    plan.placedPos = plan.contactPlan ? [...pos] :
      [pos[0] - (c * ox - s * oy), pos[1] - (s * ox + c * oy), pos[2] - oz];
    plan.holdUntil = carryT + PLACE_HOLD; // 压持窗口：模拟机器人确认放稳再松爪
    writeBox(plan.ids, plan.placedPos, plan.placedQuat);
    stagedHolds.delete(`P:${cfg.boxType}:${slotIndex}`);
    stagedHolds.set(`G:${cfg.boxType}:${slotIndex}`, { ids: plan.gids, pos: plan.stagePos, quat: STAGING_QUAT });
    writeBox(plan.gids, plan.stagePos, STAGING_QUAT);
    handedOver++;
  }

  function onStep(dt) {
    if (!cfg || (phase === 'idle' && !errorMsg)) {
      zeroQfrc(); // 未激活也兜底清零外挂残差
      return;
    }
    if (errorMsg) {
      if (cur) writeRobot(cur, 0);
      const stoppedPlan = cfg.plans[slotIndex];
      if (stoppedPlan?.gids && !stoppedPlan.executedStrategy) {
        const qa = stoppedPlan.gids.jointQadr;
        const p = held ? [...data.qpos.slice(qa, qa + 3)] : stoppedPlan.spawn.pos;
        const q = held ? [...data.qpos.slice(qa + 3, qa + 7)] : stoppedPlan.spawn.quat;
        writeBox(stoppedPlan.gids, p, q);
      }
      // 明确报错态（瞬移消除改造）：演示暂停，不再推进任何相位；仅保持潜伏箱体
      // （物理潜伏体不保持会持续下坠）并清零外挂。恢复手段 = 页面重置按钮。
      for (const h of stagedHolds.values()) writeBox(h.ids, h.pos, h.quat);
      zeroQfrc();
      return;
    }
    phaseT += dt;
    onStepRL(dt);

    // 未激活箱子按隐藏位 kinematic 保持（否则重力使其持续下坠，数值最终爆炸）
    for (const h of stagedHolds.values()) writeBox(h.ids, h.pos, h.quat);

    // stack 模式契约：平衡外挂完全停用（每子步清零，方案 4.2）
    zeroQfrc();
  }

  // ---------------- rl 路径（方案 4.2 状态机 v2 + 瞬移消除改造） ----------------
  // RL 段：不写机器人 qpos/qvel，只写 ctrl（walker 内部）+ 幽灵箱 kinematic；
  // kinematic 段（操作窗口/转身/恢复）：qpos 直写 + qvel 差分 + ctrl 跟踪。
  function onStepRL(dt) {
    const plan = cfg.plans[slotIndex];
    const rm = plan.rm;

    switch (phase) {
      case 'WALK_APPROACH':
      case 'WALK_CARRY':
      case 'WALK_HOME':
      case 'STOP_PATH': {
        const activeWalkPhase = phase === 'STOP_PATH' ? resumeWalkPhase : phase;
        const r = walker.step(dt);
        if (r.fallen) { handleWalkFailure('跌倒'); break; }
        if (r.done && r.overtime) { handleWalkFailure('超时'); break; }
        if (r.turnYaw != null) {
          resumeWalkPhase = activeWalkPhase;
          cur = captureRobotPose();
          walker.exit();
          beginPivotTurn(cur, r.turnYaw, 'TURN_PATH', '对准路径下一段', activeWalkPhase === 'WALK_CARRY' ? TURN_RATE_CARRY : TURN_RATE);
          break;
        }
        if (r.braking && !r.done) { resumeWalkPhase = activeWalkPhase; phase = 'STOP_PATH'; }
        if (r.done) {
          if (activeWalkPhase === 'WALK_APPROACH') {
            // 行走段结束基座是物理位姿；PREPARE blend 吸收位置/朝向/关节残差（4.2 序 2）。
            // 停点残差经 WALK_STOP_LEAD 已压到 ~0.03m；blend 时长按位移/朝向残差自适应
            //（峰值速度 ≤0.23m/s，N15 blend 行口径：位移 ≤0.15m 且滑移 p95 ≤400mm/s）
            startPose = captureRobotPose();
            cur = startPose;
            const goal0 = clipPoseAt(plan.rm, 0);
            const yawGap = Math.abs(Math.atan2(Math.sin(quatToYaw(goal0.quat) - quatToYaw(startPose.quat)),
              Math.cos(quatToYaw(goal0.quat) - quatToYaw(startPose.quat))));
            const posDisp = Math.hypot(startPose.pos[0] - goal0.pos[0], startPose.pos[1] - goal0.pos[1]);
            if (posDisp > 0.15) { handleWalkFailure('取箱停点偏差过大'); break; }
            if (yawGap > TURN_APPROACH_MIN) {
              pickAlignPose = goal0;
              beginPivotTurn(startPose, quatToYaw(goal0.quat), 'TURN_PICK', '对准料箱');
            } else {
              phaseDur = DUR.PREPARE + posDisp / BLEND_POS_RATE + yawGap * BLEND_YAW_RATE;
              phase = 'PREPARE'; phaseT = 0; prevRobot.pos = null;
            }
          } else if (activeWalkPhase === 'WALK_CARRY') {
            prevPhysPose = captureRobotPose();
            cur = prevPhysPose;
            // 箱 blend 起点一次性捕获：若逐子步锚随动，机器人 blend 的 slerp 会拖着
            // 箱子绕基座快速横扫（0.5m 半径 × ~2rad/s），N10 连续性爆炸。
            // blend 时长按箱-目标距离自适应，限速 0.5m/s（≤5mm/10ms 采样窗）：停站
            // 朝向/位置残差经 0.4m 锚臂放大后 dg 可达 ~1m，旧版 0.4s 硬帽会把箱体
            // 以 ~2.8m/s 强拉到位（实测 N10 单窗 28mm 横向跳变）。
            // 起点取箱的最后写值（限速输出）而非锚位姿：两者有尖峰滞后时用锚会
            // 在相位边界产生同滞后量的跳变（MERO-5）。
            alignBoxFrom = carryBoxLast ?? carryAnchorPose();
            const goalG = clipPoseAt(cfg.plans[slotIndex].rm, cfg.plans[slotIndex].lowerStart);
            const dg = Math.hypot(alignBoxFrom.pos[0] - goalG.objPos[0],
              alignBoxFrom.pos[1] - goalG.objPos[1], alignBoxFrom.pos[2] - goalG.objPos[2]);
            alignBoxDur = Math.max(DUR.BOX_BLEND, Math.min(1.2, dg / 0.5));
            // 机器人 blend 时长按停点残差自适应（峰值 ≤0.23m/s，与箱 blend 取大防相位切换瞬移）
            const yawGapA = Math.abs(Math.atan2(Math.sin(quatToYaw(goalG.quat) - quatToYaw(prevPhysPose.quat)),
              Math.cos(quatToYaw(goalG.quat) - quatToYaw(prevPhysPose.quat))));
            const posDispA = Math.hypot(prevPhysPose.pos[0] - goalG.pos[0], prevPhysPose.pos[1] - goalG.pos[1]);
            phaseDur = Math.max(DUR.ALIGN_PLACE, alignBoxDur, posDispA / BLEND_POS_RATE + yawGapA * BLEND_YAW_RATE);
            phase = 'ALIGN_PLACE'; phaseT = 0;
          } else {
            // 行走段没有 writeRobot（RL 物理自主），cur 停在行走起点——必须在完成时
            // 显式刷新，否则 finishAll 的回站立 blend 会从行走起点位姿出发（旧代码
            // 隐患：基线实测 BLEND_STAND 0.76m/2042mm/s 瞬移段的真身，由新残差守卫暴露）
            cur = captureRobotPose();
            consecutiveHomeFails = 0; // 回位成功：清连续失败计数
            if (slotIndex + 1 < cfg.slots.length) {
              const nextSlot = slotIndex + 1;
              // Settle the empty-hand stance before turning into the next task.
              failedKind = null;
              beginRecover(() => beginSlot(nextSlot));
            }
            else finishAll();
          }
        } else {
          if (activeWalkPhase === 'WALK_APPROACH') {
            writeBox(plan.gids, plan.spawn.pos, plan.spawn.quat); // 幽灵体生成位 kinematic 保持（防重力下落）
          } else if (activeWalkPhase === 'WALK_CARRY' && (boxAnchor || handsAnchor)) {
            followCarryBox(dt);
          }
        }
        break;
      }
      case 'TURN_PATH': {
        writeRobot(pivotTurnPose(), dt);
        if (resumeWalkPhase === 'WALK_APPROACH') writeBox(plan.gids, plan.spawn.pos, plan.spawn.quat);
        else if (resumeWalkPhase === 'WALK_CARRY') followCarryBox(dt);
        if (phaseT >= turnDurCur) {
          cur = captureRobotPose();
          walker.enter();
          walker.resumeAfterTurn(turnDurCur);
          phase = resumeWalkPhase; phaseT = 0; prevRobot.pos = null;
          resumeWalkPhase = null;
        }
        break;
      }
      case 'TURN_PICK': {
        writeRobot(pivotTurnPose(), dt);
        writeBox(plan.gids, plan.spawn.pos, plan.spawn.quat);
        if (phaseT >= turnDurCur) {
          startPose = captureRobotPose(); cur = startPose;
          const d = Math.hypot(startPose.pos[0]-pickAlignPose.pos[0], startPose.pos[1]-pickAlignPose.pos[1]);
          phaseDur = DUR.PREPARE + d / BLEND_POS_RATE;
          phase = 'PREPARE'; phaseT = 0; prevRobot.pos = null;
        }
        break;
      }
      case 'PREPARE': {
        const alpha = smoothstep(Math.min(1, phaseT / (phaseDur || DUR.PREPARE)));
        const goal = clipPoseAt(rm, 0);
        writeRobot(blendPose(startPose, { pos: goal.pos, quat: goal.quat, dof: goal.dof }, alpha), dt);
        writeBox(plan.gids, plan.spawn.pos, plan.spawn.quat); // 幽灵体生成位 kinematic 保持
        if (phaseT >= (phaseDur || DUR.PREPARE)) {
          phase = 'CARRY_REPLAY'; carryT = 0; prevRobot.pos = null;
          if (plan.carryFull) lastWalkPath = manipulationRoute(rm);
        }
        break;
      }
      case 'CARRY_REPLAY': {
        // Contact plans replay the solved approach, grasp, grounded carry and
        // outside-rack lowering. Experimental RL carry uses the original lift segment.
        carryT += dt;
        const f = carryT * rm.fps;
        let pose = clipPoseAt(rm, f);
        if (plan.carryFull && negCarryDrift() > 0) {
          // negcheck 第三组注入的 N14 侧（G1BOX_NEGCHECK_CARRY_DRIFT，仅防假绿消费）：
          // root 走廊漂移（−x、段头 f=0 起漂、release 时到位的连续斜坡）模拟 N14 要防的
          // "λ 压缩/整体平移"类滑移回归——支撑脚获得恒定附加速度 drift/release·fps。
          // 与箱漂移（下方 grasp 起漂、+x）独立反向：手随 root 漂、箱反向漂 → 相对位移
          // 同向叠加 → N13-mocap 必 FAIL；注入物理上就是"坏实现"，不要求自洽。
          // 量级差（阈值联定依据，见 probe-notes MERO-10）：固有滑移 max≈0.30m/s，
          // 注入 1.0m 附加 ≈0.15m/s 均匀分量 → p95/max 越过冻结阈值，两侧余量 ≥40%。
          const k = Math.min(1, f / Math.max(1, rm.release));
          pose = { ...pose, pos: [pose.pos[0] - negCarryDrift() * k, pose.pos[1], pose.pos[2]] };
        }
        writeRobot(pose, dt);
        if (!held && f >= rm.grasp) held = true; // grasp 起幽灵体跟随（含当帧）
        if (held) {
          // Contact plans already match the spawn pose at grasp. Legacy RL clips
          // retain the ramp used to compensate their source floor offset.
          const ramp = plan.contactPlan ? 1 : Math.min(1, (f - rm.grasp) / SINK_RAMP_FRAMES);
          const target = pose.objPos;
          let bp = lerp3(plan.spawn.pos, target, ramp);
          const bq = quatSlerp(plan.spawn.quat, pose.objQuat, ramp);
          // negcheck 第三组注入（G1BOX_NEGCHECK_CARRY_DRIFT，仅防假绿消费）：携带段幽灵箱
          // 按斜坡漂移至注入距离 → N13-mocap 必 FAIL（正常运行为 0，零开销直通）
          const drift = (plan.carryFull && held) ? negCarryDrift() : 0;
          if (drift > 0) {
            const k = Math.min(1, (f - rm.grasp) / Math.max(1, rm.release - rm.grasp));
            bp = [bp[0] + drift * k, bp[1], bp[2]];
          }
          writeBox(plan.gids, bp, bq);
        } else {
          writeBox(plan.gids, plan.spawn.pos, plan.spawn.quat);
        }
        if (plan.carryFull) {
          if (f >= rm.release) {
            // 全程回放段末（方案 4.3）：换位（幽灵→物理，RELEASE_DROP=0）+ 压持 + 回放
            // release..T−1 的既有 RELEASE 机制零改动接手；箱偏航由 handoverToPhysics
            // 对齐栅格（换位合法瞬移，方案 4.3 表内已接受口径）
            plan.executedStrategy = 'mocap';
            const relPose = clipPoseAt(rm, rm.release);
            handoverToPhysics(plan, relPose.objPos);
            phase = 'RELEASE'; // 幽灵体已退场，只回放机器人
          }
          break;
        }
        if (f >= plan.liftEnd) {
          // 段末 live 捕获持箱锚（4.5 / MERO-8：pelvis 系或双手中点系，零跳变进入行走段）；
          // 踏步转身对准「取箱位 → 放置站位」的行进方向后交给 WALK_CARRY（策略只有前进能力）
          captureCarryAnchor(pose.objPos, pose.objQuat);
          const from = captureRobotPose();
          cur = from;
          const goalG = clipPoseAt(rm, plan.lowerStart);
          const bearing = Math.atan2(goalG.pos[1] - from.pos[1], goalG.pos[0] - from.pos[0]);
          beginPivotTurn(from, bearing, 'TURN_CARRY', '朝放置站位', TURN_RATE_CARRY);
        }
        break;
      }
      case 'TURN_CARRY': {
        // 踏步转身（原地枢转+交替抬腿）；箱锚随动：转身时箱子绕锚系同步旋转
        //（抱箱转身，位置/朝向连续；MERO-8 起锚系默认双手中点——手相对身体静止，
        // 与 pelvis 锚位姿等价，边界零跳变）。枢转峰值 0.45rad/s × 锚臂 ~0.3m →
        // 箱切向速度 ~0.14m/s，N10 单采样窗 ≤1.4mm（旧定格转身 2.5m/s 撞穿阈值）。
        writeRobot(pivotTurnPose(), dt);
        const aT = carryAnchorPose();
        writeBox(plan.gids, aT.pos, aT.quat);
        if (phaseT >= turnDurCur) {
          cur = captureRobotPose();
          startRLWalk('carry');
        }
        break;
      }
      case 'TURN_APPROACH': {
        writeRobot(pivotTurnPose(), dt);
        writeBox(plan.gids, plan.spawn.pos, plan.spawn.quat); // 幽灵体生成位 kinematic 保持（防重力下落）
        if (phaseT >= turnDurCur) {
          cur = captureRobotPose();
          startRLWalk('approach');
        }
        break;
      }
      case 'ALIGN_PLACE': {
        // 实测位姿 blend 到 T(clip[lower_start])；箱子从锚随动 blend 回 clip obj 跟随（4.2 序 5）。
        // 相位时长自适应（phaseDur ≥ 箱 blend 时长，见 WALK_CARRY 完成分支）
        const alpha = smoothstep(Math.min(1, phaseT / (phaseDur || DUR.ALIGN_PLACE)));
        const goal = clipPoseAt(rm, plan.lowerStart);
        writeRobot(blendPose(prevPhysPose, { pos: goal.pos, quat: goal.quat, dof: goal.dof }, alpha), dt);
        // 箱从进入时捕获的锚随动位姿独立 lerp 到 clip 跟随位，与机器人的 blend 解耦
        // （连续性由 N10 断言守卫）
        const boxAlpha = smoothstep(Math.min(1, phaseT / (alignBoxDur || DUR.BOX_BLEND)));
        const a = alignBoxFrom ?? carryAnchorPose();
        writeBox(plan.gids,
          lerp3(a.pos, goal.objPos, boxAlpha),
          quatSlerp(a.quat, goal.objQuat, boxAlpha));
        if (phaseT >= (phaseDur || DUR.ALIGN_PLACE)) {
          held = true;
          carryT = plan.lowerStart / rm.fps;
          alignBoxFrom = null;
          phase = 'CARRY_REPLAY2'; prevRobot.pos = null;
        }
        break;
      }
      case 'CARRY_REPLAY2': {
        // 操作窗口 2：下放-释放（帧域 lower_start..release；释放换位/压持机制零改动）
        carryT += dt;
        const f = carryT * rm.fps;
        const pose = clipPoseAt(rm, f);
        writeRobot(pose, dt);
        writeBox(plan.gids, pose.objPos, pose.objQuat);
        if (f >= rm.release) {
          const relPose = clipPoseAt(rm, rm.release);
          plan.executedStrategy = 'rl'; // M-1 策略完整性：rl 策略槽位经 v2 全链换位
          handoverToPhysics(plan, relPose.objPos);
          phase = 'RELEASE'; // 幽灵体已退场，只回放机器人
        }
        break;
      }
      case 'RELEASE': {
        carryT += dt;
        const f = carryT * rm.fps;
        writeRobot(clipPoseAt(rm, f), dt);
        if (carryT < plan.holdUntil) {
          // 压持窗口：物理体按放箱位姿 kinematic 保持（接触零冲击建立）
          writeBox(plan.ids, plan.placedPos, plan.placedQuat);
        } else if (!boxReleased) {
          boxReleased = true;
          log(`[stack] 松爪 @t+${PLACE_HOLD}s，物理体接管`);
        }
        if (f >= rm.T - 1) { phase = 'SETTLE'; phaseT = 0; prevRobot.pos = null; }
        break;
      }
      case 'RECOVER_STAND': {
        // 原地恢复站立（瞬移消除改造）：root xy/朝向保持，z+dof blend 到 stand。
        // 从摔倒等物理位姿起身属明示演示级近似（无起身高保真数据源），连续性由
        // 无头 N17 逐子步跳变断言守门。持箱失败时箱子跟手（限速帽内连续）。
        const alpha = smoothstep(Math.min(1, phaseT / DUR.RECOVER));
        writeRobot(blendPose(startPose, {
          pos: [startPose.pos[0], startPose.pos[1], standPose.pos[2]],
          quat: yawQuat(quatToYaw(startPose.quat)),
          dof: standPose.dof,
        }, alpha), dt);
        if (failedKind === 'carry' && (boxAnchor || handsAnchor)) followCarryBox(dt);
        if (phaseT >= DUR.RECOVER) {
          cur = captureRobotPose();
          const next = recoverNext;
          recoverNext = null;
          if (next) next();
        }
        break;
      }
      case 'SETTLE': {
        carryT += dt; // 供压持窗口计时（RELEASE 段可能不足 PLACE_HOLD，在此延续）
        if (carryT < plan.holdUntil) writeBox(plan.ids, plan.placedPos, plan.placedQuat);
        writeRobot(cur, 0); // 机器人冻结（kinematic），箱子做物理
        if (phaseT >= DUR.SETTLE) {
          // 回位序列（瞬移消除改造）：先原地起身（STAND_UP，MERO-5 机制拆分——释放
          // 蹲姿直接交给 RL 行走会在零指令预热中踉跄失稳，z 必须同步升到 stand 骨架
          // 高度否则脚 ~8cm 压入地面），再踏步转身朝向原点，随后 WALK_HOME 全程前进。
          // 转身目标在 SETTLE 结束时计算（起身不移动水平位置，方位角不变）。
          turnHomeStart = captureRobotPose();
          turnHomeYaw = Math.atan2(standPose.pos[1] - turnHomeStart.pos[1], standPose.pos[0] - turnHomeStart.pos[0]);
          startPose = turnHomeStart;
          cur = turnHomeStart;
          phase = 'STAND_UP'; phaseT = 0; prevRobot.pos = null;
          log('[stack] 原地起身（释放蹲姿→站立，无水平位移）');
        }
        break;
      }
      case 'STAND_UP': {
        const alpha = smoothstep(Math.min(1, phaseT / DUR.STAND_UP));
        writeRobot(blendPose(turnHomeStart, {
          pos: [turnHomeStart.pos[0], turnHomeStart.pos[1], standPose.pos[2]],
          quat: yawQuat(quatToYaw(turnHomeStart.quat)), // 朝向保持不动，旋转全部交给踏步转身
          dof: standPose.dof,
        }, alpha), dt);
        if (phaseT >= DUR.STAND_UP) {
          cur = captureRobotPose();
          beginPivotTurn(cur, turnHomeYaw, 'TURN_HOME', '朝原点');
        }
        break;
      }
      case 'TURN_HOME': {
        writeRobot(pivotTurnPose(), dt);
        if (phaseT >= turnDurCur) {
          cur = captureRobotPose();
          startRLWalk('home');
        }
        break;
      }
      case 'BLEND_STAND': {
        // 位置/关节 blend 到 stand，朝向保持来路方向不动（旋转后置到 TURN_STAND，
        // 见 finishAll——避免 blend 段内出现原地旋转的脚底拖扫）
        const alpha = smoothstep(Math.min(1, phaseT / (phaseDur || DUR.BLEND_STAND)));
        writeRobot(blendPose(startPose, {
          pos: standPose.pos,
          quat: yawQuat(quatToYaw(startPose.quat)),
          dof: standPose.dof,
        }, alpha), dt);
        if (phaseT >= (phaseDur || DUR.BLEND_STAND)) {
          cur = captureRobotPose();
          const yawGap = Math.abs(Math.atan2(Math.sin(quatToYaw(standPose.quat) - quatToYaw(cur.quat)),
            Math.cos(quatToYaw(standPose.quat) - quatToYaw(cur.quat))));
          if (yawGap > TURN_APPROACH_MIN) {
            beginPivotTurn(cur, quatToYaw(standPose.quat), 'TURN_STAND',
              `对齐站立朝向（残差 ${(yawGap * 180 / Math.PI).toFixed(0)}°）`);
          } else {
            phase = 'idle'; done = true;
            log('[stack] 完成：机器人交还 idle（平衡外挂恢复）');
          }
        }
        break;
      }
      case 'TURN_STAND': {
        writeRobot(pivotTurnPose(), dt);
        if (phaseT >= turnDurCur) {
          phase = 'idle'; done = true;
          log('[stack] 完成：机器人交还 idle（平衡外挂恢复）');
        }
        break;
      }
      default:
        break;
    }
  }

  // ---------------- 启停 ----------------
  function start() {
    if (pendingCfg) {
      try {
        cfg = buildConfig(pendingCfg);
        log(`[stack] 新配置生效: ${pendingCfg.layers}层×${pendingCfg.cols}列 ${pendingCfg.boxType}`);
      } catch (e) {
        log(`[stack] 待生效配置无效，沿用旧配置: ${e.message}`);
      }
      pendingCfg = null;
    }
    if (!cfg) throw new Error('stack.start() 前须 loadStackConfig()');

    // 进入 stack：reset 到 stand 关键帧，随后立即重摆箱子（方案 3.4-2 契约）。
    // v2（rl 模式）：摆位初始朝向面向取箱位（yawHome）。T0 实测官方 velocity 策略只有
    // 「前进 + 弱转向」能力（vy 侧移 / vz 原地转 / vx 倒退在站立态均无效，见
    // docs/policy-probe-notes.md），倒退或侧移去取箱位不可行——改为初始摆位一次到位，
    // 全场 RL 段均为前进行走。kinematic 摆位发生在演示装载期（与 reseatBoxes 同惯例）。
    if (standKey >= 0) mujoco.mj_resetDataKeyframe(model, data, standKey);
    mujoco.mj_forward(model, data);
    if (mode === 'rl') {
      const g = cfg.plans[0].clip; // 第一槽 clip 的第 0 帧 root 位姿 = 取箱位
      const goal = clipPoseAt(cfg.plans[0].rm, 0);
      const yawHome = Math.atan2(goal.pos[1], goal.pos[0]); // 从原点指向取箱位的方向
      const yq = [Math.cos(yawHome / 2), 0, 0, Math.sin(yawHome / 2)];
      data.qpos[3] = yq[0]; data.qpos[4] = yq[1]; data.qpos[5] = yq[2]; data.qpos[6] = yq[3];
      data.qvel[3] = 0; data.qvel[4] = 0; data.qvel[5] = 0;
      mujoco.mj_forward(model, data);
      log(`[stack] rl 摆位：初始朝向 ${((yawHome * 180) / Math.PI).toFixed(0)}°（面向取箱位）`);
    }
    standPose = captureRobotPose();
    cur = standPose;
    reseatBoxes();

    done = false;
    carryT = 0;
    // 瞬移消除改造新增状态复位（错误暂停后重新 start 可用）
    errorMsg = null;
    recoverNext = null;
    failedKind = null;
    consecutiveHomeFails = 0;
    walkRetries = 0;
    // 行走避障规划快照/统计按轮复位（N19 degraded 计数口径 = 本轮运行）
    lastWalkObstacles = null;
    lastWalkPath = null;
    pendingApproachWps = null;
    resumeWalkPhase = null; pickAlignPose = null;
    walkPlanStats.total = 0;
    walkPlanStats.degraded = 0;
    for (const k of Object.keys(walkPlanStats.modes)) delete walkPlanStats.modes[k];
    beginSlot(0);
  }

  function stop() {
    phase = 'idle';
    handedOver = 0; held = false; slotIndex = 0;
    lastWalkPath = null; lastWalkObstacles = null;
    done = false;
    errorMsg = null;
    recoverNext = null;
    pendingApproachWps = null;
    if (walker && walker.entered) walker.exit(); // N2 四路径之一：停止必恢复增益
    // 保留 cfg 便于再次 start()；若挂起待生效配置则就地消费
    if (pendingCfg) {
      try { cfg = buildConfig(pendingCfg); } catch (e) { log(`[stack] 待生效配置无效，沿用旧配置: ${e.message}`); }
      pendingCfg = null;
    }
    if (standKey >= 0) {
      mujoco.mj_resetDataKeyframe(model, data, standKey);
      mujoco.mj_forward(model, data);
    }
    reseatBoxes(); // 清场：所有箱子回隐藏位
    log('[stack] 停止并清场');
  }

  function status() {
    const plan = cfg?.plans[slotIndex];
    const placement = plan?.contactPlan?.placement;
    const frame = plan ? carryT * plan.rm.fps : 0;
    let label = PHASE_LABEL[phase] || phase;
    if (placement && phase === 'CARRY_REPLAY' && frame >= placement.feetPlanted) label = '站稳送箱';
    if (placement && phase === 'RELEASE') label = frame < placement.handsWithdrawn ? '松手退手' : frame < placement.retreatEnd ? '后退' : '起身';
    return {
      phase,
      label: errorMsg ? '已暂停（错误）' : (done ? '完成' : label),
      slotIndex,
      handedOver,
      slotTotal: cfg ? cfg.slots.length : 0,
      done,
      mode, // 恒 'rl'（legacy 瞬移降级已移除；字段保留兼容 UI）
      held, // 当前槽箱子是否已抓起（MERO-9：N13 抓取瞬间基线定位用）
      carryStrategy: cfg?.plans[slotIndex]?.strategy ?? null, // 当前槽携带策略 'mocap'|'rl'（MERO-10，状态栏明示）
      error: errorMsg, // 非空 = 演示因不可瞬移恢复的失败暂停（A5 明确报错要求）
      config: cfg ? { layers: cfg.layers, cols: cfg.cols, boxType: cfg.boxType } : null,
      pending: pendingCfg,
    };
  }

  function boxes() {
    if (!cfg) return [];
    const base = boxTypeBase(cfg.boxType);
    const out = [];
    for (let i = 0; i < cfg.slots.length; i++) out.push(boxIds(base + i));
    return out;
  }

  return {
    loadStackConfig, start, stop, onStep, status, boxes, reseatBoxes,
    get active() { return phase !== 'idle'; },
    get config() { return cfg ? { layers: cfg.layers, cols: cfg.cols, boxType: cfg.boxType } : null; },
    get visibleBoxBodies() {
      if (!cfg) return [];
      const out = [];
      const base = boxTypeBase(cfg.boxType);
      for (let i = 0; i < cfg.plans.length; i++) {
        const p = cfg.plans[i];
        if (p.executedStrategy) out.push(`carton${base + i}`);
        else if (i === slotIndex && phase !== 'idle') out.push(`cartonG${base + i}`);
      }
      return out;
    },
    get intakeSlots() { return cfg ? cfg.plans.map(p => ({pos: [...p.spawn.pos], quat: [...p.spawn.quat]})) : []; },
    get currentContacts() { return cfg?.plans[slotIndex]?.contactPlan?.contacts ?? null; },
    get graspOffsets() { return graspOff; }, // lift_end 帧逐手抓握偏移（MERO-9 N13 基准；未捕获时 null）
    // 逐槽计划/执行策略（MERO-10 M-1 策略完整性断言输入）：executed 在换位时刻记录
    // 'mocap'|'rl'，未完成/被跳过槽位为 null（中断/跳槽由此显性化，防静默假绿）
    get slotStrategies() {
      return {
        planned: cfg ? cfg.plans.map((p) => p.strategy) : [],
        executed: cfg ? cfg.plans.map((p) => p.executedStrategy) : [],
      };
    },
    // 操作窗感知的 clip 最小 root z（N8b 阈值镜像输入：各槽实际回放 clip 的最低骨盆高度）
    get manipClipMinZ() {
      if (!cfg) return 1;
      let mn = 1;
      for (const p of cfg.plans) {
        if (p.rm?.rootPos) for (let f = 0; f < p.rm.T; f++) mn = Math.min(mn, p.rm.rootPos[f * 3 + 2]);
      }
      return mn;
    },
    // 最近一次行走段的避障规划快照（walk-avoid-plan 4.6；N19 断言双源之一——同源口径：
    // rects 为全量组装集（含被粗筛裁掉的），goal 为前移后的最终目标，与 walkSeq 一一对应）
    get lastWalkObstacles() { return lastWalkObstacles; },
    // 最近一段规划的原始角点折线（工厂方案 D5 小地图输入；未规划时 null）
    get lastWalkPath() { return lastWalkPath; },
    // 当前槽位的格网中心（工厂方案 D5 activeSlot 数据来源：小地图槽位高亮；未装载/无槽时 null）
    get currentSlot() {
      if (!cfg || phase === 'idle' || done) return null;
      const p = cfg.plans[slotIndex];
      return p ? { index: slotIndex, center: [p.slot.center[0], p.slot.center[1]] } : null;
    },
    // 行走避障规划累计统计（N19 detail 输入：degraded 计数默认链必须 0 + planner mode 分布）
    get walkPlanStats() { return walkPlanStats; },
  };
}
