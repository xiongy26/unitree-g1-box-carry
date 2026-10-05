// 主程序：加载 MuJoCo WASM 与 G1 模型 → three.js 渲染 → 平衡辅助 + idle/walk demo。
// 该工程从 g1-sports-demo 摘出，移除篮球 / 乒乓球 / 二号机控制器，只保留 G1 显示。
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { MujocoVisualizer } from './visualizer.js';
import {
  buildJointMap, bodyId, jointId, keyId, applyBalanceAssist, REST_POSE,
} from './util.js';
import {
  createBoxStacking, boxXmlSnippet, injectBoxes, MAX_BOXES, boxTypeBase,
} from './stackCore.js';
import { factoryStaticXml, FACTORY_SOLID_BODIES, FACTORY } from './factory.js';
import { createMinimap } from './minimap.js';
import { createRecorder } from './recorder.js';
import { createWalkController, ARMHOLD_DEFAULT } from './controller/walkController.js';
import { loadPolicy } from './controller/g1Policy.js';

const TIMESTEP = 0.002;

// 行走模式：默认 rl（RL 行走驱动水平位移段）。?walk=legacy 整段回放已随瞬移消除
// 改造移除（无真实步态数据源，见 README）——显式传入时打警告并按默认 rl 运行。
const urlWalk = new URLSearchParams(location.search).get('walk');
if (urlWalk === 'legacy') {
  console.warn('[g1-viewer] ?walk=legacy 整段回放模式已移除（无真实步态数据源，见 README），按默认 rl 链运行');
}
const WALK_MODE = 'rl';
// 携带策略（MERO-10 方案 4.6）：?carry=mocap（默认，持箱搬移=真人动捕全程回放）| rl（v2 对照）；
// 'legacy' 显式传入时由 stackCore 映射为 mocap 并打日志（legacy 整段回放已移除）
const urlCarry = new URLSearchParams(location.search).get('carry');
const CARRY_MODE = ['mocap', 'rl', 'legacy'].includes(urlCarry) ? urlCarry : 'mocap';
// softArmHold 开关（MERO-8）：?armhold=soft|off 显式选择；缺省取 ARMHOLD_DEFAULT（门控实验结论）
const urlArmhold = new URLSearchParams(location.search).get('armhold');
const ARMHOLD = (urlArmhold === 'soft' || urlArmhold === 'off') ? urlArmhold : ARMHOLD_DEFAULT;

// 码箱子模式的动作清单（hoi-retarget 管线产物转换件，复跑命令见 README"码箱子模式"）
const MOTION_FILES = {
  largebox: { low: 'carry_low_largebox', mid: 'carry_mid_largebox', high: 'carry_high_largebox' },
  plasticbox: { low: 'carry_low_plasticbox' },
};
// 预注入两套箱型各 4 槽（层×列上限 2×2，P1 两层可用格位），避免运行中重编译
const BOX_INJECT = [['largebox', 0], ['plasticbox', 100]];

// 29 自由度 G1 关节（顺序与 g1.xml 的 actuator 一致）
const JOINT_NAMES = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
];

const $ = (id) => document.getElementById(id);

// ---------------- 加载 ----------------
function showLoad(msg, frac) {
  $('loading-text').textContent = msg;
  if (frac !== undefined) $('loading-bar').style.width = `${Math.round(frac * 100)}%`;
}

async function fetchAssets() {
  // 同时取 scene.xml 与 g1.xml —— g1.xml 的 <worldbody> body 会被原样拼到 scene.xml
  // 里（避免 wasm 里 <include> 找不到 g1.xml 的问题；只把 STL 注入 VFS）。
  const [sceneXml, g1Xml] = await Promise.all([
    fetch('model/scene.xml').then((r) => r.text()),
    fetch('model/g1.xml').then((r) => r.text()),
  ]);
  // 取出 g1.xml 的内层（去掉外层 <mujoco>…</mujoco>）
  const g1Inner = g1Xml.replace(/^[\s\S]*?<mujoco[^>]*>/, '').replace(/<\/mujoco>\s*$/, '');
  // 用 g1Inner 替换 scene.xml 里的占位注释；该注释由 scene.xml 的 <!-- ... --> 提供
  const mergedXml = sceneXml.replace(
    /<!-- G1 由 main\.js[\s\S]*?-->/,
    g1Inner.trim(),
  );
  // 码箱子：预取动作 JSON（管线产物转换件）。任一失败只禁用对应箱型，不影响 idle/walk。
  const clips = {};
  const carryFiles = {}; // MERO-10：全程回放片段平面表（stem → clip，walk_meta.carry 解析后挂载）
  let stackReady = true;
  const walkMeta = await fetch('motions/walk_meta.json', { cache: 'no-store' }).then(r => {
    if (!r.ok) throw new Error(`动作清单加载失败 HTTP ${r.status}`);
    return r.json();
  }).catch(e => { console.error(e); return null; });
  const carryFilesToLoad = [...new Set(Object.values(walkMeta?.carry ?? {})
    .flatMap(tiers => Object.values(tiers).map(entry => entry?.clip))
    .filter(stem => typeof stem === 'string' && stem.length > 0))];
  const manipulation = await fetch('motions/manipulation.json', { cache: 'no-store' }).then(r => {
    if (!r.ok) throw new Error(`取放接触轨迹加载失败 HTTP ${r.status}`);
    return r.json();
  }).catch(e => { console.error(e); return null; });
  await Promise.all(Object.entries(MOTION_FILES).map(async ([type, tiers]) => {
    clips[type] = {};
    await Promise.all(Object.entries(tiers).map(async ([tier, stem]) => {
      try {
        clips[type][tier] = await fetch(`motions/${stem}.json`, { cache: 'no-store' }).then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        });
      } catch (e) {
        console.warn(`[g1-viewer] 动作加载失败 motions/${stem}.json:`, e);
      }
    }));
    if (!clips[type].low) delete clips[type];
  }));
  await Promise.all(carryFilesToLoad.map(async (stem) => {
    try {
      carryFiles[stem] = await fetch(`motions/${stem}.json`, { cache: 'no-store' }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      });
    } catch (e) {
      console.warn(`[g1-viewer] 动作加载失败 motions/${stem}.json:`, e);
    }
  }));
  if (!clips.largebox?.low) stackReady = false;

  // 注入工厂静态片段 + 箱子 XML（幽灵+物理双体 + 机器人 exclude，见 stackCore 注释）。
  // 必须拼到第一个 </worldbody> 前（g1 内层 worldbody 末尾），保持 qpos"机器人优先"布局。
  // 工厂片段（factoryStaticXml，无关节——nq 契约）先拼、箱子后拼，顺序无关；承重板
  // body 经第三参进 exclude（机器人-板接触切断，箱-板保留，见 factory.js 注释）。
  // 工厂片段与 stackReady 无关（idle/walk 模式同场景）。
  const factorySnippets = factoryStaticXml();
  const solidBodies = FACTORY_SOLID_BODIES.map((b) => b.body);
  let xml = mergedXml;
  if (stackReady) {
    let snippets = factorySnippets;
    for (const [type, base] of BOX_INJECT) {
      const box = clips[type].low.box;
      for (let i = 0; i < MAX_BOXES; i++) snippets += boxXmlSnippet(base + i, box, [0, 0, -5]) + '\n';
    }
    xml = injectBoxes(mergedXml, snippets, solidBodies);
  } else {
    xml = injectBoxes(mergedXml, factorySnippets, solidBodies);
    console.warn('[g1-viewer] 码箱子动作数据缺失，箱子未注入，模式按钮将禁用');
  }

  // 从合并后的 XML 解析网格文件名（确保 scene.xml 与 g1.xml 内嵌资源都在内）
  const meshNames = [...xml.matchAll(/<mesh(?:\s+name="[^"]*")?\s+file="([^"]+)"/g)].map((m) => m[1]);
  let done = 0;
  const bufs = await Promise.all(meshNames.map(async (name) => {
    const buf = await (await fetch(`model/assets/${name}`)).arrayBuffer();
    done++;
    showLoad(`下载机器人网格 ${done}/${meshNames.length}`, done / meshNames.length);
    return [name, new Uint8Array(buf)];
  }));
  return { mergedXml: xml, bufs, clips, carryFiles, stackReady, manipulation, walkMeta };
}

// ---------------- 场景 / 灯光 ----------------
function buildThreeScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x919ea5);

  // 工厂方案 D1/S3：场地扩至 16×14m 后阴影相机随之扩大（left/right/top/bottom ±4 → ±10、
  // far 20 → 30、光源拉远），否则大半场无阴影（R8）；第二盏低强度方向光消背光面死黑。
  const sun = new THREE.DirectionalLight(0xfff2df, 1.6);
  sun.position.set(6, -8, 12);
  sun.target.position.set(0, 0, 0.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const cam = sun.shadow.camera;
  cam.left = -10; cam.right = 10; cam.top = 10; cam.bottom = -10; cam.near = 1; cam.far = 30;
  sun.shadow.bias = -0.0004;
  scene.add(sun, sun.target);

  const fill = new THREE.HemisphereLight(0x9fb4cc, 0x2a2c34, 0.55);
  scene.add(fill);

  const fill2 = new THREE.DirectionalLight(0xbfd0e8, 0.35);
  fill2.position.set(-6, 8, 8);
  scene.add(fill2);

  return scene;
}

// ---------------- 主程序 ----------------
async function main() {
  console.log('[g1-viewer] main() 开始');
  showLoad('初始化 MuJoCo WASM（约 10 MB）…', 0.03);
  console.log('[g1-viewer] 调用 loadMujoco()…');
  const mujoco = await loadMujoco();
  console.log('[g1-viewer] loadMujoco() 完成, mujoco 类型:', typeof mujoco, ', 有 MjVFS:', !!mujoco?.MjVFS);

  const { mergedXml, bufs, clips, carryFiles, stackReady, manipulation, walkMeta: assetWalkMeta } = await fetchAssets();
  console.log('[g1-viewer] fetchAssets 完成, mesh 数:', bufs.length, ', merged XML 长度:', mergedXml.length);
  showLoad('编译 MJCF 模型…', 1);

  const vfs = new mujoco.MjVFS();
  for (const [name, buf] of bufs) vfs.addBuffer('assets/' + name, buf);
  showLoad('编译:VFS 就绪', 1);

  let model;
  try {
    showLoad('编译:解析 MJCF…', 1);
    model = mujoco.MjModel.from_xml_string(mergedXml, vfs);
    console.log('[g1-viewer] 模型编译成功 nq=', model.nq, 'nu=', model.nu);
    showLoad('编译:完成', 1);
  } catch (e) {
    throw new Error('模型编译失败: ' + (e && e.message ? e.message : e));
  }
  const data = new mujoco.MjData(model);
  console.log(`[g1-viewer] 模型规模 nq=${model.nq} nu=${model.nu} nbody=${model.nbody} ngeom=${model.ngeom}`);

  // 把机器人放回 "stand" 关键帧（g1.xml 自带）
  const standKey = keyId(mujoco, model, 'stand');
  if (standKey >= 0) mujoco.mj_resetDataKeyframe(model, data, standKey);
  mujoco.mj_forward(model, data);

  const jmap = buildJointMap(mujoco, model, JOINT_NAMES);
  const pelvisBid = bodyId(mujoco, model, 'pelvis');

  // rl 行走装配（方案 4.7）：加载策略权重与帧域元数据，装配行走控制器。
  // 任一失败 → 明确报错并禁用码箱子模式（瞬移消除改造：不再静默降级 legacy 整体
  // 平移回放，A5 要求 console + 页面可见提示；页面提示在点"码箱子"时展示）。
  let walker = null;
  let walkMode = WALK_MODE;
  let walkMeta = null;
  if (stackReady && WALK_MODE === 'rl') {
    try {
      if (!assetWalkMeta) throw new Error('动作清单未加载，无法装配上架任务');
      walkMeta = assetWalkMeta;
      const policy = await loadPolicy('vendor/policy/');
      walker = createWalkController({
        mujoco, model, data, jmap, jointNames: JOINT_NAMES,
        policy,
        armHold: ARMHOLD,
        log: (m) => console.log('[g1-viewer]', m),
      });
      console.log('[g1-viewer] RL 行走装配完成 walk=rl（权重 commit '
        + policy.manifest.commit.slice(0, 8) + '）');
      // MERO-10：walk_meta.carry 声明的全程回放片段挂载到 clips[type].carry[tier]
      // （缺文件时对应 tier 由 stackCore 降级链回落 rl，log + 状态栏明示）
      for (const [type, tiers] of Object.entries(walkMeta.carry ?? {})) {
        if (!clips[type]) continue;
        for (const [tier, entry] of Object.entries(tiers)) {
          const clip = carryFiles[entry?.clip];
          if (clip) {
            clips[type].carry ??= {};
            clips[type].carry[tier] = clip;
          } else {
            console.warn(`[g1-viewer] carrywalk 片段缺失 ${type}/${tier}: motions/${entry?.clip}.json（该 tier 降级 rl）`);
          }
        }
      }
    } catch (e) {
      console.error('[g1-viewer] RL 行走装配失败，码箱子模式禁用'
        + '（legacy 瞬移降级已移除，见 README；页面点"码箱子"会显示提示）:', e);
      walker = null;
      walkMode = 'rl';
      walkMeta = null;
    }
  }

  // 码箱子核心（纯逻辑）：先按 UI 默认参数装载配置（会重摆箱子到取箱区外潜伏位）
  function readStackUI() {
    return {
      layers: parseInt($('layers-sel').value, 10),
      cols: parseInt($('cols-sel').value, 10),
      boxType: $('boxtype-sel').value,
    };
  }
  let stack = null;
  let stackError = '动作数据或行走策略未就绪';
  if (stackReady) {
    try {
      stack = createBoxStacking({
        mujoco, model, data, clips,
        walker, walkMode, carryMode: CARRY_MODE, meta: walkMeta, manipulation,
        log: (m) => console.log('[g1-viewer]', m),
      });
      stack.loadStackConfig(readStackUI());
    } catch (e) {
      stackError = e.message;
      console.warn('[g1-viewer] 码箱子初始化失败:', e);
      stack = null;
    }
  }

  // three.js
  const scene = buildThreeScene();
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  $('app').appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.05, 80);
  camera.up.set(0, 0, 1);
  camera.position.set(5, -2.5, 2.7);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI / 2 - 0.02;
  controls.minDistance = 0.8;
  controls.maxDistance = 24;
  controls.target.set(1.8, 1.1, 0.75);
  controls.update();

  const viz = new MujocoVisualizer(mujoco, model, scene);

  // ---------------- 状态 ----------------
  let mode = 'idle';        // 'idle' | 'walk'
  let paused = false;
  let speed = 1;
  let walkPhase = 0;        // walk demo 周期相位
  const PELVIS_ANCHOR = [0, 0];

  function applyCtrl() {
    if (mode === 'stack') return; // stack 模式由 stack.onStep 接管 ctrl 与 qfrc_applied
    if (mode === 'idle') {
      // 放松站立：29 个执行器目标恒为 REST_POSE
      for (let i = 0; i < 29; i++) data.ctrl[i] = REST_POSE[i];
    } else {
      // 前倾 demo：左右腿交替前摆 + 周期地把 pelvis 向前轻推一下再复位
      const t = data.time;
      // 关节目标在 REST_POSE 基础上加小幅摆动
      for (let i = 0; i < 29; i++) {
        const base = REST_POSE[i];
        let off = 0;
        if (i < 6) {        // 左腿
          off = 0.18 * Math.sin(2 * t);
        } else if (i < 12) { // 右腿
          off = 0.18 * Math.sin(2 * t + Math.PI);
        } else if (i >= 15 && i < 22) { // 左臂
          off = 0.10 * Math.sin(2 * t + Math.PI);
        } else if (i >= 22) {            // 右臂
          off = 0.10 * Math.sin(2 * t);
        }
        data.ctrl[i] = base + off;
      }
    }
    // 无论哪种模式，都施加水平回位 + 直立力矩（演示外挂，保证不跌倒）
    // （stack 模式已提前 return：外挂停用是码箱子的硬契约，见方案 4.2）
    applyBalanceAssist(data, pelvisBid, PELVIS_ANCHOR[0], PELVIS_ANCHOR[1], 1.0, 0);
  }

  function reset() {
    // 重置前先收片：清场画面没有保存价值，录了就存、没录无副作用
    // （reset 只能经 UI/调试钩子触发，此时 recorder 已初始化）
    if (recorder.active) recorder.stop();
    if (standKey >= 0) mujoco.mj_resetDataKeyframe(model, data, standKey);
    mujoco.mj_forward(model, data);
    // 重置 = 清箱（全部回潜伏位）+ 堆垛状态归零；keyframe reset 后必须显式重摆箱子
    // （方案 3.4-2 契约：reset 会把箱子 qpos 零化）
    if (stack) stack.stop();
  }

  // keepStack：切出 stack 时保留箱子现场。仅堆垛完成自动交还 idle 时使用——
  // 此时 stack 内部 phase 已是 'idle'，无需 stop()，清场反而会把成品垛搬回潜伏位。
  // 手动切换（按钮/快捷键）不带此参数，仍走 stop() 清场。
  function setMode(m, { keepStack = false } = {}) {
    if (m === 'stack') {
      if (!stack) {
        $('hint').textContent = `料箱上架不可用：${stackError}`;
        return;
      }
      try { stack.start(); } catch (e) {
        $('hint').textContent = `料箱上架不可用：${e.message}`;
        console.error(e);
        return;
      }
    } else if (mode === 'stack' && !keepStack) {
      stack.stop(); // 离开堆垛：清场 + 回 stand
    }
    mode = m;
    $('mode-idle').classList.toggle('active', m === 'idle');
    $('mode-walk').classList.toggle('active', m === 'walk');
    $('mode-stack').classList.toggle('active', m === 'stack');
    $('hint').textContent = m === 'idle'
      ? 'G1 在原地放松站立，平衡辅助保持直立（按下 "料箱上架" 看搬运入架）'
      : m === 'walk'
        ? '演示模式：双腿 / 双臂小幅摆动 + 平衡辅助把它轻轻拉回原点'
        : '按槽位依次取料 → 搬运 → 上架 → 返回。转向前停稳，料箱箱口朝上；层数、列数和箱型在下一轮生效。';
  }

  function setCameraPreset(btn) {
    const [px, py, pz] = btn.dataset.pos.split(',').map(Number);
    const [tx, ty, tz] = btn.dataset.target.split(',').map(Number);
    camera.position.set(px, py, pz);
    controls.target.set(tx, ty, tz);
    controls.update();
  }

  // ---------------- 固定角落实景小地图（工厂方案 D5/S5） ----------------
  // 数据层 src/minimapModel.js（纯函数，Node 可测 F5）+ DOM 渲染层 src/minimap.js：
  // 静态层离屏缓存一次绘制，动态层 12Hz 节流（update 相位判断，与主循环解耦不插帧）。
  // getInput 只读 qpos/快照（零每帧分配压力；R10）。
  const quatYawOf = (q) => Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
  const minimap = createMinimap({
    factory: FACTORY,
    mountEl: $('minimap'),
    getInput: () => {
      const q = data.qpos;
      const boxes = [];
      let activeSlot = null;
      if (stack) {
        const st = stack.status();
        const conf = stack.config;
        const box = conf ? clips[conf.boxType]?.low?.box : null;
        if (box) {
          const [hx, hy] = box.half_size;
          // 已放置物理箱（qpos 直读）
          const ids = stack.boxes();
          for (let i = 0; i < st.handedOver && i < ids.length; i++) {
            const a = ids[i].jointQadr;
            boxes.push({ x: q[a], y: q[a + 1], yaw: quatYawOf([q[a + 3], q[a + 4], q[a + 5], q[a + 6]]), hx, hy, kind: 'placed' });
          }
          // 当前槽幽灵箱（生成/跟随位）
          if (stack.active && st.slotTotal > 0 && !st.done && st.slotIndex >= st.handedOver) {
            const jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value,
              `cartonG${boxTypeBase(conf.boxType) + st.slotIndex}_joint`);
            if (jid >= 0) {
              const a = model.jnt_qposadr[jid];
              boxes.push({ x: q[a], y: q[a + 1], yaw: quatYawOf([q[a + 3], q[a + 4], q[a + 5], q[a + 6]]), hx, hy, kind: 'ghost' });
            }
          }
        }
        activeSlot = stack.currentSlot;
      }
      return {
        robot: { x: q[0], y: q[1], yaw: quatYawOf([q[3], q[4], q[5], q[6]]) },
        path: stack ? stack.lastWalkPath : null,
        boxes,
        activeSlot,
      };
    },
  });
  $('minimap-zoom').onclick = () => {
    $('minimap-zoom').textContent = minimap.toggleView() ? '作业区' : '全厂';
    minimap.update();
  };
  $('minimap-btn').onclick = () => {
    $('minimap-btn').classList.toggle('active', minimap.toggle());
  };

  // ---------------- 页面内录制（干净合成画面：3D 全幅 + 右下角小地图） ----------------
  // 合成与编码在 src/recorder.js；此处只接线画布、配置与开关。
  // 自动收尾观察在主循环 stepFrame 末尾（完成/失败后留尾巴再停）。
  const REC_TAIL_MS = 2500; // 任务完成/失败后继续录的收尾时长（成品垛完整入镜）
  let recAutoStopAt = -1;   // >0 表示已布防的自动停止时刻（performance.now() 毫秒）
  let recWasFinished = false; // 上一帧的完成/失败态（只在转变时刻布防，防止旧状态误触发）
  const recorder = createRecorder({
    getViewCanvas: () => renderer.domElement,
    // 小地图面板整体入镜（底板/边框/标题/视图钮/图例），cssSize 须与 index.html 一致
    getOverlayInfo: () => ({
      canvas: $('minimap'),
      cssSize: minimap.cssSize,
      viewLabel: $('minimap-zoom').textContent,
    }),
    getConfig: () => (stack && stack.config) || null,
    button: $('record-btn'),
    hintEl: $('hint'),
  });
  $('record-btn').onclick = () => recorder.toggle();

  // ---------------- UI ----------------
  $('mode-idle').onclick = () => setMode('idle');
  $('mode-walk').onclick = () => setMode('walk');
  $('mode-stack').onclick = () => setMode('stack');
  $('reset-btn').onclick = () => reset();
  $('pause-btn').onclick = () => {
    paused = !paused;
    $('pause-btn').textContent = paused ? '继续' : '暂停';
  };
  $('speed-sel').onchange = (e) => { speed = parseFloat(e.target.value); };
  // 堆垛参数：下一轮循环生效（进行中的堆垛不被打断）
  for (const id of ['layers-sel', 'cols-sel', 'boxtype-sel']) {
    $(id).onchange = () => {
      if (!stack) return;
      try {
        stack.loadStackConfig(readStackUI());
      } catch (e) {
        $('hint').textContent = `参数无效：${e.message}`;
      }
    };
  }
  for (const b of document.querySelectorAll('button.cam')) b.onclick = () => setCameraPreset(b);

  window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); $('pause-btn').click(); }
    if (e.key === 'r' || e.key === 'R') reset();
    if (e.key === 'v' || e.key === 'V') recorder.toggle();
    if (e.key === '1') setMode('idle');
    if (e.key === '2') setMode('walk');
    if (e.key === '3') setMode('stack');
  });

  // ---------------- 主循环 ----------------
  let acc = 0, last = -1, fpsT = 0, fpsN = 0;

  function stepFrame(dt) {
    if (!paused) {
      acc += dt * speed;
      const maxN = Math.min(500, Math.ceil(dt * speed / TIMESTEP) + 8);
      let n = 0;
      while (acc >= TIMESTEP && n < maxN) {
        applyCtrl();
        if (mode === 'stack' && stack) stack.onStep(TIMESTEP); // mj_step 前写入本子步位姿
        mujoco.mj_step(model, data);
        acc -= TIMESTEP;
        n++;
      }
      if (n >= maxN) acc = 0;
      // 堆垛完成 → 交还 idle（平衡外挂恢复）；keepStack 保留成品堆垛，按 R 清场
      if (mode === 'stack' && stack && stack.status().done) setMode('idle', { keepStack: true });
    }
    updateStackStatus();
    viz.visibleBoxBodies = new Set(stack?.visibleBoxBodies ?? []);
    viz.update(data);
    // 录制时 force 刷新小地图：绕过 12Hz 节流与面板隐藏直返，保证合成帧里的小地图最新
    minimap.update(performance.now(), recorder.active);
    controls.update();
    renderer.render(scene, camera);
    recorder.frame(); // 未录制时零开销直返；录制中合成 3D 画面 + 小地图
    // 自动收尾：任务完成/失败的"转变帧"布防，留 REC_TAIL_MS 尾巴后停（此前无转变不触发）。
    // 上一轮的旧 done 不会误触发——reset 后 recWasFinished 随 status 归位。
    const stNow = stack ? stack.status() : null;
    const finished = !!(stNow && (stNow.done || stNow.error));
    if (recorder.active) {
      if (finished && !recWasFinished && recAutoStopAt < 0) {
        recAutoStopAt = performance.now() + REC_TAIL_MS;
      }
      if (recAutoStopAt > 0 && performance.now() >= recAutoStopAt) {
        recAutoStopAt = -1;
        recorder.stop();
      }
    } else {
      recAutoStopAt = -1;
    }
    recWasFinished = finished;
  }

  function updateStackStatus() {
    if (!stack) return;
    const el = $('stack-status');
    if (mode !== 'stack') { el.textContent = '—'; return; }
    const st = stack.status();
    // 行走/搬移驱动明示（方案 4.6；瞬移消除改造后 mode 恒 rl，字段保留防御性显示）：
    // MERO-10 起逐槽显示携带策略（mocap=真人动捕回放）
    const driveTag = st.mode !== 'rl'
      ? ' · 回放'
      : st.carryStrategy === 'mocap' ? ' · 接触重定向' : '';
    el.textContent = st.error
      ? `已暂停：${st.error}（按 R 重置）`
      : st.done
        ? `完成 ${st.slotTotal} 槽${driveTag}`
        : st.slotTotal > 0
          ? `槽位 ${Math.min(st.slotIndex + 1, st.slotTotal)}/${st.slotTotal} · ${st.label}${driveTag}`
          : st.label;
  }

  function frame(now) {
    try {
      if (last < 0) last = now;
      const dtRaw = Math.max(0, (now - last) / 1000);
      last = now;
      stepFrame(Math.min(0.5, dtRaw));
      fpsT += dtRaw; fpsN++;
      if (fpsT > 0.5) {
        $('fps').textContent = `${Math.round(fpsN / fpsT)} FPS`;
        $('simtime').textContent = `${data.time.toFixed(1)}s`;
        fpsT = 0; fpsN = 0;
      }
    } catch (err) {
      console.error('frame error:', err);
      $('fps').textContent = '错误: ' + (err && err.message ? err.message : err);
    }
  }

  // 双驱动：rAF + 定时器兜底（页面后台 / 无头时仍能继续推进仿真）
  let lastRaf = -1e9;
  function driverRaf(now) { lastRaf = performance.now(); frame(now); requestAnimationFrame(driverRaf); }
  requestAnimationFrame(driverRaf);
  setInterval(() => {
    if (performance.now() - lastRaf > 250) frame(performance.now());
  }, 15);

  setMode('idle');
  $('loading').classList.add('done');

  // 调试接口：window.__sim.drive(秒) 同步推进（无头自检用）
  window.__sim = {
    mujoco, model, data, viz, camera, controls, scene, renderer,
    setMode, reset, recorder,
    get mode() { return mode; },
    get stack() { return stack; },
    drive(secs = 0.1, dt = 1 / 60, render = true) {
      const n = Math.max(1, Math.round(secs / dt));
      for (let i = 0; i < n; i++) {
        if (!paused) {
          acc += dt * speed;
          const maxN = Math.min(500, Math.ceil(dt * speed / TIMESTEP) + 8);
          let k = 0;
          while (acc >= TIMESTEP && k < maxN) {
            applyCtrl();
            if (mode === 'stack' && stack) stack.onStep(TIMESTEP);
            mujoco.mj_step(model, data);
            acc -= TIMESTEP;
            k++;
          }
          if (k >= maxN) acc = 0;
        }
        if (render) viz.update(data);
      }
      if (render) { minimap.update(performance.now() + 100); controls.update(); renderer.render(scene, camera); }
      return data.time;
    },
  };
}

main().catch((err) => {
  console.error('[g1-viewer] main() 抛出:', err);
  const txt = $('loading-text');
  if (txt) {
    txt.textContent = '加载失败：' + (err && err.message ? err.message : err);
    txt.style.color = '#ff8080';
  }
  const sp = document.querySelector('.spinner');
  if (sp) sp.style.display = 'none';
});