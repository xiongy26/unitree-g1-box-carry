// 手-箱贴近度探针（MERO-7）：量出「clip 持箱帧」的手-箱面距离基线，
// 并可复现 WALK_CARRY 段实测（配合 --hold 实验）。
//
// 口径：手 = rubber_hand 视觉 geom（wrist_yaw_link 挂点前 41.5mm，mesh 尺寸即手掌）；
// 距离 = mj_geomDistance（手 geom ↔ 幽灵箱 geom），负值表示表面穿透（手指插进箱面）。
//
// 用法：
//   node tools/probes/handbox_probe.mjs            # 只量 clip 持箱帧基线（各箱型各 tier）
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import loadMujoco from '../../vendor/mujoco/mujoco.js';
import { boxXmlSnippet, injectBoxes, remapClipForSlot, MAX_BOXES } from '../../src/stackCore.js';
import { buildJointMap, quatSlerp } from '../../src/util.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

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
      catch { /* 缺文件跳过 */ }
    }
  }
  return clips;
}

const mujoco = await loadMujoco();
const clips = loadClips();
const meta = JSON.parse(readFileSync(path.join(ROOT, 'motions', 'walk_meta.json'), 'utf8'));

const sceneXml = readFileSync(path.join(ROOT, 'model', 'scene.xml'), 'utf8');
const g1Xml = readFileSync(path.join(ROOT, 'model', 'g1.xml'), 'utf8');
const g1Inner = g1Xml.replace(/^[\s\S]*?<mujoco[^>]*>/, '').replace(/<\/mujoco>\s*$/, '');
const merged = sceneXml.replace(/<!-- G1 由 main\.js[\s\S]*?-->/, g1Inner.trim());
let snippets = '';
for (const [type, base] of [['largebox', 0], ['plasticbox', 100]]) {
  const box = clips[type]?.low?.box;
  if (!box) continue;
  for (let i = 0; i < MAX_BOXES; i++) snippets += boxXmlSnippet(base + i, box, [0, 0, -5]) + '\n';
}
const xml = injectBoxes(merged, snippets);
const vfs = new mujoco.MjVFS();
const assetsDir = path.join(ROOT, 'model', 'assets');
for (const f of readdirSync(assetsDir)) {
  if (f.toLowerCase().endsWith('.stl')) vfs.addBuffer('assets/' + f, new Uint8Array(readFileSync(path.join(assetsDir, f))));
}
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);

const mjOBJ = mujoco.mjtObj;
function gid(bodyName, meshName) {
  const bid = mujoco.mj_name2id(model, mjOBJ.mjOBJ_BODY.value, bodyName);
  const gadr = model.body_geomadr[bid], gnum = model.body_geomnum[bid];
  for (let g = gadr; g < gadr + gnum; g++) {
    const mn = mujoco.mj_id2name(model, mjOBJ.mjOBJ_MESH.value, model.geom_dataid[g]);
    if (mn === meshName) return g;
  }
  return -1;
}
const handL = gid('left_wrist_yaw_link', 'left_rubber_hand');
const handR = gid('right_wrist_yaw_link', 'right_rubber_hand');
const boxGid = (k) => mujoco.mj_name2id(model, mjOBJ.mjOBJ_GEOM.value, `cartonG${k}_geom`);
const fromto = new Float64Array(6);
const dist = (g1, g2, cap = 0.2) => mujoco.mj_geomDistance(model, data, g1, g2, cap, fromto);
const gcenter = (g) => [data.geom_xpos[3 * g], data.geom_xpos[3 * g + 1], data.geom_xpos[3 * g + 2]];

// 点到 OBB 箱面的解析 SDF（world 点 → 箱体局部系，q=|local|-half；外距=norm(max(q,0))，
// 内距=min(max q,0)，返回 <0 表示点在箱内即"手插进箱里"的深度）
function obbSdf(p, center, quat, half) {
  const c = [center[0] - p[0], center[1] - p[1], center[2] - p[2]];
  // q⁻¹ ⊗ v（quatRotVec 的逆：conj）
  const q = quat;
  const v = (qr, v2) => {
    const [w, x, y, z] = qr;
    const tx = 2 * (y * v2[2] - z * v2[1]), ty = 2 * (z * v2[0] - x * v2[2]), tz = 2 * (x * v2[1] - y * v2[0]);
    return [v2[0] + w * tx + (y * tz - z * ty), v2[1] + w * ty + (z * tx - x * tz), v2[2] + w * tz + (x * ty - y * tx)];
  };
  const l = v([q[0], -q[1], -q[2], -q[3]], c);
  const d = [Math.abs(l[0]) - half[0], Math.abs(l[1]) - half[1], Math.abs(l[2]) - half[2]];
  const outD = Math.hypot(Math.max(d[0], 0), Math.max(d[1], 0), Math.max(d[2], 0));
  const inD = Math.min(Math.max(d[0], d[1], d[2]), 0);
  return { sdf: outD + inD, local: l };
}

console.log(`geom id: handL=${handL} handR=${handR}`);

function sample(clip, f, objPos, objQuat) {
  const jmap = buildJointMap(mujoco, model, clip.joint_names);
  for (let i = 0; i < 36; i++) data.qpos[i] = 0;
  const rp = clip.root_pos[f], rq = clip.root_quat[f];
  for (let k = 0; k < 3; k++) data.qpos[k] = rp[k];
  for (let k = 0; k < 4; k++) data.qpos[3 + k] = rq[k];
  for (let i = 0; i < clip.joint_names.length; i++) data.qpos[jmap[clip.joint_names[i]].q] = clip.dof_pos[f][i];
  mujoco.mj_forward(model, data);
  const g = boxGid(0);
  const bc = gcenter(g);
  const bx = [model.geom_size[3 * g], model.geom_size[3 * g + 1], model.geom_size[3 * g + 2]];
  // 箱体写入（幽灵体 qpos 直写）
  const gj = mujoco.mj_name2id(model, mjOBJ.mjOBJ_JOINT.value, 'cartonG0_joint');
  const qa = model.jnt_qposadr[gj];
  for (let k = 0; k < 3; k++) data.qpos[qa + k] = objPos[k];
  for (let k = 0; k < 4; k++) data.qpos[qa + 3 + k] = objQuat[k];
  mujoco.mj_forward(model, data);
  const cl = gcenter(handL), cr = gcenter(handR);
  const gl = gcenter(g);
  const sl = obbSdf(cl, gl, objQuat, bx);
  const sr = obbSdf(cr, gl, objQuat, bx);
  return { dL: dist(handL, g), dR: dist(handR, g), sdfL: sl.sdf, sdfR: sr.sdf, cl, cr, gl };
}

console.log('\n== 距离口径自检：箱体挪远后 mj_geomDistance 应给出正距离 ==');
{
  const clip = clips.largebox.low;
  const r = sample(clip, 39, [clip.obj_pos[39][0] + 0.6, clip.obj_pos[39][1], clip.obj_pos[39][2]], clip.obj_quat[39]);
  console.log(`  箱心外移 0.6m: geomL=${(r.dL * 1000).toFixed(1)}mm geomR=${(r.dR * 1000).toFixed(1)}mm sdfL=${(r.sdfL * 1000).toFixed(1)} sdfR=${(r.sdfR * 1000).toFixed(1)}`);
}

console.log('\n== 双腕 body 原点中点（palmMid）→ 箱几何中心距离（lift_end 帧，MERO-8 N12 阈值依据） ==');
{
  const bidL = mujoco.mj_name2id(model, mjOBJ.mjOBJ_BODY.value, 'left_wrist_yaw_link');
  const bidR = mujoco.mj_name2id(model, mjOBJ.mjOBJ_BODY.value, 'right_wrist_yaw_link');
  for (const [type, set] of Object.entries(clips)) {
    for (const [tier, clip] of Object.entries(set)) {
      const m = meta[type]?.[tier];
      if (!m) continue;
      const f = m.lift_end;
      const r = sample(clip, f, clip.obj_pos[f], clip.obj_quat[f]); // 内部 mj_forward 后 data.xpos 即该帧 FK
      const b = r.gl; // 箱 geom 中心
      const pm = [(r.cl[0] + r.cr[0]) / 2, (r.cl[1] + r.cr[1]) / 2, (r.cl[2] + r.cr[2]) / 2];
      // cl/cr 是 rubber_hand geom 中心（近似掌心）；再补一列腕 body 原点中点（N12 口径）
      const wl = [data.xpos[3 * bidL], data.xpos[3 * bidL + 1], data.xpos[3 * bidL + 2]];
      const wr = [data.xpos[3 * bidR], data.xpos[3 * bidR + 1], data.xpos[3 * bidR + 2]];
      const pmw = [(wl[0] + wr[0]) / 2, (wl[1] + wr[1]) / 2, (wl[2] + wr[2]) / 2];
      const dGeom = Math.hypot(pm[0] - b[0], pm[1] - b[1], pm[2] - b[2]);
      const dWrist = Math.hypot(pmw[0] - b[0], pmw[1] - b[1], pmw[2] - b[2]);
      console.log(`${type}/${tier} lift_end f${f}: |箱心−掌心中点|=${(dGeom * 1000).toFixed(1)}mm  |箱心−腕原点中点|=${(dWrist * 1000).toFixed(1)}mm`);
    }
  }
}

console.log('\n== clip 持箱帧手-箱面距离（未重映射的原始 clip 位姿） ==');
for (const [type, set] of Object.entries(clips)) {
  for (const [tier, clip] of Object.entries(set)) {
    const m = meta[type]?.[tier];
    if (!m) continue;
    const frames = [clip.grasp_frame, m.lift_end, Math.round((m.lift_end + m.lower_start) / 2), m.lower_start];
    const rows = frames.map((f) => {
      const r = sample(clip, f, clip.obj_pos[f], clip.obj_quat[f]);
      return `f${f}(${(f / clip.fps).toFixed(2)}s) geomL=${(r.dL * 1000).toFixed(1)} geomR=${(r.dR * 1000).toFixed(1)}`
        + ` | sdfL=${(r.sdfL * 1000).toFixed(1)} sdfR=${(r.sdfR * 1000).toFixed(1)}`;
    });
    console.log(`${type}/${tier} [g=${clip.grasp_frame} lift=${m.lift_end} low=${m.lower_start}]:`);
    for (const r of rows) console.log(`  ${r}mm`);
  }
}