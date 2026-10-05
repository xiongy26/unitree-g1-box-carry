import { FACTORY } from './factory.js';
// Contact trajectories are solved offline against the final viewer model. Runtime
// only validates and translates them to a column; it never moves the walking root.
export function motionSignature(clip, box, height, yaw) {
  const text = JSON.stringify([clip.root_pos, clip.root_quat, clip.dof_pos,
    clip.obj_pos, clip.obj_quat, clip.grasp_frame, clip.release_frame, box, height, yaw, FACTORY.rack, FACTORY.rackCenter, FACTORY.obstacles]);
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), 16777619);
  return (hash >>> 0).toString(16);
}

export function applyContactPlan(rm, clip, box, slot, yaw, library) {
  const key = `${box.type}:${slot.layer}`;
  const entry = library?.format === 'g1-shelf-contact/1' ? library.plans?.[key] : null;
  if (!entry) throw new Error(`缺少 ${key} 取放接触轨迹，请运行 tools/pipeline/solve_manipulation.py`);
  const signature = motionSignature(clip, box, slot.center[2], yaw);
  if (entry.signature !== signature || entry.sourceT !== rm.T || entry.fps !== rm.fps ||
      entry.grasp !== rm.grasp || entry.sourceRelease !== rm.release) {
    throw new Error(`${key} 取放轨迹已过期，请重新求解（动作、箱型或货架高度已变化）`);
  }
  const sourceT = rm.T;
  if (!Number.isInteger(entry.T) || entry.T < sourceT || entry.T > sourceT + 240 || !Number.isInteger(entry.release) || entry.release < rm.release || entry.release >= entry.T - 1) throw new Error(`${key} 轨迹帧数无效`);
  const finiteRows = (rows, count, width) => Array.isArray(rows) && rows.length === count &&
    rows.every(row => Array.isArray(row) && row.length === width && row.every(Number.isFinite));
  if (!finiteRows(entry.rootPos, entry.T, 3) || !finiteRows(entry.rootQuat, entry.T, 4) || !finiteRows(entry.objPos, entry.T, 3) || !finiteRows(entry.dof, entry.T, rm.nDof) ||
      !entry.rootQuat.every(q => Math.abs(Math.hypot(...q) - 1) < 1e-5) ||
      !finiteRows(entry.contact_points, 2, 3) || !Array.isArray(entry.slot) || entry.slot.length !== 3 || !entry.slot.every(Number.isFinite) ||
      !Array.isArray(entry.objQuat) || entry.objQuat.length !== 4 || !entry.objQuat.every(Number.isFinite) ||
      Math.abs(Math.hypot(...entry.objQuat) - 1) > 1e-5 || !Number.isFinite(entry.playbackFps) || entry.playbackFps <= 0) throw new Error(`${key} 取放轨迹数据损坏`);
  const placement = entry.placement;
  if (placement && (![placement.insertStart, placement.feetPlanted, placement.handsWithdrawn, placement.retreatEnd].every(Number.isInteger) ||
      placement.insertStart < entry.grasp || placement.insertStart >= placement.feetPlanted || placement.feetPlanted >= entry.release ||
      placement.handsWithdrawn <= entry.release || placement.retreatEnd <= placement.handsWithdrawn || placement.retreatEnd >= entry.T)) {
    throw new Error(`${key} 取放阶段数据损坏`);
  }
  rm.T = entry.T;
  rm.release = entry.release;
  rm.rootPos = new Float64Array(rm.T * 3);
  rm.rootQuat = new Float64Array(rm.T * 4);
  rm.objPos = new Float64Array(rm.T * 3);
  rm.objQuat = new Float64Array(rm.T * 4);
  rm.dof = new Float64Array(rm.T * rm.nDof);
  const dx = slot.center[0] - entry.slot[0], dy = slot.center[1] - entry.slot[1];
  for (let t = 0; t < rm.T; t++) {
    rm.objPos.set([entry.objPos[t][0] + dx, entry.objPos[t][1] + dy, entry.objPos[t][2]], t * 3);
    rm.objQuat.set(entry.objQuat, t * 4);
    rm.dof.set(entry.dof[t], t * rm.nDof);
    rm.rootPos.set([entry.rootPos[t][0] + dx, entry.rootPos[t][1] + dy, entry.rootPos[t][2]], t * 3);
    rm.rootQuat.set(entry.rootQuat[t], t * 4);
  }
  rm.fps = entry.playbackFps;
  return { contacts: entry.contact_points, signature, placement: placement ? { ...placement } : null };
}

export function manipulationRoute(rm) {
  const points = [];
  for (let f = rm.grasp; f <= rm.release; f += 4) points.push({ x: rm.rootPos[f * 3], y: rm.rootPos[f * 3 + 1] });
  const r = rm.release * 3;
  points.push({ x: rm.rootPos[r], y: rm.rootPos[r + 1] });
  return { kind: 'carry', points, goal: [rm.rootPos[r], rm.rootPos[r + 1]] };
}
