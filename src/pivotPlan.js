// Kinematic step-turn: world-fixed stance feet, alternating yawed footfalls.
const smooth = u => { u = Math.max(0, Math.min(1, u)); return u * u * (3 - 2 * u); };
const yawQuat = a => [Math.cos(a / 2), 0, 0, Math.sin(a / 2)];
const yawMatrix = a => [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0, 0, 0, 1];
function slerp(a, b, u) {
  let dot = a.reduce((v, x, i) => v + x * b[i], 0);
  if (dot < 0) { b = b.map(x => -x); dot = -dot; }
  const angle = Math.acos(Math.min(1, dot));
  const weights = angle < 1e-5 ? [1 - u, u] : [Math.sin((1 - u) * angle) / Math.sin(angle), Math.sin(u * angle) / Math.sin(angle)];
  const q = a.map((x, i) => x * weights[0] + b[i] * weights[1]);
  const norm = Math.hypot(...q); return q.map(x => x / norm);
}
function linearSolve(matrix, rhs) {
  const rows = matrix.map((row, i) => [...row, rhs[i]]);
  for (let k = 0; k < rhs.length; k++) {
    let pivot = k;
    for (let i = k + 1; i < rhs.length; i++) if (Math.abs(rows[i][k]) > Math.abs(rows[pivot][k])) pivot = i;
    [rows[k], rows[pivot]] = [rows[pivot], rows[k]];
    const scale = rows[k][k]; if (Math.abs(scale) < 1e-12) return rhs.map(() => 0);
    for (let j = k; j <= rhs.length; j++) rows[k][j] /= scale;
    for (let i = 0; i < rhs.length; i++) if (i !== k) {
      const factor = rows[i][k];
      for (let j = k; j <= rhs.length; j++) rows[i][j] -= factor * rows[k][j];
    }
  }
  return rows.map(row => row[rhs.length]);
}
export function buildPivotPlan({ mujoco, model, data, jmap, jointNames, from, targetYaw, rate = .5 }) {
  const yaw0 = Math.atan2(2 * (from.quat[0] * from.quat[3] + from.quat[1] * from.quat[2]), 1 - 2 * (from.quat[2] ** 2 + from.quat[3] ** 2));
  const angle = Math.atan2(Math.sin(targetYaw - yaw0), Math.cos(targetYaw - yaw0));
  const stepTime = .5, settleTime = .6, fps = 40;
  const count = Math.max(2, Math.ceil(Math.abs(angle) / Math.min(.24, rate * stepTime)));
  const duration = settleTime + (count + 1) * stepTime;
  const qsave = Float64Array.from(data.qpos), vsave = Float64Array.from(data.qvel);
  const feet = ['left', 'right'].map(side => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, `${side}_ankle_roll_link`));
  const jointIds = jointNames.slice(0, 12).map(n => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n));
  const ankleHeight = feet.map(b => {
    let bottom = Infinity;
    for (let g = model.body_geomadr[b]; g < model.body_geomadr[b] + model.body_geomnum[b]; g++) {
      if (model.geom_type[g] === 2) bottom = Math.min(bottom, model.geom_pos[g * 3 + 2] - model.geom_size[g * 3]);
    }
    return .002 - bottom;
  });
  const poses = []; let worstPosition = 0;
  try {
    data.qpos.set(from.pos, 0); data.qpos.set(from.quat, 3);
    for (let i = 0; i < jointNames.length; i++) data.qpos[jmap[jointNames[i]].q] = from.dof[i];
    mujoco.mj_kinematics(model, data);
    const anchors = feet.map((b, side) => ({ pos: Array.from(data.xpos.slice(b * 3, b * 3 + 3)), yaw: Math.atan2(data.xmat[b * 9 + 3], data.xmat[b * 9]), side }));
    const initial = anchors.map(a => ({ ...a, pos: [...a.pos] }));
    let activeStep = -1, stepFrom, stepTo;
    const firstSide = angle >= 0 ? 0 : 1;
    let previous = [...from.dof];
    for (let frame = 0; frame <= Math.ceil(duration * fps); frame++) {
      const time = Math.min(duration, frame / fps), settle = smooth(time / settleTime);
      const elapsed = Math.max(0, time - settleTime);
      const k = Math.min(count, Math.floor(elapsed / stepTime));
      const u = Math.min(1, (elapsed - k * stepTime) / stepTime);
      const moving = (firstSide + k) % 2, support = 1 - moving;
      if (time >= settleTime && k !== activeStep) {
        if (activeStep >= 0) anchors[(firstSide + activeStep) % 2] = stepTo;
        activeStep = k; stepFrom = { ...anchors[moving], pos: [...anchors[moving].pos] };
        const landingYaw = yaw0 + angle * Math.min(count, k + 1) / count;
        const sideOffset = moving === 0 ? .105 : -.105;
        stepTo = { side: moving, yaw: landingYaw, pos: [from.pos[0] - Math.sin(landingYaw) * sideOffset, from.pos[1] + Math.cos(landingYaw) * sideOffset, ankleHeight[moving]] };
      }
      const progress = k < count ? (k + smooth((u - .15) / .7)) / count : 1;
      const heading = yaw0 + angle * progress;
      const targets = anchors.map((a, side) => ({ yaw: a.yaw, pos: [...a.pos] }));
      if (time < settleTime) {
        for (let side = 0; side < 2; side++) {
          targets[side].pos[2] = initial[side].pos[2] * (1 - settle) + ankleHeight[side] * settle;
          targets[side].yaw = initial[side].yaw + Math.atan2(Math.sin(yaw0 - initial[side].yaw), Math.cos(yaw0 - initial[side].yaw)) * settle;
          anchors[side] = { ...targets[side], pos: [...targets[side].pos], side };
        }
      } else {
        const swing = smooth(u);
        targets[moving] = { yaw: stepFrom.yaw + Math.atan2(Math.sin(stepTo.yaw - stepFrom.yaw), Math.cos(stepTo.yaw - stepFrom.yaw)) * swing,
          pos: stepFrom.pos.map((v, i) => v * (1 - swing) + stepTo.pos[i] * swing) };
        targets[moving].pos[2] += .038 * Math.sin(Math.PI * u) ** 2;
      }
      const sway = time >= settleTime ? .15 * Math.sin(Math.PI * u) ** 2 : 0;
      const pos = [from.pos[0] + (anchors[support].pos[0] - from.pos[0]) * sway,
        from.pos[1] + (anchors[support].pos[1] - from.pos[1]) * sway, from.pos[2] * (1 - settle) + .75 * settle];
      const quat = time < settleTime ? slerp(from.quat, yawQuat(yaw0), settle) : yawQuat(heading);
      const dof = [...previous];
      // A straight knee is an IK singularity; seed the forward bending branch.
      if (frame > 0) for (let side = 0; side < 2; side++) {
        const i = side * 6;
        if (dof[i + 3] < .1) { dof[i] -= .025; dof[i + 3] += .05; dof[i + 4] -= .025; }
      }
      for (let i = 12; i < dof.length; i++) dof[i] = from.dof[i];
      data.qpos.set(pos, 0); data.qpos.set(quat, 3);
      for (let i = 0; i < jointNames.length; i++) data.qpos[jmap[jointNames[i]].q] = dof[i];
      for (let side = 0; side < 2; side++) {
        const bid = feet[side], targetR = yawMatrix(targets[side].yaw), indices = Array.from({ length: 6 }, (_, j) => side * 6 + j);
        const residual = () => {
          mujoco.mj_kinematics(model, data);
          const result = targets[side].pos.map((v, i) => data.xpos[bid * 3 + i] - v);
          for (let i = 0; i < 9; i++) result.push((data.xmat[bid * 9 + i] - targetR[i]) * .1);
          return result;
        };
        for (let iteration = 0; iteration < 12; iteration++) {
          const error = residual();
          if (Math.hypot(...error) < .00015) break;
          const jac = indices.map(i => {
            const address = jmap[jointNames[i]].q, value = data.qpos[address];
            data.qpos[address] = value + 1e-4; const perturbed = residual(); data.qpos[address] = value;
            return error.map((v, r) => (perturbed[r] - v) / 1e-4);
          });
          const matrix = jac.map((a, i) => jac.map((b, j) => a.reduce((s, v, r) => s + v * b[r], i === j ? .00002 : 0)));
          const gradient = jac.map(a => -a.reduce((s, v, r) => s + v * error[r], 0));
          const delta = linearSolve(matrix, gradient);
          for (let j = 0; j < 6; j++) {
            const i = indices[j], jid = jointIds[i];
            const lower = Math.max(model.jnt_range[jid * 2], previous[i] - 6 / fps);
            const upper = Math.min(model.jnt_range[jid * 2 + 1], previous[i] + 6 / fps);
            dof[i] = Math.max(lower, Math.min(upper, dof[i] + Math.max(-.18, Math.min(.18, delta[j]))));
            data.qpos[jmap[jointNames[i]].q] = dof[i];
          }
        }
        const error = residual(); worstPosition = Math.max(worstPosition, Math.hypot(...error.slice(0, 3)));
      }
      poses.push({ pos, quat, dof }); previous = dof;
    }
  } finally {
    data.qpos.set(qsave); data.qvel.set(vsave); mujoco.mj_kinematics(model, data);
  }
  return { poses, fps, duration, steps: count + 1, worstPosition };
}
export function samplePivotPlan(plan, time) {
  const f = Math.min(plan.poses.length - 1, Math.max(0, time * plan.fps));
  const i = Math.min(plan.poses.length - 2, Math.floor(f)), a = f - i;
  const p = plan.poses[i], n = plan.poses[i + 1];
  return { pos: p.pos.map((v, j) => v + (n.pos[j] - v) * a), quat: slerp(p.quat, n.quat, a), dof: p.dof.map((v, j) => v + (n.dof[j] - v) * a) };
}
