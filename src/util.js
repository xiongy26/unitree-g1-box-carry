// 共享工具：关节索引表、四元数数学、平衡辅助、基座位姿设置、待机姿态。
// 该工程从 g1-sports-demo 摘出，移除了乒乓球相关常量，但保留与上游同名的导出，
// 以便以后需要时复用 g1.xml 的关节顺序与平衡代码。

export function mjName2id(mujoco, model, objType, name) {
  return mujoco.mj_name2id(model, objType, name);
}

// name -> {q: qposadr, d: dofadr, c: actuator adr}
export function buildJointMap(mujoco, model, jointNames) {
  const mjOBJ_JOINT = mujoco.mjtObj.mjOBJ_JOINT.value;
  const mjOBJ_ACTUATOR = mujoco.mjtObj.mjOBJ_ACTUATOR.value;
  const map = {};
  for (const name of jointNames) {
    const jid = mjName2id(mujoco, model, mjOBJ_JOINT, name);
    const aid = mjName2id(mujoco, model, mjOBJ_ACTUATOR, name);
    if (jid < 0 || aid < 0) throw new Error('joint/actuator not found: ' + name);
    map[name] = {
      q: model.jnt_qposadr[jid],
      d: model.jnt_dofadr[jid],
      c: aid,
    };
  }
  return map;
}

export function bodyId(mujoco, model, name) {
  return mjName2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY.value, name);
}
export function jointId(mujoco, model, name) {
  return mjName2id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT.value, name);
}
export function keyId(mujoco, model, name) {
  return mjName2id(mujoco, model, mujoco.mjtObj.mjOBJ_KEY.value, name);
}
export function geomId(mujoco, model, name) {
  return mjName2id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM.value, name);
}

// ---------- 四元数（mujoco wxyz 约定） ----------
export function quatRotVec(q, v) {
  const [w, x, y, z] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}
export function quatConj(q) { return [q[0], -q[1], -q[2], -q[3]]; }
export function quatMul(a, b) {
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}
export function yawQuat(yaw) { return [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)]; }

// 球面插值（wxyz）：码箱子模式的回放过渡段与箱子朝向采样用。
// dot<0 时取短路径；夹角极小退化为 lerp+归一化，避免除零。
export function quatSlerp(a, b, t) {
  let bw = b[0], bx = b[1], by = b[2], bz = b[3];
  let dot = a[0] * bw + a[1] * bx + a[2] * by + a[3] * bz;
  if (dot < 0) { bw = -bw; bx = -bx; by = -by; bz = -bz; dot = -dot; }
  if (dot > 0.9995) {
    const w = a[0] + (bw - a[0]) * t, x = a[1] + (bx - a[1]) * t;
    const y = a[2] + (by - a[2]) * t, z = a[3] + (bz - a[3]) * t;
    const n = Math.hypot(w, x, y, z) || 1;
    return [w / n, x / n, y / n, z / n];
  }
  const th = Math.acos(Math.min(1, dot));
  const s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return [
    a[0] * wa + bw * wb, a[1] * wa + bx * wb,
    a[2] * wa + by * wb, a[3] * wa + bz * wb,
  ];
}

// ---------- 平衡辅助 ----------
// 把 pelvis 锚到 (anchorX, anchorY) 附近，并对倾斜施加恢复力矩。
// 在 29 自由度 G1 上能可靠地保持直立展示。
export function applyBalanceAssist(data, pelvisBid, anchorX, anchorY, strength = 1.0, dofOff = 0) {
  const px = data.xpos[3 * pelvisBid], py = data.xpos[3 * pelvisBid + 1];
  const vx = data.qvel[dofOff], vy = data.qvel[dofOff + 1];
  let fx = 140 * (anchorX - px) - 45 * vx;
  let fy = 140 * (anchorY - py) - 45 * vy;
  const cap = 55 * strength;
  fx = Math.max(-cap, Math.min(cap, fx));
  fy = Math.max(-cap, Math.min(cap, fy));
  data.qfrc_applied[dofOff] = fx;
  data.qfrc_applied[dofOff + 1] = fy;

  // 直立力矩：把世界系重力 (-z) 投影到机体系，与 [0, -1, 0] 的差作为误差
  const q = [data.xquat[4 * pelvisBid], data.xquat[4 * pelvisBid + 1], data.xquat[4 * pelvisBid + 2], data.xquat[4 * pelvisBid + 3]];
  const w = [data.qvel[dofOff + 3], data.qvel[dofOff + 4], data.qvel[dofOff + 5]];
  const gravBody = quatRotVec(quatConj(q), [0, 0, -1]);
  const kTilt = 90 * strength, kDamp = 14 * strength, capT = 22 * strength;
  let tx = kTilt * gravBody[1] - kDamp * w[0];
  let ty = -kTilt * gravBody[0] - kDamp * w[1];
  let tz = -kDamp * w[2] * 0.5;
  tx = Math.max(-capT, Math.min(capT, tx));
  ty = Math.max(-capT, Math.min(capT, ty));
  tz = Math.max(-capT, Math.min(capT, tz));
  data.qfrc_applied[dofOff + 3] = tx;
  data.qfrc_applied[dofOff + 4] = ty;
  data.qfrc_applied[dofOff + 5] = tz;
}

// 把自由基座放到 world 位姿
export function setBasePose(data, x, y, z, yaw, qadr = 0, dadr = 0) {
  data.qpos[qadr] = x; data.qpos[qadr + 1] = y; data.qpos[qadr + 2] = z;
  const q = yawQuat(yaw);
  data.qpos[qadr + 3] = q[0]; data.qpos[qadr + 4] = q[1]; data.qpos[qadr + 5] = q[2]; data.qpos[qadr + 6] = q[3];
  for (let i = 0; i < 6; i++) data.qvel[dadr + i] = 0;
}

// 放松待机姿态（29 执行器）：双臂略前伸，自然站立
export const REST_POSE = [
  -0.1, 0, 0, 0.3, -0.2, 0,
  -0.1, 0, 0, 0.3, -0.2, 0,
  0, 0, 0,
  0.2, 0.2, 0, 1.28, 0, 0, 0,
  0.2, -0.2, 0, 1.28, 0, 0, 0,
];

// 指数平滑目标：cur += (goal-cur) * min(1, dt/tau)
export function smoothInto(cur, goal, dt, tau) {
  const a = Math.min(1, dt / tau);
  for (let i = 0; i < goal.length; i++) cur[i] += (goal[i] - cur[i]) * a;
}

export const G = 9.81;
