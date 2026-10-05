// 行走段 2D 避障路径规划（纯函数、零依赖、Node/浏览器双端 ESM；无 DOM/three/process 依赖）。
//
// 【算法蓝本与出处（许可与出处，勿删）】AtsushiSakai/PythonRobotics
// `PathPlanning/VisibilityRoadMap`（MIT License）结构移植 + 本项目语境改写：
// 解析膨胀几何 → 可见性图建图 → Dijkstra 搜索 → 角点过渡平滑。不引入任何运行时依赖、
// 不 vendor 第三方文件（复用/借鉴/自研边界见 docs/walk-avoid-plan.md 5.2；MIT 允许移植，
// 出处在此注明）。角点过渡思想（非代码移植）参考 Nav2 Regulated Pure Pursuit 的
// 拐角插过渡点/最大航向变化检查。
//
// 【设计依据】docs/walk-avoid-plan.md 4.4（规划器流程逐条对应）/ 4.3.3（失败语义三级阶梯）
// / 4.6（接口契约）。本模块只吃矩形列表，不知道箱子/槽位/业务——障碍集组装方是
// stackCore.collectWalkObstacles（4.1 层职责切分：规划是任务层关切，与 RL 控制域无关）。
//
// 确定性契约：全部纯函数、无随机、无时钟，同输入同输出（无头断言可复现；Node/浏览器
// 同一 ESM 实现，IEEE 基本函数双端逐位一致——headless M1 自测含两次调用逐位一致断言）。
//
// rect 契约：{ cx, cy, yaw, hx, hy, r?, role?, key? }——hx/hy 为原始半边长；
// r 为 per-obstacle 膨胀半径覆盖（缺省 robotR+margin；target 箱缺省取 targetShrinkR，
// 组装方也可显式传 r 覆盖）。
//
// 失败语义（4.3.3 三级阶梯；调用方 stackCore 负责明示日志与 degraded 计数）：
//   1 常规：全膨胀（robotR+margin）可见性图求路径，成功即用；
//   2 预期内贴近机制（不算降级）：per-obstacle 收缩半径（target 箱）+ 起点/终点落入膨胀
//     矩形的"入射边透明化" + 起点脱出路点（最近膨胀边界外 0.05m 径向）；
//   3 真降级：全膨胀无可行路径 → 全障碍收缩一轮（r_min=targetShrinkR）再求；仍无可行 →
//     直线兜底（mode='straight-degraded'，degraded=true）。收缩一轮成功时 degraded 仍为
//     false（按方案不计入降级计数），但以 shrunk:true 显性化供调用方打日志——绝不静默。

export const PLANNER_DEFAULTS = {
  robotR: 0.30,         // 机器人等效半径（N19b 阈 0.35 = 0.30 + 0.05 的定标基准，方案 4.5.1）
  margin: 0.10,         // 膨胀余量：覆盖物理滑移 ≤5mm / 幽灵外推 20mm / yaw 对齐跳变（4.2/R5）
  targetShrinkR: 0.12,  // 目标箱收缩膨胀半径（取箱站位设计性贴近 0.139 的贴近机制，4.3.3-2）
  cornerOffset: 0.30,   // 角点过渡点偏移（沿前后边各取，不超过邻段长一半）
  turnMax: Math.PI / 6, // 平滑触发转角阈（30°）。【P2 第二轮实测记录】曾试 15°（让 17-27°
  // 中小转角也走过渡弦以适配 RL 转弯能力）——弦端点会落入 target 膨胀层触发整链回退、
  // 保留原角点，且配对试验实测无净收益，按最优实测态回滚 30°；RL 可跟转角上限 ≈17°
  //（pure pursuit，wz≤0.2 / vx=0.3 → 最小转弯半径 ≈1.5m）的约束记录在案，绕行走廊
  // 布局须使全部转角 ≤17°（pillar_1 联定依据，见 factory.js）。
  maxNodes: 256,        // 节点预算（2 + 4×矩形数；工厂方案 D4 起矩形最坏 ~32（静态 14 +
                        // staging 12 + placed 6 + target 1）→ 节点 ~130，128 会静默截断
                        // 把可行问题变假降级，故扩容；超限仍截断保底）
  escapeMargin: 0.05,   // 脱出路点越过膨胀边界的外距（4.4 步 4）
  pruneMargin: 0.6,     // 走廊 AABB 粗筛外距（潜伏行 y≥2.4 在此被裁掉，4.4 步 1）
};

// 判交语义 eps："严格内交才算相交"——路径顶点本就是膨胀矩形角点，擦着膨胀边界走是合法
// 路径（顶点处到原始矩形距离恰 = r）；eps 吸收角点坐标回代 pointRectDist 的浮点噪声
//（~1e-16 量级），勿调大到能吞掉真实间隙的量级（最小判定间隙 0.02，相差 7 个数量级）。
const EPS = 1e-9;

// 2D 点到旋转矩形距离（<0 为在矩形内，值 = 穿透深度）。导出供 stackCore（negcheck 注入
// 的最近障碍查找）与 headless（N19 同源距离口径）复用——三处必须同一实现，防口径漂移。
export function pointRectDist(p, rect) {
  const c = Math.cos(rect.yaw), s = Math.sin(rect.yaw);
  const dx = p.x - rect.cx, dy = p.y - rect.cy;
  const lx = c * dx + s * dy, ly = -s * dx + c * dy; // 点在矩形局部系
  const qx = Math.abs(lx) - rect.hx, qy = Math.abs(ly) - rect.hy;
  const outD = Math.hypot(Math.max(qx, 0), Math.max(qy, 0));
  return outD + Math.min(Math.max(qx, qy), 0);
}

// 膨胀后四角顶点（r=0 即原始角点）。顺序固定（局部 +x+y 角起、逆时针）——建图确定性依赖。
export function rectCorners(rect, r = 0) {
  const hx = rect.hx + r, hy = rect.hy + r;
  const c = Math.cos(rect.yaw), s = Math.sin(rect.yaw);
  return [
    { x: rect.cx + c * hx - s * hy, y: rect.cy + s * hx + c * hy },
    { x: rect.cx - c * hx - s * hy, y: rect.cy - s * hx + c * hy },
    { x: rect.cx - c * hx + s * hy, y: rect.cy - s * hx - c * hy },
    { x: rect.cx + c * hx + s * hy, y: rect.cy + s * hx - c * hy },
  ];
}

// 解析膨胀：旋转矩形 ⊕ 半径 r 的保守覆盖——放大矩形直角覆盖圆角（真实 Minkowski 和的
// 圆角不参与路径规划时，保守安全；调研结论问题 2：无需引库，解析可得）。
export function inflateRect(rect, r) {
  return { cx: rect.cx, cy: rect.cy, yaw: rect.yaw, hx: rect.hx + r, hy: rect.hy + r };
}

// 线段-旋转矩形相交（严格内交语义）。凸集完备性：线段与凸矩形的交是子线段（可空/点），
// 交有正长度时必居其一——① 某端点严格在内；② 与某边严格正交穿越（进出点至少一个是边的
// 内点）；③ 交的内部在内而两端都在边界/角点上（如对角弦）——此时由凸性线段中点必严格
// 在内。擦边/触角点/共线重叠不算相交（合法擦碰）。三条全查，缺一会漏判对角弦穿箱。
export function segIntersectsRect(a, b, rect) {
  if (pointRectDist(a, rect) < -EPS || pointRectDist(b, rect) < -EPS) return true;
  if (pointRectDist({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, rect) < -EPS) return true;
  const cs = rectCorners(rect, 0);
  for (let i = 0; i < 4; i++) {
    if (segSegStrict(a, b, cs[i], cs[(i + 1) % 4])) return true;
  }
  return false;
}

// 线段严格正交穿越（orientation 叉积法；端点触碰/共线重叠均不算——擦碰合法语义）。
function segSegStrict(p1, p2, p3, p4) {
  const d1 = cross3(p3, p4, p1), d2 = cross3(p3, p4, p2);
  const d3 = cross3(p1, p2, p3), d4 = cross3(p1, p2, p4);
  return ((d1 > EPS && d2 < -EPS) || (d1 < -EPS && d2 > EPS))
    && ((d3 > EPS && d4 < -EPS) || (d3 < -EPS && d4 > EPS));
}

function cross3(a, b, p) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// 脱出点（4.4 步 4）：从 p 沿"矩形中心→p"径向推到膨胀边界外 margin（射线-矩形局部系解析
// 求交；tExit 为沿方向到各轴边界的最小正参数）。p 恒在膨胀内（containment 已筛），tExit>0。
function escapePointOn(p, rect, margin) {
  const dx = p.x - rect.cx, dy = p.y - rect.cy;
  const l = Math.hypot(dx, dy);
  const ux = l < 1e-9 ? 1 : dx / l, uy = l < 1e-9 ? 0 : dy / l; // 退化（p=中心）：取 +x
  const c = Math.cos(rect.yaw), s = Math.sin(rect.yaw);
  const lx = c * dx + s * dy, ly = -s * dx + c * dy;
  const lu = c * ux + s * uy, lv = -s * ux + c * uy;
  let tExit = Infinity;
  if (lu > 1e-12) tExit = Math.min(tExit, (rect.hx - lx) / lu);
  if (lu < -1e-12) tExit = Math.min(tExit, (-rect.hx - lx) / lu);
  if (lv > 1e-12) tExit = Math.min(tExit, (rect.hy - ly) / lv);
  if (lv < -1e-12) tExit = Math.min(tExit, (-rect.hy - ly) / lv);
  if (!Number.isFinite(tExit) || tExit < 0) tExit = 0; // 防御：p 恰在边界上时不外推距离
  return { x: p.x + ux * (tExit + margin), y: p.y + uy * (tExit + margin) };
}

// 主入口：from/to {x,y}；rects 矩形列表；opts 覆盖 PLANNER_DEFAULTS。
// 返回 { points: [{x, y, yaw}], mode: 'visibility'|'straight'|'straight-degraded',
//        pruned, degraded, usedRects, shrunk? }——points 不含起点（调用方从 from 起密化，
// 与旧 walker.planPath 语义一致）；yaw = 段行进方向，末点 yaw = 末段航向（4.4 步 8）。
export function planPathAvoid(from, to, rects, opts = {}) {
  const o = { ...PLANNER_DEFAULTS, ...opts };
  const all = Array.isArray(rects) ? rects : [];
  const dx0 = to.x - from.x, dy0 = to.y - from.y;
  const len0 = Math.hypot(dx0, dy0);
  const dir0 = len0 < 1e-6 ? (to.yaw ?? 0) : Math.atan2(dy0, dx0);
  // 每矩形膨胀半径：per-obstacle r 覆盖 → target 箱收缩 → 缺省 robotR+margin（4.6 契约）
  const rOf = (rc) => rc.r ?? (rc.role === 'target' ? o.targetShrinkR : o.robotR + o.margin);

  // 1) 粗筛：from→to 包围盒 + pruneMargin（膨胀半径计入矩形 AABB；潜伏行在此被裁掉）
  const minX = Math.min(from.x, to.x) - o.pruneMargin, maxX = Math.max(from.x, to.x) + o.pruneMargin;
  const minY = Math.min(from.y, to.y) - o.pruneMargin, maxY = Math.max(from.y, to.y) + o.pruneMargin;
  const kept = [];
  for (const rc of all) {
    const r = rOf(rc);
    const ex = Math.abs(Math.cos(rc.yaw)) * (rc.hx + r) + Math.abs(Math.sin(rc.yaw)) * (rc.hy + r);
    const ey = Math.abs(Math.sin(rc.yaw)) * (rc.hx + r) + Math.abs(Math.cos(rc.yaw)) * (rc.hy + r);
    if (rc.cx + ex < minX || rc.cx - ex > maxX || rc.cy + ey < minY || rc.cy - ey > maxY) continue;
    kept.push(rc);
  }
  const finish = (points, mode, degraded, shrunk = false) => ({
    points, mode, pruned: all.length - kept.length, degraded, usedRects: kept, ...(shrunk ? { shrunk: true } : {}),
  });

  // 零长位移 / 无相关障碍：直线短路（未启用可见性图搜索，mode='straight'）
  if (len0 < 1e-6 || kept.length === 0) {
    return finish([{ x: to.x, y: to.y, yaw: dir0 }], 'straight', false);
  }

  // 单轮求解：给定膨胀矩形列表 → 节点集 + 贴近机制 + 建图 + Dijkstra → 节点链（含起终点）
  // 或 null（无可行路径）。贴近机制（透明化/脱出）随膨胀列表重算——降级收缩一轮后
  // 起终点包含关系随之变化。
  const solve = (infL) => {
    // 3) 节点集：起点、终点 + 各膨胀矩形 4 顶点（maxNodes 截断保底：丢顶点只会让该矩形处
    //    无可行边而走向降级阶梯，失败方向安全）
    const nodes = [{ x: from.x, y: from.y, kind: 'start' }, { x: to.x, y: to.y, kind: 'end' }];
    for (let i = 0; i < infL.length && nodes.length < o.maxNodes; i++) {
      for (const cn of rectCorners(infL[i], 0)) {
        if (nodes.length >= o.maxNodes) break;
        nodes.push({ x: cn.x, y: cn.y, kind: 'corner' });
      }
    }
    // 4) 贴近机制：起点/终点落入某膨胀矩形 → 该矩形对该节点入射边透明化（只跳过膨胀检查，
    //    原始矩形仍硬检查——任何边都不得穿过箱体实体）；起点在膨胀内 → 前置脱出路点
    //    （最近膨胀边界外 escapeMargin 沿径向），把"膨胀内行走"确定性压到 <0.3m。
    //    脱出点继承起点的透明化集合（它是起点在同一包含区内的代理节点，保证图连通）。
    const contStart = [], contEnd = [];
    for (let i = 0; i < infL.length; i++) {
      if (pointRectDist(from, infL[i]) < -EPS) contStart.push(i);
      if (pointRectDist(to, infL[i]) < -EPS) contEnd.push(i);
    }
    if (contStart.length > 0) {
      let bi = contStart[0], bd = Infinity;
      for (const i of contStart) {
        const d = pointRectDist(from, infL[i]);
        if (d < bd) { bd = d; bi = i; }
      }
      const ep = escapePointOn(from, infL[bi], o.escapeMargin);
      nodes.push({ x: ep.x, y: ep.y, kind: 'escape' });
    }
    // 边可见性：触及 start/escape 的边对 contStart 内矩形透明、触及 end 的边对 contEnd 内
    // 矩形透明；透明矩形退回原始矩形硬检查（安全底线）。
    // 【P2 审查修复】target 矩形例外：起点/终点可能深入其**原矩形**内部（货架外廓内的
    // 释放站位 root y=1.89 > 矩形下缘 1.8、贴箱站位），原矩形硬检查会封死一切出发/
    // 到达边 → 起点/终点孤立 → 双重求解失败 → 假降级（home 段"无可行路径直线降级"
    // 的根因）。target 是"设计性贴近"语义，对其连原矩形一并豁免，安全底线由 N19c
    // （target 行进段 ≥0.02）兜底；非 target（placed/staging/静态普通）仍退回原矩形
    // 硬检查——穿过箱体实体始终被禁止。
    const segClear = (a, b) => {
      const ts = a.kind === 'start' || a.kind === 'escape' || b.kind === 'start' || b.kind === 'escape';
      const te = a.kind === 'end' || b.kind === 'end';
      for (let ri = 0; ri < infL.length; ri++) {
        const transparent = (ts && contStart.includes(ri)) || (te && contEnd.includes(ri));
        if (transparent) {
          if (kept[ri].role === 'target') continue;
          if (segIntersectsRect(a, b, kept[ri])) return false;
          continue;
        }
        if (segIntersectsRect(a, b, infL[ri])) return false;
      }
      return true;
    };
    // 5) 建图：节点对连边当且仅当线段不与任何非透明矩形相交
    const nN = nodes.length;
    const adj = Array.from({ length: nN }, () => []);
    for (let i = 0; i < nN; i++) {
      for (let j = i + 1; j < nN; j++) {
        if (!segClear(nodes[i], nodes[j])) continue;
        const w = Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y);
        adj[i].push([j, w]);
        adj[j].push([i, w]);
      }
    }
    // 6) 搜索：Dijkstra（O(V²)，V≤maxNodes；欧氏启发 A* 为可选优化，本规模不需要）。
    //    等距取最小下标（严格 < 比较），确定性。
    const dist = new Array(nN).fill(Infinity);
    const prev = new Array(nN).fill(-1);
    const done = new Array(nN).fill(false);
    dist[0] = 0;
    for (;;) {
      let u = -1, du = Infinity;
      for (let i = 0; i < nN; i++) {
        if (!done[i] && dist[i] < du) { du = dist[i]; u = i; }
      }
      if (u < 0 || u === 1) break; // 起点 0 / 终点 1
      done[u] = true;
      for (const [v, w] of adj[u]) {
        const nd = du + w;
        if (nd < dist[v]) { dist[v] = nd; prev[v] = u; }
      }
    }
    if (!Number.isFinite(dist[1])) return null;
    const chain = [];
    for (let v = 1; v >= 0; v = prev[v]) chain.push(nodes[v]);
    chain.reverse();
    // 7) 平滑：对每个中间 corner 顶点，转角 > turnMax 时沿前后边各取过渡点（偏移
    //    cornerOffset，不超过邻段长一半）并以过渡弦替代角点——跟踪器提前转向，捕获半径
    //    （0.15）+ 跟踪漂移的切角留在膨胀外（膨胀 − 底线的 0.05 预算 + 0.3 偏移净空，R1）。
    //    过渡弦重跑判交校验（透明化语义同建图），失败回退保留原角点。start/end/escape
    //    顶点不平滑（escape 是机制点、邻段短）。
    const smoothed = [chain[0]];
    for (let i = 1; i < chain.length - 1; i++) {
      const p = smoothed[smoothed.length - 1], v = chain[i], nx = chain[i + 1];
      const l1 = Math.hypot(v.x - p.x, v.y - p.y), l2 = Math.hypot(nx.x - v.x, nx.y - v.y);
      const d1x = (v.x - p.x) / (l1 || 1), d1y = (v.y - p.y) / (l1 || 1);
      const d2x = (nx.x - v.x) / (l2 || 1), d2y = (nx.y - v.y) / (l2 || 1);
      const turn = Math.atan2(Math.abs(d1x * d2y - d1y * d2x), d1x * d2x + d1y * d2y);
      const off = Math.min(o.cornerOffset, l1 / 2, l2 / 2);
      if (v.kind !== 'corner' || turn <= o.turnMax || off <= EPS) {
        smoothed.push(v);
        continue;
      }
      const t1 = { x: v.x - d1x * off, y: v.y - d1y * off, kind: 'smooth' };
      const t2 = { x: v.x + d2x * off, y: v.y + d2y * off, kind: 'smooth' };
      if (segClear(t1, t2)) smoothed.push(t1, t2);
      else smoothed.push(v); // 校验失败：回退未平滑角点（该角点保留，路径仍为已验证折线）
    }
    smoothed.push(chain[chain.length - 1]);
    // 输出前全段复检（4.4 步 7：判交校验失败回退未平滑折线——逐角回退后理论不再触发，
    // 保留为防御性兜底）
    let finalChain = smoothed;
    for (let i = 1; i < smoothed.length; i++) {
      if (!segClear(smoothed[i - 1], smoothed[i])) { finalChain = chain; break; }
    }
    return finalChain;
  };

  const inf = kept.map((rc) => inflateRect(rc, rOf(rc)));
  let chain = solve(inf);
  let shrunk = false;
  if (!chain) {
    // 3) 真降级阶梯第一级：全膨胀无可行 → 全障碍收缩一轮（r_min=targetShrinkR，4.3.3）再求
    const rMin = Math.min(o.robotR + o.margin, o.targetShrinkR);
    chain = solve(kept.map((rc) => inflateRect(rc, rMin)));
    shrunk = !!chain;
  }
  if (!chain) {
    // 仍无可行路径：直线兜底（degraded=true，调用方明示日志 + degraded 计数）
    return finish([{ x: to.x, y: to.y, yaw: dir0 }], 'straight-degraded', true);
  }
  // 8) 输出：points = 链去起点；yaw = 段行进方向，末点 yaw = 末段航向
  const pts = chain.slice(1);
  const points = pts.map((p, i) => {
    let yaw;
    if (i < pts.length - 1) yaw = Math.atan2(pts[i + 1].y - p.y, pts[i + 1].x - p.x);
    else if (pts.length >= 2) yaw = Math.atan2(p.y - pts[i - 1].y, p.x - pts[i - 1].x);
    else yaw = dir0;
    return { x: p.x, y: p.y, yaw };
  });
  return finish(points, 'visibility', false, shrunk);
}
