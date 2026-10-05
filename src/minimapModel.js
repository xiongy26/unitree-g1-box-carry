// 小地图纯数据层（工厂方案 D5）：绘制指令全部为世界坐标纯数据 {op, ...}，
// 渲染层（src/minimap.js）负责世界→像素映射。Node 可直接 import（headless F5 断言），
// 无 DOM/three/process 依赖（同 pathPlanner/factory 确定性契约）。
//
// 指令契约：
//   { op:'rect',   x, y, yaw=0, hx, hy, stroke, fill?, width? }   矩形（中心+半边长）
//   { op:'circle', x, y, r, stroke, fill?, width? }              圆
//   { op:'poly',   pts:[{x,y}...], stroke, width?, closed? }     折线/多边形
//   { op:'tri',    x, y, yaw, size, fill, stroke? }              三角（机器人位姿）
//   { op:'cross',  x, y, size, stroke, width? }                  十字（目标/槽位高亮）
//   { op:'dash',   x, y, yaw=0, hx, hy, stroke, width? }         虚线矩形（区域标示）
//
// 三处消费一致性（D2）：静态指令直接读 FACTORY 表 + rackRect()/factoryObstacleRects()
// 同源推导，headless F2c/F5 按同表对账。

import { rackRect } from './factory.js';
import { MAX_BOXES } from './stackCore.js';

// 世界视图窗口：FACTORY.floorHalf 外扩 0.5m（方案 D5：x∈[-8.5,8.5], y∈[-7.5,7.5]）
export function mapView(factory) {
  const [fhx, fhy] = factory.floorHalf;
  return { minX: -fhx - 0.5, maxX: fhx + 0.5, minY: -fhy - 0.5, maxY: fhy + 0.5 };
}

export function workView(factory) {
  const [rx, ry] = factory.rackCenter;
  const [px, py] = factory.pickZone.center;
  return { minX: Math.min(0, rx, px) - 1, maxX: Math.max(0, rx, px) + 1.3,
    minY: Math.min(0, ry, py) - 1, maxY: Math.max(0, ry, py) + 1.3 };
}

// 世界→像素线性映射闭包：等比缩放取 min、居中、y 翻转（世界 y 向上 / canvas y 向下）。
// 返回 { scale, toPixel(x,y)->[px,py], toWorld(px,py)->[x,y] }（toWorld 供 F5 往返断言
// 与未来的指针交互，当前渲染只消费 toPixel）。
export function worldToPixelMapper(view, w, h) {
  const sx = w / (view.maxX - view.minX);
  const sy = h / (view.maxY - view.minY);
  const s = Math.min(sx, sy);
  const ox = w / 2 - ((view.minX + view.maxX) / 2) * s;
  const oy = h / 2 + ((view.minY + view.maxY) / 2) * s;
  return {
    scale: s,
    toPixel: (x, y) => [ox + x * s, oy - y * s],
    toWorld: (px, py) => [(px - ox) / s, (oy - py) / s],
  };
}

// 静态层绘制指令（世界坐标）：厂房外廓/货架/障碍/暂存区标示/潜伏行。
// 一次性生成、离屏缓存（渲染层职责），内容与 FACTORY 表构造性一致。
export function buildStaticDrawList(factory) {
  const ops = [];
  const [fhx, fhy] = factory.floorHalf;
  // 厂房外廓 + 围界
  ops.push({ op: 'rect', x: 0, y: 0, yaw: 0, hx: fhx, hy: fhy, stroke: '#55607a', width: 2 });
  for (const o of factory.obstacles) {
    if (o.kind === 'wall') {
      ops.push({ op: 'rect', x: o.cx, y: o.cy, yaw: o.yaw, hx: o.hx, hy: o.hy, stroke: '#55607a', fill: 'rgba(60,66,80,0.5)' });
    } else if (o.kind === 'pillar') {
      ops.push({ op: 'circle', x: o.cx, y: o.cy, r: o.hx, stroke: '#9aa3b5', fill: 'rgba(140,150,170,0.55)' });
    } else if (o.kind === 'rack') {
      ops.push({ op: 'rect', x: o.cx, y: o.cy, yaw: o.yaw, hx: o.hx, hy: o.hy, stroke: '#7a8496', fill: 'rgba(90,100,115,0.35)' });
    } else if (o.kind === 'equipment') {
      ops.push({ op: 'rect', x: o.cx, y: o.cy, yaw: o.yaw, hx: o.hx, hy: o.hy, stroke: '#7fb5c4', fill: 'rgba(110,160,175,0.3)' });
    } else if (o.kind === 'fence') {
      ops.push({ op: 'rect', x: o.cx, y: o.cy, yaw: o.yaw, hx: o.hx, hy: o.hy, stroke: '#c8a24a', fill: 'rgba(170,130,50,0.35)' });
    } else {
      ops.push({ op: 'rect', x: o.cx, y: o.cy, yaw: o.yaw, hx: o.hx, hy: o.hy, stroke: '#8a6a45', fill: 'rgba(120,90,55,0.4)' });
    }
  }
  // 活动货架：外廓高亮 + 格位列分隔线（俯视；层板格位 = 2 列 ×3 层，俯视只表达列）
  {
    const r = rackRect();
    ops.push({ op: 'rect', x: r.cx, y: r.cy, yaw: r.yaw, hx: r.hx, hy: r.hy, stroke: '#6fb2ff', fill: 'rgba(111,178,255,0.22)', width: 1.5 });
    ops.push({
      op: 'poly',
      pts: [
        { x: r.cx, y: r.cy - factory.rack.depthHalf },
        { x: r.cx, y: r.cy + factory.rack.depthHalf },
      ],
      stroke: 'rgba(111,178,255,0.8)', width: 1,
    });
  }
  // 取箱暂存区（虚线框）
  ops.push({
    op: 'dash',
    x: factory.pickZone.center[0], y: factory.pickZone.center[1], yaw: 0,
    hx: factory.pickZone.half[0], hy: factory.pickZone.half[1],
    stroke: '#ffd166', width: 1.5,
  });
  return ops;
}

// 动态层绘制指令（世界坐标）：机器人三角/路径折线/箱子矩形/当前槽位与目标高亮。
// path 为 stack.lastWalkPath（规划器原始角点，非密化）；activeSlot 为 stack.currentSlot。
export function buildDynamicDrawList({ robot, path, boxes, activeSlot } = {}) {
  const ops = [];
  // 路径折线（规划器输出原始角点）+ 目标十字
  if (path && Array.isArray(path.points) && path.points.length > 0) {
    ops.push({ op: 'poly', pts: path.points.map((p) => ({ x: p.x, y: p.y })), stroke: '#4dc9ff', width: 1.5 });
    if (path.goal) ops.push({ op: 'cross', x: path.goal[0], y: path.goal[1], size: 0.18, stroke: '#ffd166', width: 1.5 });
  }
  // 箱子矩形（placed 成品垛 + 当前幽灵箱）
  for (const b of boxes ?? []) {
    if (b.kind === 'ghost') {
      ops.push({ op: 'rect', x: b.x, y: b.y, yaw: b.yaw, hx: b.hx, hy: b.hy, stroke: '#7fe08f', fill: 'rgba(120,220,140,0.35)', width: 1.2 });
    } else {
      ops.push({ op: 'rect', x: b.x, y: b.y, yaw: b.yaw, hx: b.hx, hy: b.hy, stroke: '#e8a24a', fill: 'rgba(230,150,60,0.55)', width: 1 });
    }
  }
  // 当前槽位高亮（货架格位中心十字）
  if (activeSlot && activeSlot.center) {
    ops.push({ op: 'cross', x: activeSlot.center[0], y: activeSlot.center[1], size: 0.22, stroke: '#ff8fd0', width: 1.5 });
  }
  // 机器人位姿三角（root qpos + 偏航）
  if (robot && Number.isFinite(robot.x) && Number.isFinite(robot.y)) {
    ops.push({ op: 'tri', x: robot.x, y: robot.y, yaw: robot.yaw ?? 0, size: 0.35, fill: '#ffffff', stroke: '#2f6fed' });
  }
  return ops;
}

// 取箱区标示内派生取箱点检查（F2/R2 消费）：基准 clip 的派生点是否落在 pickZone 内
export function pickZoneCovers(factory, pickPos) {
  const [px, py] = pickPos;
  const [cx, cy] = factory.pickZone.center;
  const [hx, hy] = factory.pickZone.half;
  return Math.abs(px - cx) <= hx && Math.abs(py - cy) <= hy;
}
