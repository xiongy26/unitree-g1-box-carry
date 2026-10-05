// 小地图 DOM 渲染层（工厂方案 D5/S5）：Canvas 2D 分层渲染。
//   - 静态层：buildStaticDrawList 指令一次绘制进离屏 canvas 缓存（视口固定不重建）；
//   - 动态层：buildDynamicDrawList 指令按 12Hz 节流重绘（清屏 → 贴静态缓存 → 画动态），
//     performance.now() 相位判断，与主渲染循环解耦、不插帧（R10）；
//   - 开关：setVisible(false) 后画布隐藏且 update 直返不再重绘（B3）。
// 纯渲染：不读仿真状态，全部输入来自 getInput()（main.js 提供的只读快照闭包）。
import { worldToPixelMapper, mapView, workView, buildStaticDrawList, buildDynamicDrawList } from './minimapModel.js';

const DRAW_HZ = 12;
const DRAW_MIN_DT = 1000 / DRAW_HZ;

// 12Hz 节流下的最小帧间隔（px 视口固定，DPR 只影响清晰度不影响节流）

export function createMinimap({ factory, getInput, mountEl, size = { w: 320, h: 280 } }) {
  const canvas = mountEl;
  const panel = canvas.closest('.panel') ?? canvas.parentElement;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = size.w * dpr;
  canvas.height = size.h * dpr;
  const ctx = canvas.getContext('2d');

  let overview = false;
  let mapper = worldToPixelMapper(workView(factory), size.w, size.h);

  // 静态层离屏缓存（一次绘制；窗口尺寸/视口为编译期常量，无 resize 重建路径）
  const staticCanvas = document.createElement('canvas');
  staticCanvas.width = canvas.width;
  staticCanvas.height = canvas.height;
  function rebuildStatic() {
    const sctx = staticCanvas.getContext('2d');
    sctx.setTransform(1, 0, 0, 1, 0, 0);
    sctx.clearRect(0, 0, staticCanvas.width, staticCanvas.height);
    sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    renderOps(sctx, buildStaticDrawList(factory), mapper);
  }

  rebuildStatic();

  let visible = true;
  let lastDraw = -Infinity;

  function renderRect(c, op) {
    const [px, py] = mapper.toPixel(op.x, op.y);
    const hw = op.hx * mapper.scale, hh = op.hy * mapper.scale;
    c.save();
    c.translate(px, py);
    c.rotate(-(op.yaw ?? 0)); // 世界逆时针 → canvas 顺时针
    if (op.op === 'dash') c.setLineDash([4, 3]);
    c.beginPath();
    c.rect(-hw, -hh, hw * 2, hh * 2);
    if (op.fill) { c.fillStyle = op.fill; c.fill(); }
    c.strokeStyle = op.stroke;
    c.lineWidth = op.width ?? 1;
    c.stroke();
    c.restore();
  }

  function renderOps(c, ops) {
    for (const op of ops) {
      switch (op.op) {
        case 'rect':
        case 'dash':
          renderRect(c, op);
          break;
        case 'circle': {
          const [px, py] = mapper.toPixel(op.x, op.y);
          c.beginPath();
          c.arc(px, py, Math.max(1.5, op.r * mapper.scale), 0, Math.PI * 2);
          if (op.fill) { c.fillStyle = op.fill; c.fill(); }
          c.strokeStyle = op.stroke;
          c.lineWidth = op.width ?? 1;
          c.stroke();
          break;
        }
        case 'poly': {
          if (op.pts.length < 2) break;
          c.beginPath();
          const [x0, y0] = mapper.toPixel(op.pts[0].x, op.pts[0].y);
          c.moveTo(x0, y0);
          for (let i = 1; i < op.pts.length; i++) {
            const [px, py] = mapper.toPixel(op.pts[i].x, op.pts[i].y);
            c.lineTo(px, py);
          }
          if (op.closed) c.closePath();
          c.strokeStyle = op.stroke;
          c.lineWidth = op.width ?? 1;
          c.stroke();
          break;
        }
        case 'tri': {
          // 机头 + 双翼：世界系三角顶点 → 像素
          const L = op.size, W = op.size * 0.62;
          const pts = [
            [op.x + Math.cos(op.yaw) * L, op.y + Math.sin(op.yaw) * L],
            [op.x + Math.cos(op.yaw + 2.5) * W, op.y + Math.sin(op.yaw + 2.5) * W],
            [op.x + Math.cos(op.yaw - 2.5) * W, op.y + Math.sin(op.yaw - 2.5) * W],
          ];
          c.beginPath();
          for (let i = 0; i < pts.length; i++) {
            const [px, py] = mapper.toPixel(pts[i][0], pts[i][1]);
            if (i === 0) c.moveTo(px, py); else c.lineTo(px, py);
          }
          c.closePath();
          c.fillStyle = op.fill;
          c.fill();
          if (op.stroke) { c.strokeStyle = op.stroke; c.lineWidth = 1; c.stroke(); }
          break;
        }
        case 'cross': {
          const [px, py] = mapper.toPixel(op.x, op.y);
          const r = op.size * mapper.scale;
          c.beginPath();
          c.moveTo(px - r, py); c.lineTo(px + r, py);
          c.moveTo(px, py - r); c.lineTo(px, py + r);
          c.strokeStyle = op.stroke;
          c.lineWidth = op.width ?? 1;
          c.stroke();
          break;
        }
        default:
          break;
      }
    }
  }

  // force=true：录制合成用，绕过隐藏直返与 12Hz 节流（画面里的小地图必须每帧新鲜）
  function drawDynamic(now, force = false) {
    if (!force && !visible) return;
    if (!force && now - lastDraw < DRAW_MIN_DT) return; // 12Hz 相位节流（与主循环解耦，不插帧）
    lastDraw = now;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(staticCanvas, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    renderOps(ctx, buildDynamicDrawList(getInput()));
  }

  return {
    toggleView() {
      overview = !overview;
      mapper = worldToPixelMapper(overview ? mapView(factory) : workView(factory), size.w, size.h);
      rebuildStatic(); lastDraw = -Infinity;
      return overview;
    },
    setVisible(v) {
      visible = !!v;
      panel.classList.toggle('hidden', !visible);
      if (visible) lastDraw = -Infinity; // 重新开启即强制重绘一帧
      else ctx.clearRect(0, 0, canvas.width, canvas.height);
    },
    toggle() {
      this.setVisible(!visible);
      return visible;
    },
    // 主循环每帧调用（main.js frame()）；内部 12Hz 节流 + 关闭直返
    // （录制时传 force=true，见 drawDynamic）
    update(now = performance.now(), force = false) {
      drawDynamic(now, force);
    },
    get visible() { return visible; },
    // CSS 逻辑尺寸（录制合成按此换算面板缩放，须与 index.html 的 #minimap 尺寸一致）
    get cssSize() { return { ...size }; },
  };
}
