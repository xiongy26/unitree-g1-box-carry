// 页面内视频录制（干净合成画面）：主 3D 画布全幅 + 右下角小地图面板
// （底板、边框、标题、视图切换钮与图例一并入镜，不含控制面板与 HUD）。
// 每帧把画布合成进离屏画布，canvas.captureStream → MediaRecorder 编码；
// 优先 H.264 MP4（Chrome 126+），编码器不可用时回退 WebM（转换命令见 README）。
// 纯展示端模块：只读画布与堆垛配置，不触碰仿真状态。

const MP4_CANDIDATES = [
  'video/mp4;codecs=avc1.640028', // H.264 High
  'video/mp4;codecs=avc1.42E01E', // H.264 Baseline
  'video/mp4',
];
const WEBM_CANDIDATES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
];

// 返回当前浏览器可用的最优封装+编码；无 MediaRecorder 或无任何支持时返回 null
export function pickRecorderMime() {
  if (typeof MediaRecorder === 'undefined') return null;
  for (const t of MP4_CANDIDATES) if (MediaRecorder.isTypeSupported(t)) return t;
  for (const t of WEBM_CANDIDATES) if (MediaRecorder.isTypeSupported(t)) return t;
  return null;
}

// 比特率：像素量自适应，1080p 约 10 Mbps，封顶 20 Mbps
function pickBitrate(w, h) {
  return Math.min(20e6, Math.max(8e6, Math.round(w * h * 5)));
}

const TWO = (n) => String(n).padStart(2, '0');

function stampName(cfg, ext) {
  const d = new Date();
  const tag = cfg ? `${cfg.boxType}-${cfg.layers}x${cfg.cols}` : 'scene';
  return `g1-${tag}-${d.getFullYear()}${TWO(d.getMonth() + 1)}${TWO(d.getDate())}-${TWO(d.getHours())}${TWO(d.getMinutes())}${TWO(d.getSeconds())}.${ext}`;
}

export function createRecorder({
  getViewCanvas,     // () => HTMLCanvasElement，主 3D 画布（WebGL backing store）
  getOverlayInfo,    // () => { canvas, cssSize:{w,h}, viewLabel }，右下角小地图面板
  getConfig,         // () => {boxType,layers,cols} | null，用于文件名
  button,            // 录制开关按钮（本模块负责其文案与高亮态）
  hintEl,            // 页面提示条（复用 #hint，与 setMode 的提示共用位置）
}) {
  const comp = document.createElement('canvas'); // 合成画布（不进 DOM）
  const ctx = comp.getContext('2d');
  const IDLE_LABEL = button?.textContent ?? '● 录制 (V)';

  let rec = null;
  let mime = null;
  let chunks = [];
  let active = false;
  let startedAt = 0;
  let ticker = 0;
  let lastBlob = null; // 验证钩子：最后一次录制的 Blob（自动化检查 size/type 用）

  // 小地图面板设计稿（CSS 像素，与 index.html #minimap-panel 同构）：
  // 半透明圆角底板 + 细描边、标题行（规划路线 / 视图切换钮）、地图画布、图例行。
  // 录制时按 k 倍缩放绘出（k = 地图画布 backing 宽 / CSS 宽，页面与合成同 DPR 封顶 2）。
  const PANEL = { pad: 8, heading: 18, legend: 16, radius: 14, mapRadius: 8 };
  const LEGEND = [['━', '路线', '#4dc9ff'], ['＋', '目标', '#ffd166'], ['▲', '机器人', '#ffffff']];
  const FONT_STACK = 'system-ui, "PingFang SC", "Microsoft YaHei", sans-serif';

  function drawOverlayPanel({ canvas, cssSize, viewLabel }) {
    if (!cssSize?.w || !cssSize?.h) return;
    const k = canvas.width / cssSize.w;
    const W = cssSize.w + PANEL.pad * 2;
    const H = cssSize.h + PANEL.pad * 2 + PANEL.heading + PANEL.legend;
    const margin = Math.round(16 * k); // 页面面板 right/bottom 16px 的等比距离
    const x0 = Math.max(margin, comp.width - Math.round(W * k) - margin);
    const y0 = Math.max(margin, comp.height - Math.round(H * k) - margin);
    ctx.save();
    ctx.translate(x0, y0);
    ctx.scale(k, k);
    // 底板与边框（页面 .panel 同色：半透明深底 + 白 9% 描边）
    ctx.beginPath();
    ctx.roundRect(0, 0, W, H, PANEL.radius);
    ctx.fillStyle = 'rgba(16, 20, 28, 0.72)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.09)';
    ctx.lineWidth = 1;
    ctx.stroke();
    // 标题行：左「规划路线」，右视图切换钮（全厂 / 作业区，随页面状态）
    const cy = PANEL.pad + PANEL.heading / 2;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#e8ecf4';
    ctx.font = `12px ${FONT_STACK}`;
    ctx.textAlign = 'left';
    ctx.fillText('规划路线', PANEL.pad + 2, cy);
    ctx.font = `11px ${FONT_STACK}`;
    const chip = viewLabel ?? '';
    const cw = ctx.measureText(chip).width + 16;
    const chx = W - PANEL.pad - cw;
    ctx.beginPath();
    ctx.roundRect(chx, cy - 10, cw, 20, 9);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.stroke();
    ctx.fillStyle = '#dfe6f2';
    ctx.textAlign = 'center';
    ctx.fillText(chip, chx + cw / 2, cy + 0.5);
    // 地图画布（圆角裁剪内贴）
    const my = PANEL.pad + PANEL.heading;
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(PANEL.pad, my, cssSize.w, cssSize.h, PANEL.mapRadius);
    ctx.clip();
    ctx.drawImage(canvas, PANEL.pad, my, cssSize.w, cssSize.h);
    ctx.restore();
    // 图例行：━ 路线 / ＋ 目标 / ▲ 机器人（左·中·右，与页面 space-between 一致）
    const ly = my + cssSize.h + PANEL.legend / 2;
    ctx.font = `10px ${FONT_STACK}`;
    LEGEND.forEach(([glyph, label, color], i) => {
      ctx.fillStyle = color;
      ctx.textAlign = i === 0 ? 'left' : i === 1 ? 'center' : 'right';
      ctx.fillText(`${glyph} ${label}`, i === 0 ? PANEL.pad : i === 1 ? W / 2 : W - PANEL.pad, ly);
    });
    ctx.restore();
  }

  // 合成一帧：黑底 → 主画面等比居中（窗口录制中缩放时留黑边）→ 右下角小地图面板
  function drawFrame() {
    const view = getViewCanvas();
    if (!view || !view.width || !view.height) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#0d1119'; // 与页面加载背景一致
    ctx.fillRect(0, 0, comp.width, comp.height);
    const s = Math.min(comp.width / view.width, comp.height / view.height);
    const dw = view.width * s, dh = view.height * s;
    ctx.drawImage(view, (comp.width - dw) / 2, (comp.height - dh) / 2, dw, dh);
    const info = getOverlayInfo?.();
    if (info?.canvas?.width) drawOverlayPanel(info);
  }

  function fmtElapsed(ms) {
    const t = Math.floor(ms / 1000);
    return `${TWO(Math.floor(t / 60))}:${TWO(t % 60)}`;
  }

  function extOf(m) { return m.includes('mp4') ? 'mp4' : 'webm'; }

  function setButton(label, recState) {
    if (!button) return;
    button.textContent = label;
    button.classList.toggle('rec', !!recState);
  }

  function start() {
    if (active) return true;
    mime = pickRecorderMime();
    if (!mime) {
      if (hintEl) hintEl.textContent = '录制不可用：当前浏览器缺少 MediaRecorder 或可用的视频编码器';
      return false;
    }
    const view = getViewCanvas();
    // 分辨率在开始时锁定（取偶：H.264 要求偶数宽高，奇数窗口会被编码器裁边）；
    // 录制中改窗口大小只会等比加黑边
    comp.width = view.width - (view.width % 2);
    comp.height = view.height - (view.height % 2);
    drawFrame();               // 先合成一帧，避免首帧黑屏
    let stream;
    try {
      stream = comp.captureStream(60);
    } catch (e) {
      if (hintEl) hintEl.textContent = `录制不可用：captureStream 失败（${e.message}）`;
      return false;
    }
    try {
      rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: pickBitrate(comp.width, comp.height) });
    } catch (e) {
      if (hintEl) hintEl.textContent = `录制不可用：${e.message}`;
      return false;
    }
    chunks = [];
    rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    rec.onerror = (e) => {
      if (hintEl) hintEl.textContent = `录制出错：${e.error?.message ?? e.error ?? '未知错误'}，已停止`;
      stop();
    };
    rec.onstop = finalize;
    rec.start(1000); // 每秒切一次 chunk，避免长时间录制在 stop 时一次性编码大块
    active = true;
    startedAt = performance.now();
    setButton(`■ 停止 00:00`, true);
    ticker = setInterval(() => setButton(`■ 停止 ${fmtElapsed(performance.now() - startedAt)}`, true), 250);
    if (hintEl) {
      const ext = extOf(mime);
      hintEl.textContent = ext === 'mp4'
        ? `录制中 ${comp.width}×${comp.height}（H.264 MP4）：任务完成后自动收片下载，也可按 V 手动停止`
        : `录制中 ${comp.width}×${comp.height}（浏览器不支持 MP4，已回退 WebM，转换命令见 README）`;
    }
    return true;
  }

  function stop() {
    if (!active) return;
    active = false;
    clearInterval(ticker);
    setButton(IDLE_LABEL, false);
    if (rec && rec.state !== 'inactive') rec.stop(); // onstop → finalize 下载
    else finalize();
  }

  function finalize() {
    const type = mime ?? 'video/webm';
    const blob = new Blob(chunks, { type });
    chunks = [];
    rec = null;
    if (!blob.size) {
      if (hintEl) hintEl.textContent = '录制结果为空，未保存文件';
      return;
    }
    const ms = performance.now() - startedAt;
    const name = stampName(getConfig?.() ?? null, extOf(type));
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    if (hintEl) {
      hintEl.textContent = `已保存 ${name}（${fmtElapsed(ms)}，${(blob.size / 1048576).toFixed(1)} MB）`;
    }
    lastBlob = blob;
  }

  // 主循环每帧调用；未录制时零开销直返
  function frame() {
    if (active) drawFrame();
  }

  function toggle() {
    if (active) { stop(); return false; }
    return start();
  }

  return {
    start, stop, toggle, frame,
    get active() { return active; },
    get mimeType() { return mime; },
    get lastBlob() { return lastBlob; },
    get size() { return { w: comp.width, h: comp.height }; },
  };
}
