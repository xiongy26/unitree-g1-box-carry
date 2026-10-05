// 录制功能回归检查（真实页面 + MediaRecorder + 下载落盘）。
// 先启动带调试端口的 Chrome，再运行本检查：
//   google-chrome --headless=new --remote-debugging-port=9223 \
//     --user-data-dir=/tmp/chrome-recorder-acc --no-first-run --window-size=1600,900 about:blank &
//   node tools/checks/recorder_check.mjs http://localhost:8000 http://localhost:9223
//   node tools/checks/recorder_check.mjs http://localhost:8000 http://localhost:9223 --full-loop
// 默认快速模式：录制数秒料箱上架起步段，校验编码选择、Blob 与下载文件。
// --full-loop：1×1 塑料箱完整跑完上架循环，校验任务完成后自动收片（耗时数分钟）。
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdirSync, readdirSync, statSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';

const site = process.argv[2] ?? 'http://127.0.0.1:8000';
const debug = process.argv[3] ?? 'http://127.0.0.1:9223';
const fullLoop = process.argv.includes('--full-loop');
const downloadDir = '/tmp/recorder-check-dl';
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// 无头软件渲染下 rAF 只有几 Hz，视频帧率低但时长/封装/编码仍须正确：
// 用 ffprobe 独立校验容器（有 ffprobe 时），不硬性要求帧率。
const hasFfprobe = spawnSync('ffprobe', ['-version'], { stdio: 'ignore' }).status === 0;
function probe(path) {
  const fmt = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', path]).toString());
  const v = fmt.streams.find((s) => s.codec_type === 'video');
  return { codec: v.codec_name, width: v.width, height: v.height, duration: Number(fmt.format.duration) };
}

mkdirSync(downloadDir, { recursive: true });
for (const f of readdirSync(downloadDir)) { // 只认本次运行产生的文件
  if (f.startsWith('g1-')) { try { unlinkSync(join(downloadDir, f)); } catch {} }
}

const page = await (await fetch(`${debug}/json/new?about:blank`, { method: 'PUT' })).json();
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.addEventListener('open', resolve, { once: true }); ws.addEventListener('error', reject, { once: true }); });
let sequence = 0;
const pending = new Map();
ws.addEventListener('message', (event) => {
  const message = JSON.parse(event.data);
  if (message.id) { pending.get(message.id)?.(message); pending.delete(message.id); }
});
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++sequence;
  const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 30000);
  pending.set(id, (message) => { clearTimeout(timer); message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result); });
  ws.send(JSON.stringify({ id, method, params }));
});
async function evaluate(expression) {
  const result = await send('Runtime.evaluate', { expression, returnByValue: true });
  assert.ok(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
try {
  await send('Page.enable');
  await send('Runtime.enable');
  // 下载落盘位置由 CDP 指定，不依赖浏览器默认下载目录
  await send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: downloadDir });
  await send('Page.navigate', { url: site });
  let ready = false;
  for (let i = 0; i < 120; i++) {
    await sleep(250);
    if (await evaluate('!!window.__sim && !!window.__sim.recorder')) { ready = true; break; }
  }
  assert.ok(ready, 'page did not finish initialization');

  const mimes = await evaluate(
    `['video/mp4;codecs=avc1.640028','video/mp4','video/webm;codecs=vp9','video/webm'].filter((t) => MediaRecorder.isTypeSupported(t))`,
  );
  assert.ok(mimes.length > 0, 'browser has no supported recorder mime');

  assert.equal(await evaluate(`window.__sim.recorder.start()`), true);
  assert.equal(await evaluate(`window.__sim.recorder.active`), true);

  if (fullLoop) {
    // 1×1 最小循环：录制先于任务开始，完成后应留尾巴自动收片
    await evaluate(`
      document.getElementById('layers-sel').value = '1';
      document.getElementById('cols-sel').value = '1';
      document.getElementById('boxtype-sel').dispatchEvent(new Event('change'));
      window.__sim.setMode('stack');
      window.__sim.mode`);
    let stopped = false;
    for (let i = 0; i < 600; i++) { // 上限 10 分钟
      await sleep(1000);
      if (!(await evaluate(`window.__sim.recorder.active`))) { stopped = true; break; }
    }
    assert.ok(stopped, 'recorder did not auto-stop after task completion');
  } else {
    await evaluate(`window.__sim.setMode('stack')`);
    await sleep(6000);
    await evaluate(`window.__sim.recorder.stop()`);
  }
  for (let i = 0; i < 40; i++) {
    if (!(await evaluate(`window.__sim.recorder.active`))) break;
    await sleep(250);
  }

  const out = await evaluate(`(() => {
    const r = window.__sim.recorder;
    return {
      active: r.active, mime: r.mimeType, blobSize: r.lastBlob?.size ?? 0, blobType: r.lastBlob?.type ?? '',
      hint: document.getElementById('hint').textContent, btn: document.getElementById('record-btn').textContent,
      w: r.size.w, h: r.size.h,
    };
  })()`);
  assert.equal(out.active, false, 'recorder still active after stop');
  assert.ok(out.blobSize > 50_000, `recorded blob too small: ${out.blobSize}`);
  assert.ok(out.blobType.startsWith('video/'), `unexpected blob type: ${out.blobType}`);
  assert.ok(out.hint.includes('已保存'), `hint after stop: ${out.hint}`);
  assert.match(out.btn, /录制/, `button label after stop: ${out.btn}`);

  let file = null;
  for (let i = 0; i < 60 && !file; i++) {
    await sleep(500);
    file = readdirSync(downloadDir).find((f) => f.startsWith('g1-') && (f.endsWith('.mp4') || f.endsWith('.webm')));
  }
  assert.ok(file, 'no downloaded file appeared in download dir');
  const path = join(downloadDir, file);
  const size = statSync(path).size;
  assert.ok(Math.abs(size - out.blobSize) / out.blobSize < 0.05, `file size ${size} diverges from blob ${out.blobSize}`);
  if (hasFfprobe) {
    const p = probe(path);
    assert.ok(['h264', 'vp8', 'vp9', 'av01'].includes(p.codec), `unexpected codec: ${p.codec}`);
    assert.equal(`${p.width}x${p.height}`, `${out.w}x${out.h}`, 'resolution mismatch');
    const minDur = fullLoop ? 15 : 3.5; // 快速模式录制约 6 秒；完整循环至少 15 秒
    const maxDur = fullLoop ? 1500 : 30;
    assert.ok(p.duration > minDur && p.duration < maxDur, `duration out of range: ${p.duration}s`);
    console.log(`PASS ffprobe: ${p.codec} ${p.width}x${p.height} ${p.duration.toFixed(1)}s`);
  }
  console.log(`PASS recorder: ${file} (${(size / 1048576).toFixed(1)} MB, ${out.mime}, ${out.w}x${out.h}${fullLoop ? ', auto-stop' : ''})`);
  console.log(`FILE ${path}`);
} finally {
  ws.close();
  await fetch(`${debug}/json/close/${page.id}`);
}
