// Actual page integration check, including the default 2x2 plastic-box setup.
// Start Chrome with --remote-debugging-port=9223, then:
// node tools/checks/browser_smoke.mjs http://localhost:8000 http://localhost:9223
import assert from 'node:assert/strict';
const site = process.argv[2] ?? 'http://127.0.0.1:8000';
const debug = process.argv[3] ?? 'http://127.0.0.1:9223';
const page = await (await fetch(`${debug}/json/new?about:blank`, { method: 'PUT' })).json();
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.addEventListener('open', resolve, { once: true }); ws.addEventListener('error', reject, { once: true }); });
let sequence = 0;
const pending = new Map();
ws.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (message.id) { pending.get(message.id)?.(message); pending.delete(message.id); }
});
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++sequence;
  const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 30000);
  pending.set(id, message => { clearTimeout(timer); message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result); });
  ws.send(JSON.stringify({ id, method, params }));
});
async function evaluate(expression) {
  const result = await send('Runtime.evaluate', { expression, returnByValue: true });
  assert.ok(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
try {
  await send('Page.enable'); await send('Runtime.enable');
  await send('Page.navigate', { url: site });
  let ready = false;
  for (let i = 0; i < 120; i++) {
    await new Promise(resolve => setTimeout(resolve, 250));
    if (await evaluate('!!window.__sim')) { ready = true; break; }
  }
  assert.ok(ready, 'page did not finish initialization');
  // No preload or selector injection: this catches failures in the shipped defaults.
  const first = await evaluate(`window.__sim.setMode('stack'); ({status:window.__sim.stack?.status(),hint:document.getElementById('hint').textContent})`);
  assert.ok(first.status, first.hint);
  assert.equal(first.status.config.boxType, 'plasticbox');
  assert.equal(first.status.slotTotal, 4);
  assert.notEqual(first.status.phase, 'idle');
  assert.equal(first.status.error, null);
  console.log('PASS browser default: plasticbox 2x2 starts');
  for (const box of ['largebox', 'plasticbox']) {
    for (const layers of [1, 2]) {
      const result = await evaluate(`
        window.__sim.setMode('idle');
        document.getElementById('boxtype-sel').value=${JSON.stringify(box)};
        document.getElementById('layers-sel').value=${layers};
        document.getElementById('cols-sel').value='2';
        document.getElementById('boxtype-sel').dispatchEvent(new Event('change'));
        window.__sim.setMode('stack'); window.__sim.drive(.02,1/60,false);
        ({status:window.__sim.stack?.status(),mode:window.__sim.mode,hint:document.getElementById('hint').textContent});
      `);
      assert.equal(result.mode, 'stack', result.hint);
      assert.equal(result.status.config.boxType, box);
      assert.equal(result.status.config.layers, layers);
      assert.equal(result.status.slotTotal, layers * 2);
      assert.equal(result.status.error, null);
      console.log(`PASS browser selectors: ${box} ${layers}x2 starts`);
    }
  }
  await evaluate(`window.__sim.setMode('idle')`);
} finally {
  ws.close();
  await fetch(`${debug}/json/close/${page.id}`);
}
