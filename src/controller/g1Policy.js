// G1 velocity 策略运行时：权重加载 + MLP 前向 + golden 自测（方案 docs/rl-walking-plan.md 5.1）。
// 纯 ESM、无 DOM/three 依赖，Node 可直接 import（headless 与浏览器共用同一实现）。
//
// 装订参数的单一事实源是 vendor/policy/manifest.json（T0 探针定论后冻结，见
// docs/policy-probe-notes.md）；本模块只按 manifest 描述驱动，不硬编码层结构。
// 前向数值已与 onnxruntime 参考比对（导出期 1e-5，运行期 golden 1e-4，断言 N1）。

// 解析 golden.bin：int32 count + count 组交错 [obs 480][action 29] float32。
export function parseGolden(buffer) {
  const dv = new DataView(buffer);
  const count = dv.getInt32(0, true);
  const f32 = new Float32Array(buffer, 4, (buffer.byteLength - 4) / 4);
  const obsDim = 480, actDim = 29;
  const obs = [], act = [];
  for (let g = 0; g < count; g++) {
    obs.push(f32.slice(g * (obsDim + actDim), g * (obsDim + actDim) + obsDim));
    act.push(f32.slice(g * (obsDim + actDim) + obsDim, (g + 1) * (obsDim + actDim)));
  }
  return { count, obs, act };
}

// 从 ArrayBuffer 构建 weights：逐层 W(out×in 行主序) 与 b，Float32Array 预分配 scratch（零 GC）。
export function buildWeights(buffer, manifest) {
  const dims = manifest.arch.layers;
  const view = new Float32Array(buffer, 0, buffer.byteLength / 4);
  const layers = [];
  let off = 0;
  for (let l = 0; l < dims.length - 1; l++) {
    const outD = dims[l + 1], inD = dims[l];
    const w = view.slice(off, off + outD * inD); off += outD * inD;
    const b = view.slice(off, off + outD); off += outD;
    layers.push({ w, b, outD, inD });
  }
  // 隐层中间缓冲（dims 交替复用，前向零分配）
  const scratch = dims.slice(1).map((d) => new Float32Array(d));
  return { layers, dims, scratch };
}

// 加载策略资产。baseUrl 指向 vendor/policy/（尾斜杠可有可无）；
// fetchImpl 缺省用全局 fetch（浏览器），Node 无头侧传入基于 fs 的实现。
export async function loadPolicy(baseUrl, fetchImpl) {
  const f = fetchImpl ?? ((u) => fetch(u));
  const base = String(baseUrl).replace(/\/+$/, '');
  const manifest = await f(`${base}/manifest.json`).then((r) => {
    if (!r.ok) throw new Error(`manifest 加载失败 HTTP ${r.status}`);
    return r.json();
  });
  const files = manifest.files ?? { weights: 'g1_velocity_v0.weights.bin', golden: 'golden.bin' };
  const weightsBuf = await f(`${base}/${files.weights}`).then((r) => {
    if (!r.ok) throw new Error(`权重加载失败 HTTP ${r.status}`);
    return r.arrayBuffer();
  });
  const weights = buildWeights(weightsBuf, manifest);
  let golden = null;
  try {
    const goldenBuf = await f(`${base}/${files.golden}`).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.arrayBuffer();
    });
    golden = parseGolden(goldenBuf);
  } catch (e) {
    console.warn('[g1Policy] golden.bin 加载失败，policy-selftest 不可用:', e && e.message);
  }
  return { weights, manifest, golden };
}

// MLP 前向：y = W·x + b（W 按 out×in 行主序），隐层 ELU，输出层线性。
// 全 Float32Array + 预分配 scratch，零 GC 压力（方案 4.4）。
export function policyForward(weights, obs) {
  const { layers, scratch } = weights;
  let x = obs;
  for (let l = 0; l < layers.length; l++) {
    const { w, b, outD, inD } = layers[l];
    const y = scratch[l];
    for (let o = 0; o < outD; o++) {
      let s = b[o];
      const row = o * inD;
      for (let i = 0; i < inD; i++) s += x[i] * w[row + i];
      y[o] = l < layers.length - 1 ? (s > 0 ? s : Math.expm1(s)) : s;
    }
    x = y;
  }
  return x; // 复用最后一层 scratch，调用方当帧消费
}

// golden fixture 前向一致性自测（断言 N1）：max|Δ| < 1e-4（方案 8.2）。
export function policySelfTest(weights, golden) {
  if (!golden || !golden.count) return { ok: false, maxAbsErr: NaN, detail: '无 golden 数据' };
  let maxErr = 0;
  for (let g = 0; g < golden.count; g++) {
    const out = policyForward(weights, golden.obs[g]);
    for (let i = 0; i < golden.act[g].length; i++) {
      maxErr = Math.max(maxErr, Math.abs(out[i] - golden.act[g][i]));
    }
  }
  return { ok: maxErr < 1e-4, maxAbsErr: maxErr, detail: `${golden.count} 组 max|Δ|=${maxErr.toExponential(2)}` };
}
