// 工厂码垛场景唯一布局源：MJCF、路径避障、小地图共同消费。
// 作业目标为蓝色钢制料箱架，两个可用层板均参与真实箱体接触。
// 工作区由多排料箱架和地面交接区组成。
// 静态设施除作业货架承重面外均为视觉几何；机器人避障由路径规划处理。
// 动捕取放仍属于演示近似，不能等同于完整接触控制。

// ---------------- 布局常量表（三处消费的唯一事实源） ----------------
export const FACTORY = {
  // 地面半边长（scene.xml floor size 同步改；总场地 16×14m）
  floorHalf: [8, 7],
  // 走廊进抵方向：机器人朝 +y 走向货架（取箱区在货架 −y 侧）
  rackPsi: Math.PI / 2,
  // 活动货架格网中心 S_rack（stackLayout 槽位基准；取箱点由它反推派生，见 derivePickPos）
  rackCenter: [2.2, 2.2],
  // 作业货架：两列下部作业槽位，上部两层用于静态料箱展示。
  rack: {
    cells: 2,
    slotGap: 0.065, // Keep space for bin lips and hands between occupied columns.
    layers: 2,
    cellHalfW: 0.26,
    depthHalf: 0.35,
    boardTopZ: [0.04, 0.65],
    decoTopZ: 1.85, // 上部展示层
    boardT: 0.03,   // 层板厚（板中心 z = boardTop − boardT/2）
    uprightH: 2.15,
    postR: 0.05,    // 立柱半径
  },
  // 取箱暂存区标示（地面标记 geom + 小地图虚线框；1.4×0.8m 完整覆盖两箱型派生取箱点
  // largebox (2.2,0.487) / plasticbox (2.2,0.536)，R2）
  pickZone: { center: [2.2, 0.5], half: [0.7, 0.4] },
  // 潜伏行只读记录（stackCore stagingPos 的既有语义，不得移动；y 行与 stackCore
  // STAGING_Y 同源，headless F2c 校验一致性）
  staging: { x0: -3, dx: 0.6, y: { largebox: 3.0, plasticbox: 4.2, smallbox: 5.4 } },
  // 静态障碍矩形表（pillar 用 hx=hy=r 的等价矩形 + 圆形渲染标记）。
  // z = 底面高、height = 高度（geom 中心 z = z + height/2）；rgba 缺省按 kind 取色。
  obstacles: [
    // 立柱设在外围，作业通道保持开放。
    { kind: 'pillar', name: 'pillar_1', cx: -6.4, cy: -3.5, yaw: 0, hx: 0.15, hy: 0.15, z: 0, height: 2.6 },
    { kind: 'pillar', name: 'pillar_2', cx: -1.6, cy: -2.0, yaw: 0, hx: 0.15, hy: 0.15, z: 0, height: 2.6 },
    { kind: 'pillar', name: 'pillar_3', cx: 4.6, cy: 4.6, yaw: 0, hx: 0.15, hy: 0.15, z: 0, height: 2.6 },
    // 安全围栏 ×2（低栏）
    { kind: 'fence', name: 'fence_a', cx: -4.0, cy: -0.5, yaw: 0, hx: 1.5, hy: 0.12, z: 0, height: 0.6 },
    { kind: 'fence', name: 'fence_b', cx: 3.0, cy: -2.5, yaw: 0, hx: 1.2, hy: 0.12, z: 0, height: 0.6 },
    // 托盘堆 ×2
    { kind: 'pallet', name: 'pallet_a', cx: -3.2, cy: -2.0, yaw: 0, hx: 0.6, hy: 0.4, z: 0, height: 0.15 },
    { kind: 'pallet', name: 'pallet_b', cx: 4.8, cy: 1.2, yaw: 0, hx: 0.6, hy: 0.4, z: 0, height: 0.15 },
    // 道具货架 ×2（纯装饰，全 ghost 含板——无放置语义，不参与 exclude）
    { kind: 'rack', name: 'deco_rack_a', cx: -2.8, cy: 0.8, yaw: 0, hx: 0.75, hy: 0.35, z: 0, height: 0.9 },
    { kind: 'rack', name: 'deco_rack_b', cx: -2.8, cy: 2.4, yaw: 0, hx: 0.75, hy: 0.35, z: 0, height: 0.9 },
    { kind: 'rack', name: 'storage_c', cx: 4.0, cy: 2.2, yaw: 0, hx: 0.75, hy: 0.35, z: 0, height: 1.85 },
    { kind: 'rack', name: 'storage_d', cx: 5.65, cy: 2.2, yaw: 0, hx: 0.75, hy: 0.35, z: 0, height: 1.85 },
    { kind: 'rack', name: 'storage_e', cx: -5.0, cy: 0.8, yaw: 0, hx: 0.75, hy: 0.35, z: 0, height: 1.85 },
    { kind: 'rack', name: 'storage_f', cx: -5.0, cy: 2.4, yaw: 0, hx: 0.75, hy: 0.35, z: 0, height: 1.85 },
    // 围界 ×4（场地封边）
    { kind: 'wall', name: 'wall_n', cx: 0, cy: 7, yaw: 0, hx: 8, hy: 0.1, z: 0, height: 0.3 },
    { kind: 'wall', name: 'wall_s', cx: 0, cy: -7, yaw: 0, hx: 8, hy: 0.1, z: 0, height: 0.3 },
    { kind: 'wall', name: 'wall_e', cx: 8, cy: 0, yaw: 0, hx: 0.1, hy: 7, z: 0, height: 0.3 },
    { kind: 'wall', name: 'wall_w', cx: -8, cy: 0, yaw: 0, hx: 0.1, hy: 7, z: 0, height: 0.3 },
  ],
};

// 按 kind 的渲染配色（MJCF rgba 与小地图色由表+kind 推导，避免逐条目重复）
const KIND_RGBA = {
  pillar: '0.50 0.52 0.55 1',
  fence: '0.85 0.65 0.15 1',
  pallet: '0.60 0.45 0.28 1',
  rack: '0.055 0.13 0.46 1', // 蓝色钢制（实际绘制走 DECO_RACK_*；此为兜底色）
  wall: '0.42 0.44 0.47 1',
};
// 货架蓝色钢制配色（2026-10 料箱工厂改造，参考 #1560BD~#1D4E9E 区间微调）。
// 只改颜色；FACTORY.rack 几何字段与 obstacles 表逐字未动。
const RACK_BOARD_RGBA = '0.055 0.13 0.46 1';     // 活动货架层板（钢蓝）
const RACK_POST_RGBA = '0.045 0.10 0.36 1';      // 活动货架立柱（深钢蓝）
const RACK_BEAM_RGBA = '0.035 0.075 0.28 1';      // 活动货架横梁装饰（fx_rack_beam_N/S）
const DECO_RACK_POST_RGBA = '0.045 0.10 0.36 1'; // 装饰货架角柱
const DECO_RACK_BOARD_RGBA = '0.055 0.13 0.46 1';// 装饰货架层板
const DECO_BIN_RGBA = '0.045 0.10 0.36 1';       // 装饰料箱主色（深蓝塑料，与搬运箱同族）
const DECO_PARTS_RGBA = '0.60 0.47 0.30 1';     // 料箱内“零件”块（约半数装饰箱）
const PICKZONE_RGBA = '0.95 0.80 0.25 0.35';  // 取箱区标示（半透明黄，保留）
// 黄黑警示胶带（floorTapeXml，切片 D）
const TAPE_YELLOW = '0.93 0.78 0.10 1';
const TAPE_BLACK = '0.07 0.07 0.08 1';
const TAPE_HW = 0.03;   // 胶带半宽（60mm 胶带）
const TAPE_HT = 0.004;  // 半厚：段体 z∈[0,0.008]，底面贴地（底面朝下被背面剔除，与地面
                        // 无 z-fight——与 fx_pickzone_mark 同构）
const TAPE_SEG = 0.4;   // 黄黑交替周期（0.2 黄 + 0.2 黑）
// 主通道双实线（东西向）：取箱区/活动货架与潜伏区之间的工作走廊；x 内缩 0.4m 避开
// 围界。线位经过 pallet_b 上方属地面划线常态（pallet 实体压在划线上）。
const MAIN_AISLE_Y = [1.25, 1.55];

// geom 命名前缀（F1 无头断言按名抽全查）
const FX_PREFIX = 'fx_';

// ---------------- 小工具：XML 数值格式化与 rgba 明暗派生（视觉层专用） ----------------
// f4：4 位小数去尾零（视觉壳/胶带大量派生数值，避免 0.18550000000000002 进 XML）
function f4(v) {
  const s = (Math.abs(v) < 5e-5 ? 0 : v).toFixed(4);
  return s.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
}
function parseRgba(s) {
  return s.trim().split(/\s+/).map(Number);
}
// 同色明暗派生（k>1 提亮/“箱内”观感，k<1 压暗/翻边肋条观感），clamp 到 [0,1]
function fmtRgba(c, k = 1) {
  const cl = (v) => Math.max(0, Math.min(1, v));
  return `${f4(cl(c[0] * k))} ${f4(cl(c[1] * k))} ${f4(cl(c[2] * k))} ${c[3]}`;
}

// ---------------- 装饰货架与装饰料箱常量（2026-10 料箱工厂改造，切片 C） ----------------
// 装饰货架改蓝色钢制并加高加密（纯装饰 ghost）：obstacles 表 deco_rack_a/b 的
// cx/cy/yaw/hx/hy/height 逐字未动——避障只消费平面 footprint，视觉加高无碰撞影响。
const DECO_RACK_H = 1.8;                        // 视觉柱高（表值 height=0.9 只管避障口径）
const DECO_BOARD_T = 0.03;                      // 层板厚
const DECO_BOARD_Z = [0.30, 0.70, 1.10, 1.50];  // 4 层板中心高（顶箱含翻边 1.787 < 柱高）
const DECO_BIN = { hx: 0.20, hy: 0.135, hz: 0.13 }; // 装饰料箱半尺寸（0.40×0.27×0.26m）
// 每层摆箱数（0-2，部分层留空更自然，摆满度参考用户照片；两架各异且确定性）
const DECO_BIN_FILL = { deco_rack_a: [2, 1, 2, 0], deco_rack_b: [1, 2, 0, 2] };

// ---------------- 开口料箱视觉壳生成器（切片 A/C 共用，单一事实源） ----------------
// 输出“料箱中心局部系”的 geom 参数表（type 恒 box）：底板 + 4 面薄壁 + 箱口翻边
//（顶缘外挑一圈）± 外壁横向肋条。壳外廓与物理外接盒严格重合（壁外表面 = ±hx/±hy、
// 底 = −hz、壁顶 = +hz，翻边高出 lipT）。调用方负责拼 XML 并声明 contype="0"
// conaffinity="0" mass="0"（binShellGeomXml 统一带出；R7：一切碰撞解耦在 XML 期声明）。
// 角部避让规则：x 向壁全长、y 向壁嵌在两 x 向壁内侧面之间；翻边/肋条同规则——任意
// 两件不出现共面重叠面（只允许背靠背或边贴边），避免同色 z-fighting。
function openBinShellGeoms(prefix, hx, hy, hz, opts = {}) {
  const t = opts.wall ?? 0.010;  // 壁厚（8-12mm 观感档中值）
  const lipW = opts.lipW ?? 0.018; // 箱口翻边外挑
  const lipT = opts.lipT ?? 0.012; // 翻边厚
  const ribD = opts.ribD ?? 0.005; // 外壁横向肋条外挑
  const ribH = opts.ribH ?? 0.024; // 肋条高
  const base = parseRgba(opts.rgba ?? DECO_BIN_RGBA);
  const wallR = fmtRgba(base);
  const botR = fmtRgba(base, 1.15); // 内底板略亮（箱内观感）
  const lipR = fmtRgba(base, 0.78); // 翻边略深
  const ribR = fmtRgba(base, 0.9);
  const g = [];
  const add = (suffix, size, pos, rgba) => g.push({ name: `${prefix}_${suffix}`, size, pos, rgba });
  add('bottom', [hx, hy, t / 2], [0, 0, -hz + t / 2], botR);
  const wz = t / 2, wh = hz - t / 2; // 壁坐在底板上沿、顶面与外接盒顶齐平
  add('wall_xp', [t / 2, hy, wh], [hx - t / 2, 0, wz], wallR);
  add('wall_xn', [t / 2, hy, wh], [-(hx - t / 2), 0, wz], wallR);
  add('wall_yp', [hx - t, t / 2, wh], [0, hy - t / 2, wz], wallR);
  add('wall_yn', [hx - t, t / 2, wh], [0, -(hy - t / 2), wz], wallR);
  add('lip_yp', [hx + lipW, lipW / 2, lipT / 2], [0, hy + lipW / 2, hz + lipT / 2], lipR);
  add('lip_yn', [hx + lipW, lipW / 2, lipT / 2], [0, -(hy + lipW / 2), hz + lipT / 2], lipR);
  add('lip_xp', [lipW / 2, hy, lipT / 2], [hx + lipW / 2, 0, hz + lipT / 2], lipR);
  add('lip_xn', [lipW / 2, hy, lipT / 2], [-(hx + lipW / 2), 0, hz + lipT / 2], lipR);
  if (opts.rib) {
    const zr = -0.45 * hz; // 下腹部一条横向肋（避开底板与翻边的贴面区）
    add('rib_xp', [ribD / 2, hy - 2 * t, ribH / 2], [hx + ribD / 2, 0, zr], ribR);
    add('rib_xn', [ribD / 2, hy - 2 * t, ribH / 2], [-(hx + ribD / 2), 0, zr], ribR);
    add('rib_yp', [hx - 2 * t, ribD / 2, ribH / 2], [0, hy + ribD / 2, zr], ribR);
    add('rib_yn', [hx - 2 * t, ribD / 2, ribH / 2], [0, -(hy + ribD / 2), zr], ribR);
  }
  return g;
}

// 视觉壳 → MJCF geom 行（切片 A 的 carton 双体与切片 C 的装饰料箱统一出口）：
// off = 壳中心相对挂载系原点的偏移（carton 传 bbox_center_offset 与主 geom 同心；
// 装饰料箱传 [0,0,0] 并包在静态 body 里）。全部 XML 期声明零物理属性（R7 惯例）。
export function binShellGeomXml(prefix, hx, hy, hz, opts = {}, off = [0, 0, 0], indent = '    ') {
  return openBinShellGeoms(prefix, hx, hy, hz, opts).map((g) =>
    `${indent}<geom name="${g.name}" type="box" size="${g.size.map(f4).join(' ')}"`
    + ` pos="${g.pos.map((v, i) => f4(v + off[i])).join(' ')}" rgba="${g.rgba}"`
    + ` contype="0" conaffinity="0" mass="0"/>`
  ).join('\n');
}

// ---------------- 纯函数：货架几何 ----------------

// 层板顶高副本（boardTopZ[layer]===null 表示该层落地=地面格，z 贡献 0）。
// 返回切片防外部改表（D2 单一事实源只读语义）。
export function shelfBoardTops() {
  return [...FACTORY.rack.boardTopZ, 1.25];
}

// 活动货架外廓避障矩形（含立柱外缘）。role='target'：机器人放置站位距货架立面实测
// 可为负（D4），全膨胀会让 WALK 末段必踩膨胀区 → N19b 假红；target 化后走
// targetShrinkR 收缩 + 终点透明化（原矩形仍硬检查），与"目标箱设计性贴近"同一机制。
export function rackRect() {
  const r = FACTORY.rack;
  return {
    cx: FACTORY.rackCenter[0], cy: FACTORY.rackCenter[1],
    yaw: 0, // 货架长轴沿 x（列沿 x 排布，进抵方向 rackPsi=+y 由 stackLayout yaw 承担）
    hx: 2 * r.cellHalfW + 2 * r.postR,
    hy: r.depthHalf + r.postR,
    role: 'target', key: 'static:active_rack',
  };
}

// 取箱点派生（D2）：P0 = S_rack − carryDisp(clip)·û(rackPsi)，使 buildConfig 里
// computeStackFrameCarry(clip, P0, rackPsi) 输出的 frame.S 精确回到货架格网中心
// （函数本体零改动，浮点残差 ~1e-16）。carryDisp 与 stackCore.computeStackFrameCarry
// 同公式（grasp→release 的 obj 水平位移）；此处独立实现以免 factory→stackCore 依赖。
export function derivePickPos(clip) {
  const g = clip.grasp_frame, r = clip.release_frame;
  const dx = clip.obj_pos[r][0] - clip.obj_pos[g][0];
  const dy = clip.obj_pos[r][1] - clip.obj_pos[g][1];
  const disp = Math.hypot(dx, dy);
  const ux = Math.cos(FACTORY.rackPsi), uy = Math.sin(FACTORY.rackPsi);
  return [FACTORY.rackCenter[0] - disp * ux, FACTORY.rackCenter[1] - disp * uy];
}

// 静态避障矩形全集（D4）：活动货架 + obstacles 表逐条。rect 契约
// {cx,cy,yaw,hx,hy,role,key} 不变；key='static:<name>'（活动货架 'static:active_rack'）。
export function factoryObstacleRects() {
  const rack = rackRect();
  return [
    rack,
    ...FACTORY.obstacles.map((o) => ({
      cx: o.cx, cy: o.cy, yaw: o.yaw, hx: o.hx, hy: o.hy,
      role: 'static', key: `static:${o.name}`,
    })),
  ];
}

// 两层承重板与机器人排除接触，保留完整箱体-层板物理接触。
export const FACTORY_SOLID_BODIES = [
  { body: 'fx_rack_board_L1', geom: 'fx_rack_board_L1_geom' },
  { body: 'fx_rack_board_L2', geom: 'fx_rack_board_L2_geom' },
];

// 活动货架板的世界位姿/尺寸（factoryStaticXml 与 headless F2a 同源；yaw 恒 0——
// 板长轴沿 x，与 rackRect 一致）。装饰板（ghost）不在此表——它不承重、不进格位。
export function rackBoardSpecs() {
  const r = FACTORY.rack;
  const [cx, cy] = FACTORY.rackCenter;
  return FACTORY_SOLID_BODIES.map((b, i) => {
    const layer = i;
    const top = r.boardTopZ[layer];
    return {
      ...b, layer,
      pos: [cx, cy, top - r.boardT / 2],
      size: [2 * r.cellHalfW, r.depthHalf, r.boardT / 2],
      top,
    };
  });
}

// ---------------- XML 片段生成（消费 1） ----------------

function obstacleXml(o) {
  const rgba = o.rgba ?? KIND_RGBA[o.kind] ?? '0.5 0.5 0.5 1';
  const name = `${FX_PREFIX}${o.name}`;
  const zc = o.z + o.height / 2;
  if (o.kind === 'equipment') return visualGeom(o.name, [o.hx, o.hy, o.height / 2], [o.cx, o.cy, zc], rgba, 'group="5"');
  if (o.kind === 'pallet') return palletXml(o.name, o.cx, o.cy, o.hx, o.hy);
  if (o.kind === 'fence') {
    const out = [];
    for (let i = 0; i <= 6; i++) out.push(visualGeom(`${o.name}_post${i}`, [0.035, 0.035, 0.5], [o.cx - o.hx + i * o.hx / 3, o.cy, 0.5], '0.90 0.65 0.10 1'));
    for (const z of [0.15, 0.95]) out.push(visualGeom(`${o.name}_rail${z}`, [o.hx, 0.025, 0.025], [o.cx, o.cy, z], '0.90 0.65 0.10 1'));
    for (let i = 0; i < 24; i++) out.push(visualGeom(`${o.name}_mesh${i}`, [0.007, 0.008, 0.38], [o.cx - o.hx + (i + 0.5) * o.hx / 12, o.cy, 0.55], '0.25 0.29 0.30 1'));
    return out.join('\n');
  }
  if (o.kind === 'pillar') {
    // 立柱：cylinder（size = r, 半高）
    return `    <geom name="${name}" type="cylinder" size="${o.hx} ${o.height / 2}" pos="${o.cx} ${o.cy} ${zc}" rgba="${rgba}"/>`;
  }
  return `    <geom name="${name}" type="box" size="${o.hx} ${o.hy} ${o.height / 2}" pos="${o.cx} ${o.cy} ${zc}"${o.yaw ? ` euler="0 0 ${(o.yaw * 180) / Math.PI}"` : ''} rgba="${rgba}"/>`;
}

// 道具货架（装饰，全 ghost 含板/柱/料箱）：蓝色钢制 4 层板 + 4 角柱 + 每层 0-2 只
// 装饰料箱（binShellGeomXml 简化版：无肋条；约半数箱内放“零件”块，参考照片摆满度）。
// 外廓 = 表内矩形（hx/hy 含柱），板/柱几何由外廓推导，与 rackRect 无格位语义耦合；
// 避障 footprint 不变（不新增 factoryObstacleRects 条目、不改表值）。
// geom 命名保持 F1 断言兼容：fx_deco_rack_{a,b}_board1..、_post{W|E}{S|N}。
function decoRackXml(o) {
  const pr = 0.05, t = DECO_BOARD_T;
  const bx = o.hx - pr, by = o.hy - pr;
  const deg = (o.yaw * 180) / Math.PI;
  const rot = deg ? ` euler="0 0 ${deg}"` : '';
  const cy = Math.cos(o.yaw), sy = Math.sin(o.yaw);
  const lines = [];
  for (const [i, zc] of DECO_BOARD_Z.entries()) {
    lines.push(`    <geom name="${FX_PREFIX}${o.name}_board${i + 1}" type="box" size="${f4(bx)} ${f4(by)} ${f4(t / 2)}" pos="${o.cx} ${o.cy} ${f4(zc)}"${rot} rgba="${DECO_RACK_BOARD_RGBA}" contype="0" conaffinity="0" mass="0"/>`);
  }
  for (const sx of [-1, 1]) for (const sy2 of [-1, 1]) {
    // 柱心在旋转后局部角点 (±bx, ±by) → 世界系
    const px = o.cx + cy * sx * bx - sy * sy2 * by;
    const py = o.cy + sy * sx * bx + cy * sy2 * by;
    const corner = `${sx < 0 ? 'W' : 'E'}${sy2 < 0 ? 'S' : 'N'}`;
    lines.push(`    <geom name="${FX_PREFIX}${o.name}_post${corner}" type="cylinder" size="${pr} ${f4(DECO_RACK_H / 2)}" pos="${f4(px)} ${f4(py)} ${f4(DECO_RACK_H / 2)}" rgba="${DECO_RACK_POST_RGBA}" contype="0" conaffinity="0" mass="0"/>`);
  }
  // 每层装饰料箱：静态 body（无关节，bin 局部系 = 料箱中心，随货架 euler 旋转）；
  // 2 箱沿货架长轴对称摆放、单箱居中。长轴偏移按货架 yaw 旋转到世界系。
  const fill = DECO_BIN_FILL[o.name] ?? DECO_BIN_FILL.deco_rack_a;
  fill.forEach((n, l) => {
    const zTop = DECO_BOARD_Z[l] + t / 2;
    const spots = n >= 2 ? [-(DECO_BIN.hx + 0.02), DECO_BIN.hx + 0.02] : n === 1 ? [0] : [];
    spots.forEach((x, c) => {
      const p = `${FX_PREFIX}${o.name}_bin_l${l}c${c}`;
      const bxw = o.cx + x * cy, byw = o.cy + x * sy;
      lines.push(`    <body name="${p}" pos="${f4(bxw)} ${f4(byw)} ${f4(zTop + DECO_BIN.hz)}"${rot}>`);
      lines.push(binShellGeomXml(p, DECO_BIN.hx, DECO_BIN.hy, DECO_BIN.hz,
        { rgba: DECO_BIN_RGBA }, [0, 0, 0], '      '));
      if ((l + c) % 2 === 0) {
        // 箱内“零件”块：坐在内底板上表面，约占半箱高（部分箱内装有零件）
        const ph = DECO_BIN.hz * 0.5;
        lines.push(`      <geom name="${p}_parts" type="box" size="${f4(DECO_BIN.hx * 0.62)} ${f4(DECO_BIN.hy * 0.62)} ${f4(ph / 2)}" pos="0 0 ${f4(-DECO_BIN.hz + 0.010 + ph / 2)}" rgba="${DECO_PARTS_RGBA}" contype="0" conaffinity="0" mass="0"/>`);
      }
      lines.push(`    </body>`);
    });
  });
  return lines.join('\n');
}

// 两个物理承重板 + 上部展示层；作业层保持前侧开放。
function activeRackXml() {
  const [cx, cy] = FACTORY.rackCenter;
  const r = FACTORY.rack, out = [];
  for (const b of rackBoardSpecs()) {
    out.push(`<body name="${b.body}" pos="${b.pos.join(' ')}"><geom name="${b.geom}" type="box" size="${b.size.join(' ')}" rgba="${RACK_BOARD_RGBA}" contype="1" conaffinity="1"/></body>`);
  }
  for (const z of [1.25, r.decoTopZ]) {
    out.push(visualGeom(`active_board${z}`, [0.52, 0.35, 0.015], [cx, cy, z - 0.015], RACK_BOARD_RGBA));
    for (const dx of [-0.23, 0.23]) {
      out.push(`<body name="fx_active_bin${z}_${dx}" pos="${cx + dx} ${cy} ${z + DECO_BIN.hz}">${binShellGeomXml(`fx_active_bin${z}_${dx}`, DECO_BIN.hx, DECO_BIN.hy, DECO_BIN.hz)}</body>`);
    }
  }
  for (const dx of [-0.57, 0.57]) for (const dy of [-0.35, 0.35]) {
    out.push(visualGeom(`active_post${dx}_${dy}`, [0.028, 0.028, r.uprightH / 2], [cx + dx, cy + dy, r.uprightH / 2], RACK_POST_RGBA));
    for (let k = 0; k < 12; k++) out.push(visualGeom(`post_hole${dx}_${dy}_${k}`, [0.01, 0.001, 0.014], [cx + dx, cy + dy - 0.029, 0.12 + k * 0.14], '0.04 0.09 0.18 1'));
  }
  for (const z of [...r.boardTopZ, 1.25, r.decoTopZ]) for (const dy of [-0.33, 0.33]) {
    out.push(visualGeom(`active_beam${z}_${dy}`, [0.52, 0.02, 0.018], [cx, cy + dy, z - 0.033], RACK_BEAM_RGBA));
  }
  return out.join('\n');
}

function visualGeom(name, size, pos, rgba, extra = '') {
  return `<geom name="fx_${name}" type="box" size="${size.map(f4).join(' ')}" pos="${pos.map(f4).join(' ')}" rgba="${rgba}" contype="0" conaffinity="0" mass="0" ${extra}/>`;
}

function palletXml(name, cx, cy, hx, hy, solid = false) {
  const wood = '0.62 0.43 0.24 1';
  const out = [];
  if (solid) {
    const b = rackBoardSpecs()[0];
    out.push(`<body name="${b.body}" pos="${b.pos.join(' ')}"><geom name="${b.geom}" type="box" size="${b.size.join(' ')}" rgba="${wood}" contype="1" conaffinity="1"/></body>`);
  }
  for (let i = 0; i < 7; i++) {
    const x = cx - hx + (i + 0.5) * 2 * hx / 7;
    out.push(visualGeom(`${name}_slat${i}`, [hx / 7 - 0.009, hy, 0.012], [x, cy, 0.108], wood));
  }
  for (const x of [-hx + 0.09, 0, hx - 0.09]) {
    out.push(visualGeom(`${name}_runner${x}`, [0.065, hy, 0.04], [cx + x, cy, 0.05], '0.43 0.29 0.16 1'));
  }
  return out.join('\n');
}

// 供料线放在取箱区侧面；末端地面取料位沿用真实动捕抓取高度。
function productionXml() {
  const out = [];
  // 车间背景墙：前侧开放，便于观察工作区。
  out.push(visualGeom('back_wall', [8, 0.06, 1.8], [0, 7, 1.8], '0.65 0.70 0.72 1'));
  for (let i = 0; i < 8; i++) {
    out.push(visualGeom(`wall_panel${i}`, [0.018, 0.025, 1.8], [-7 + i * 2, 6.92, 1.8], '0.42 0.48 0.51 1'));
    out.push(visualGeom(`window${i}`, [0.78, 0.02, 0.42], [-7 + i * 2, 6.91, 2.55], '0.24 0.39 0.46 1'));
  }
  return out.join('\n');
}

// ---------------- 地面黄黑警示胶带（切片 D，纯视觉 ghost 可踩过） ----------------
// 轴对齐胶带线：TAPE_SEG 周期黄黑交替段模拟斜纹警示胶带观感。段体 z∈[0,0.008]、
// 顶面朝上/底面朝下被背面剔除，与地面无 z-fight；潜伏行线整体压在箱体足迹内侧
//（箱底板 z∈[0,0.01] 包住线段，仅在箱间空隙露出——实仓“地面划线被货物压住”常态）。
function tapeLaneXml(lane, x0, y0, x1, y1) {
  const horiz = Math.abs(y1 - y0) < 1e-9;
  const a0 = horiz ? Math.min(x0, x1) : Math.min(y0, y1);
  const a1 = horiz ? Math.max(x0, x1) : Math.max(y0, y1);
  const fixed = horiz ? y0 : x0;
  const lines = [];
  let i = 0;
  for (let s = a0; s < a1 - 1e-9; s += TAPE_SEG, i++) {
    const e = Math.min(s + TAPE_SEG, a1);
    if (e - s < 1e-6) break; // 防浮点漂移残余段：半长四舍五入为 0 的 geom 会编译失败
    const half = (e - s) / 2, c = (s + e) / 2;
    const rgba = i % 2 === 0 ? TAPE_YELLOW : TAPE_BLACK;
    const size = horiz ? `${f4(half)} ${TAPE_HW} ${TAPE_HT}` : `${TAPE_HW} ${f4(half)} ${TAPE_HT}`;
    const pos = horiz ? `${f4(c)} ${f4(fixed)} ${TAPE_HT}` : `${f4(fixed)} ${f4(c)} ${TAPE_HT}`;
    lines.push(`    <geom name="${FX_PREFIX}tape_${lane}_${i}" type="box" size="${size}" pos="${pos}" rgba="${rgba}" contype="0" conaffinity="0" mass="0"/>`);
  }
  return lines.join('\n');
}

// 胶带线位（全部轴对齐）：取箱区边界一圈（fx_pickzone_mark 外圈）+ 潜伏区每箱型一行
// 中心线（6 潜伏位 = stackCore.MAX_BOXES，x 两侧各让 0.5m；不引入新 FACTORY 字段）
// + 主通道双实线（MAIN_AISLE_Y）。命名见文件头“装饰件命名总表”。
function floorTapeXml() {
  const [pzx, pzy] = FACTORY.pickZone.center, [pzhx, pzhy] = FACTORY.pickZone.half;
  const b = TAPE_HW;
  const stg = FACTORY.staging;
  const lanes = [
    ['pzn', pzx - pzhx - b, pzy + pzhy + b, pzx + pzhx + b, pzy + pzhy + b],
    ['pzs', pzx - pzhx - b, pzy - pzhy - b, pzx + pzhx + b, pzy - pzhy - b],
    ['pzw', pzx - pzhx - b, pzy - pzhy, pzx - pzhx - b, pzy + pzhy],
    ['pze', pzx + pzhx + b, pzy - pzhy, pzx + pzhx + b, pzy + pzhy],

    ...FACTORY.obstacles.filter(o => o.kind === 'rack').flatMap(o => [
      [`rack_${o.name}_s`, o.cx-o.hx-0.08, o.cy-o.hy-0.08, o.cx+o.hx+0.08, o.cy-o.hy-0.08],
      [`rack_${o.name}_n`, o.cx-o.hx-0.08, o.cy+o.hy+0.08, o.cx+o.hx+0.08, o.cy+o.hy+0.08],
    ]),
    ['work_s', -0.7, -0.8, 3, -0.8],
    ['work_n', 0.8, 2.8, 3, 2.8],
  ];
  return lanes.map((l) => tapeLaneXml(...l)).join('\n');
}

// 静态 MJCF 片段（无任何 joint；直接挂 worldbody 的 geom 属 world body，
// model.geom_pos 即世界坐标——headless F2a 按名读位姿与表对账的口径基础）。
export function factoryStaticXml() {
  const [pzx, pzy] = FACTORY.pickZone.center, [pzhx, pzhy] = FACTORY.pickZone.half;
  const parts = [
    '    <!-- 工厂静态几何（src/factory.js 生成：层板物理+exclude，其余 ghost；无关节，nq 契约） -->',
    activeRackXml(),
    productionXml(),
    ...FACTORY.obstacles.map((o) => (o.kind === 'rack' ? decoRackXml(o) : obstacleXml(o))),
    // 取箱暂存区地面标示（薄板 ghost，机器人可踩过）
    `    <geom name="${FX_PREFIX}pickzone_mark" type="box" size="${pzhx} ${pzhy} 0.004" pos="${pzx} ${pzy} 0.004" rgba="${PICKZONE_RGBA}"/>`,
    // 黄黑警示胶带线（取箱区边界/潜伏区行/主通道，切片 D）
    floorTapeXml(),
  ];
  return parts.join('\n') + '\n';
}
