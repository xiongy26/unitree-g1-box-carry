// Shared viewer transforms, so offline contact solving uses the final shelf heights.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { remapClipCarryFull, stackLayout } from '../../src/stackCore.js';
import { FACTORY, derivePickPos, shelfBoardTops, factoryStaticXml } from '../../src/factory.js';
import { motionSignature } from '../../src/manipulationPlan.js';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = stem => JSON.parse(readFileSync(path.join(root, 'motions', `${stem}.json`)));
export function buildReference() {
  const meta = read('walk_meta');
  const tasks = [];
  for (const type of ['largebox', 'plasticbox']) {
    const base = read(meta.carry[type].low.clip);
    const slots = stackLayout(base.box, FACTORY.rackCenter, FACTORY.rackPsi, shelfBoardTops());
    for (let layer = 0; layer < 2; layer++) {
      const clip = read(meta.carry[type][layer === 0 ? 'low' : 'mid'].clip);
      const other = slots.find(s => s.layer === layer && s.col === 1);
      const sl = slots.find(s => s.layer === layer && s.col === 0);
      const g = clip.grasp_frame, r = clip.release_frame;
      const theta = FACTORY.rackPsi - Math.atan2(clip.obj_pos[r][1] - clip.obj_pos[g][1], clip.obj_pos[r][0] - clip.obj_pos[g][0]);
      const rm = remapClipCarryFull(clip, sl.center, base.box.half_size[2], base.box.bbox_center_offset[2], {
        theta, anchor: clip.obj_pos[r].slice(0, 2), target: sl.center.slice(0, 2),
      });
      tasks.push({ signature: motionSignature(clip,base.box,sl.center[2],FACTORY.rackPsi), key: `${type}:${layer}`, box: base.box, columnDelta: other.center[0]-sl.center[0], slot: sl.center, clip, pick: derivePickPos(base),
        rm: Object.fromEntries(Object.entries(rm).map(([k,v]) => [k, ArrayBuffer.isView(v) ? Array.from(v) : v])) });
    }
  }
  let xml = readFileSync(path.join(root,'model/g1.xml'),'utf8');
  xml = xml.replace('</worldbody>', `${factoryStaticXml()}\n<body name="task_box"><freejoint name="task_box_joint"/><geom name="task_box_geom" type="box" size="0.2 0.2 0.17" mass="3"/></body></worldbody>`);
  xml = xml.replace(/file="([^"/]+\.STL)"/g, (_,n) => `file="${path.join(root,'model/assets',n)}"`);
  return { tasks, xml, rack: FACTORY.rack, yaw: FACTORY.rackPsi };
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) console.log(JSON.stringify(buildReference()));
