// Validate the persisted plans and runtime remapping, without loading WASM.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildReference } from '../lib/manipulation_reference.mjs';
import { applyContactPlan, manipulationRoute } from '../../src/manipulationPlan.js';
const library = JSON.parse(readFileSync(new URL('../../motions/manipulation.json', import.meta.url)));
const reference = buildReference();
for (const task of reference.tasks) {
  const fresh = () => structuredClone(task.rm);
  // Runtime arrays are allocated from the solved frame count, including retreat.
  const rm = fresh();
  const contactPlan = applyContactPlan(rm, task.clip, task.box, { layer: Number(task.key.split(':')[1]), center: task.slot }, reference.yaw, library);
  assert.equal(rm.T, library.plans[task.key].T);
  assert.deepEqual(contactPlan.placement, library.plans[task.key].placement);
  assert.equal(rm.objPos.length, rm.T * 3);
  assert.ok(rm.T > rm.release + 24, 'retreat precedes standing');
  const second = fresh();
  const center = [task.slot[0] + task.columnDelta, ...task.slot.slice(1)];
  applyContactPlan(second, task.clip, task.box, { layer: Number(task.key.split(':')[1]), center }, reference.yaw, library);
  for (let t = 0; t < rm.T; t++) {
    assert.ok(Math.abs(second.rootPos[t * 3] - rm.rootPos[t * 3] - task.columnDelta) < 1e-9);
    assert.ok(Math.abs(second.objPos[t * 3] - rm.objPos[t * 3] - task.columnDelta) < 1e-9);
    assert.deepEqual(second.dof.slice(t * rm.nDof, (t + 1) * rm.nDof), rm.dof.slice(t * rm.nDof, (t + 1) * rm.nDof));
  }
  assert.equal(manipulationRoute(rm).kind, 'carry');
  const broken = structuredClone(library);
  broken.plans[task.key].signature = 'stale';
  assert.throws(() => applyContactPlan(fresh(), task.clip, task.box, { layer: Number(task.key.split(':')[1]), center: task.slot }, reference.yaw, broken), /过期/);
  broken.plans[task.key].signature = library.plans[task.key].signature;
  broken.plans[task.key].rootQuat[0] = [0, 0, 0, 0];
  assert.throws(() => applyContactPlan(fresh(), task.clip, task.box, { layer: Number(task.key.split(':')[1]), center: task.slot }, reference.yaw, broken), /损坏/);
  const badPhase = structuredClone(library);
  badPhase.plans[task.key].placement.feetPlanted = badPhase.plans[task.key].release;
  assert.throws(() => applyContactPlan(fresh(), task.clip, task.box, { layer: Number(task.key.split(':')[1]), center: task.slot }, reference.yaw, badPhase), /阶段数据损坏/);
  console.log(`PASS ${task.key}: playback, column translation, stale and corrupt data rejection`);
}
