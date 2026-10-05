// three.js 渲染器：直接从 MjModel/MjData 构建 geom 网格并逐帧同步位姿。
// 不走 mjvScene，直接读 geom_xpos / geom_xmat 实时视图（零拷贝）。
import * as THREE from 'three';

const mjGEOM = { PLANE: 0, HFIELD: 1, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7 };

export class MujocoVisualizer {
  constructor(mujoco, model, scene) {
    this.mujoco = mujoco;
    this.model = model;
    this.scene = scene;
    this.meshes = [];
    this.visibleBoxBodies = null;

    this._m4 = new THREE.Matrix4();
    this.build(model, scene);
  }

  build(model, scene) {
    const n = model.ngeom;
    const gtype = model.geom_type;
    const gsize = model.geom_size;
    const grgba = model.geom_rgba;
    const ggroup = model.geom_group;
    const gdataid = model.geom_dataid;

    for (let i = 0; i < n; i++) {
      // 跳过碰撞凸包 (group 3) 与辅助组：只渲染视觉网格 (group 2) 与场景道具 (group 0)
      if (ggroup[i] === 3 || ggroup[i] === 4 || ggroup[i] === 5) { this.meshes[i] = null; continue; }
      const t = gtype[i];
      const s = [gsize[3 * i], gsize[3 * i + 1], gsize[3 * i + 2]];
      let geo = null;
      if (t === mjGEOM.PLANE) {
        geo = new THREE.PlaneGeometry(2 * s[0], 2 * s[1]);
      } else if (t === mjGEOM.SPHERE) {
        geo = new THREE.SphereGeometry(s[0], 32, 20);
      } else if (t === mjGEOM.CAPSULE) {
        geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 14);
        geo.rotateX(Math.PI / 2);
      } else if (t === mjGEOM.CYLINDER) {
        geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24);
        geo.rotateX(Math.PI / 2);
      } else if (t === mjGEOM.BOX) {
        geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      } else if (t === mjGEOM.MESH) {
        geo = this.meshGeometry(model, gdataid[i]);
      } else {
        this.meshes[i] = null;
        continue;
      }

      const rgba = [grgba[4 * i], grgba[4 * i + 1], grgba[4 * i + 2], grgba[4 * i + 3]];
      const dark = rgba[0] + rgba[1] + rgba[2] < 0.6;
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
        roughness: dark ? 0.55 : 0.7,
        metalness: dark ? 0.15 : 0.05,
        transparent: rgba[3] < 1,
        opacity: rgba[3],
        depthWrite: rgba[3] >= 1,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      const bodyName = this.mujoco.mj_id2name(model, this.mujoco.mjtObj.mjOBJ_BODY.value, model.geom_bodyid[i]);
      mesh.userData.boxBody = /^cartonG?\d+$/.test(bodyName ?? '') ? bodyName : null;
      scene.add(mesh);
      this.meshes[i] = mesh;
    }
  }

  meshGeometry(model, meshId) {
    const vadr = model.mesh_vertadr[meshId], vnum = model.mesh_vertnum[meshId];
    const fadr = model.mesh_faceadr[meshId], fnum = model.mesh_facenum[meshId];
    const verts = model.mesh_vert, faces = model.mesh_face;
    const pos = new Float32Array(vnum * 3);
    for (let i = 0; i < vnum * 3; i++) pos[i] = verts[vadr * 3 + i];
    const idx = new Uint32Array(fnum * 3);
    for (let i = 0; i < fnum * 3; i++) idx[i] = faces[fadr * 3 + i];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    return geo;
  }

  update(data) {
    const xp = data.geom_xpos, xr = data.geom_xmat;
    for (let i = 0; i < this.meshes.length; i++) {
      const mesh = this.meshes[i];
      if (!mesh) continue;
      if (mesh.userData.boxBody && this.visibleBoxBodies) mesh.visible = this.visibleBoxBodies.has(mesh.userData.boxBody);
      const o = 3 * i, q = 9 * i;
      this._m4.set(
        xr[q], xr[q + 1], xr[q + 2], xp[o],
        xr[q + 3], xr[q + 4], xr[q + 5], xp[o + 1],
        xr[q + 6], xr[q + 7], xr[q + 8], xp[o + 2],
        0, 0, 0, 1);
      mesh.matrix.copy(this._m4);
      mesh.matrixWorldNeedsUpdate = true;
    }
  }
}
