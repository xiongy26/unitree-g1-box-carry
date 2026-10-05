#!/usr/bin/env python3
"""Check interpolated contact plans against actual MuJoCo geometry, including bin lips."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import mujoco as mj
ROOT=Path(__file__).resolve().parents[2]

def slerp(a,b,u):
    a=np.asarray(a);b=np.asarray(b);dot=float(a@b)
    if dot<0:b=-b;dot=-dot
    if dot>.9995:
        q=(1-u)*a+u*b;return q/np.linalg.norm(q)
    theta=np.arccos(np.clip(dot,-1,1));return (np.sin((1-u)*theta)*a+np.sin(u*theta)*b)/np.sin(theta)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only');parser.add_argument('--hz',type=int,default=120);parser.add_argument('--library',type=Path,default=ROOT/'motions/manipulation.json');args=parser.parse_args()
    reference=json.loads(subprocess.check_output(['node',str(ROOT/'tools/lib/manipulation_reference.mjs')],text=True))
    library=json.loads(args.library.read_text());failures=[]
    for task in reference['tasks']:
        key=task['key']
        if args.only and key!=args.only:continue
        entry=library['plans'][key]
        assert entry['signature']==task['signature'],f'{key}: stale source/layout'
        box=task['box'];half=np.array(box['half_size']);off=np.array(box['bbox_center_offset'])
        size=half+[.018,.018,.006];offset=off+[0,0,.006]
        xml=reference['xml'].replace('name="task_box_geom" type="box" size="0.2 0.2 0.17" mass="3"',
            f'name="task_box_geom" type="box" size="{" ".join(map(str,size))}" pos="{" ".join(map(str,offset))}" mass="3"')
        # Model already placed bins explicitly, independently of the solver.
        neighbors=[]
        current=np.array(task['slot']); current[2]+=.006
        locations=[current]
        if key.endswith(':1'):
            for x in (task['slot'][0],task['slot'][0]+task['columnDelta']):
                locations.append(np.array([x,task['slot'][1],reference['rack']['boardTopZ'][0]+half[2]+.006]))
        for i,loc in enumerate(locations):
            neighbors.append(f'check_neighbor_{i}')
            xml=xml.replace('</worldbody>',f'<geom name="check_neighbor_{i}" type="box" size="{" ".join(map(str,size))}" pos="{" ".join(map(str,loc))}" euler="0 0 {reference["yaw"]}" contype="0" conaffinity="0" mass="0"/></worldbody>')
        model=mj.MjModel.from_xml_string(xml);data=mj.MjData(model)
        neighbor_ids=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,n) for n in neighbors]
        gid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'task_box_geom')
        jid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'task_box_joint');ba=model.jnt_qposadr[jid]
        joints=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n) for n in task['clip']['joint_names']];qa=[model.jnt_qposadr[j] for j in joints]
        wrists=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,f'{s}_wrist_yaw_link') for s in ('left','right')]
        robot=[];shelf=[];feet=[]
        for k in range(model.ngeom):
            name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,k) or '';body=mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,model.geom_bodyid[k]) or '';mesh=model.geom_dataid[k]
            rubber=mesh>=0 and 'rubber_hand' in (mj.mj_id2name(model,mj.mjtObj.mjOBJ_MESH,mesh) or '')
            if (model.geom_contype[k] and body not in ('world','task_box') and not body.startswith('fx_')) or rubber:robot.append(k)
            if (name.startswith('fx_active_') and 'hole' not in name) or name.startswith('fx_rack_board'):shelf.append(k)
            if 'ankle_roll' in body and model.geom_type[k]==mj.mjtGeom.mjGEOM_SPHERE:feet.append(k)
        rp=np.array(entry['rootPos']);rq=np.array(entry['rootQuat']);q=np.array(entry['dof']);op=np.array(entry['objPos']);oq=np.array(entry['objQuat'])
        placement=entry.get('placement')
        if placement:
            planted=placement['feetPlanted'];withdrawn=placement['handsWithdrawn'];release=entry['release']
            drift=np.linalg.norm(rp[planted:withdrawn+1]-rp[planted],axis=1).max()
            legcols=[i for i,n in enumerate(task['clip']['joint_names']) if any(w in n for w in ('hip','knee','ankle'))]
            assert drift<.001 and np.max(np.abs(q[planted:withdrawn+1,legcols]-q[planted,legcols]))<1e-5, f'{key}: lower body moves during planted insertion'
            targets=np.array(entry['footTargets'])
            swing=np.abs(targets[:,0,2]-targets[:,1,2])>.01
            support=np.argmin(targets[:,:,2],axis=1)
            sway=rp[:,0]-task['slot'][0]
            support_x=targets[np.arange(len(targets)),support,0]-task['slot'][0]
            assert np.all(sway[swing]*support_x[swing]>=-1e-6), f'{key}: weight shift points toward swinging foot'
            travel=op[release,1]-op[planted,1]
            assert travel>.08, f'{key}: planted arm insertion is too short'
            insertion_jump=np.linalg.norm(np.diff(op[placement['insertStart']:release+1],axis=0),axis=1).max()
            assert insertion_jump<.03, f'{key}: abrupt bin displacement during insertion'
            print(f'{key}: planted insertion {travel*1000:.1f}mm, pelvis drift {drift*1000:.3f}mm',flush=True)
        lo=model.jnt_range[joints,0];hi=model.jnt_range[joints,1]
        assert np.all(q>=lo-1e-6) and np.all(q<=hi+1e-6),f'{key}: joint limits'
        armidx=[i for i,n in enumerate(task['clip']['joint_names']) if any(w in n for w in ('shoulder','elbow','wrist'))];legidx=[i for i,n in enumerate(task['clip']['joint_names']) if any(w in n for w in ('hip','knee','ankle'))]
        speed=np.abs(np.diff(q,axis=0))*entry['playbackFps']
        assert speed[:,armidx].max()<=4.01 and speed[:,legidx].max()<=6.01,f'{key}: velocity limits'
        stats={'bin_body':(1.,None),'bin_shelf':(1.,None),'robot_shelf':(1.,None),'neighbors':(1.,None),'foot_floor':(1.,None),'palm_error':(0.,None)};fromto=np.zeros(6)
        def minimum(label,value,where):
            if value<stats[label][0]:stats[label]=(float(value),where)
        bounds_center=np.zeros((model.ngeom,3));bounds_half=np.zeros((model.ngeom,3))
        def kinematics():
            mj.mj_kinematics(model,data)
            rotations=data.geom_xmat.reshape(-1,3,3)
            bounds_center[:]=data.geom_xpos+np.einsum('nij,nj->ni',rotations,model.geom_aabb[:,:3])
            bounds_half[:]=np.einsum('nij,nj->ni',np.abs(rotations),model.geom_aabb[:,3:])
        def distance(a,b):
            # AABB separation is a conservative lower bound, not a contact estimate.
            separation=np.maximum(np.abs(bounds_center[a]-bounds_center[b])-bounds_half[a]-bounds_half[b],0)
            bound=float(np.linalg.norm(separation))
            if bound>.003:return min(.5,bound)
            return mj.mj_geomDistance(model,data,a,b,.5,fromto)
        for f in np.append(np.arange(0,entry['T']-1,entry['playbackFps']/args.hz),entry['T']-1):
            t=int(f);u=float(f-t);n=min(t+1,entry['T']-1)
            data.qpos[:3]=rp[t]*(1-u)+rp[n]*u;data.qpos[3:7]=slerp(rq[t],rq[n],u);data.qpos[qa]=q[t]*(1-u)+q[n]*u
            data.qpos[ba:ba+3]=op[t]*(1-u)+op[n]*u;data.qpos[ba+3:ba+7]=oq;kinematics()
            for k in robot:minimum('bin_body',distance(gid,k),(round(float(f),2),mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,model.geom_bodyid[k])))
            for k in shelf:
                minimum('bin_shelf',distance(gid,k),(round(float(f),2),mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,k)))
                for rob in robot:
                    if np.linalg.norm(data.geom_xpos[k]-data.geom_xpos[rob])>model.geom_rbound[k]+model.geom_rbound[rob]+.002:continue
                    minimum('robot_shelf',distance(rob,k),(round(float(f),2),mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,model.geom_bodyid[rob]),mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,k)))
            for k in feet:minimum('foot_floor',data.geom_xpos[k,2]-model.geom_size[k,0],round(float(f),2))
            R=data.geom_xmat[gid].reshape(3,3);center=data.qpos[ba:ba+3]+R@off
            for side,w in enumerate(wrists):
                palm=data.xpos[w]+data.xmat[w].reshape(3,3)@np.array([.085,0,0]);err=np.linalg.norm(palm-center-R@entry['contact_points'][side])
                if entry['grasp']<=f<=entry['release'] and err>stats['palm_error'][0]:stats['palm_error']=(float(err),(round(float(f),2),side))
            # Replay the translated second column against the real opposite post
            # and the bin that was placed in column zero earlier in the cycle.
            data.qpos[0]+=task['columnDelta'];data.qpos[ba]+=task['columnDelta'];kinematics()
            for k in shelf:
                minimum('bin_shelf',distance(gid,k),(round(float(f),2),'column1',mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,k)))
                for rob in robot:
                    if np.linalg.norm(data.geom_xpos[k]-data.geom_xpos[rob])<=model.geom_rbound[k]+model.geom_rbound[rob]+.002:
                        minimum('robot_shelf',distance(rob,k),(round(float(f),2),'column1',mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,model.geom_bodyid[rob]),mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,k)))
            for k in neighbor_ids:
                minimum('neighbors',distance(gid,k),(round(float(f),2),'bin',mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,k)))
                for rob in robot:
                    if np.linalg.norm(data.geom_xpos[k]-data.geom_xpos[rob])<=model.geom_rbound[k]+model.geom_rbound[rob]+.002:
                        minimum('neighbors',distance(rob,k),(round(float(f),2),mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,model.geom_bodyid[rob]),mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,k)))
        passed=all(stats[k][0]>=-.001 for k in ('bin_body','bin_shelf','robot_shelf','neighbors','foot_floor')) and stats['palm_error'][0]<=.04
        print(('PASS' if passed else 'FAIL'),key,', '.join(f'{k}={v[0]*1000:.1f}mm @{v[1]}' for k,v in stats.items()),flush=True)
        if not passed:failures.append(key)
    if failures:raise SystemExit('Invalid manipulation trajectories: '+', '.join(failures))
if __name__=='__main__':main()
