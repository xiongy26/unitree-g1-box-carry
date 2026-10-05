#!/usr/bin/env python3
"""Contact-centric shelf retargeting, inspired by hoi-retarget's q-only solve.

Uses the viewer's final geometry, fixed object-frame palm targets, joint limits,
and temporal tracking. Source motion files stay unchanged. No upstream code copied.
Run with a Python environment containing numpy, scipy, mujoco and Node on PATH.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares, minimize
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]

def solve(task, xml, yaw, rack):
    box=task['box']
    proxy_half=np.array(box['half_size'])+[.018,.018,.006]
    proxy_off=np.array(box['bbox_center_offset'])+[0,0,.006]
    # Compile the offset, rather than mutate geom_pos: MuJoCo's sameframe fast
    # path is selected at compile time and would otherwise ignore the new offset.
    xml=xml.replace('name="task_box_geom" type="box" size="0.2 0.2 0.17" mass="3"',
        f'name="task_box_geom" type="box" size="{" ".join(map(str,proxy_half))}" pos="{" ".join(map(str,proxy_off))}" mass="3"')
    mirror_x=2*task['slot'][0]-(2.2-.57)
    xml=xml.replace('</worldbody>',f'<geom name="fx_active_post_virtual" type="box" size=".028 .028 {rack["uprightH"]/2}" pos="{mirror_x} {task["slot"][1]-rack["depthHalf"]} {rack["uprightH"]/2}" contype="0" conaffinity="0" mass="0"/></worldbody>')
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'task_box_geom')
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'task_box_joint')
    ba = model.jnt_qposadr[jid]
    box = task['box']
    half = np.array(box['half_size'])
    off = np.array(box['bbox_center_offset'])
    clip, rm = task['clip'], task['rm']
    source_T, g, source_r, fps = rm['T'], rm['grasp'], rm['release'], rm['fps']
    r=source_r+60
    T=r+85
    source_insert=source_r-int(.9*fps)
    def extend(values,width):
        rows=np.array(values).reshape(source_T,width)
        clock=np.arange(T,dtype=float)
        clock[source_insert:]=np.minimum(source_r,source_insert+(clock[source_insert:]-source_insert)*(source_r-source_insert)/(r-source_insert))
        return np.column_stack([np.interp(clock,np.arange(source_T),rows[:,i]) for i in range(width)])
    rp = extend(rm['rootPos'],3)
    rq = extend(rm['rootQuat'],4)
    dof = extend(rm['dof'],len(clip['joint_names']))
    op = extend(rm['objPos'],3)
    original = dof.copy()
    quat = np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])
    R = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    # The authored task keeps an open bin upright. Object pose is fixed in the IK solve.
    # Convert centers explicitly; the viewer bodies have an offset origin.
    center = op.copy()
    center[:,2] += off[2]
    center[:g+1] = [op[g,0],op[g,1],half[2]]
    center[r:] = task['slot']
    center[:,2] = np.maximum(center[:,2],half[2])
    center[:,0] = task['slot'][0]
    # Lower outside the rack, then insert horizontally into the selected compartment.
    # Original motion lowering through the upper shelf is not a legal placement path.
    front_y = task['slot'][1] - rack['depthHalf'] - half[0] - .04
    center[:,1] = np.minimum(center[:,1],front_y)
    lower_start = max(g+1,source_r-int(1.8*fps))
    insert_start = max(lower_start+1,source_insert)
    center[g:r,2]=np.minimum(center[g:r,2],max(.70,task['slot'][2]+.20))
    start_height = center[lower_start,2]
    for t in range(lower_start,insert_start+1):
        u=(t-lower_start)/(insert_start-lower_start); u=u*u*(3-2*u)
        center[t,2]=(1-u)*start_height+u*(task['slot'][2]+.003)
    insert_from=center[insert_start].copy()
    for t in range(insert_start,r+1):
        u=(t-insert_start)/(r-insert_start); u=u*u*(3-2*u)
        center[t]=(1-u)*insert_from+u*np.array(task['slot'])
    center[r:]=task['slot']
    qadr = np.array([model.jnt_qposadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n)] for n in clip['joint_names']])
    joint_ids = np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n) for n in clip['joint_names']])
    arms = [np.array([i for i,n in enumerate(clip['joint_names']) if n.startswith(side) and any(s in n for s in ('shoulder','elbow','wrist'))]) for side in ('left','right')]
    wrists = [mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,f'{side}_wrist_yaw_link') for side in ('left','right')]
    def geoms(names, hands=False):
        out=[]
        for name in names:
            bid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)
            for k in range(model.body_geomadr[bid],model.body_geomadr[bid]+model.body_geomnum[bid]):
                mesh = model.geom_dataid[k]
                meshname = mj.mj_id2name(model,mj.mjtObj.mjOBJ_MESH,mesh) if mesh>=0 else ''
                if model.geom_contype[k] or (hands and 'rubber_hand' in (meshname or '')): out.append(k)
        return out
    bodies=geoms(['pelvis','torso_link','waist_yaw_link','waist_roll_link'] + [f'{s}_{j}_link' for s in ('left','right') for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')])
    environment=[]
    for k in range(model.ngeom):
        name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,k) or ''
        if (name.startswith('fx_active_') and 'hole' not in name) or name.startswith('fx_rack_board'):
            environment.append(k)
    arm_geoms=[geoms([f'{s}_{j}_link' for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')],True) for s in ('left','right')]
    palm_geoms=[]
    for side in ('left','right'):
        palm_geoms.append(next(k for k in arm_geoms[len(palm_geoms)] if model.geom_dataid[k]>=0 and 'rubber_hand' in mj.mj_id2name(model,mj.mjtObj.mjOBJ_MESH,model.geom_dataid[k])))
    fromto=np.zeros(6)
    aabb_center=np.zeros((model.ngeom,3));aabb_half=np.zeros((model.ngeom,3))
    def pair_distance(a,b):
        gap=np.maximum(np.abs(aabb_center[a]-aabb_center[b])-aabb_half[a]-aabb_half[b],0)
        lower=float(np.linalg.norm(gap))
        if lower>.01:return min(.5,lower)
        return mj.mj_geomDistance(model,data,a,b,.5,fromto)
    def distance(k):return pair_distance(gid,k)
    def forward(t, p=None):
        data.qpos[:3]=rp[t]; data.qpos[3:7]=rq[t]; data.qpos[qadr]=dof[t]
        data.qpos[ba:ba+3]=(center[t] if p is None else p)-R@off
        data.qpos[ba+3:ba+7]=quat
        mj.mj_kinematics(model,data)
        rotations=data.geom_xmat.reshape(-1,3,3)
        aabb_center[:]=data.geom_xpos+np.einsum('nij,nj->ni',rotations,model.geom_aabb[:,:3])
        aabb_half[:]=np.einsum('nij,nj->ni',np.abs(rotations),model.geom_aabb[:,3:])
    # Replan the pelvis along the legal bin route, then retarget the legs to
    # stationary stance anchors. Simply translating a mocap skeleton would slide feet.
    feet=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,f'{side}_ankle_roll_link') for side in ('left','right')]
    foot_geoms=[geoms([f'{side}_ankle_roll_link']) for side in ('left','right')]
    # The warehouse task faces the bin throughout; recover ground contacts through
    # leg IK while the pelvis and waist share the crouching tilt.
    rq[:]=quat
    for name in ('waist_yaw_joint','waist_roll_joint'):
        dof[:,clip['joint_names'].index(name)]=0
    # Keep the bin within arm reach while approaching; reserve the final extension
    # for insertion, with the pelvis outside the front beam.
    desired_y=np.minimum(center[:,1]-.60,task['slot'][1]-rack['depthHalf']-.27)
    pregrasp_gap=.05*(1-np.clip((np.arange(T)-g)/20,0,1))
    desired_y-=pregrasp_gap
    # Walk close enough to reach the shelf, then plant both feet. The remaining
    # bin travel comes from arm extension, rather than translating the whole body.
    plant = insert_start + int(round((r-insert_start)*.50))
    placement_y = task['slot'][1]-rack['depthHalf']-.35
    u=np.clip((np.arange(T)-insert_start)/(plant-insert_start),0,1)
    u=u*u*(3-2*u)
    desired_y[insert_start:]=desired_y[insert_start]*(1-u[insert_start:])+placement_y*u[insert_start:]
    # Unload and withdraw the palms before moving the pelvis backward.
    retreat=np.clip((np.arange(T)-r-24)/24,0,1);retreat=retreat*retreat*(3-2*retreat)
    desired_y[r:]=placement_y-.25*retreat[r:]
    blend=np.clip((np.arange(T)-(g-24))/24,0,1)
    blend=blend*blend*(3-2*blend)
    rp[:,1]=desired_y
    rp[:,0]=center[:,0]
    desired_z=np.clip(center[:,2]+.19,.34,.74)
    rise=np.clip((np.arange(T)-r-48)/36,0,1);rise=rise*rise*(3-2*rise)
    desired_z[r:]=desired_z[r]*(1-rise[r:])+.79*rise[r:]
    rp[:,2]=.79*(1-blend)+desired_z*blend
    tilt=np.clip((.65-center[:,2])/.45,0,1)*.5*blend
    tilt[r:]*=(1-rise[r:])
    for t in range(T):
        pitch=np.array([np.cos(tilt[t]/2),0,np.sin(tilt[t]/2),0])
        # yaw(y) * pitch(y): wxyz quaternion composition.
        rq[t]=[quat[0]*pitch[0],-quat[3]*pitch[2],quat[0]*pitch[2],quat[3]*pitch[0]]
    # Plan alternating stance/swing contacts in the replanned corridor. Stance
    # anchors are fixed in world coordinates; swing feet land before handover.
    flat_z=[]
    for gs in foot_geoms:
        flat_z.append(.002-min(model.geom_pos[k,2]-model.geom_size[k,0] for k in gs if model.geom_type[k]==mj.mjtGeom.mjGEOM_SPHERE))
    waist_idx=clip['joint_names'].index('waist_pitch_joint')
    reach=np.clip((np.arange(T)-plant)/(r-plant),0,1);reach=reach*reach*(3-2*reach)
    # A small torso hinge shares the reach on the low shelf; legs stay planted.
    torso_pitch=.35+.15*reach*np.clip((.66-desired_z)/.25,0,1)
    dof[:,waist_idx]=torso_pitch*(1-rise)-tilt
    squat=np.clip((.66-desired_z)/.25,0,1)*blend
    foot_width=.11+(.15-.11)*squat
    foot_forward=-.06+.25*squat
    anchors=[np.array([rp[0,0]-.11,rp[0,1]-.06,flat_z[0]]),
             np.array([rp[0,0]+.11,rp[0,1]-.06,flat_z[1]])]
    foot_targets=np.zeros((T,2,3)); moving_side=0; swing_start=None; cooldown=0
    swing_frames=8; step_from=None; step_to=None
    for t in range(T):
        midpoint=(anchors[0][:2]+anchors[1][:2])/2
        may_step = t < plant-swing_frames or t >= r+24
        if may_step and swing_start is None and cooldown<=0 and np.linalg.norm(rp[t,:2]+np.array([0,foot_forward[min(T-1,t+16)]])-midpoint)>.055:
            swing_start=t;step_from=anchors[moving_side].copy()
            future=min(T-1,t+swing_frames)
            contact_future=min(T-1,t+16)
            step_to=np.array([rp[future,0]+(-foot_width[contact_future] if moving_side==0 else foot_width[contact_future]),min(rp[future,1]+foot_forward[contact_future],task['slot'][1]-rack['depthHalf']-.16),flat_z[moving_side]])
            stride=step_to[:2]-step_from[:2]; length=np.linalg.norm(stride)
            if length>.45:step_to[:2]=step_from[:2]+stride*.45/length
        foot_targets[t]=anchors
        if swing_start is not None:
            u=min(1.,(t-swing_start)/swing_frames);v=u*u*(3-2*u)
            foot_targets[t,moving_side]=(1-v)*step_from+v*step_to
            foot_targets[t,moving_side,2]+=.038*np.sin(np.pi*u)**2
            # Transfer weight toward the planted foot while the other foot swings.
            rp[t,0]+=(-.022 if moving_side==1 else .022)*(1-squat[t])*np.sin(np.pi*u)**2
            if u>=1:
                anchors[moving_side]=step_to.copy();moving_side=1-moving_side;swing_start=None;cooldown=0
        else:cooldown-=1
    legs=[np.array([i for i,n in enumerate(clip['joint_names']) if n.startswith(side) and any(w in n for w in ('hip','knee','ankle'))]) for side in ('left','right')]
    legidx=np.concatenate(legs)
    ranges=model.jnt_range[joint_ids[legidx]]
    for t in range(T):
        reference_z=rp[t,2]
        reference=original[t,legidx]
        seed=reference.copy()
        for side in range(2):
            local=[clip['joint_names'][i] for i in legs[side]]
            for j,name in enumerate(local):
                if 'hip_pitch' in name:seed[side*6+j]=-.3-1.5*squat[t]
                if 'knee' in name:seed[side*6+j]=.6+1.8*squat[t]
                if 'ankle_pitch' in name:seed[side*6+j]=-.3-.3*squat[t]
        lo,hi=ranges[:,0].copy(),ranges[:,1].copy()
        if t>0:
            lo=np.maximum(lo,dof[t-1,legidx]-6/13)
            hi=np.minimum(hi,dof[t-1,legidx]+6/13)
        def residual(x):
            rp[t,2]=x[0];dof[t,legidx]=x[1:];forward(t)
            values=[]
            for side,wi in enumerate(feet):
                values.extend((data.xpos[wi]-foot_targets[t,side])*30)
                values.extend((data.xmat[wi].reshape(3,3)-R).ravel()*8)
            values.extend((x[1:]-reference)*.004)
            values.append((x[0]-reference_z)*100)
            return np.array(values)
        lower=np.r_[max(.30,reference_z-.06),lo];upper=np.r_[min(.86,reference_z+.06),hi]
        start=np.clip(np.r_[reference_z,seed],lower+1e-8,upper-1e-8)
        opt=least_squares(residual,start,bounds=(lower,upper),max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        rp[t,2]=opt.x[0];dof[t,legidx]=opt.x[1:]
        forward(t)
        actual_min=min(data.geom_xpos[k,2]-model.geom_size[k,0] for gs in foot_geoms for k in gs if model.geom_type[k]==mj.mjtGeom.mjGEOM_SPHERE)
        rp[t,2]+=.003-actual_min
    # Reuse the planted lower-body pose exactly through insertion and withdrawal.
    # This prevents tiny independent per-frame IK changes from looking like drift.
    rp[plant:r+25]=rp[plant]
    rq[plant:r+25]=rq[plant]
    dof[plant:r+25,legidx]=dof[plant,legidx]
    # Plan a collision-free object reference BEFORE solving robot contacts.
    corrections=np.zeros((T,3))
    for t in range(g,r):
        forward(t)
        if min(distance(k) for k in bodies)>=0.004 and min(distance(k) for k in environment)>=.001: continue
        ref=center[t].copy()
        def constraints(delta):
            forward(t,ref+delta)
            return np.array([distance(k)-0.004 for k in bodies]+[distance(k)-.001 for k in environment])
        opt=minimize(lambda x: float(np.dot(x*x,[16.,1.,4.])),np.array([0.,0.,.02]),method='SLSQP',
            constraints={'type':'ineq','fun':constraints},bounds=[(-.32,.32),(-.32,.32),(0,.20)],options={'maxiter':60,'ftol':1e-8})
        if not opt.success:
            candidates=[]
            for start in ([.18,0,0],[-.18,0,0],[0,-.12,.05],[0,.12,0],[0,.30,0],[0,.30,.06]):
                attempt=minimize(lambda x:float(np.dot(x*x,[16.,1.,4.])),np.array(start),method='SLSQP',
                    constraints={'type':'ineq','fun':constraints},bounds=[(-.45,.45),(-.32,.32),(0,.20)],options={'maxiter':100,'ftol':1e-8})
                if attempt.success: candidates.append(attempt)
            if candidates: opt=min(candidates,key=lambda a:a.fun)
            else:
                forward(t,ref+opt.x)
                details=sorted([(distance(k),mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,model.geom_bodyid[k])) for k in bodies+environment])[:4]
                raise RuntimeError(f"{task['key']} frame {t}: object clearance {opt.message}; {details}")
        corrections[t]=opt.x
    # Smooth the correction with an overlapping window, like the upstream temporal solve.
    # Reproject afterward; smoothing alone is never treated as evidence of clearance.
    corrections=gaussian_filter1d(corrections,2,axis=0)
    center+=corrections
    center[:g+1]=center[g]; center[:g+1,2]=half[2]
    center[r:]=task['slot']
    for t in range(g,r):
        forward(t)
        if min(distance(k) for k in bodies)<0.01 or min(distance(k) for k in environment)<.0005:
            ref=center[t].copy()
            def constraints(delta):
                forward(t,ref+delta); return np.array([distance(k)-0.004 for k in bodies]+[distance(k)-.001 for k in environment])
            opt=minimize(lambda x: float(np.dot(x*x,[16.,1.,4.])),np.zeros(3),method='SLSQP',constraints={'type':'ineq','fun':constraints},
                bounds=[(-.2,.2),(-.2,.4),(0,.2)],options={'maxiter':80,'ftol':1e-9})
            if not opt.success or min(constraints(opt.x)) < -1e-5:
                attempts=[]
                for seed in ([0,.15,.01],[0,.30,.02]):
                    attempt=minimize(lambda x:float(np.dot(x*x,[16.,1.,4.])),np.array(seed),method='SLSQP',constraints={'type':'ineq','fun':constraints},
                        bounds=[(-.2,.2),(-.2,.4),(0,.2)],options={'maxiter':100,'ftol':1e-9})
                    if min(constraints(attempt.x))>=-1e-5: attempts.append(attempt)
                if not attempts: raise RuntimeError(f'{task["key"]} frame {t}: smoothed object path cannot recover clearance')
                opt=min(attempts,key=lambda a:a.fun)
            center[t]+=opt.x
    # Fixed front-side grip targets below the rim leave room under the next board.
    # Keep the same object-frame point throughout carrying, without contact switching.
    forward(g)
    contacts=[]
    wrist_rot=[data.xmat[wi].reshape(3,3).copy() for wi in wrists]
    for side,wi in enumerate(wrists):
        front=.075 if box['type']=='largebox' else .025
        lateral=.035 if box['type']=='largebox' else .055
        contacts.append(np.array([-half[0]-front,(1 if side==0 else -1)*(half[1]+lateral),half[2]-.070]))
    # Each arm tracks its contact, original motion, previous correction, and collision gaps.
    # Box pose, pelvis and leg DOFs remain fixed throughout the arm optimization.
    arm_environment=[k for k in environment if
        (mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,k) or '').startswith(('fx_rack_board','fx_active_beam','fx_active_post'))]
    for side,idx in enumerate(arms):
        original[r+1:,idx]=[.3,.2 if side==0 else -.2,0,1.28,0,0,0]
        dof[r+1:,idx]=original[r+1:,idx]
    prev=[None,None]
    worst=[1.,1.]
    for t in range(T):
        if t<max(0,g-24): continue
        engage=min(1.,max(0.,(t-(g-24))/24))
        release=max(0.,1.-max(0,t-r)/24)
        alpha=engage*release
        forward(t)
        for side,idx in enumerate(arms):
            ref=original[t,idx]
            wi=wrists[side]
            reference_palm=data.xpos[wi].copy()+data.xmat[wi].reshape(3,3)@np.array([.085,0,0])
            target=center[t]+R@contacts[side]
            target=alpha*target+(1-alpha)*reference_palm
            previous=prev[side] if prev[side] is not None else dof[max(0,t-1),idx].copy()
            ranges=model.jnt_range[joint_ids[idx]]
            if t>0:
                # Conservative 4rad/s cap; do not let finite-difference IK snap a wrist.
                lo=np.maximum(ranges[:,0],dof[t-1,idx]-4/13)
                hi=np.minimum(ranges[:,1],dof[t-1,idx]+4/13)
            else: lo,hi=ranges[:,0],ranges[:,1]
            def residual(q):
                dof[t,idx]=q; forward(t)
                palm=data.xpos[wi]+data.xmat[wi].reshape(3,3)@np.array([.085,0,0])
                ds=np.array([distance(k) for k in arm_geoms[side]])
                clearance=[min(0,distance_arm-.010) for k in arm_geoms[side] for env in arm_environment
                    if abs(model.geom_pos[env,0]-task['slot'][0])<.7
                    for distance_arm in [pair_distance(k,env)]]
                return np.concatenate([(palm-target)*8,(data.xmat[wi].reshape(3,3)-wrist_rot[side]).ravel()*.03*alpha,(q-ref)*.015,(q-previous)*.10,
                    np.minimum(ds-.012,0)*70,np.array(clearance)*100,[max(0,distance(palm_geoms[side])-.025)*10*alpha]])
            start=np.clip(previous,lo+1e-8,hi-1e-8)
            opt=least_squares(residual,start,bounds=(lo,hi),max_nfev=55,ftol=1e-5,xtol=1e-5,gtol=1e-5)
            dof[t,idx]=opt.x; prev[side]=opt.x.copy()
            forward(t)
            if g<=t<=r: worst[side]=min(worst[side],min(distance(k) for k in arm_geoms[side]))
    forward(r)
    nonhand=min(distance(k) for k in bodies)
    print(f"{task['key']}: arm clearance {np.round(np.array(worst)*1000,1)} mm; final body {nonhand*1000:.1f} mm",flush=True)
    return {'signature':task['signature'],'source_clip':clip['source_clip'],'T':T,'sourceT':source_T,'fps':fps,'playbackFps':13,'grasp':g,'release':r,'sourceRelease':source_r,
            'slot':task['slot'],'box':box,'rootPos':np.round(rp,6).tolist(),'rootQuat':np.round(rq,8).tolist(),'footTargets':np.round(foot_targets,6).tolist(),'objPos':np.round(center-R@off,9).tolist(),
            'objQuat':quat.tolist(),'dof':np.round(dof,6).tolist(),
            'contact_points':[c.tolist() for c in contacts],
            'placement':{'insertStart':insert_start,'feetPlanted':plant,'handsWithdrawn':r+24,'retreatEnd':r+48},
            'arm_clearance_m':worst}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=ROOT/'motions/manipulation.json'); parser.add_argument('--only'); args=parser.parse_args()
    ref=json.loads(subprocess.check_output(['node',str(ROOT/'tools/lib/manipulation_reference.mjs')],text=True))
    result={'format':'g1-shelf-contact/1','method':'Fixed object-frame grip targets, weight-shifting steps, planted arm insertion, hand withdrawal before retreat, bounded joint IK, temporal tracking, signed-distance clearance. Derived from original hoi-retarget motion; not an unmodified mocap replay.', 'plans':{}}
    if args.only and args.out.exists(): result=json.loads(args.out.read_text())
    for task in ref['tasks']:
        if args.only and task['key']!=args.only: continue
        result['plans'][task['key']]=solve(task,ref['xml'],ref['yaw'],ref['rack'])
    temp=args.out.with_suffix('.tmp')
    temp.write_text(json.dumps(result,separators=(',',':'))+'\n')
    temp.replace(args.out)
if __name__=='__main__': main()
