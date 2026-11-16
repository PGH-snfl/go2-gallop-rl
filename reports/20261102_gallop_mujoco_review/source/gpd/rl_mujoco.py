"""CPU MuJoCo execution of saved 47/12 Gallop policies and their startup bundle.

The checkpoint's simulator-independent core and GallopState are reused by hash.
No Isaac application, SDK/DDS, optimizer update or physical hardware is used.
"""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
import torch
from tensordict import TensorDict

from gpd.paths import ROOT
from isaac_task.gallop_runtime import checkpoint_source_directory
from scripts.analysis.collect_unitree_baseline import load_yaml

LEGS=('FL','FR','RL','RR')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unitree_torque(target,q,dq,motor):
    effort=float(motor['stiffness'])*(target-q)-float(motor['damping'])*dq
    full=np.where(effort*dq>0,float(motor['Y1']),float(motor['Y2']))
    ratio=np.clip((float(motor['X2'])-np.abs(dq))/(float(motor['X2'])-float(motor['X1'])),0,1)
    return np.clip(effort,-full*ratio,full*ratio)


def euler_xyz(quat):
    w,x,y,z=quat.unbind(1)
    return (torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
            torch.asin((2*(w*y-z*x)).clamp(-1,1)),
            torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def saved_state_class(checkpoint,manifest):
    directory=checkpoint_source_directory(checkpoint,manifest)
    name='_go2_saved_core_'+manifest['gallop_core_sha256'][:16]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,directory/'gallop_core.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    core=sys.modules[name]
    tree=ast.parse((directory/'gallop_env.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='GallopState')
    namespace=dict(core.__dict__);namespace.update(LEGS=LEGS,euler_xyz_from_quat=euler_xyz)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(directory/'gallop_env.py'),'exec'),namespace)
    return core,namespace['GallopState']


def make_scene(source,out,dt,friction,motor):
    tree=ET.parse(source);xml=tree.getroot()
    xml.find('compiler').set('meshdir',str(source.parent/'assets'))
    option=xml.find('option');option.set('timestep',str(dt));option.set('gravity','0 0 -9.81')
    for node in xml.findall('.//default/joint'):
        node.set('damping','0');node.set('armature','0');node.set('frictionloss',str(motor.get('friction') or 0.))
    for node in xml.findall('.//default/motor'):node.set('ctrlrange',f"{-motor['Y2']} {motor['Y2']}")
    for node in xml.findall('.//default/geom'):
        if node.get('contype')!='0':node.set('friction',f'{friction} 0 0');node.set('condim','3')
    visual=ET.SubElement(xml,'visual');ET.SubElement(visual,'global',offwidth='640',offheight='480')
    ET.SubElement(visual,'headlight',ambient='.4 .4 .4',diffuse='.7 .7 .7')
    world=xml.find('worldbody')
    ET.SubElement(world,'light',pos='0 0 5',dir='0 0 -1',directional='true')
    ET.SubElement(world,'geom',name='floor',type='plane',size='0 0 .1',friction=f'{friction} 0 0',condim='3',rgba='.3 .35 .4 1')
    tree.write(out/'scene.xml')
    return mj.MjModel.from_xml_path(str(out/'scene.xml'))


class Scene(dict):pass


class GallopMuJoCo:
    def __init__(self,checkpoint,source_model,out,noise=False,seed=42):
        from rsl_rl.runners import OnPolicyRunner
        checkpoint=Path(checkpoint).resolve();self.manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
        self.checkpoint=checkpoint;self.checkpoint_hash=sha(checkpoint)
        if self.manifest['interface']!='unitree_go2_gallop_47_62_12_v1':raise ValueError('Expected Gallop 47/62/12 checkpoint')
        self.core,State=saved_state_class(checkpoint,self.manifest)
        self.settings=self.core.GallopSettings(**self.manifest['gallop_settings'])
        self.config=load_yaml(checkpoint.parent/'params/env.yaml');agent=load_yaml(checkpoint.parent/'params/agent.yaml')
        if agent['clip_actions'] is not None:raise ValueError('Unimplemented RSL wrapper action clipping')
        robot_cfg=self.config['scene']['robot'];self.motor=robot_cfg['actuators']['GO2HV']
        if self.motor['min_delay'] or self.motor['max_delay'] or self.motor['Fs'] or self.motor['Fd']:raise ValueError('Nonzero actuator delay/friction requires another adapter')
        self.physics_dt=float(self.config['sim']['dt']);self.decimation=int(self.config['decimation']);self.control_dt=self.physics_dt*self.decimation
        self.model=make_scene(Path(source_model).resolve(),out,self.physics_dt,1.,self.motor)
        self.data=mj.MjData(self.model);self.noise=noise;self.rng=np.random.default_rng(seed)
        contract=json.loads((ROOT/'assets/go2_isaac_contract.json').read_text())
        if contract['isaac_asset_sha256']!=self.manifest['asset_sha256']:raise ValueError('Robot USD contract mismatch')
        self.native=contract['native_joint_order'];self.canonical=[f'{leg}_{j}_joint' for leg in LEGS for j in ('hip','thigh','calf')]
        joints=[mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_JOINT,n) for n in self.native]
        if min(joints)<0:raise ValueError('Missing named MuJoCo joint')
        self.qadr=self.model.jnt_qposadr[joints];self.vadr=self.model.jnt_dofadr[joints]
        self.act=[]
        for j in joints:
            matches=np.flatnonzero(self.model.actuator_trnid[:,0]==j)
            if len(matches)!=1:raise ValueError('Joint actuator mapping is ambiguous')
            self.act.append(int(matches[0]))
        if not np.allclose(self.model.actuator_gear[self.act,0],1):raise ValueError('Non-unit motor gear')
        self.default=np.array([next(v for expr,v in robot_cfg['init_state']['joint_pos'].items() if re.fullmatch(expr,n)) for n in self.native])
        limits={j['name']:np.deg2rad([j['lower'],j['upper']]) for j in contract['joints']}
        hard=np.array([limits[n] for n in self.native]);mid=hard.mean(1);radius=np.ptp(hard,axis=1)*.5*robot_cfg['soft_joint_pos_limit_factor']
        self.soft=np.column_stack((mid-radius,mid+radius))
        if not np.allclose(hard,self.model.jnt_range[joints],atol=2e-4):raise ValueError('Isaac and MuJoCo hard joint limits differ')
        self.base=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_BODY,'base_link')
        self.feet=[mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_BODY,leg+'_foot') for leg in LEGS]
        self.foot_geoms={mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_GEOM,leg):i for i,leg in enumerate(LEGS)}
        self.data.qpos[:3]=robot_cfg['init_state']['pos'];self.data.qpos[3:7]=robot_cfg['init_state']['rot'];self.data.qpos[self.qadr]=self.default
        self.data.qvel[:]=0.;mj.mj_forward(self.model,self.data)
        self.tensor=NS(joint_pos=torch.zeros(1,12),default_joint_pos=torch.tensor(self.default,dtype=torch.float32)[None],
            soft_joint_pos_limits=torch.tensor(self.soft,dtype=torch.float32)[None],root_pos_w=torch.zeros(1,3),root_quat_w=torch.zeros(1,4),
            root_lin_vel_w=torch.zeros(1,3),root_lin_vel_b=torch.zeros(1,3),root_ang_vel_b=torch.zeros(1,3),
            body_pos_w=torch.zeros(1,4,3),body_lin_vel_w=torch.zeros(1,4,3))
        self.sensor=NS(body_names=[leg+'_foot' for leg in LEGS],data=NS(net_forces_w=torch.zeros(1,4,3)))
        robot=NS(data=self.tensor,find_joints=lambda names,**kw:([self.native.index(n) for n in names],names),
            find_bodies=lambda names,**kw:(list(range(4)),names),root_physx_view=NS(get_masses=lambda:torch.tensor(self.model.body_mass,dtype=torch.float32)[None]))
        scene=Scene(robot=robot,contact_forces=self.sensor);scene.env_origins=torch.zeros(1,3)
        self.command=torch.zeros(1,3)
        self.env=NS(scene=scene,num_envs=1,device='cpu',step_dt=self.control_dt,max_episode_length=round(self.config['episode_length_s']/self.control_dt),
            common_step_counter=0,cfg=NS(decimation=self.decimation,sim=NS(gravity=(0,0,-9.81))),
            reset_terminated=torch.zeros(1,dtype=torch.bool),reset_time_outs=torch.zeros(1,dtype=torch.bool),
            command_manager=NS(get_command=lambda name:self.command))
        self.sync()
        teacher=None
        if self.settings.startup_enabled:
            teacher=self.core.FrozenWalkingPolicy(checkpoint.parent/'source/walking_teacher.pt','cpu',self.manifest['walking_teacher']['action_scale'],self.settings.action_scale)
        self.state=State(self.env,self.settings,None,out/'episodes.jsonl',False,False,teacher)
        class Probe:
            num_envs=1;num_actions=12
            def get_observations(self):return TensorDict(dict(policy=torch.zeros(1,47),critic=torch.zeros(1,62)),batch_size=[1])
        self.runner=OnPolicyRunner(Probe(),agent,log_dir=None,device='cpu')
        self.runner.load(str(checkpoint),load_optimizer=False,map_location='cpu')
        def forbid(*a,**kw):raise RuntimeError('PPO updates are forbidden during MuJoCo validation')
        self.runner.learn=forbid;self.runner.alg.update=forbid
        self.actor=self.runner.get_inference_policy(device='cpu')
        self.initial_policy={k:v.detach().clone() for k,v in self.runner.alg.policy.state_dict().items()}
        self.target=self.default.copy();self.last_action=np.zeros(12,dtype=np.float32)
        self.base_contact_force=0.;self.last_torque=np.zeros(12);self.fault=None
        self.physics=[]
        self.contract=dict(native_joint_order=self.native,canonical_joint_order=self.canonical,
            qpos_addresses=self.qadr.tolist(),qvel_addresses=self.vadr.tolist(),actuator_ids=self.act,
            default_joint_positions=self.default.tolist(),soft_joint_limits=self.soft.tolist(),physics_dt=self.physics_dt,control_dt=self.control_dt,
            nominal_mujoco_mass_kg=float(self.model.body_mass.sum()),nominal_usd_mass_kg=sum(b['mass'] for b in contract['bodies']),
            source_model_sha256=sha(source_model),isaac_asset_sha256=contract['isaac_asset_sha256'],motor=self.motor,
            model_adjustments=['flat plane only, friction1/condim3','zero passive damping/armature; frictionloss0.01Nm approximation',
                'saved PD25/0.5 and Go2HV torque-speed clipping for all joints','USD hard/soft limits, saved default pose/height'])

    def sync(self):
        d=self.data;m=self.model;t=self.tensor
        R=d.xmat[self.base].reshape(3,3)
        omega_b=d.qvel[3:6];v_com=d.qvel[:3]+np.cross(R@omega_b,R@m.body_ipos[self.base])
        for name,value in [('joint_pos',d.qpos[self.qadr]),('root_pos_w',d.qpos[:3]),('root_quat_w',d.qpos[3:7]),
                           ('root_lin_vel_w',v_com),('root_lin_vel_b',R.T@v_com),('root_ang_vel_b',omega_b)]:
            getattr(t,name).copy_(torch.as_tensor(value,dtype=torch.float32)[None])
        t.body_pos_w.copy_(torch.tensor(d.xpos[self.feet],dtype=torch.float32)[None])
        for i,body in enumerate(self.feet):
            jac=np.zeros((3,m.nv));mj.mj_jacBody(m,d,jac,None,body)
            t.body_lin_vel_w[0,i]=torch.tensor(jac@d.qvel,dtype=torch.float32)
        forces=np.zeros((4,3));base=np.zeros(3);buf=np.zeros(6)
        for i in range(d.ncon):
            c=d.contact[i];mj.mj_contactForce(m,d,i,buf)
            world=c.frame.reshape(3,3).T@buf[:3]
            for geom,sign in [(int(c.geom1),-1),(int(c.geom2),1)]:
                if geom in self.foot_geoms:forces[self.foot_geoms[geom]]+=sign*world
                if m.geom_bodyid[geom]==self.base:base+=sign*world
        self.sensor.data.net_forces_w.copy_(torch.tensor(forces,dtype=torch.float32)[None])
        self.base_contact_force=float(np.linalg.norm(base))

    def observation(self):
        R=self.data.xmat[self.base].reshape(3,3);t=self.tensor
        radius=1. if self.state.startup is None else (0. if self.state.startup.stage[0]==0 else .5+.5*float(self.state.startup.mix[0]))
        p=2*math.pi*float(self.state.phase[0])
        raw=[t.root_ang_vel_b[0].numpy().copy(),R.T@np.array([0,0,-1.]),self.command[0].numpy().copy(),
             self.data.qpos[self.qadr]-self.default,self.data.qvel[self.vadr],self.last_action.copy(),np.array([math.sin(p),math.cos(p)])*radius]
        terms=self.config['observations']['policy'];names=['base_ang_vel','projected_gravity','velocity_commands','joint_pos_rel','joint_vel_rel','last_action','gallop_phase']
        values=[]
        for value,name in zip(raw,names):
            term=terms[name]
            if self.noise and term.get('noise'):
                noise=term['noise'];value=value+self.rng.uniform(noise['n_min'],noise['n_max'],size=value.shape)
            if term.get('clip') is not None:value=np.clip(value,*term['clip'])
            if term.get('scale') is not None:value=value*term['scale']
            values.append(value)
        return torch.tensor(np.concatenate(values),dtype=torch.float32)[None]

    @torch.inference_mode()
    def step(self,last=False):
        obs=self.observation();sampled=self.actor(TensorDict(dict(policy=obs),batch_size=[1]))
        self.state.begin_step()
        executed=sampled
        if self.state.startup is not None:
            executed=self.core.blend_startup_actions(sampled,self.state.walking_policy(obs),self.state.startup.stage,self.state.startup.mix)
        self.last_action=executed[0].numpy().copy()
        self.state.executed_actions.copy_(executed)
        proposed=np.clip(self.default+self.settings.action_scale*self.last_action,-100,100)
        self.target=np.clip(proposed,self.soft[:,0],self.soft[:,1])
        self.state.action_clip[:]=float(np.mean(np.abs(proposed-self.target)>1e-6))
        for _ in range(self.decimation):
            self.last_torque=unitree_torque(self.target,self.data.qpos[self.qadr],self.data.qvel[self.vadr],self.motor)
            self.data.ctrl[self.act]=self.last_torque;mj.mj_step(self.model,self.data);mj.mj_forward(self.model,self.data)
            self.sync();self.state.observe_leap_physics(self.physics_dt)
            self.physics.append(dict(time=float(self.data.time),position=self.data.qpos[:3].copy(),
                velocity=self.tensor.root_lin_vel_b[0].numpy().copy(),angular_velocity=self.tensor.root_ang_vel_b[0].numpy().copy(),
                quaternion=self.data.qpos[3:7].copy(),foot_force=self.sensor.data.net_forces_w[0].numpy().copy(),
                foot_velocity=self.tensor.body_lin_vel_w[0].numpy().copy(),contact=self.state.leap.contact[0].numpy().copy(),
                recovering=bool(self.state.flight.recovering[0]),base_contact_force=self.base_contact_force))
        self.env.common_step_counter+=1
        tilt=math.acos(float(np.clip(self.data.xmat[self.base].reshape(3,3)[2,2],-1,1)))
        reasons=[]
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():reasons.append('nonfinite')
        if self.base_contact_force>1:reasons.append('base_contact')
        if tilt>self.config['terminations']['bad_orientation']['params']['limit_angle']:reasons.append('bad_orientation')
        if not self.settings.min_height<=self.data.qpos[2]<=self.settings.max_height:reasons.append('height')
        self.fault=','.join(reasons) if reasons else None
        self.env.reset_terminated[:]=bool(reasons);self.env.reset_time_outs[:]=last
        self.state.update_measurements()
        if reasons or last:self.state.complete_episodes(torch.tensor([0]))
        elif self.state.startup is not None:
            d=self.tensor;rpy=torch.stack(euler_xyz(d.root_quat_w),dim=1)
            self.state.startup.advance(self.control_dt,d.root_lin_vel_b,d.root_ang_vel_b,rpy,d.root_pos_w[:,2],self.state.age>0)
            self.command.zero_();self.command[:,0]=self.state.startup.command
        sample=self.state.snapshot();sample['action']=sampled.numpy().copy();sample['mujoco_qpos']=self.data.qpos[None].copy()
        sample['joint_velocity']=self.data.qvel[self.vadr][None].copy();sample['motor_torque']=self.last_torque[None].copy()
        return sample

    def unchanged(self):
        return sha(self.checkpoint)==self.checkpoint_hash and all(torch.equal(v,self.initial_policy[k]) for k,v in self.runner.alg.policy.state_dict().items())
