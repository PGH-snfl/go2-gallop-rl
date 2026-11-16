"""Conditional Go2 kinematic/ballistic budgets, never a hardware or learning maximum."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import brentq

from gpd.paths import ROOT


def stance_screen(speed,height,stance_seconds,length,limits,velocity_factor=1.):
    # Planar hip-roll0, constant thigh-root height, symmetric fixed-foot stance.
    # x is foot centre relative to thigh root; dx/dt=-body forward speed.
    x=np.linspace(-speed*stance_seconds/2,speed*stance_seconds/2,2001)
    r2=x*x+height*height
    cosine=(r2-2*length*length)/(2*length*length)
    if np.max(np.abs(cosine))>1+1e-10:
        return dict(feasible=False,reason='unreachable')
    knee=-np.arccos(np.clip(cosine,-1,1))
    thigh=np.arctan2(-x,height)-knee/2
    dk_dx=-x/(length*length*np.sin(knee))
    dt_dx=-height/r2-dk_dx/2
    calf_speed=float(np.max(np.abs(dk_dx*speed)))
    thigh_speed=float(np.max(np.abs(dt_dx*speed)))
    position_ok=bool((knee>=limits['calf']['lower']-1e-10).all() and (knee<=limits['calf']['upper']+1e-10).all())
    for group in ('front_thigh','rear_thigh'):
        position_ok &= bool((thigh>=limits[group]['lower']-1e-10).all() and (thigh<=limits[group]['upper']+1e-10).all())
    ratio=max(calf_speed/(limits['calf']['velocity']*velocity_factor),
              thigh_speed/(min(limits['front_thigh']['velocity'],limits['rear_thigh']['velocity'])*velocity_factor))
    return dict(feasible=position_ok and ratio<=1+1e-8,position_ok=position_ok,velocity_ratio=ratio,
        calf_peak_rad_s=calf_speed,thigh_peak_rad_s=thigh_speed,
        thigh_min_max_rad=[float(thigh.min()),float(thigh.max())],
        calf_min_max_rad=[float(knee.min()),float(knee.max())])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    urdf=ROOT/'reports/20261111_gallop_46000_sim2sim_review/sources/official_go2_description.urdf'
    scene=ROOT/'reports/20261111_gallop_46000_sim2sim_review/sources/local_original_go2.xml'
    nodes={n.attrib['name']:n for n in ET.parse(urdf).getroot().findall('joint')}
    def limit(name):return {k:float(v) for k,v in nodes[name].find('limit').attrib.items()}
    limits={group:limit(name) for group,name in [('hip','FL_hip_joint'),('front_thigh','FL_thigh_joint'),
        ('rear_thigh','RL_thigh_joint'),('calf','FL_calf_joint')]}
    body=ET.parse(scene).getroot().find(".//body[@name='FL_calf']")
    length=abs(float(body.attrib['pos'].split()[2]))
    lower_length=abs(float(body.find("body[@name='FL_foot']").attrib['pos'].split()[2]))
    if not math.isclose(length,lower_length):raise ValueError('Expected equal213mm links')
    rmax=2*length*math.cos(abs(limits['calf']['upper'])/2)
    screens=[]
    for period,duty,label in [(.39,.45,'current18cm_reference'),(.60,.28,'final80cm_reference')]:
        for h in (.24,.28,.32):
            geometry_speed=2*math.sqrt(rmax*rmax-h*h)/(duty*period)
            for factor in (1.,.9):
                def cost(v):return stance_screen(v,h,period*duty,length,limits,factor)['velocity_ratio']-1
                cap=geometry_speed if cost(geometry_speed)<=0 else brentq(cost,.001,geometry_speed)
                detail=stance_screen(cap,h,period*duty,length,limits,factor)
                if not detail['position_ok']:raise ValueError('Joint position constraint needs another search')
                screens.append(dict(reference=label,period_s=period,max_leg_duty=duty,
                    thigh_root_to_foot_centre_height_m=h,stance_seconds=period*duty,
                    angular_velocity_factor=factor,geometry_only_speed_mps=geometry_speed,
                    conditional_stance_speed_mps=cap,**detail))
    rows=[];g=9.81;distance=.8
    for v in (2.,2.5,2.75,3.,3.7):
        t=distance/v;vz=g*t/2;rise=vz*vz/(2*g)
        rows.append(dict(forward_speed_mps=v,air_distance_m=distance,flight_seconds=t,
            vertical_takeoff_and_landing_speed_mps=vz,com_apex_rise_m=rise,
            required_min_average_vertical_force_bw_period060=.6/(.6-t) if t<.6 else None,
            required_min_average_vertical_force_bw_period039=.39/(.39-t) if t<.39 else None,
            air_distance_in030s_m=v*.3,air_distance_in0078s_m=v*.078,
            minimum_landing_compression_m_if_force_globally_capped3bw=vz*vz/(2*g*(3-1))))
    with (out/'ballistic80cm.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'conditional_stance_screens.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(screens[0]));w.writeheader();w.writerows(screens)
    manifest=json.loads((ROOT/'runs/isaac/train/unitree_go2_gallop/20261110_223314_cef16b/manifest.json').read_text())
    cfg=manifest['gallop_settings'];references=[]
    for goal in manifest['staged_design']['plan']:
        v=goal['speed'];r=np.clip((v-cfg['reference_start_speed'])/(cfg['reference_max_speed']-cfg['reference_start_speed']),0,1)
        thigh=cfg['thigh_amplitude_slow']+(cfg['thigh_amplitude_fast']-cfg['thigh_amplitude_slow'])*r
        calf=cfg['calf_amplitude_slow']+(cfg['calf_amplitude_fast']-cfg['calf_amplitude_slow'])*r
        references.append(dict(name=goal['name'],speed_mps=v,period_s=goal['period'],reference_air_s=goal['air_seconds'],
            reference_air_distance_budget_m=v*goal['air_seconds'],
            reference_thigh_peak_rad_s=max(thigh*math.pi/(goal['period']*min(d,1-d)) for d in goal['duty']),
            reference_calf_peak_rad_s=max(calf*math.pi/(goal['period']*(1-d)) for d in goal['duty'])))
    result=dict(exact_trainable_maximum_speed_mps=None,exact_trainable_maximum_air_distance_m=None,
        reason='Independent position/velocity/effort maxima do not identify joint torque-speed-temperature capability, contact dynamics or a feasible stable periodic gait.',
        official_product_spec=dict(edu_listed_speed_mps=3.7,laboratory_max_approx_mps=5.,
            url='https://www.unitree.com/go2/',not_a_custom_gallop_certification=True),
        inputs=dict(official_joint_limits=limits,thigh_length_m=length,calf_length_m=lower_length,
            maximum_sagittal_thigh_root_to_foot_centre_reach_m=rmax,
            urdf_path=str(urdf),urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
            mjcf_path=str(scene),mjcf_sha256=hashlib.sha256(scene.read_bytes()).hexdigest()),
        stance_screens=screens,ballistic80cm=rows,reference_shape_speed_budgets=references,
        assumptions=[
            'Stance screen: symmetric sagittal stance, zero body rotation, constant thigh-root height, no foot slip; swing return and forces/inertia omitted.',
            'Height is thigh-root to foot-centre vertical distance, not measured trunk height or ground clearance.',
            'The90percent velocity factor is a proposed engineering margin, not a Unitree official rating.',
            'Ballistics: constant horizontal COM velocity, equal COM takeoff/first-touchdown height, gravity9.81, no external aerial force.',
            'The ballistic model and constant-height stance screen are separate necessary-budget models; not a dynamically consistent complete trajectory.',
            'The3BW global-force assumption is an illustrative project constraint, not a manufacturer rating; existing3BW landing checks do not impose a whole-cycle cap.',
            'Periodic impulse assumes no other flight phase; any additional flight makes the required mean stance force larger.',
            'Body-root air distance in existing logs can differ from COM air distance as the robot changes orientation/configuration.',
            'Reference-shape derivatives do not prove actual joint tracking or dynamically attainable speed; the calf reference has derivative changes at stance/swing boundaries.',
        ],policy_updates=0,physics_rollouts=0,physical_hardware=False)
    (out/'calculation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    # Necessary checks on formulas, not tests of a robot performance claim.
    assert math.isclose(rows[3]['flight_seconds'],.8/3)
    assert math.isclose(rows[3]['vertical_takeoff_and_landing_speed_mps'],1.308)
    assert math.isclose(rows[3]['com_apex_rise_m'],.0872)
    assert math.isclose(rows[3]['required_min_average_vertical_force_bw_period060'],1.8)
    assert all(s['feasible'] for s in screens)
    for s in screens:
        assert not stance_screen(s['conditional_stance_speed_mps']*1.00001,
            s['thigh_root_to_foot_centre_height_m'],s['stance_seconds'],length,limits,s['angular_velocity_factor'])['feasible']
    (out/'verification.json').write_text(json.dumps(dict(analytic_ballistic_checks=4,conditional_boundary_checks=len(screens),
        feasible_in_stance_screen_only=True,exact_robot_maximum_not_identified=True,policy_updates=0,physics_rollouts=0),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),reference_speeds=references,
        nominal_height_screens=[s for s in screens if s['thigh_root_to_foot_centre_height_m']==.28],ballistic80cm=rows),indent=2))


if __name__=='__main__':main()
