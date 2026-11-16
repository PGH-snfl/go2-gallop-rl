"""Render recorded baseline-evaluation traces; no simulation/training imports."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    a=p.parse_args();run=a.run
    report=json.loads((run/'report.json').read_text())
    with np.load(run/'trace.npz') as archive:
        data={key:archive[key] for key in archive.files}
    assert report.get('observation_commands_verified'), 'Unverified evaluator output'
    np.testing.assert_allclose(data['policy_command'],data['command'],atol=1e-6,rtol=0)
    with (run/'state_trace.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['time','trial','repeat','valid_before_first_failure','vx_cmd','vy_cmd','yaw_cmd','vx','vy','yaw_rate','height','roll','pitch','FL_force','FR_force','RL_force','RR_force','peak_abs_torque'])
        for step,t in enumerate(data['time']):
            for i,trial in enumerate(report['trials']):
                w.writerow([t,trial['name'],trial['repeat'],int(data['valid'][step,i]),*data['command'][step,i],
                    *data['body_velocity'][step,i,:2],data['angular_velocity'][step,i,2],data['position'][step,i,2],
                    *data['rpy'][step,i,:2],*data['foot_force'][step,i],np.abs(data['torque'][step,i]).max()])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(dict.fromkeys(t['name'] for t in report['trials']))
    fig,axes=plt.subplots(len(names),3,figsize=(15,3*len(names)),layout='constrained')
    time=data['time']
    for row,name in enumerate(names):
        ids=[i for i,t in enumerate(report['trials']) if t['name']==name]
        for i in ids:
            mask=data['valid'][:,i];x=time[mask]
            axes[row,0].plot(x,data['body_velocity'][mask,i,0],lw=1,alpha=.8)
            axes[row,1].plot(x,data['angular_velocity'][mask,i,2],lw=1,alpha=.8)
            axes[row,2].plot(x,np.rad2deg(data['rpy'][mask,i,0]),lw=1,alpha=.8)
            axes[row,2].plot(x,np.rad2deg(data['rpy'][mask,i,1]),lw=.7,alpha=.5,linestyle=':')
        axes[row,0].plot(time,data['command'][:,ids[0],0],'k--',label='command')
        axes[row,1].plot(time,data['command'][:,ids[0],2],'k--',label='command')
        for axis,label in zip(axes[row],['vx (m/s)','yaw rate (rad/s)','roll / pitch (deg)']):
            axis.set_title(name+' — '+label);axis.grid(alpha=.2);axis.set_xlabel('Protocol time (s)')
    fig.suptitle('Flat-ground policy evaluation: trace ends at first failure; repetitions overlay')
    fig.savefig(run/'tracking.png',dpi=140);plt.close(fig)
    focal_name=report.get('focal_case','forward_04')
    focal=next(i for i,t in enumerate(report['trials']) if t['name']==focal_name)
    mask=data['valid'][:,focal];x=time[mask]
    fig,axes=plt.subplots(2,1,figsize=(12,5),layout='constrained',sharex=True)
    axes[0].imshow((data['foot_force'][mask,focal]>5).T,origin='lower',aspect='auto',interpolation='nearest',extent=[x[0],x[-1],-.5,3.5],cmap='Blues',vmin=0,vmax=1)
    axes[0].set_yticks(range(4),['FL','FR','RL','RR']);axes[0].set_ylabel('Contact >5N')
    axes[1].plot(x,data['body_velocity'][mask,focal,0],label='measured vx');axes[1].plot(x,data['command'][mask,focal,0],'k--',label='command')
    axes[1].set_ylabel('m/s');axes[1].set_xlabel('Time (s)');axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(run/f'{focal_name}_contacts.png',dpi=160);plt.close(fig)
    print(run/'tracking.png')
    for case in report['trials']:
        m=case['metrics'];failed=[key for key,v in case['gates'].items() if not v]
        print(case['name'],case['repeat'],'PASS' if case['passed'] else 'FAIL',dict(vx_rmse=m['vx_rmse'],yaw_rmse=m['yaw_rmse'],stop_speed=m['stop_speed_rms'],slip=m['contact_slip_rms']),failed)

if __name__=='__main__':main()
