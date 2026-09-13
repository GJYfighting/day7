#!/usr/bin/env python3
"""Day7 offline checks and real Fortress smoke + 50 x 4 acceptance."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from day7_env import (ROOT, CurriculumManager, DomainRandomizer, PerceptionNoise,
                      atomic_json, local_path, run_episodes)
import numpy as np
import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external_cache_manifest():
    manifest = {}
    for name in ('.ignition', '.sdformat', '.gazebo', '.ros'):
        directory = Path.home()/name
        if directory.exists():
            for path in directory.rglob('*'):
                if path.is_file():
                    manifest[str(path.relative_to(Path.home()))] = [digest(path), path.stat().st_mtime_ns]
    return manifest


def isolation():
    previous = ROOT.parent / ('day' + '6')
    before = json.loads((ROOT / 'results/day6_sha256_before.json').read_text())
    after = {str(p.relative_to(previous)): digest(p) for p in sorted(previous.rglob('*')) if p.is_file()}
    atomic_json('results/day6_sha256_after.json', after)
    assert before == after, 'baseline SHA256 changed'
    foreign = set()
    for directory in ROOT.parent.iterdir():
        if directory.is_dir() and directory != ROOT and re.match(r'day\d', directory.name):
            foreign.update((p.stat().st_dev, p.stat().st_ino) for p in directory.rglob('*') if p.is_file())
    references = 0
    for p in ROOT.rglob('*'):
        if p.is_symlink():
            p.resolve().relative_to(ROOT)
        if not p.is_file():
            continue
        assert (p.stat().st_dev, p.stat().st_ino) not in foreign, f'foreign hardlink: {p}'
        try:
            text = p.read_text()
        except (UnicodeError, OSError):
            continue
        for match in re.finditer(r'/home/[^\s\"\'<>]*?/day\d[^\s\"\'<>]*', text):
            value = match.group()
            assert value.startswith(str(ROOT)), f'foreign absolute path: {p}: {value}'
        if p.suffix in ('.sdf', '.urdf'):
            for uri in re.findall(r'file://([^<\"\']+)', text):
                resource = Path(uri)
                if not resource.is_absolute():
                    continue  # Gazebo built-in media/materials resource.
                resource.resolve().relative_to(ROOT)
                assert resource.exists(), uri
                references += 1
    protected = ('day4_env.py', 'residual_env.py', 'visual_grasp_v5.py',
                 'grasp_geometry_v5.py', 'models/sac_smoke.zip',
                 'simulations/robot_gazebo/worlds/grasp_table.sdf')
    for name in protected:
        assert digest(ROOT/name) == before[name], f'protected baseline changed: {name}'
    current_config = yaml.safe_load((ROOT/'config.yaml').read_text())
    current_config.pop('day7')
    assert current_config == yaml.safe_load((previous/'config.yaml').read_text()), 'baseline configuration changed'
    changes = {'copied_unchanged': [], 'copied_modified': [], 'added': [], 'deleted': []}
    for name, old in before.items():
        p = ROOT / name
        if not p.exists():
            changes['deleted'].append(name)
        else:
            changes['copied_unchanged' if digest(p) == old else 'copied_modified'].append(name)
    changes['added'] = [str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and str(p.relative_to(ROOT)) not in before]
    atomic_json('results/day7_file_changes.json', changes)
    return dict(baseline_files=len(before), unchanged=True, local_model_references=references,
                modified=changes['copied_modified'], deleted=changes['deleted'])


def syntax(config):
    counts = dict(python=0, yaml=0, xml=0, shell=0)
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.relative_to(ROOT).parts for x in ('vendor', 'runtime')):
            continue
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(p)); counts['python'] += 1
        elif p.suffix in ('.yaml', '.yml'):
            yaml.safe_load(p.read_text()); counts['yaml'] += 1
        elif p.suffix in ('.sdf', '.urdf', '.srdf'):
            ET.parse(p); counts['xml'] += 1
        elif p.suffix == '.sh':
            subprocess.run(['bash', '-n', str(p)], check=True); counts['shell'] += 1
    randomizer = DomainRandomizer(config)
    for level in config['day7']['levels']:
        path = randomizer.write_sdf(randomizer.sample(123, level))
        p = subprocess.run(['ign', 'sdf', '-k', str(path)], capture_output=True, text=True, timeout=30)
        assert p.returncode == 0 and 'Valid' in p.stdout + p.stderr, p.stdout + p.stderr
        model = ET.parse(path).find('model')
        assert model.attrib['name'] == 'wood_block'
        assert model.find("link/sensor/contact/collision").text == 'block_collision'
        assert model.find('plugin/odom_topic').text == config['topics']['block_odometry']
    original = ET.parse(local_path(config['day7']['world_sdf']))
    block = original.find(".//model[@name='wood_block']")
    assert float(block.findtext('link/inertial/mass')) == config['day7']['nominal']['mass_kg']
    assert original.findtext(".//plugin[@name='ignition::gazebo::systems::Sensors']/render_engine") == 'ogre2'
    for key in ('mu', 'mu2'):
        assert float(block.findtext('link/collision/surface/friction/ode/'+key)) == config['day7']['nominal'][key]
    return counts


def numerical(config):
    from residual_env import ResidualActionController
    import day6_check as baseline
    versions = {}
    for name in ('numpy', 'cv2', 'gymnasium', 'stable_baselines3', 'torch'):
        module = importlib.import_module(name)
        versions[name] = dict(version=module.__version__, path=module.__file__)
    expected = dict(numpy='1.26.4', cv2='4.8.1', gymnasium='0.29.1', stable_baselines3='2.3.2')
    assert all(versions[k]['version'] == v for k, v in expected.items())
    controller = ResidualActionController(ROOT/'config.yaml')
    assert controller.model_loaded, controller.model_load_error
    assert controller.model.observation_space.shape == (10,)
    assert controller.model.action_space.shape == (4,)
    baseline.check_model(controller)
    baseline.check_base_equivalence(ROOT/'config.yaml')
    baseline.check_residual_finite(controller)
    baseline.check_residual_limit(ROOT/'config.yaml')
    baseline.check_low_pass(ROOT/'config.yaml')
    baseline.check_final_limit(ROOT/'config.yaml')
    baseline.check_invalid_guard(ROOT/'config.yaml')
    baseline.check_no_confidence_gate()
    # Exact off-domain fusion including the frozen SAC and filter state.
    other = ResidualActionController(ROOT/'config.yaml')
    p = DomainRandomizer(config).sample(22, 'L3', 'off')
    other.base_gain *= p['gain_scale']
    controller.reset_episode()
    for i in range(100):
        obs = baseline.observation(ex=i*0.0001)
        a, b = controller.decide(obs), other.decide(obs)
        for name in ('base_action', 'mapped_residual', 'filtered_residual', 'final_action'):
            assert np.array_equal(getattr(a, name), getattr(b, name))
    # Transport failures are promoted before the inherited TestFailure handler.
    from day7_env import technical_guard, TechnicalFailure
    from visual_grasp_v5 import TestFailure
    for reason, expected_technical in [('INTERFACE_NOT_READY', True), ('NO_DUAL_CONTACT', False)]:
        def operation():
            raise TestFailure(reason)
        try:
            try:
                technical_guard(operation)()
            except TestFailure:
                pass
            assert not expected_technical
        except TechnicalFailure:
            assert expected_technical
    # A delayed command must not block receipt of the six perception topics.
    from day7_env import DelayedPublisher
    import threading
    from types import SimpleNamespace
    received=[]
    done=threading.Event()
    class Sink:
        def publish(self, message):
            received.append((time.monotonic(), message))
            if len(received)==2: done.set()
    delay=config['day7']['levels']['L3']['delay_ms']/1000
    publisher=DelayedPublisher(Sink(),SimpleNamespace(params={'delay_sec':delay}))
    try:
        start=time.monotonic()
        message={'value':1}
        publisher.publish(message)
        message['value']=2
        publisher.publish(message)
        assert time.monotonic()-start < delay/2, 'command delay blocks executor'
        assert done.wait(2), 'delayed messages missing'
        assert [x[1]['value'] for x in received]==[1,2], 'mutable message/order changed'
        assert all(stamp-start >= delay for stamp,_ in received), 'command sent early'
        publisher.publish({'value':3})
        publisher.clear()
        time.sleep(delay*1.2)
        assert len(received)==2, 'old episode command leaked'
    finally:
        publisher.close()
    return dict(versions=versions, observation_dim=10, action_dim=4,
                final_limits=controller.final_limit.tolist(), residual_limits=controller.residual_limit.tolist(),
                off_fusion_bitwise_identical=True, nonblocking_command_delay=True)


def randomization(config):
    d = DomainRandomizer(config)
    for level in config['day7']['levels']:
        a = [d.sample(s, level) for s in range(100)]
        assert a == [d.sample(s, level) for s in range(100)]
        assert a != [d.sample(s+1, level) for s in range(100)]
        for p in a:
            for i, key in enumerate(('base_x_m', 'base_y_m')):
                x = p['position_world_m'][i] - config['robot_spawn_world'][i]
                lo, hi = config['day4']['sampling'][key]
                assert lo <= x <= hi
            json.dumps(p, allow_nan=False)
    p = d.sample(4, 'L3', 'off')
    assert all(p[k] == 0 for k in ('pixel_std','depth_std_m','invalid_fraction','delay_sec'))
    assert p['translation_m'] == [0,0,0] and p['rotation_rad'] == [0,0,0]
    assert all(p[k] == config['day7']['nominal'][k] for k in ('mass_kg','mu','mu2','brightness','contrast','gain_scale'))
    rgb, depth = np.full((24,32,3), 100, np.uint8), np.ones((24,32),np.float32)
    noise = PerceptionNoise(p)
    x,y = noise.images(rgb,depth)
    assert np.array_equal(rgb,x) and np.array_equal(depth,y)
    p = d.sample(4,'L3')
    first, second = PerceptionNoise(p), PerceptionNoise(p)
    for _ in range(5):
        a,b = first.images(rgb,depth), second.images(rgb,depth)
        assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert not np.array_equal(a[0],rgb) and not np.array_equal(a[1],depth)
    r,t = first.extrinsics(np.eye(3),np.zeros(3))
    assert np.allclose(r.T@r,np.eye(3)) and np.isclose(np.linalg.det(r),1)
    invalid_config=json.loads(json.dumps(config))
    invalid_config['day7']['levels']['L3']['position_fraction']=1.01
    try:
        DomainRandomizer(invalid_config)
    except ValueError:
        pass
    else:
        raise AssertionError('expanded safety rectangle was accepted')
    return dict(seeds_per_level=100, deterministic_frames=5, off_zero=True, enlarged_rectangle_rejected=True)


def curriculum(config):
    d = config['day7']
    for start in range(4):
        for successes in (55,56,79,80):
            m = CurriculumManager(d,'auto',f'L{start}')
            for i in range(99):
                m.update(i < successes)
                assert m.level == start
            saved = m.state()
            for _ in range(10): m.update(False,True)
            assert m.state() == saved
            m.update(False)
            expected = min(3,start+1) if successes >= 80 else max(0,start-1) if successes <=55 else start
            assert m.level == expected
            if expected != start:
                for _ in range(99): m.update(True)
                assert m.level == expected
    for mode in ('off','fixed'):
        m = CurriculumManager(d,mode,'L1')
        for _ in range(300): m.update(True)
        assert m.label == 'L1'
    return dict(thresholds=[0.55,0.56,0.79,0.80], dwell=100, technical_excluded=True, boundaries=True)


def environment():
    assert Path(os.environ.get('DAY7_ROOT','')).resolve() == ROOT, 'source session_env.sh first'
    for key in ('ROS_HOME','ROS_LOG_DIR','IGN_LOG_PATH','IGN_FUEL_CACHE_PATH','IGN_HOMEDIR',
                'PYTHONPYCACHEPREFIX','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_RUNTIME_DIR',
                'MPLCONFIGDIR','QML_DISK_CACHE_PATH','TORCH_HOME','CUDA_CACHE_PATH','TMPDIR'):
        Path(os.environ[key]).resolve().relative_to(ROOT/'runtime')
    for key in ('PATH','PYTHONPATH'):
        assert not any(re.search(r'/day[0-6](?:/|$)', p) for p in os.environ.get(key,'').split(':'))
    assert Path(shutil.which('ign')).resolve() == ROOT/'runtime/bin/ign'
    subprocess.run(['ruby','-c',str(ROOT/'runtime/bin/ign')],check=True,capture_output=True)
    assert os.environ['ROS_DOMAIN_ID'] != '88'
    assert os.environ['IGN_PARTITION'].startswith('day7_')
    return dict(domain=os.environ['ROS_DOMAIN_ID'],partition=os.environ['IGN_PARTITION'],python=sys.executable)


def live(report, save, only_level=None, smoke_only=False):
    from day6_check import stop_process
    logroot = ROOT/'runtime/logs/day7_check'/time.strftime('%Y%m%d_%H%M%S')
    logroot.mkdir(parents=True,exist_ok=True)
    (logroot/'config.yaml').write_text((ROOT/'config.yaml').read_text())
    specs = [('world',['ros2','launch',str(ROOT/'world.launch.py'),'gui:=false']),
             ('moveit',['ros2','launch',str(ROOT/'moveit.launch.py')]),
             ('perception',['/usr/bin/python3',str(ROOT/'perception_v5.py'),'--ros-args','--params-file',str(ROOT/'perception_v5.yaml')])]
    processes=[]
    evaluation_config=yaml.safe_load((ROOT/'config.yaml').read_text())['day7']
    evaluation_seed=evaluation_config['seed']
    external_log=Path.home()/'.ignition/rendering/ogre2.log'
    external_before=(digest(external_log),external_log.stat().st_mtime_ns) if external_log.exists() else None
    env = os.environ.copy()
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR'): env.pop(key,None)
    try:
        for name,cmd in specs:
            stream=(logroot/(name+'.log')).open('w')
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((p,stream))
        time.sleep(5)
        for phase,count in ([('smoke',1)] if smoke_only else [('smoke',1),('full',50)]):
            for index in ([int(only_level[1])] if only_level else range(4)):
                name=f'{phase}_L{index}'
                if report[name]['status'] == 'PASS':
                    continue
                prior=report[name]
                retry=prior.get('stage_retry',0)+(1 if prior['status']=='FAIL' else 0)
                if retry > evaluation_config['stage_restarts']:
                    raise RuntimeError(f'{name}: stage restart limit reached')
                tag=name if retry==0 else name+f'_retry{retry}'
                if any(p.poll() is not None for p,_ in processes[:3]):
                    raise RuntimeError('simulation component exited; inspect runtime/logs/day7_check')
                logpath=logroot/(tag+'.log')
                with logpath.open('w') as stream:
                    code="from day7_env import run_episodes; import sys; r=run_episodes('fixed',sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),tag=sys.argv[4]); sys.exit(0 if sum(not x['technical_failure'] for x in r)==int(sys.argv[2]) else 1)"
                    p=subprocess.Popen(['/usr/bin/python3','-c',code,f'L{index}',str(count),str(evaluation_seed+index),tag],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    processes.append((p,stream))
                    while p.poll() is None:
                        time.sleep(10)
                    status='PASS' if p.returncode==0 else 'FAIL'
                report[name]=dict(status=status,log=str(logpath.relative_to(ROOT)),requested_valid=count,record_tag=tag,stage_retry=retry,previous_runs=prior.get('previous_runs',[])+([prior] if prior['status']=='FAIL' else []))
                save()
            if phase == 'smoke' and any(report[f'smoke_L{i}']['status'] == 'FAIL' for i in range(4)):
                raise RuntimeError('smoke failed; full acceptance was not started')
        if any(report[f'full_L{i}']['status'] == 'FAIL' for i in range(4)):
            raise RuntimeError('one or more full evaluation levels failed')
        return dict(headless=True,valid_episodes=0 if smoke_only else (50 if only_level else 200),smoke_episodes=1 if only_level else 4)
    finally:
        for p,stream in reversed(processes):
            stop_process(p)
            stream.close()
        external_after=(digest(external_log),external_log.stat().st_mtime_ns) if external_log.exists() else None
        internal_log=ROOT/'runtime/home/.ignition/rendering/ogre2.log'
        report['runtime_cache_isolation']=dict(status='PASS' if external_before==external_after and internal_log.exists() else 'FAIL', external_ogre_log_unchanged=external_before==external_after, internal_ogre_log_exists=internal_log.exists())
        save()



def episode_audit(config, run_id, report=None):
    from residual_env import ResidualActionController
    controller = ResidualActionController(ROOT/'config.yaml')
    randomizer = DomainRandomizer(config)
    path = local_path(config['day7']['results_path'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [r for r in rows if r.get('run_id') == run_id]
    counts = {}
    actions = 0
    for phase, expected in [('smoke', 1), ('full', 50)]:
        for level in config['day7']['levels']:
            key = phase+'_'+level
            tag = (report or {}).get(key,{}).get('record_tag',key)
            group = [r for r in rows if r['tag'] == tag]
            valid = [r for r in group if not r['technical_failure']]
            assert len(valid) == expected, f'{tag}: {len(valid)}/{expected} valid'
            assert len({r['attempt'] for r in group}) == len(group), f'{tag}: duplicate attempts'
            for r in group:
                json.dumps(r, allow_nan=False)
                assert r['steps'] == len(r['actions'])
                if r['technical_failure']:
                    assert r['technical_reason']
                    assert not any(word in r['technical_reason'].lower() for word in ('non-finite', 'outside limits', 'nan', 'step budget exceeded', 'mismatch')), r['technical_reason']
                else:
                    assert r['terminated'] or r['truncated']
                    params = r['parameters']
                    expected_params = randomizer.sample(r['seed'], level, r['mode'])
                    assert all(params[k] == v for k,v in expected_params.items()), f'{tag}: parameter replay differs'
                    assert params['model_rebuild_acknowledged'] and params['sdf_sha256']
                    for key in ('mass_kg','mu','mu2'):
                        measured = params['physics_readback'][key]
                        nominal = params[key]
                        rounded = float(format(nominal, '.'+str(config['day7']['sdf_readback_significant_digits'])+'g'))
                        assert any(np.isclose(measured,v,rtol=1e-12,atol=1e-15) for v in (nominal, rounded)), f'{tag}: unapplied {key}'
                for action in r['actions']:
                    for key, limit in [('final_action', controller.final_limit), ('mapped_residual', controller.residual_limit), ('filtered_residual', controller.residual_limit)]:
                        value = np.asarray(action[key], dtype=np.float32)
                        assert value.shape == (4,) and np.isfinite(value).all()
                        assert np.all(np.abs(value) <= limit), f'{tag}: {key} bound'
                    actions += 1
            counts[tag] = dict(valid=len(valid), technical=len(group)-len(valid))
    return dict(counts=counts, audited_actions=actions, deterministic_parameter_replay=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-check',action='store_true',help='restart only failed/incomplete levels, preserving successful runs and failed history')
    parser.add_argument('--skip-live',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--only-level',choices=['L0','L1','L2','L3'])
    args=parser.parse_args()
    external_before=external_cache_manifest()
    config=yaml.safe_load((ROOT/'config.yaml').read_text())
    report={}
    run_id=str(time.time_ns())
    if args.resume_check:
        previous=json.loads((ROOT/'results/day7_check.json').read_text())
        report=previous['checks']
        run_id=previous['run_id']
        # The new simulator must complete its own cache and environment checks.
        for key in ('gazebo','episode_audit','runtime_cache_isolation','external_cache_unchanged'):
            report.pop(key,None)
    os.environ['DAY7_CHECK_RUN_ID']=run_id
    for phase in ('smoke','full'):
        for i in range(4): report.setdefault(f'{phase}_L{i}',dict(status='SKIP',reason='not yet run'))
    def save():
        incident_path=ROOT/'results/day7_isolation_incident.json'
        history=dict(isolation_incident='results/day7_isolation_incident.json', entire_workflow_isolation_compliant=False, corrected_before_this_run=True) if incident_path.exists() else {}
        passed=bool(report) and all(v['status']=='PASS' for v in report.values())
        atomic_json('results/day7_check.json',dict(run_id=run_id,status='PASS' if passed else 'NOT_PASS',checks=report,history=history))
        path=local_path(config['day7']['results_path'])
        records=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        summary={}
        for phase in ('smoke','full'):
            for i in range(4):
                key=f'{phase}_L{i}'; tag=report[key].get('record_tag',key); rows=[r for r in records if r['tag']==tag and r.get('run_id')==run_id]
                valid=[r for r in rows if not r['technical_failure']]
                summary[key]=dict(valid=len(valid),technical_failures=len(rows)-len(valid),
                                  successes=sum(r['success'] for r in valid),status=report[key]['status'],
                                  success_rate=sum(r['success'] for r in valid)/len(valid) if valid else None,
                                  mean_return=sum(r['return'] for r in valid)/len(valid) if valid else None,
                                  mean_steps=sum(r['steps'] for r in valid)/len(valid) if valid else None)
        atomic_json('results/day7_summary.json',dict(run_id=run_id,status='PASS' if passed else 'NOT_PASS',levels=summary,checks=report,history=history))
    def check(name,fn):
        try: report[name]=dict(status='PASS',detail=fn())
        except Exception as exc: report[name]=dict(status='FAIL',reason=f'{type(exc).__name__}: {exc}')
        print(name+'='+report[name]['status']+' '+str(report[name].get('reason','')),flush=True)
        save()
    check('environment',environment)
    check('syntax',lambda:syntax(config))
    check('numerical',lambda:numerical(config))
    check('randomization',lambda:randomization(config))
    check('curriculum',lambda:curriculum(config))
    check('isolation',isolation)
    if not args.skip_live and all(v['status'] != 'FAIL' for k,v in report.items() if not k.startswith(('smoke_','full_'))):
        check('gazebo',lambda:live(report,save,args.only_level,args.smoke_only))
    else:
        report['gazebo']=dict(status='SKIP',reason='--skip-live or prerequisite failed')
    if all(report[f'{phase}_L{i}']['status'] == 'PASS' for phase in ('smoke','full') for i in range(4)):
        check('episode_audit',lambda:episode_audit(config,run_id,report))
    else:
        report['episode_audit']=dict(status='SKIP',reason='full dataset incomplete')
    check('final_isolation',isolation)
    external_after=external_cache_manifest()
    report['external_cache_unchanged']=dict(status='PASS' if external_before==external_after else 'FAIL', checked_files=len(external_before), changed=[k for k in set(external_before)|set(external_after) if external_before.get(k)!=external_after.get(k)])
    save()
    changes=json.loads((ROOT/'results/day7_file_changes.json').read_text())
    print(f"FILES: copied unchanged={len(changes['copied_unchanged'])}, copied modified={len(changes['copied_modified'])}, added={len(changes['added'])}, deleted={len(changes['deleted'])}",flush=True)
    for name in changes['copied_modified']: print('MODIFIED: '+name,flush=True)
    summary=json.loads((ROOT/'results/day7_summary.json').read_text())
    for name,item in summary['levels'].items(): print(name+' '+json.dumps(item),flush=True)
    if (ROOT/'results/day7_isolation_incident.json').exists():
        print('HISTORY: early diagnostics wrote external Gazebo/SDFormat logs; fixed before final run. See results/day7_isolation_incident.json; entire workflow isolation was not fully compliant.',flush=True)
    print('REPRODUCE: source ~/ros2_ws/day7/session_env.sh && /usr/bin/python3 day7_check.py',flush=True)
    passed=all(v['status']=='PASS' for v in report.values())
    print('DAY7_CHECK='+('PASS' if passed else 'NOT_PASS'),flush=True)
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
