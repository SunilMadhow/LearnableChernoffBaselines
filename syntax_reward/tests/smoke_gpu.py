"""Small GPU plumbing checks with real models; no fitted paper files are changed."""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from util import load_core_and_tokenizer, seed_everything, LLaDAReward
from nets_language_classifier import ValueClassifier, ClassificationDataset
from lcb_with_offline import LCB, LCBOfflineDataset
from bon import run_standard_generation
from rejection_sampling import run_rejection_sampling
from collection import SequentialCollector
from sweep_smc import run_one
from analysis.plot_lcb_evolution import collect_trajectories_via_api

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True)
parser.add_argument('--run_id', default='20260205_195651_634076')
args = parser.parse_args()
OUT = Path(args.output_dir).resolve()
assert OUT.is_relative_to(ROOT), 'Keep test outputs inside the camera-ready directory.'
OUT.mkdir(parents=True, exist_ok=True)
RUN = ROOT/'data_runs'/args.run_id

class Tee:
    def __init__(self, console, log): self.console, self.log = console, log
    def write(self, text):
        self.console.write(text); self.log.write(text); self.log.flush()
    def flush(self): self.console.flush(); self.log.flush()

log = (OUT/'gpu_smoke.log').open('w')
sys.stdout = Tee(sys.stdout, log)
sys.stderr = Tee(sys.stderr, log)
assert torch.cuda.is_available()
seed_everything(42)
report = {
    'scope': 'Small GPU plumbing checks, not a scientific rerun or training run',
    'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
    'cuda': torch.version.cuda, 'python': sys.version,
    'packages': {p: importlib.metadata.version(p) for p in ['transformers','accelerate','bitsandbytes','numpy']},
    'paper_checkpoints_overwritten': False, 'optimizer_learning_rates': 0,
    'checks': [],
}

def save_report():
    (OUT/'gpu_smoke_report.json').write_text(json.dumps(report, indent=2))

def check(name, fn):
    print('\n=== '+name+' ===', flush=True)
    started = time.monotonic()
    try:
        details = fn()
        torch.cuda.synchronize()
        item = {'name':name, 'status':'passed', 'details':details}
    except Exception as exc:
        traceback.print_exc()
        item = {'name':name, 'status':'failed', 'error':repr(exc)}
    item['seconds'] = round(time.monotonic()-started, 2)
    report['checks'].append(item)
    save_report()
    print(name+': '+item['status'], flush=True)
    return item['status']=='passed'

print('Loading the real 4-bit LLaDA model, critic and LCB on '+report['gpu'], flush=True)
core, tokenizer = load_core_and_tokenizer(2)
vc = ValueClassifier.load(str(RUN), device='cuda')
vc.reward_temperature = 0.5
lcb = LCB.load(str(RUN), None, vc, core, 'lcb_lambda_only_SYM_post_trained_temp_0p5')
assert next(core.model.parameters()).is_cuda and next(vc.net.parameters()).is_cuda
assert next(lcb.net.parameters()).is_cuda
report['models_loaded_on_gpu'] = True
report['llada_4bit'] = bool(getattr(core.model, 'is_loaded_in_4bit', False))
report['model_revisions'] = {
    'llada': getattr(core.model.config, '_commit_hash', None),
    'critic_backbone': getattr(vc.net.config, '_commit_hash', None),
}
save_report()
reward = LLaDAReward(tokenizer)

def sample_details(tokens):
    assert tokens.ndim==2 and tokens.shape[1]==32
    assert not np.any(tokens==core.mask_id)
    scores=np.asarray(reward(tokens))
    assert np.isfinite(scores).all()
    return {'texts':tokenizer.batch_decode(tokens,skip_special_tokens=True),'scores':scores.tolist()}

def unguided_and_bon():
    core.S=4
    tokens=run_standard_generation(core)
    details=sample_details(tokens)
    grouped=np.asarray(details['scores']).reshape(2,2)
    selected=np.argmax(grouped,axis=1)+np.arange(2)*2
    details['best_of_2']=sample_details(tokens[selected])
    (OUT/'unguided_and_bon.json').write_text(json.dumps(details,indent=2))
    return {'unguided_candidates':4,'best_of_2_winners':2,'steps':core.T}

def rs():
    core.S=2
    tokens, counts, total=run_rejection_sampling(core,vc,temperature=0.5,max_retries=12)
    details=sample_details(tokens)
    details['num_proposals']=counts
    (OUT/'rejection_sampling.json').write_text(json.dumps(details,indent=2))
    return {'samples':len(tokens),'total_proposals':int(total)}

def smc():
    core.S=2
    folder=OUT/'smc';folder.mkdir(exist_ok=True)
    run_one(core,vc,tokenizer,reward,2,2,2,str(folder),0.5)
    data=json.loads((folder/'smc_results_temp_0.5_K_2_ess_2.json').read_text())
    assert len(data['samples'])==4 and {s['run_idx'] for s in data['samples']}=={0,1}
    assert all(np.isfinite(s['log_weight']) for s in data['samples'])
    return {'particles':2,'independent_runs':2,'samples':4}

def collection():
    core.S=2
    result=SequentialCollector(core,reward,base_dir=str(OUT/'collection'),batch_size=1).collect()
    states=np.load(result['states']); rewards=np.load(result['rewards'])
    assert states.shape==(33,2,32) and rewards.shape==(2,)
    assert np.allclose(reward(states[0]),rewards)
    return {'states_shape':list(states.shape),'run_dir':result['run_dir']}

def trajectory_trace():
    core.S=2
    texts,traces=collect_trajectories_via_api(lcb,core,num_samples=2)
    assert len(texts)==len(traces)==2
    assert all(len(x['t'])==31 and np.isfinite(x['v']).all() and np.isfinite(x['b']).all() for x in traces)
    (OUT/'trajectory_trace.json').write_text(json.dumps({'texts':texts,'logs':traces},default=lambda x:np.asarray(x).tolist()))
    return {'samples':2,'trace_points_each':31,'scope':'execution only; timestep semantics not assessed'}

check('Unguided generation and BoN selection',unguided_and_bon)
check('Rejection sampling API',rs)
check('SMC sweep API and output serialization',smc)
check('Trajectory diagnostic execution',trajectory_trace)
check('Trajectory collection and reward serialization',collection)

# Small slices of the actual saved training data, retaining all 33 timesteps.
small=OUT/'training_inputs';small.mkdir(exist_ok=True)
states=np.array(np.load(next(RUN.glob('*_states.npy')),mmap_mode='r')[:,:2])
rewards=np.array(np.load(next(RUN.glob('*_rewards.npy')),mmap_mode='r')[:2])
np.save(small/'smoke_states.npy',states);np.save(small/'smoke_rewards.npy',rewards)

def critic_training():
    dataset=ClassificationDataset(str(small/'smoke_states.npy'),str(small/'smoke_rewards.npy'),vc.index_set,vc.model_name)
    vc.loader=DataLoader(Subset(dataset,[0,32,64,65]),batch_size=4)
    for group in vc.optimizer.param_groups: group['lr']=0.0
    before=vc.net.head.weight.detach().clone()
    vc.fit(epochs=1)
    assert torch.isfinite(vc.net.head.weight.grad).all()
    assert torch.equal(before,vc.net.head.weight)
    return {'batches':1,'examples':4,'lr':0,'gradient_norm':float(vc.net.head.weight.grad.norm())}

def offline_training():
    pre=RUN/'lcb_pretrain_data'/'20260205_223939_643203'
    offline=OUT/'offline_inputs';offline.mkdir(exist_ok=True)
    np.save(offline/'smoke_states.npy',np.array(np.load(next(pre.glob('*_states.npy')),mmap_mode='r')[:,:2]))
    targets=offline/'targets.npy'
    np.save(targets,np.array(np.load(pre/'lcb_targets_temp_0p5.npy',mmap_mode='r')[:,:2]))
    dataset=LCBOfflineDataset(str(offline),lcb.index_set,str(targets))
    lcb.offline_dataset=Subset(dataset,[0,32,64,65])
    for group in lcb.net_opt.param_groups: group['lr']=0.0
    before=lcb.net.head.weight.detach().clone()
    lcb.regression_fit(epochs=1,batch_size=4)
    assert torch.isfinite(lcb.net.head.weight.grad).all()
    assert torch.equal(before,lcb.net.head.weight)
    return {'batches':1,'examples':4,'lr':0,'gradient_norm':float(lcb.net.head.weight.grad.norm())}

def post_training():
    core.S=2;core.reset()
    source=core.X.copy();successors=core.propose()
    for group in lcb.net_opt.param_groups: group['lr']=0.0
    lcb.lambda_mode='search';lcb.lam_max=6;lcb.delta=0.5;lcb.symmetric_objective=True
    before=lcb.net.head.weight.detach().clone()
    lcb.train_on_step(successors,source,core.t)
    lcb.calibrate_tau(successors,source,core.t)
    assert torch.isfinite(lcb.net.head.weight.grad).all()
    assert torch.equal(before,lcb.net.head.weight)
    assert np.isfinite(lcb.taus[core.t])
    return {'timesteps':1,'examples':2,'lr':0,'tau':float(lcb.taus[core.t]),'lambda':float(lcb.lambdas[core.t-1])}

def checkpoint_roundtrip():
    vc.net.eval();lcb.net.eval()
    critic_expected=vc.predict_under_log(states[16],16)
    lcb_expected=lcb.predict_under_log(states[16],16)
    with tempfile.TemporaryDirectory(prefix='syntax_gpu_roundtrip_') as folder:
        vc.save(folder);lcb.save(folder,'smoke_lcb')
        new_vc=ValueClassifier.load(folder,device='cuda');new_vc.reward_temperature=0.5
        new_lcb=LCB.load(folder,None,new_vc,core,'smoke_lcb')
        assert np.allclose(new_vc.predict_under_log(states[16],16),critic_expected)
        assert np.allclose(new_lcb.predict_under_log(states[16],16),lcb_expected)
        del new_vc,new_lcb;gc.collect();torch.cuda.empty_cache()
    return {'critic':'same predictions','lcb':'same predictions','saved_to':'temporary runtime directory only'}

check('Critic training forward/backward at zero learning rate',critic_training)
check('Offline LCB training forward/backward at zero learning rate',offline_training)
check('LCB post-training step and tau calibration at zero learning rate',post_training)
check('GPU checkpoint save/load roundtrip',checkpoint_roundtrip)
report['peak_gpu_memory_gb']=torch.cuda.max_memory_allocated()/1e9
report['code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
save_report()
print('\nGPU SMOKE RESULTS',flush=True)
for item in report['checks']:print(item['status'].upper(),item['name'],flush=True)
print('Report:',OUT/'gpu_smoke_report.json',flush=True)
sys.exit(0 if all(x['status']=='passed' for x in report['checks']) else 1)
