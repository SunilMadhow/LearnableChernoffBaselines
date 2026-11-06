"""Rebuild numerical summaries from the saved syntax-reward samples (CPU only)."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'data_runs/20260205_195651_634076'
OLD = ROOT / 'data_runs/20251224_233551_699407/results'
OUT = ROOT / 'figures/regenerated'
OUT.mkdir(parents=True, exist_ok=True)


def summarize(label, path):
    data = json.loads(path.read_text())
    samples = data['samples']
    scores = np.array([x['score'] for x in samples])
    nfes = np.array([x.get('num_proposals', x.get('proposals', 0)) for x in samples])
    assert len(samples) > 0, path
    assert np.all(nfes > 0), f'Missing NFE counts: {path}'
    p = np.mean(scores == 1)
    row = dict(method=label, n=len(samples), mean_reward=scores.mean(),
               reward_ci95=1.96*scores.std(ddof=1)/np.sqrt(len(scores)),
               mean_nfe=nfes.mean(), nfe_ci95=1.96*nfes.std(ddof=1)/np.sqrt(len(nfes)),
               effective_n=nfes.mean()/32, perfect_rate=p,
               perfect_ci95=1.96*np.sqrt(p*(1-p)/len(scores)),
               source=str(path.relative_to(ROOT)))
    return row


rows=[]
for a in [0.5, 0.3, 0.1]:
    rows.append(summarize(f'RS alpha={a}', OLD/f'rejection_sampling_results_temp_{a}.json'))
    tag=str(a).replace('.','p')
    rows.append(summarize(f'LCB alpha={a}', NEW/'inference_results'/f'lcb_lambda_only_SYM_post_trained_temp_{tag}_inference_results.json'))
rows.append(summarize('Bo2', OLD/'best_of_2_results.json'))
rows.append(summarize('Unguided', OLD/'unguided_results_1000.json'))
with (OUT/'main_results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
for row in rows:
    print(f"{row['method']:18s} n={row['n']:4d} reward={row['mean_reward']:.6f} NFE={row['mean_nfe']:.6f} effN={row['effective_n']:.6f} perfect={row['perfect_rate']:.3f}")

sweeps=[]
for a in [0.5,0.3,0.1]:
    for d in [0.9,0.7,0.5,0.3]:
        tag=f"lcb_lambda_only_SYM_post_trained_temp_{str(a).replace('.','p')}_del_{str(d).replace('.','p')}_inference_results"
        path=NEW/'inference_results/sweeps'/tag
        if not path.exists(): path=path.with_suffix('.json')
        row=summarize(f'LCB alpha={a}, delta={d}',path)
        assert row['n']==500, path
        sweeps.append(dict(alpha=a,delta=d,**row))
with (OUT/'sweep_results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=sweeps[0]); writer.writeheader(); writer.writerows(sweeps)
print(f'Validated {len(sweeps)} sweep results, 500 samples each.')

smc=[]
for path in sorted((NEW/'SMC').glob('smc_results_*.json')):
    data=json.loads(path.read_text()); cfg=data['config']
    assert len(data['samples'])==cfg['M']*cfg['num_particles'], path
    assert all('run_idx' in x for x in data['samples']), path
    smc.append(dict(K=cfg['num_particles'],ess=cfg['ess_threshold'],M=cfg['M']))
smc.sort(key=lambda r:(r['K'],r['ess']))
print(f'Validated {len(smc)} SMC configurations with particle run labels.')
print('Saved main_results.csv and sweep_results.csv to',OUT)
