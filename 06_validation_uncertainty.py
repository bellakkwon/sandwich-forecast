"""Exploratory week-level bootstrap; selected model held fixed, final test unused."""
from pathlib import Path
import pandas as pd,numpy as np,json,math
P=Path(__file__).resolve().parent/'model_validation'
e=pd.read_csv(P/'fixed_set_model_comparison.csv');e=e[e.candidate.eq('leaves_7')].copy()
e['model_ae']=(e.prediction-e.actual).abs();e['baseline_ae']=(e.last_week-e.actual).abs()
a=e.groupby('week',sort=True)[['actual','model_ae','baseline_ae']].sum().to_numpy();assert len(a)==14
idx=np.random.default_rng(0).integers(0,14,size=(10000,14));b=a[idx].sum(axis=1);d=(b[:,1]-b[:,2])/b[:,0]
g=e.groupby(['Store_num','INV_NUMBER'])[['model_ae','baseline_ae']].sum()
pd.DataFrame({'replicate':range(1,10001),'model_minus_last_week_wape':d}).to_csv(P/'bootstrap_wape_difference.csv',index=False)
s={'bootstrap':'iid resampling of 14 whole-week aggregate triples with replacement; selected model fixed','replicates':10000,'rng':'numpy.random.default_rng(0)','percentile_quantile_method':'numpy linear default','ci_95_percentage_points':(np.quantile(d,[.025,.975])*100).tolist(),'strict_pair_wins':int(g.model_ae.lt(g.baseline_ae).sum()),'pair_denominator':len(g),'either_direction_8_of_12_iid_fair_sign_probability':2*sum(math.comb(12,k) for k in range(8,13))/2**12,'limitations':['Adjacent-week dependence not preserved','Candidate selection uncertainty not included','Pairs share calendar weeks and are not independent']}
(P/'uncertainty_summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8');print(json.dumps(s,indent=2))
