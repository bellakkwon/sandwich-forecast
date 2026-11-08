"""Run 2 frozen LightGBM candidates on validation only. No final-test metrics.
Requires pandas,numpy,lightgbm==4.6.0. Existing output directory is never overwritten.
"""
from pathlib import Path
import json,hashlib,platform
import numpy as np
import pandas as pd
import lightgbm as lgb
P=Path(__file__).resolve().parent;O=P/'model_validation'
if O.exists():raise SystemExit('Output exists; preserve it before any intentional rerun.')
spec=json.loads((P/'05_model_spec.json').read_text());s=pd.read_csv(P/'raw/WEEKLY_SALES_10STORES.csv');s['week']=pd.to_datetime(s.Date,format='%m/%d/%Y');s=s[s.week.le('2012-12-30')]
z=pd.read_csv(P/'selection/selected_targets.csv');keys=pd.read_csv(P/'validation_baselines/common_evaluation_keys.csv',parse_dates=['week']);base=pd.read_csv(P/'validation_baselines/common_evaluation_predictions.csv',parse_dates=['week'])
weeks=pd.date_range('2012-04-01','2012-12-30',freq='7D');parts=[]
for r in z.itertuples():
 a=s[(s.Store_num==r.Store_num)&(s.INV_NUMBER==r.INV_NUMBER)].set_index('week').reindex(weeks)
 y=a.Sold.where(a.Sold.ge(0));prev=y.shift(1)
 f=pd.DataFrame({'week':weeks,'target':y.to_numpy(),'Store_num':r.Store_num,'INV_NUMBER':r.INV_NUMBER})
 for k in range(1,5):f[f'lag{k}']=y.shift(k).to_numpy()
 f['mean4']=prev.rolling(4,min_periods=4).mean().to_numpy();f['std4_ddof0']=prev.rolling(4,min_periods=4).std(ddof=0).to_numpy();f['diff1_2']=(y.shift(1)-y.shift(2)).to_numpy();f['lag_price']=a.Price.shift(1).to_numpy()
 f['store']=str(r.Store_num);f['code']=str(r.INV_NUMBER);f['type']=r.category;parts.append(f.iloc[4:])
f=pd.concat(parts,ignore_index=True)
for c in spec['categorical']:f[c]=pd.Categorical(f[c],categories=sorted(f[c].unique()))
features=spec['features'];params={k:spec[k] for k in ['objective','learning_rate','seed','num_threads','deterministic','force_col_wise','min_data_in_leaf','max_depth','max_bin','feature_fraction','bagging_fraction','bagging_freq','lambda_l1','lambda_l2']};params['verbosity']=-1
O.mkdir();(O/'models').mkdir();preds=[];logs=[]
for leaves in spec['num_leaves_candidates']:
 for week in weeks[26:]:
  tr=f[f.week.lt(week)&f.target.notna()];te=f[f.week.eq(week)]
  assert tr.week.max()<week and len(te)==172
  model=lgb.train(dict(params,num_leaves=leaves),lgb.Dataset(tr[features],label=tr.target,categorical_feature=spec['categorical']),num_boost_round=spec['num_boost_round'])
  raw=model.predict(te[features]);out=te[['Store_num','INV_NUMBER','week']].copy();out['candidate']=f'leaves_{leaves}';out['prediction_raw']=raw;out['prediction']=np.maximum(raw,0);preds.append(out)
  logs.append({'candidate':f'leaves_{leaves}','forecast_week':str(week.date()),'training_rows':len(tr),'training_max_week':str(tr.week.max().date()),'predicted_n':len(out),'negative_raw_n':int((raw<0).sum())})
  if week==weeks[-1]:model.save_model(str(O/'models'/f'leaves_{leaves}_last_validation.txt'))
 print('Completed',leaves,flush=True)
pred=pd.concat(preds,ignore_index=True);metrics=[];comp=[]
for name,g in pred.groupby('candidate'):
 e=base.merge(g,on=['Store_num','INV_NUMBER','week'],how='left',validate='one_to_one');assert len(e)==len(keys)==2348
 ok=np.isfinite(e.prediction);assert ok.all(),'Model missing predictions: cannot use full-set selection'
 metrics.append({'candidate':name,'n':len(e),'WAPE':(e.prediction-e.actual).abs().sum()/e.actual.sum(),'MAE':(e.prediction-e.actual).abs().mean(),'Bias':(e.prediction-e.actual).sum()/e.actual.sum(),'missing_n':int((~ok).sum())});comp.append(e)
m=pd.DataFrame(metrics);best=min(metrics,key=lambda r:(round(r['WAPE'],12),int(r['candidate'].split('_')[1])))['candidate']
e=pd.concat(comp,ignore_index=True);e.to_csv(O/'fixed_set_model_comparison.csv',index=False,encoding='utf-8-sig');pred.to_csv(O/'all_validation_model_predictions.csv',index=False);pd.DataFrame(logs).to_csv(O/'training_log.csv',index=False);m.to_csv(O/'candidate_metrics.csv',index=False)
# Diagnose only after both candidate training runs. All 4 methods are on same keys.
a=base.merge(pred[pred.candidate.eq(best)],on=['Store_num','INV_NUMBER','week'],validate='one_to_one');methods=['last_week','mean_4weeks','prediction'];rows=[]
for dim,groups in [('overall',[('all',a)]),('week',a.groupby('week')),('category',a.groupby('category')),('volume_band',a.groupby('volume_band'))]:
 for group,g in groups:
  for method in methods:
   err=g[method]-g.actual;rows.append({'dimension':dim,'group':str(group),'method':best if method=='prediction' else method,'n':len(g),'actual_sum':g.actual.sum(),'absolute_error_sum':err.abs().sum(),'MAE':err.abs().mean(),'WAPE':err.abs().sum()/g.actual.sum(),'Bias':err.sum()/g.actual.sum()})
pd.DataFrame(rows).to_csv(O/'three_method_diagnostics.csv',index=False,encoding='utf-8-sig')
# User's October observation checked on frozen evaluation set.
w=base.assign(diff=(base.mean_4weeks-base.actual).abs()-(base.last_week-base.actual).abs()).groupby('week')['diff'].sum();october=w.loc['2012-10-07':'2012-10-28'].sum()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'selected_candidate':best,'lightgbm_version':lgb.__version__,'python':platform.python_version(),'final_test_evaluated':False,'spec_sha256':sha(P/'05_model_spec.json'),'frozen_key_sha256':sha(P/'validation_baselines/common_evaluation_keys.csv'),'script_sha256':sha(Path(__file__)),'october_4week_error_difference':float(october),'remaining_error_difference':float(w.sum()-october),'outputs_sha256':{p.name:sha(p) for p in O.glob('*.csv')}}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2));print(m.to_string(index=False));print(json.dumps(manifest,indent=2))
