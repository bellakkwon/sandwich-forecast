"""Frozen validation-only baselines. Requires pandas/numpy. Never evaluates final 12 weeks.
Usage: python 04_validation_baselines.py [--output-dir PATH]
Refuses to overwrite an existing output directory to preserve frozen evaluation artifacts.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=P/'validation_baselines');args=ap.parse_args();O=args.output_dir
if O.exists():raise SystemExit('Output already exists; use a NEW --output-dir to preserve frozen results.')
# Parse dates only to filter. Sales after 2012-12-30 are discarded before any target computation.
raw=P/'raw/WEEKLY_SALES_10STORES.csv';target=P/'selection/selected_targets.csv'
s=pd.read_csv(raw);s['week']=pd.to_datetime(s.Date,format='%m/%d/%Y');s=s[s.week.le('2012-12-30')].copy()
z=pd.read_csv(target);z=z.sort_values(['Store_num','INV_NUMBER']).reset_index(drop=True)
assert len(z)==172 and not z.duplicated(['Store_num','INV_NUMBER']).any()
z['volume_band']=pd.cut(z.weekly_average,bins=[20,40,80,np.inf],right=False,labels=['20_to_lt40','40_to_lt80','80_plus']).astype(str)
assert not z.volume_band.eq('nan').any()
weeks=pd.date_range('2012-04-01','2012-12-30',freq='7D');assert len(weeks)==40
s=s.merge(z[['Store_num','INV_NUMBER']],on=['Store_num','INV_NUMBER'],validate='many_to_one')
assert not s.duplicated(['Store_num','INV_NUMBER','week']).any()
parts=[]
for _,r in z.iterrows():
 a=s[(s.Store_num==r.Store_num)&(s.INV_NUMBER==r.INV_NUMBER)].set_index('week').reindex(weeks)
 x=pd.to_numeric(a.Sold,errors='raise').where(lambda v:v.ge(0))
 p=pd.DataFrame({'week':weeks,'actual':x.to_numpy(),'last_week':x.shift(1).to_numpy(),'mean_4weeks':x.shift(1).rolling(4,min_periods=4).mean().to_numpy()})
 p['actual_status']=np.where(a.Sold.isna(),'missing_row',np.where(a.Sold.lt(0),'negative_sold','valid'))
 p['Store_num']=int(r.Store_num);p['INV_NUMBER']=int(r.INV_NUMBER);p['category']=r.category;p['volume_band']=r.volume_band
 p['training_weekly_average']=r.weekly_average
 # Independent first validation forecast checks (weeks 23..26; no week27 actual used).
 assert p.iloc[26].last_week==x.iloc[25]
 assert np.isclose(p.iloc[26].mean_4weeks,x.iloc[22:26].mean()) and x.iloc[22:26].notna().all()
 parts.append(p.iloc[26:].copy())
v=pd.concat(parts,ignore_index=True).sort_values(['Store_num','INV_NUMBER','week']).reset_index(drop=True)
v['common_eligible']=v[['actual','last_week','mean_4weeks']].notna().all(axis=1)
for col in ['actual','last_week','mean_4weeks']:v['missing_'+col]=v[col].isna()
c=v[v.common_eligible].copy();keys=['Store_num','INV_NUMBER','week']
assert len(v)==2408 and v.week.min()==pd.Timestamp('2012-09-30') and v.week.max()==pd.Timestamp('2012-12-30')
assert not c.duplicated(keys).any() and c.actual.gt(0).all()
rows=[]
for dimension,groups in [('overall',[('all',v)]),('category',list(v.groupby('category'))),('volume_band',[(b,v[v.volume_band==b]) for b in ['20_to_lt40','40_to_lt80','80_plus']])]:
 for group,g in groups:
  e=g[g.common_eligible];n=len(e);den=e.actual.sum() if n else 0
  for model in ['last_week','mean_4weeks']:
   err=e[model]-e.actual;ae=err.abs().sum() if n else np.nan
   rows.append({'dimension':dimension,'group':group,'model':model,'scheduled_n':len(g),'common_n':n,'coverage':n/len(g) if len(g) else np.nan,'actual_sum':den if n else np.nan,'absolute_error_sum':ae,'MAE':ae/n if n else np.nan,'WAPE':ae/den if n and den!=0 else np.nan,'Bias':err.sum()/den if n and den!=0 else np.nan})
m=pd.DataFrame(rows)
# Meaningful missing-propagation/lag checks, independent from real data.
x=pd.Series([1.,2.,np.nan,4.,5.,6.,7.,8.]);q=x.shift(1).rolling(4,min_periods=4).mean()
assert pd.isna(q.iloc[6]) and q.iloc[7]==5.5
alter=x.copy();alter.iloc[7]=99999
assert alter.shift(1).rolling(4,min_periods=4).mean().iloc[7]==q.iloc[7]
O.mkdir(parents=True)
v.to_csv(O/'validation_predictions_all.csv',index=False,encoding='utf-8-sig',na_rep='NA')
c[keys].to_csv(O/'common_evaluation_keys.csv',index=False)
c.to_csv(O/'common_evaluation_predictions.csv',index=False,encoding='utf-8-sig',na_rep='NA')
v[~v.common_eligible].to_csv(O/'excluded_evaluation_rows.csv',index=False,encoding='utf-8-sig',na_rep='NA')
z.to_csv(O/'frozen_targets_and_bands.csv',index=False,encoding='utf-8-sig')
m.to_csv(O/'validation_metrics.csv',index=False,encoding='utf-8-sig',na_rep='NA')
weekly=v.groupby('week').agg(scheduled_n=('common_eligible','size'),common_n=('common_eligible','sum'));weekly['coverage']=weekly.common_n/weekly.scheduled_n;weekly.to_csv(O/'weekly_coverage.csv')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
manifest={'validation_start':'2012-09-30','validation_end':'2012-12-30','final_test_not_evaluated':True,'scheduled_n':len(v),'common_n':len(c),'excluded_n':len(v)-len(c),'coverage':len(c)/len(v),'forecast_rounding':False,'volume_bands':'[20,40), [40,80), [80,infinity); initial 26-week mean','source_sha256':sha(raw),'target_sha256':sha(target),'script_sha256':sha(Path(__file__)),'output_sha256':{f.name:sha(f) for f in O.glob('*.csv')}}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(m.to_string(index=False));print('Missing flags (overlapping):',v[['missing_actual','missing_last_week','missing_mean_4weeks']].sum().to_dict());print('Bands:',z.volume_band.value_counts().to_dict())
