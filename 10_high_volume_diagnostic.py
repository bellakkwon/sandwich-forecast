"""Post-hoc hypothesis check. No fitting, tuning or re-evaluation of models."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
P=Path(__file__).resolve().parent;O=P/'diagnostics_bundle'
e=pd.read_csv(P/'final_evaluation/final_common_predictions.csv',parse_dates=['week'])
# Diagnostic bins, introduced AFTER final results; distinct from frozen initial-training volume bands.
bins=[0,40,80,150,300,np.inf];labels=['0_to_lt40','40_to_lt80','80_to_lt150','150_to_lt300','300_plus']
e['lag1_band_posthoc']=pd.cut(e.last_week,bins=bins,right=False,labels=labels)
e['model_minus_lastweek_prediction']=e.prediction-e.last_week
rows=[]
for band,g in e.groupby('lag1_band_posthoc',observed=False):
 d=g.model_minus_lastweek_prediction;den=g.actual.sum()
 rows.append({'lag1_band_posthoc':str(band),'n':len(g),'unique_pairs':g[['Store_num','INV_NUMBER']].drop_duplicates().shape[0],'weeks':g.week.nunique(),'mean_lag1':g.last_week.mean(),'mean_actual':g.actual.mean(),'mean_model':g.prediction.mean(),'mean_model_minus_lastweek':d.mean(),'median_model_minus_lastweek':d.median(),'share_model_below_lastweek':d.lt(0).mean(),'model_absolute_error_sum':(g.prediction-g.actual).abs().sum(),'lastweek_absolute_error_sum':(g.last_week-g.actual).abs().sum(),'model_WAPE':(g.prediction-g.actual).abs().sum()/den if den else np.nan,'lastweek_WAPE':(g.last_week-g.actual).abs().sum()/den if den else np.nan,'model_Bias':(g.prediction-g.actual).sum()/den if den else np.nan,'lastweek_Bias':(g.last_week-g.actual).sum()/den if den else np.nan})
r=pd.DataFrame(rows);r.to_csv(O/'high_lag1_prediction_comparison.csv',index=False)
e[['Store_num','INV_NUMBER','week','actual','last_week','prediction','lag1_band_posthoc','model_minus_lastweek_prediction']].to_csv(O/'high_lag1_diagnostic_rows.csv',index=False)
# Reconstruct exact candidate training rows at forecast date 2013-02-24.
s=pd.read_csv(P/'raw/WEEKLY_SALES_10STORES.csv');s['week']=pd.to_datetime(s.Date,format='%m/%d/%Y');s=s[s.week.lt('2013-02-24')];z=pd.read_csv(P/'selection/selected_targets.csv');weeks=pd.date_range('2012-04-01','2013-02-17',freq='7D');parts=[]
for st,code in z[['Store_num','INV_NUMBER']].itertuples(index=False,name=None):
 a=s[(s.Store_num==st)&(s.INV_NUMBER==code)].set_index('week').reindex(weeks);y=a.Sold.where(a.Sold.ge(0));g=pd.DataFrame({'Store_num':st,'INV_NUMBER':code,'week':weeks,'actual':y.to_numpy(),'lag1':y.shift(1).to_numpy()}).iloc[4:];parts.append(g[g.actual.notna()])
t=pd.concat(parts);hi=t[t.lag1.ge(300)].copy();hi.to_csv(O/'training_high_lag1_before_2013-02-24.csv',index=False);mid=hi[hi.lag1.ge(320)&hi.lag1.le(400)]
pair=s[(s.Store_num==2)&(s.INV_NUMBER==1651)][['week','Sold']].sort_values('week');full=pd.read_csv(P/'raw/WEEKLY_SALES_10STORES.csv');full['week']=pd.to_datetime(full.Date,format='%m/%d/%Y');series=full[(full.Store_num==2)&(full.INV_NUMBER==1651)][['week','Sold']].sort_values('week');series.to_csv(O/'store2_item1651_full_history.csv',index=False)
summary={'diagnostic_only':True,'bins_posthoc_not_model_inputs_or_selection_rules':labels,'training_rows_before_2013_02_24_lag1_ge300':len(hi),'training_rows_lag1_320_to400_inclusive':len(mid),'of_these_store2_item1651':int(((mid.Store_num==2)&(mid.INV_NUMBER==1651)).sum())}
(O/'high_volume_hypothesis_summary.json').write_text(json.dumps(summary,indent=2));print(r.to_string(index=False));print(summary);print(series[series.week.between('2012-09-02','2012-10-14')].to_string(index=False));print(series[series.week.between('2013-01-27','2013-03-03')].to_string(index=False))
# Guard: original final predictions remain unchanged.
f=P/'final_evaluation';m=json.loads((f/'manifest.json').read_text())
for name,h in m['output_sha256'].items():assert hashlib.sha256((f/name).read_bytes()).hexdigest()==h
