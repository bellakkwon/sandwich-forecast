"""Post-hoc descriptive diagnostics only. No model training or tuning.
Cases: max/min(last-week absolute error - model absolute error) on final frozen keys.
Tie: earlier week, then ascending store and item code. Outputs never overwritten.
"""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
P=Path(__file__).resolve().parent;O=P/'diagnostics_bundle'
if O.exists():raise SystemExit('Preserve existing diagnostics; no overwrite.')
O.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
final=P/'final_evaluation';e=pd.read_csv(final/'final_common_predictions.csv',parse_dates=['week']);v=pd.read_csv(P/'model_validation/fixed_set_model_comparison.csv',parse_dates=['week']);v=v[v.candidate.eq('leaves_7')].copy();allf=pd.read_csv(final/'final_predictions_all.csv',parse_dates=['week']);z=pd.read_csv(P/'selection/selected_targets.csv');names=z.set_index('INV_NUMBER')['name'].to_dict();key=['Store_num','INV_NUMBER','week'];pair=key[:2]
e['baseline_ae']=(e.last_week-e.actual).abs();e['model_ae']=(e.prediction-e.actual).abs();e['model_minus_baseline_ae']=e.model_ae-e.baseline_ae;e['model_gain']=e.baseline_ae-e.model_ae
# 1 Weekly and pair contribution, positive means more model error.
w=e.groupby('week').agg(common_n=('actual','size'),actual_sum=('actual','sum'),baseline_ae=('baseline_ae','sum'),model_ae=('model_ae','sum'),model_minus_baseline_ae=('model_minus_baseline_ae','sum'));w['baseline_WAPE']=w.baseline_ae/w.actual_sum;w['model_WAPE']=w.model_ae/w.actual_sum;w.to_csv(O/'weekly_error_contribution.csv')
g=e.groupby(pair).agg(n=('actual','size'),actual_sum=('actual','sum'),baseline_ae=('baseline_ae','sum'),model_ae=('model_ae','sum'),model_minus_baseline_ae=('model_minus_baseline_ae','sum')).reset_index();g['name']=g.INV_NUMBER.map(names);g.sort_values(['model_minus_baseline_ae','Store_num','INV_NUMBER'],ascending=[False,True,True]).to_csv(O/'pair_error_contribution.csv',index=False)
# 2 Balanced panels based on common evaluation membership, descriptive only.
vc=v.groupby(pair).week.nunique();fc=e.groupby(pair).week.nunique();both=vc[vc.eq(14)].index.intersection(fc[fc.eq(12)].index);finalonly=fc[fc.eq(12)].index
pd.DataFrame(list(both),columns=pair).to_csv(O/'balanced_26week_pairs.csv',index=False);pd.DataFrame(list(finalonly),columns=pair).to_csv(O/'balanced_final12_pairs.csv',index=False)
frames=[]
for period,a in [('validation',v),('final',e)]:
 b=a.merge(pd.DataFrame(list(both),columns=pair),on=pair,validate='many_to_one');q=b.groupby('week')[['actual','last_week','mean_4weeks','prediction']].sum().reset_index();q['period']=period;q['pairs']=len(both);frames.append(q)
trend=pd.concat(frames).sort_values('week');trend.to_csv(O/'balanced_26week_trend.csv',index=False)
ps=[]
for period,a in [('validation',v),('final',e)]:
 b=a.merge(pd.DataFrame(list(both),columns=pair),on=pair,validate='many_to_one');den=b.actual.sum()
 for method in ['last_week','mean_4weeks','prediction']:
  err=b[method]-b.actual;ps.append({'period':period,'method':method,'pairs':len(both),'n':len(b),'mean_actual':b.actual.mean(),'WAPE':err.abs().sum()/den,'Bias':err.sum()/den})
pd.DataFrame(ps).to_csv(O/'balanced_period_metrics.csv',index=False)
# 3 Missing attribution from raw records and calendar lag lookups, no imputation.
raw=pd.read_csv(P/'raw/WEEKLY_SALES_10STORES.csv');raw['week']=pd.to_datetime(raw.Date,format='%m/%d/%Y');raw=raw[raw.Store_num.ne(24)];assert not raw.duplicated(key).any();lookup=raw.set_index(key).Sold
missing=[]
for r in allf.itertuples():
 if r.common_eligible:continue
 causes=[]
 for lag in range(5):
  dt=r.week-pd.Timedelta(weeks=lag);val=lookup.get((r.Store_num,r.INV_NUMBER,dt),np.nan)
  if pd.isna(val) or val<0:
   cause='missing_row' if pd.isna(val) else 'negative_sold';causes.append(f'lag{lag}:{cause}')
   missing.append({'Store_num':r.Store_num,'INV_NUMBER':r.INV_NUMBER,'evaluation_week':r.week,'source_week':dt,'lag':lag,'cause':cause,'affects_actual':lag==0,'affects_last_week':lag==1,'affects_mean4':lag in [1,2,3,4]})
 assert causes
md=pd.DataFrame(missing);md.to_csv(O/'missing_source_links.csv',index=False);sources=md.drop_duplicates(['Store_num','INV_NUMBER','source_week']);sources.to_csv(O/'missing_unique_source_observations.csv',index=False);md.groupby(['lag','cause']).size().rename('links').to_csv(O/'missing_reason_counts.csv')
flags=pd.read_csv(final/'pair_review_flags.csv');missingpairs=flags[(flags.method=='prediction')&(flags.status=='insufficient_data')][pair+['n']];missingpairs['name']=missingpairs.INV_NUMBER.map(names);missingpairs.to_csv(O/'insufficient_data_pairs.csv',index=False)
# Exact deterministic SHAP case selection.
ordered=e.sort_values(['week','Store_num','INV_NUMBER'],kind='stable')
best=ordered.sort_values('model_gain',ascending=False,kind='stable').iloc[0];worst=ordered.sort_values('model_gain',ascending=True,kind='stable').iloc[0];cases=[]
for label,r in [('largest_model_gain',best),('largest_model_loss',worst)]:
 tag=r.week.strftime('%Y-%m-%d');sh=pd.read_csv(final/'shap'/f'{tag}.csv',parse_dates=['week']);ss=sh[(sh.Store_num==r.Store_num)&(sh.INV_NUMBER==r.INV_NUMBER)].iloc[0];inp=pd.read_csv(final/'inputs'/f'{tag}.csv');ii=inp[(inp.Store_num==r.Store_num)&(inp.INV_NUMBER==r.INV_NUMBER)].iloc[0];features=json.loads((P/'05_model_spec.json').read_text())['features'];detail=[]
 for feature in features:detail.append({'feature':feature,'input_value':ii[feature],'shap_contribution':ss['shap_'+feature]})
 detail=pd.DataFrame(detail);detail['abs_contribution']=detail.shap_contribution.abs();detail=detail.sort_values('abs_contribution',ascending=False);detail.to_csv(O/f'{label}_shap.csv',index=False)
 expected=float(ss.expected_value);assert np.isclose(detail.shap_contribution.sum()+expected,r.prediction_raw,atol=1e-8)
 cases.append({'case':label,'Store_num':int(r.Store_num),'INV_NUMBER':int(r.INV_NUMBER),'name':names[r.INV_NUMBER],'week':tag,'actual':float(r.actual),'last_week':float(r.last_week),'model_raw':float(r.prediction_raw),'model_final':float(r.prediction),'baseline_ae':float(r.baseline_ae),'model_ae':float(r.model_ae),'model_gain':float(r.model_gain),'shap_expected_value':expected,'top_features':detail.head(4).to_dict(orient='records')})
 (O/f'{label}_input.json').write_text(ii.to_json(force_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame([{k:v for k,v in c.items() if k!='top_features'} for c in cases]).to_csv(O/'selected_cases.csv',index=False)
# Shareable figures. English labels keep rendering portable.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11.5,4.8));colors=np.where(w.model_minus_baseline_ae>=0,'#bc5750','#297b7a');ax.bar(w.index,w.model_minus_baseline_ae,width=4,color=colors);ax.axhline(0,color='#334155',lw=.8);ax.set(title='Final 12 weeks: model error minus last-week error',ylabel='Difference in absolute-error sums');ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'));ax.set_xticks(w.index);fig.text(.07,.01,'Positive = model worse. Same frozen evaluation keys within each week.',fontsize=10);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(O/'weekly_error_contribution.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(12,5.2));mapping=[('actual','Actual','#1d2939'),('last_week','Last week','#a0a8b5'),('prediction','LightGBM','#267c9e')]
for field,label,color in mapping:ax.plot(trend.week,trend[field],label=label,color=color,lw=2 if field=='actual' else 1.5)
ax.axvline(pd.Timestamp('2013-01-02'),color='#b56e32',ls='--');ax.set(title=f'Same {len(both)} pairs across all 26 evaluation weeks',ylabel='Sum of recorded POS units');ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU,interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'));ax.legend(frameon=False,ncol=3);fig.text(.07,.01,'Descriptive balanced panel selected after observation; not used for model selection. Dashed line = final period begins.',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(O/'balanced_26week_trend.png',dpi=180);plt.close(fig)
for c in cases:
 d=pd.read_csv(O/f"{c['case']}_shap.csv").sort_values('shap_contribution');fig,ax=plt.subplots(figsize=(9,5.8));ax.barh(d.feature,d.shap_contribution,color=np.where(d.shap_contribution>=0,'#bc5750','#347b9f'));ax.axvline(0,color='#334155',lw=.8);ax.set(title=f"{c['case'].replace('_',' ')} | store {c['Store_num']} / item {c['INV_NUMBER']} / {c['week']}",xlabel='Contribution to raw predicted POS units');fig.text(.06,.01,f"Base {c['shap_expected_value']:.2f} + contributions = raw {c['model_raw']:.2f}; actual {c['actual']:.0f}. Post-hoc selected case; not causal.",fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(O/f"{c['case']}_shap.png",dpi=180);plt.close(fig)
summary={'post_hoc_descriptive_only':True,'case_rule':'max/min baseline_AE-model_AE on frozen final common keys; tie week/store/code ascending','final_absolute_error_difference':float(w.model_minus_baseline_ae.sum()),'model_winning_weeks':int((w.model_minus_baseline_ae<0).sum()),'balanced_final12_pairs':len(finalonly),'balanced_both_periods_pairs':len(both),'excluded_evaluation_rows':int((~allf.common_eligible).sum()),'missing_source_links':len(md),'unique_missing_source_observations':len(sources),'missing_reason_counts':sources.cause.value_counts().to_dict(),'insufficient_data_pairs':len(missingpairs),'cases':cases}
(O/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
assert np.isclose(w.model_minus_baseline_ae.sum(),e.model_ae.sum()-e.baseline_ae.sum())
assert len(trend)==26 and trend.pairs.eq(len(both)).all()
# Original final outputs unchanged.
manifest=json.loads((final/'manifest.json').read_text())
for n,h in manifest['output_sha256'].items():assert sha(final/n)==h
print(json.dumps(summary,indent=2,ensure_ascii=False));print(pd.DataFrame(ps).to_string(index=False));print(w.to_string())
