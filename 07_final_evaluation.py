"""One frozen final evaluation. Never overwrites outputs. Requires lightgbm4.6.0,pandas,numpy."""
from pathlib import Path
import json,hashlib,datetime,platform
import pandas as pd,numpy as np
import lightgbm as lgb
P=Path(__file__).resolve().parent;O=P/'final_evaluation'
if O.exists():raise SystemExit('Output exists. Preserve original. Corrections require documented new run, never overwrite.')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=json.loads((P/'05_model_spec.json').read_text());rules=json.loads((P/'07_final_rules.json').read_text());val=json.loads((P/'model_validation/manifest.json').read_text());assert val['selected_candidate']=='leaves_7'
# Write pre-run frozen rules, code hashes and empty correction register BEFORE any final metrics.
O.mkdir();(O/'models').mkdir();(O/'inputs').mkdir();(O/'shap').mkdir()
pre={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rules':rules,'model_spec':spec,'script_sha256':sha(Path(__file__)),'rules_sha256':sha(P/'07_final_rules.json'),'spec_sha256':sha(P/'05_model_spec.json'),'target_sha256':sha(P/'selection/selected_targets.csv')}
(O/'prerun_freeze.json').write_text(json.dumps(pre,indent=2,ensure_ascii=False));(O/'CORRECTIONS.md').write_text('# Correction register\nNo implementation corrections or reruns at initialization. Preserve this run before any correction.\n')
s=pd.read_csv(P/'raw/WEEKLY_SALES_10STORES.csv');s['week']=pd.to_datetime(s.Date,format='%m/%d/%Y');z=pd.read_csv(P/'selection/selected_targets.csv');z['volume_band']=pd.cut(z.weekly_average,[20,40,80,np.inf],right=False,labels=['20_to_lt40','40_to_lt80','80_plus']).astype(str)
weeks=pd.date_range('2012-04-01','2013-03-24',freq='7D');parts=[]
for r in z.itertuples():
 a=s[(s.Store_num==r.Store_num)&(s.INV_NUMBER==r.INV_NUMBER)].set_index('week').reindex(weeks);y=a.Sold.where(a.Sold.ge(0));prev=y.shift(1)
 f=pd.DataFrame({'week':weeks,'target':y.to_numpy(),'Store_num':r.Store_num,'INV_NUMBER':r.INV_NUMBER,'category':r.category,'volume_band':r.volume_band})
 for k in range(1,5):f[f'lag{k}']=y.shift(k).to_numpy()
 f['mean4']=prev.rolling(4,min_periods=4).mean().to_numpy();f['std4_ddof0']=prev.rolling(4,min_periods=4).std(ddof=0).to_numpy();f['diff1_2']=(y.shift(1)-y.shift(2)).to_numpy();f['lag_price']=a.Price.shift(1).to_numpy();f['store']=str(r.Store_num);f['code']=str(r.INV_NUMBER);f['type']=r.category
 assert np.isclose(f.iloc[40].mean4,y.iloc[36:40].mean()) if y.iloc[36:40].notna().all() else pd.isna(f.iloc[40].mean4)
 assert list(weeks[36:40].strftime('%Y-%m-%d'))==['2012-12-09','2012-12-16','2012-12-23','2012-12-30']
 parts.append(f.iloc[4:])
f=pd.concat(parts,ignore_index=True)
catmap={}
for c in spec['categorical']:
 cats=sorted(f[c].unique());catmap[c]=cats;f[c]=pd.Categorical(f[c],categories=cats)
(O/'categorical_levels.json').write_text(json.dumps(catmap,indent=2))
key=['Store_num','INV_NUMBER','week'];v=f[f.week.ge(rules['test_start'])].copy();v['actual']=v.target;v['last_week']=v.lag1;v['mean_4weeks']=v.mean4;v['common_eligible']=v[['actual','last_week','mean_4weeks']].notna().all(axis=1)
assert len(v)==172*12 and v.week.nunique()==12
for c in ['actual','last_week','mean_4weeks']:v['missing_'+c]=v[c].isna()
cols=key+['category','volume_band','actual','last_week','mean_4weeks','common_eligible','missing_actual','missing_last_week','missing_mean_4weeks']
bas=v[cols].sort_values(key);common=bas[bas.common_eligible].copy();assert common.actual.gt(0).all() and not common.duplicated(key).any()
common[key].to_csv(O/'final_common_evaluation_keys.csv',index=False);bas.to_csv(O/'final_baselines_all.csv',index=False,na_rep='NA');common.to_csv(O/'final_common_baselines.csv',index=False);bas[~bas.common_eligible].to_csv(O/'final_excluded_rows.csv',index=False,na_rep='NA')
keyhash=sha(O/'final_common_evaluation_keys.csv');freeze={'saved_before_model_training':True,'key_sha256':keyhash,'saved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'common_n':len(common),'scheduled_n':len(bas)};(O/'key_freeze.json').write_text(json.dumps(freeze,indent=2))
params={k:spec[k] for k in ['objective','learning_rate','seed','num_threads','deterministic','force_col_wise','min_data_in_leaf','max_depth','max_bin','feature_fraction','bagging_fraction','bagging_freq','lambda_l1','lambda_l2']};params.update(num_leaves=7,verbosity=-1);features=spec['features'];preds=[];logs=[];shap_parts=[]
for week in weeks[40:]:
 tr=f[f.week.lt(week)&f.target.notna()];te=f[f.week.eq(week)].sort_values(['Store_num','INV_NUMBER']);assert len(te)==172 and tr.week.max()<week
 tag=week.strftime('%Y-%m-%d');te[key+features].to_csv(O/'inputs'/f'{tag}.csv',index=False,na_rep='NA')
 model=lgb.train(params,lgb.Dataset(tr[features],label=tr.target,categorical_feature=spec['categorical']),num_boost_round=300);model.save_model(str(O/'models'/f'{tag}.txt'))
 raw=model.predict(te[features]);contrib=model.predict(te[features],pred_contrib=True);assert np.allclose(contrib.sum(axis=1),raw,rtol=1e-8,atol=1e-8)
 pp=te[key].copy();pp['prediction_raw']=raw;pp['prediction']=np.maximum(raw,0);preds.append(pp)
 sp=te[key].copy()
 for j,c in enumerate(features):sp['shap_'+c]=contrib[:,j]
 sp['expected_value']=contrib[:,-1];sp['prediction_raw']=raw;sp.to_csv(O/'shap'/f'{tag}.csv',index=False);shap_parts.append(sp)
 logs.append({'week':tag,'train_n':len(tr),'train_max_week':str(tr.week.max().date()),'predict_n':len(te),'negative_raw_n':int((raw<0).sum()),'max_shap_additivity_error':float(np.max(np.abs(contrib.sum(axis=1)-raw)))})
 print('Finished',tag,flush=True)
pred=pd.concat(preds,ignore_index=True);allrows=bas.merge(pred,on=key,validate='one_to_one');e=common.merge(pred,on=key,how='left',validate='one_to_one');missing=int((~np.isfinite(e.prediction)).sum());assert missing==0
methods=['last_week','mean_4weeks','prediction']
def calc(g,m):
 n=len(g);den=g.actual.sum();err=g[m]-g.actual;return {'n':n,'actual_sum':den if n else np.nan,'absolute_error_sum':err.abs().sum() if n else np.nan,'MAE':err.abs().mean() if n else np.nan,'WAPE':err.abs().sum()/den if n and den else np.nan,'Bias':err.sum()/den if n and den else np.nan}
metrics=[]
for dim,groups in [('overall',[('all',e)]),('week',e.groupby('week')),('category',e.groupby('category')),('volume_band',[(x,e[e.volume_band.eq(x)]) for x in ['20_to_lt40','40_to_lt80','80_plus']])]:
 for group,g in groups:
  for m in methods:metrics.append({'dimension':dim,'group':str(group),'method':m,**calc(g,m)})
m=pd.DataFrame(metrics);flags=[]
for st,code in z[['Store_num','INV_NUMBER']].itertuples(index=False,name=None):
 g=e[(e.Store_num==st)&(e.INV_NUMBER==code)]
 for method in methods:
  vals=calc(g,method);over=int((g[method]>g.actual).sum());under=int((g[method]<g.actual).sum());status='insufficient_data' if len(g)<10 else ('overprediction_review' if over>=8 and vals['Bias']>=.1 else ('underprediction_review' if under>=8 and vals['Bias']<=-.1 else 'not_flagged'))
  flags.append({'Store_num':st,'INV_NUMBER':code,'method':method,'over_weeks':over,'under_weeks':under,'status':status,**vals})
fl=pd.DataFrame(flags);overview=m[m.dimension.eq('overall')];bw=float(overview[overview.method.eq('last_week')].WAPE.iloc[0]);mw=float(overview[overview.method.eq('prediction')].WAPE.iloc[0]);decision='select_model' if mw<bw else 'retain_last_week'
allrows.to_csv(O/'final_predictions_all.csv',index=False,na_rep='NA');e.to_csv(O/'final_common_predictions.csv',index=False);m.to_csv(O/'final_metrics.csv',index=False,na_rep='NA');fl.to_csv(O/'pair_review_flags.csv',index=False,na_rep='NA');pd.DataFrame(logs).to_csv(O/'training_log.csv',index=False)
fl.groupby(['method','status']).size().unstack(fill_value=0).to_csv(O/'review_counts.csv')
# Last4week reporting uses the exact predictions already produced, not a new forecast run.
r=e[e.week.ge('2013-03-03')];r.to_csv(O/'last4weeks_predictions.csv',index=False);pd.DataFrame([{'method':x,**calc(r,x)} for x in methods]).to_csv(O/'last4weeks_metrics.csv',index=False)
sh=pd.concat(shap_parts,ignore_index=True);sh.to_csv(O/'shap_all_predictions.csv',index=False);shapcols=['shap_'+x for x in features];shkey=sh.merge(common[key],on=key,validate='one_to_one');shkey[shapcols].abs().mean().sort_values(ascending=False).rename('mean_abs_shap').to_csv(O/'shap_importance_common.csv')
assert sha(O/'final_common_evaluation_keys.csv')==keyhash
res={'decision':decision,'common_n':len(e),'scheduled_n':len(bas),'coverage':len(e)/len(bas),'missing_model_predictions':missing,'key_sha256':keyhash,'lightgbm':lgb.__version__,'num_weekly_models':12,'first_test_mean4_dates':['2012-12-09','2012-12-16','2012-12-23','2012-12-30'],'corrections':[],'output_sha256':{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()}}
(O/'manifest.json').write_text(json.dumps(res,indent=2,ensure_ascii=False));print(overview.to_string(index=False));print('DECISION',decision,'COMMON',len(e),len(bas));print(fl.groupby(['method','status']).size().to_string())
