"""Independent audit of saved predictions; no training or specification changes."""
from pathlib import Path
import csv,datetime,json,hashlib,math
import pandas as pd,numpy as np,lightgbm as lgb
P=Path(__file__).resolve().parent;O=P/'final_evaluation';spec=json.loads((P/'05_model_spec.json').read_text());levels=json.loads((O/'categorical_levels.json').read_text());pred=pd.read_csv(O/'final_predictions_all.csv',parse_dates=['week']);tested=0
for fp in sorted((O/'inputs').glob('*.csv')):
 x=pd.read_csv(fp,dtype={c:str for c in spec['categorical']})
 for c in spec['categorical']:x[c]=pd.Categorical(x[c],categories=levels[c])
 model=lgb.Booster(model_file=str(O/'models'/(fp.stem+'.txt')));v=model.predict(x[spec['features']]);e=pred[pred.week.eq(fp.stem)].sort_values(['Store_num','INV_NUMBER']);assert np.allclose(v,e.prediction_raw,atol=1e-10)
 sh=pd.read_csv(O/'shap'/(fp.stem+'.csv'));assert np.allclose(sh[['shap_'+c for c in spec['features']]].sum(axis=1)+sh.expected_value,v,atol=1e-8)
 tested+=len(v)
# Plain-Python lookup independently confirms calendar lags and all final baselines.
lookup={}
with (P/'raw/WEEKLY_SALES_10STORES.csv').open() as f:
 for r in csv.DictReader(f):
  if int(r['Store_num'])==24:continue
  key=(int(r['Store_num']),int(r['INV_NUMBER']),datetime.datetime.strptime(r['Date'],'%m/%d/%Y').date());v=float(r['Sold']);lookup[key]=v if v>=0 else None
for r in pred.itertuples():
 d=r.week.date();h=[lookup.get((r.Store_num,r.INV_NUMBER,d-datetime.timedelta(weeks=i))) for i in range(1,5)];want={'actual':lookup.get((r.Store_num,r.INV_NUMBER,d)),'last_week':h[0],'mean_4weeks':sum(h)/4 if all(v is not None for v in h) else None}
 for c,v in want.items():assert pd.isna(getattr(r,c)) if v is None else math.isclose(getattr(r,c),v,rel_tol=1e-12)
manifest=json.loads((O/'manifest.json').read_text())
for name,h in manifest['output_sha256'].items():assert hashlib.sha256((O/name).read_bytes()).hexdigest()==h
base=json.loads((P/'validation_baselines/manifest.json').read_text())
for name,h in base['output_sha256'].items():assert hashlib.sha256((P/'validation_baselines'/name).read_bytes()).hexdigest()==h
print('Verified saved model predictions and SHAP:',tested,'; independent baseline rows:',len(pred),'; hashes intact')
