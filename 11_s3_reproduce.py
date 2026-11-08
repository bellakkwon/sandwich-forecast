"""Reproduce one saved forecast through S3; no training. package / upload / run.
Credentials: named profile or boto3 standard credential chain. Never embed keys.
Runtime: lightgbm==4.6.0, pandas, numpy, boto3. Test-only: moto[s3].
"""
from __future__ import annotations
import argparse,hashlib,io,json,math,platform,shutil,sys,uuid
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
KEYS=['Store_num','INV_NUMBER','week'];CATS=['store','code','type'];STR={c:str for c in CATS}
FILES={'features.csv':'input','expected_predictions.csv':'input','category_levels.json':'input','feature_order.json':'input','model.txt':'models','package_manifest.json':'input'}
UTC=lambda:datetime.now(timezone.utc).isoformat()
sha=lambda b:hashlib.sha256(b).hexdigest()
def read_csv(b):return pd.read_csv(io.BytesIO(b),dtype=STR,encoding='utf-8-sig',float_precision='round_trip')
def norm(df):
 d=df.copy()
 for k in KEYS[:2]:
  v=pd.to_numeric(d[k],errors='raise')
  if v.isna().any() or not np.isfinite(v).all() or not v.eq(np.floor(v)).all():raise ValueError('Invalid integer key: '+k)
  d[k]=v.astype('int64')
 dates=pd.to_datetime(d.week,errors='raise')
 if dates.isna().any():raise ValueError('Null week')
 d['week']=dates.dt.strftime('%Y-%m-%d')
 if d.duplicated(KEYS).any():raise ValueError('Duplicate keys')
 return d

def prepare_features(f,booster,order,levels,week,rows):
 if len(f)!=rows or not f.week.eq(week).all():raise ValueError('Feature row count/week mismatch')
 if booster.feature_name()!=order:raise ValueError('Feature order mismatch')
 if [c for c in order if c in CATS]!=CATS:raise ValueError('Categorical feature order mismatch')
 if len(booster.pandas_categorical or [])!=len(CATS):raise ValueError('Model category metadata missing')
 model_levels={c:[str(x) for x in v] for c,v in zip(CATS,booster.pandas_categorical)}
 if model_levels!=levels:raise ValueError('Category levels mismatch')
 for c in CATS:
  if f[c].isna().any() or not set(f[c]).issubset(set(levels[c])):raise ValueError('Unknown/null category: '+c)
 if not f.store.eq(f.Store_num.astype(str)).all() or not f.code.eq(f.INV_NUMBER.astype(str)).all():raise ValueError('Category/key mismatch')
 for c in order:
  if c in CATS:f[c]=pd.Categorical(f[c],categories=levels[c])
  else:
   f[c]=pd.to_numeric(f[c],errors='raise')
   if np.isinf(f[c]).any():raise ValueError('Infinite feature: '+c)
 return f

def compare(pred,exp,week,rows,tol):
 if len(exp)!=rows or not exp.week.eq(week).all():raise ValueError('Expected row count/week mismatch')
 for c in ['prediction_raw','prediction']:
  if c not in exp or not np.isfinite(pd.to_numeric(exp[c],errors='raise')).all():raise ValueError('Expected finite column required: '+c)
 if not np.array_equal(np.maximum(exp.prediction_raw.to_numpy(),0),exp.prediction.to_numpy()):raise ValueError('Expected clipping rule mismatch')
 m=pred.merge(exp[KEYS+['prediction_raw','prediction']],on=KEYS,how='outer',validate='one_to_one',indicator=True,suffixes=('_new','_expected'))
 if len(m)!=rows or not m['_merge'].eq('both').all():raise ValueError('Key sets differ')
 diffs={c:float((m[c+'_new']-m[c+'_expected']).abs().max()) for c in ['prediction_raw','prediction']}
 if not all(math.isfinite(x) for x in diffs.values()):raise ValueError('Nonfinite comparison')
 return diffs,all(x<=tol for x in diffs.values())

def forecast(model_bytes,feat_bytes,order,levels,week,rows):
 import lightgbm as lgb
 model=lgb.Booster(model_str=model_bytes.decode('utf-8'))
 f=prepare_features(norm(read_csv(feat_bytes)),model,order,levels,week,rows)
 raw=model.predict(f[order],num_threads=2)
 if not np.isfinite(raw).all():raise ValueError('Nonfinite model prediction')
 p=f[KEYS].copy();p['prediction_raw']=raw;p['prediction']=np.maximum(raw,0)
 return p

def package(a):
 import lightgbm as lgb
 if lgb.__version__!='4.6.0':raise ValueError('Use LightGBM 4.6.0')
 if a.out.exists():raise ValueError('Output exists; refusing overwrite')
 model_bytes=a.model.read_bytes();feat_bytes=a.features.read_bytes();exp_bytes=a.expected.read_bytes()
 booster=lgb.Booster(model_str=model_bytes.decode());order=booster.feature_name()
 if len(booster.pandas_categorical or [])!=3:raise ValueError('No category metadata')
 levels={c:[str(x) for x in v] for c,v in zip(CATS,booster.pandas_categorical)}
 exp=norm(read_csv(exp_bytes));pred=forecast(model_bytes,feat_bytes,order,levels,a.week,a.rows);diffs,ok=compare(pred,exp,a.week,a.rows,1e-9)
 if not ok:raise ValueError('Local reproduction failed; do NOT adjust input values: '+str(diffs))
 a.out.mkdir(parents=True)
 # Preserve exact source bytes: no CSV serialization or threshold adjustment.
 (a.out/'features.csv').write_bytes(feat_bytes);(a.out/'expected_predictions.csv').write_bytes(exp_bytes);(a.out/'model.txt').write_bytes(model_bytes)
 (a.out/'category_levels.json').write_text(json.dumps(levels,ensure_ascii=False,indent=2),encoding='utf-8');(a.out/'feature_order.json').write_text(json.dumps(order,indent=2),encoding='utf-8')
 manifest={'week':a.week,'rows':a.rows,'required_lightgbm_version':'4.6.0','tolerance':1e-9,'created_utc':UTC(),'sha256':{n:sha((a.out/n).read_bytes()) for n in FILES if n!='package_manifest.json'},'local_max_abs_diff':diffs,'source_bytes_preserved':True}
 b=json.dumps(manifest,indent=2).encode();(a.out/'package_manifest.json').write_bytes(b)
 digest=sha(b);(a.out/'trusted_manifest.sha256').write_text(digest+'\n');print(json.dumps({'package':str(a.out),'manifest_sha256':digest,'local_max_abs_diff':diffs},indent=2))

def client(a):
 import boto3
 return boto3.Session(profile_name=a.profile,region_name=a.region).client('s3')

def upload(a):
 s3=client(a);pkg=a.package;manifest=json.loads((pkg/'package_manifest.json').read_text());contents={n:(pkg/n).read_bytes() for n in FILES}
 for n,h in manifest['sha256'].items():
  if sha(contents[n])!=h:raise ValueError('Local package hash mismatch: '+n)
 for n,sub in FILES.items():s3.put_object(Bucket=a.bucket,Key=a.prefix.strip('/')+'/'+sub+'/'+n,Body=contents[n])
 print('Uploaded package; trusted manifest SHA256:',sha(contents['package_manifest.json']))

def run(a):
 import lightgbm as lgb,boto3
 run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
 log={'run_id':run_id,'started_utc':UTC(),'passed':False,'status':'started','max_abs_diff':None,'rows':None,'input_sha256':{},'checks':{},'versions':{'python':sys.version.split()[0],'platform':platform.platform(),'lightgbm':lgb.__version__,'pandas':pd.__version__,'numpy':np.__version__,'boto3':boto3.__version__}}
 s3=None;output=None;prefix=a.prefix.strip('/');local=Path(a.local_copy)/run_id if a.local_copy else None
 if local:local.mkdir(parents=True,exist_ok=False)
 try:
  if not a.manifest_sha256 or len(a.manifest_sha256)!=64:raise ValueError('Trusted manifest SHA256 required')
  s3=client(a)
  def get(n):return s3.get_object(Bucket=a.bucket,Key=prefix+'/'+FILES[n]+'/'+n)['Body'].read()
  mb=get('package_manifest.json');log['manifest_sha256']=sha(mb);log['checks']['manifest_trusted']=sha(mb)==a.manifest_sha256
  if not log['checks']['manifest_trusted']:raise ValueError('Manifest hash differs from trusted local digest')
  manifest=json.loads(mb);log['week']=manifest['week'];tol=manifest['tolerance'];log['tolerance']=tol
  if not math.isfinite(tol) or tol!=1e-9:raise ValueError('Frozen tolerance must be 1e-9')
  if lgb.__version__!=manifest['required_lightgbm_version']:raise ValueError('LightGBM version mismatch')
  raw={n:get(n) for n in FILES if n!='package_manifest.json'};hashes={n:sha(b) for n,b in raw.items()};log['input_sha256']=hashes
  log['checks']['hash_match']=hashes==manifest['sha256']
  if not log['checks']['hash_match']:raise ValueError('Package content hash mismatch; prediction not attempted')
  order=json.loads(raw['feature_order.json']);levels=json.loads(raw['category_levels.json']);exp=norm(read_csv(raw['expected_predictions.csv']))
  pred=forecast(raw['model.txt'],raw['features.csv'],order,levels,manifest['week'],manifest['rows']);log['checks']['feature_and_category_checks']=True
  diffs,ok=compare(pred,exp,manifest['week'],manifest['rows'],tol);log['max_abs_diff']=diffs;log['rows']={'features':len(pred),'expected':len(exp),'matched':len(pred)};log['checks']['keys_one_to_one']=True;log['checks']['within_tolerance']=ok
  output=pred.to_csv(index=False).encode();outkey=prefix+'/output/'+run_id+'/predictions.csv';log['output_key']=outkey;log['output_sha256']=sha(output)
  s3.put_object(Bucket=a.bucket,Key=outkey,Body=output,ContentType='text/csv')
  if not ok:raise ValueError('Prediction difference exceeds frozen tolerance')
  log['passed']=True;log['status']='passed'
 except Exception as ex:
  log['status']='failed';log['error_type']=type(ex).__name__
  # Avoid copying SDK exception text that could expose request/environment details.
  log['error']=str(ex) if isinstance(ex,ValueError) else 'Execution failed; inspect the local environment and object access.'
 log['finished_utc']=UTC();logkey=prefix+'/logs/'+run_id+'/run_log.json';log['log_key']=logkey
 if s3 is not None:
  try:s3.put_object(Bucket=a.bucket,Key=logkey,Body=json.dumps(log,indent=2,allow_nan=False).encode(),ContentType='application/json')
  except Exception as ex:log['passed']=False;log['status']='failed';log['log_upload_error_type']=type(ex).__name__
 else:log['log_upload_error_type']='NoClient'
 if local:
  if output is not None:(local/'predictions.csv').write_bytes(output)
  (local/'run_log.json').write_text(json.dumps(log,indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps(log,indent=2,allow_nan=False))
 return 0 if log['passed'] else 1

def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
 p=sub.add_parser('package');p.add_argument('--week',required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--rows',type=int,default=172)
 for name in ['upload','run']:
  q=sub.add_parser(name);q.add_argument('--bucket',required=True);q.add_argument('--prefix',required=True);q.add_argument('--profile');q.add_argument('--region',default='ap-northeast-2')
  if name=='upload':q.add_argument('--package',type=Path,required=True)
  else:q.add_argument('--manifest-sha256',required=True);q.add_argument('--local-copy',default='s3_run_copies')
 a=ap.parse_args()
 try:
  if a.cmd=='package':package(a);return 0
  if a.cmd=='upload':upload(a);return 0
  return run(a)
 except Exception as ex:
  print(type(ex).__name__+': '+str(ex),file=sys.stderr);return 1
if __name__=='__main__':sys.exit(main())
