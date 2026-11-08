"""Read-only source audit. Run: python 01_audit.py. Requires pandas.
No source correction, imputation or forecasting is performed.
"""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
s=pd.read_csv(ROOT/'raw/WEEKLY_SALES_10STORES.csv')
a=pd.read_csv(ROOT/'raw/AppWichStoreAttributes.csv')
w=pd.read_csv(ROOT/'raw/WEEKLY_WEATHER.csv')
s['week']=pd.to_datetime(s['Date'],format='%m/%d/%Y',errors='raise')
k=['Store_num','INV_NUMBER','week']
weeks=pd.date_range(s.week.min(),s.week.max(),freq='7D')
expected=pd.MultiIndex.from_product([sorted(s.Store_num.unique()),weeks],names=['Store_num','week'])
actual=pd.MultiIndex.from_frame(s[['Store_num','week']].drop_duplicates())
missing=expected.difference(actual).to_frame(index=False)
dup=s[s.duplicated(k,keep=False)]
bad_day=s[s.week.dt.day.ne(s.Day)]
for name,df in [('duplicate_keys',dup),('date_component_mismatch',bad_day),('negative_sold',s[s.Sold.lt(0)]),('missing_store_weeks',missing)]:
 df.to_csv(OUT/f'{name}.csv',index=False)
summary={'rows':len(s),'stores':int(s.Store_num.nunique()),'item_codes':int(s.INV_NUMBER.nunique()),'weeks':len(weeks),'first_week':str(weeks[0].date()),'last_week':str(weeks[-1].date()),'observed_store_weeks':len(actual),'expected_store_weeks':len(expected),'exact_duplicate_excess':int(s.duplicated().sum()),'duplicate_key_excess':int(s.duplicated(k).sum()),'duplicate_key_rows':len(dup),'negative_sold_rows':int(s.Sold.lt(0).sum()),'zero_sold_rows':int(s.Sold.eq(0).sum()),'zero_price_rows':int(s.Price.eq(0).sum()),'zero_unit_cost_rows':int(s.Unit_Cost.eq(0).sum()),'date_day_mismatch_rows':len(bad_day),'sales_null_cells':int(s.isna().sum().sum()),'store_metadata_rows':len(a),'weather_rows':len(w)}
(OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
s.groupby('Store_num').agg(observed_weeks=('week','nunique'),rows=('week','size')).to_csv(OUT/'store_coverage.csv')
# Candidate selection evidence uses ONLY first 32 calendar weeks and 9 unambiguous stores.
train_end=weeks[31]; eligible=s[s.Store_num.ne(24)&s.week.le(train_end)]
g=eligible.groupby(['Store_num','INV_NUMBER']).agg(observed_weeks=('week','nunique'),negative_rows=('Sold',lambda x:int(x.lt(0).sum())),training_units=('Sold','sum'),description=('Description','first')).reset_index()
g.to_csv(OUT/'training_only_item_coverage.csv',index=False)
# This is an inspection table, not a final menu selection: codes may be drinks, toppings or discounts.
assert len(actual)+len(missing)==len(expected)
assert not s[s.Store_num.ne(24)].duplicated(k).any()
assert s[s.Store_num.ne(24)].groupby('Store_num').week.nunique().eq(len(weeks)).all()
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('Proposed first training end:',train_end.date())
print('Validation weeks:',weeks[32].date(),weeks[39].date())
print('Test weeks:',weeks[40].date(),weeks[-1].date())
