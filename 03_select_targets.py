"""Freeze targets using first 26 weeks only. No forecasting or test-set selection."""
from pathlib import Path
import pandas as pd,json
P=Path(__file__).resolve().parent;O=P/'selection';O.mkdir(exist_ok=True)
s=pd.read_csv(P/'raw/WEEKLY_SALES_10STORES.csv');s['week']=pd.to_datetime(s.Date,format='%m/%d/%Y')
r=pd.read_csv(P/'review/product_classification_review.csv');weeks=sorted(s.week.unique());stores=sorted(set(s.Store_num)-{24});t=s[s.Store_num.isin(stores)&s.week.le(weeks[25])].copy()
# Recompute training-only screen from raw, not manually edited volume totals.
vol=t[t.Sold.ge(0)].groupby('INV_NUMBER').Sold.sum()
sand=[int(x) for x in r.loc[r['후보분류'].eq('샌드위치 후보'),'상품코드'] if vol.get(x,0)>=520]
sets=[7335,1287,1283,35020,35016];ids=sorted(sand+sets)
assert len(sand)==52 and len(ids)==57
c=t[t.INV_NUMBER.isin(ids)];assert not c.duplicated(['Store_num','INV_NUMBER','week']).any()
g=c.groupby(['Store_num','INV_NUMBER']).agg(observed_weeks=('week','nunique'),valid_weeks=('Sold',lambda x:int(x.ge(0).sum())),valid_units=('Sold',lambda x:x[x>=0].sum()))
idx=pd.MultiIndex.from_product([stores,ids],names=['Store_num','INV_NUMBER']);g=g.reindex(idx,fill_value=0).reset_index()
g['weekly_average']=g.valid_units/26
g['selected']=(g.observed_weeks==26)&(g.valid_weeks==26)&(g.valid_units>=520)
g['reason']=g.apply(lambda x:'선정' if x.selected else ('26주 유효 관측 미충족' if x.valid_weeks<26 else '주평균20 미만'),axis=1)
g['category']=g.INV_NUMBER.map(lambda x:'세트·구성 미확인' if x in sets else '샌드위치')
names=r.set_index('상품코드')['상품명_정규화'];g['name']=g.INV_NUMBER.map(names)
g['tag']=g.INV_NUMBER.map(lambda x:'프로모션성 명칭·구성 변경 여부 미확인' if x in [14746,14747] else '')
g.to_csv(O/'all_513_candidate_pairs.csv',index=False,encoding='utf-8-sig');z=g[g.selected].copy();z.to_csv(O/'selected_targets.csv',index=False,encoding='utf-8-sig')
by=z.groupby('Store_num').agg(selected_pairs=('INV_NUMBER','size'),selected_units=('valid_units','sum'))
den=c[c.Sold.ge(0)].groupby('Store_num').Sold.sum();by['candidate_57_units']=den;by['coverage_pct']=100*by.selected_units/by.candidate_57_units
by.to_csv(O/'store_summary.csv',encoding='utf-8-sig')
audit=c.groupby(['INV_NUMBER','Description']).agg(rows=('Sold','size'),sold=('Sold','sum'),price_min=('Price','min'),price_max=('Price','max'),sales=('Sales','sum'),negative_sales_rows=('Sales',lambda x:int(x.lt(0).sum())),zero_sales_rows=('Sales',lambda x:int(x.eq(0).sum())))
audit.to_csv(O/'training_price_sales_audit.csv',encoding='utf-8-sig')
summary={'candidate_codes':len(ids),'candidate_pairs':len(g),'selected_pairs':len(z),'selected_codes':int(z.INV_NUMBER.nunique()),'sandwich_pairs':int(z.category.eq('샌드위치').sum()),'set_pairs':int(z.category.ne('샌드위치').sum()),'selected_units':float(z.valid_units.sum()),'candidate_57_nonnegative_units':float(c[c.Sold.ge(0)].Sold.sum()),'coverage_pct':float(100*z.valid_units.sum()/c[c.Sold.ge(0)].Sold.sum()),'training_end':str(pd.Timestamp(weeks[25]).date())}
(O/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
assert z.valid_weeks.eq(26).all() and z.weekly_average.ge(20).all()
assert len(g)==9*57 and len(g)==len(z)+len(g[~g.selected])
print(json.dumps(summary,indent=2));print(by.to_string());print(g.reason.value_counts().to_string())
