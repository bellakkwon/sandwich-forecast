"""Candidate labels only: no final inclusion decisions or evaluation-sales ranking."""
from pathlib import Path
import re,html,json
import pandas as pd
P=Path(__file__).resolve().parent;O=P/'review';O.mkdir(exist_ok=True)
s=pd.read_csv(P/'raw/WEEKLY_SALES_10STORES.csv');s['week']=pd.to_datetime(s.Date,format='%m/%d/%Y');s['name']=s.Description.str.strip().str.upper().str.replace(r'\s+',' ',regex=True)
cut=sorted(s.week.unique())[25]; train=s[(s.Store_num!=24)&(s.week<=cut)]
def label(n):
 if re.search(r'FREE|REWARD',n):return '무료 제공·리워드','무료/리워드 표기: 실물 제공 또는 할인 기록인지 확인 필요'
 if re.search(r'COUPON|CASH CARD|TRANSFER|\bOFF\b|DISCOUNT|OVERRIDE|BOGO|BUY FL',n):return '비상품·할인 후보','쿠폰/결제/이동/할인 관련 명칭'
 if re.search(r'VAL MEAL|COMBO|KIDS? PAK|KID PAK|\bKP\b|LUNCH|LUNCK|LUNCH BOX|\bFVM\b|TO GO MEAL',n):return '세트 후보','세트/묶음 명칭: 구성 확인 전 단품 합산 금지'
 if re.search(r'PARTY|PLATTER|CATERING',n):return '대형·케이터링','일반 단품과 수량 단위가 다를 수 있음'
 if re.search(r'BREAD|\bBRD\b|CHEESE|SAUCE|MUSTARD|BACON',n) and not re.search(r'FLATBREAD|MINIFLAT|MINIFB|EGG|STEAK|PHILLY|PIZZA SUB|MELT|RANCH|TURKEY|HONEY MUSTARD HAM|HAM HONEY MUSTARD|SALAD',n) and re.search(r'BREAD|\bBRD\b|MUNSTER CHEESE|SWISS CHEESE|BACON/|MUSTARD',n):return '구성품·추가재료','빵/치즈/소스 등 단품 완제품 여부 확인'
 if re.search(r'^ADD |^EXTRA |MAYONNAISE|OLIVE OIL|VINAIG|LETTUCE|PEPPERS|OLIVES|ONIONS|PICKLES|TOMATO|SAUCE|NO DRESSING|ALL VEGGIES|SAUERKRAUT|DIPPING CUP|BEV UPGRADE',n):return '구성품·추가재료','추가재료/조미료/변경 옵션'
 if re.search(r'SALAD|SOUP|COOKIE|CHIPS|DRINK|BEVERAGE|BEV$|WATER|MILK|JUICE|COFFEE|CHOCOLATE|COKE|SODA|CUP|NACHOS|POPCORN|PRETZEL|HOTDOG|COTTON|ICEE|YOGURT|FRUIT PUNCH|CAPPA|APPLE SLICES|CINNAMON|BISCUIT',n) or ('PIZZA' in n and 'SUB' not in n):return '사이드·음료·기타음식','샌드위치 외 음식/음료 또는 용기'
 if re.search(r'^(MINI|REGULAR|FLATBREAD|SUNRISE MELT)',n):return '샌드위치 후보','명칭상 샌드위치: 코드 단위 유지, 검토 전 미확정'
 return '확인 필요','명칭만으로 분류 불충분'
rows=[]
for code,g in s.groupby('INV_NUMBER',sort=True):
 names=sorted(g.name.unique()); cats=[label(n) for n in names];cat,reason=cats[0]
 if len(set(x[0] for x in cats))>1:cat,reason='확인 필요','동일 코드 내 명칭 분류 충돌'
 t=train[train.INV_NUMBER==code];valid=t[t.Sold>=0]
 rows.append({'상품코드':int(code),'상품명_정규화':' | '.join(names),'후보분류':cat,'분류근거':reason,'검토상태':'미확정','최종분류':'','포함여부':'','검토메모':'','최초26주_9매장_행수':len(t),'최초26주_9매장_음수제외수량':float(valid.Sold.sum()),'최초26주_음수행수':int(t.Sold.lt(0).sum()),'명칭수':len(names)})
df=pd.DataFrame(rows);df.to_csv(O/'product_classification_review.csv',index=False,encoding='utf-8-sig')
counts=df['후보분류'].value_counts();counts.to_csv(O/'candidate_counts.csv',encoding='utf-8-sig')
# Static browser-viewable table; this is a review document, not a web application.
body=df.to_html(index=False,escape=True)
(O/'product_classification_review.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>상품 분류 검토표</title><style>body{font-family:Arial,sans-serif;margin:30px;color:#182638}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background:#e8eef5;position:sticky;top:0}h1{font-size:26px}</style><h1>샌드위치 상품 분류 검토표 · 초안</h1><p>505개 코드 전체의 명칭 기반 후보 분류. 전부 미확정입니다. 최종 분류·포함 여부는 CSV에서 수정하세요.</p><p>수량은 최초 26주, 24번 제외 9매장에 한정합니다. 음수는 수량 합에서 제외하고 별도 집계했습니다. 전체 기간 명칭은 사전 구축용이며 평가 판매량을 선정에 사용하지 않습니다.</p>'+counts.to_frame('코드 수').to_html()+body,encoding='utf-8')
assert len(df)==s.INV_NUMBER.nunique() and df['상품코드'].is_unique
print(counts.to_string()); print('Training cutoff',pd.Timestamp(cut).date()); print(df[df['후보분류']=='확인 필요'][['상품코드','상품명_정규화']].to_string(index=False))
