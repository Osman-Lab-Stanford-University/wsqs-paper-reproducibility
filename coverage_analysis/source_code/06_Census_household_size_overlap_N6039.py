#!/usr/bin/env python3
"""Recalculate the seven-category Census 2011 Tamil Nadu household-size overlap.

No models, plots, respondent IDs or fine geographic coordinates are used.
Requires the analyst's harmonized 6,039-cohort household-size aggregate and
original audited Census 2011 Tamil Nadu reference workbook.
"""
from pathlib import Path
from openpyxl import load_workbook
import csv, argparse, json, re


def key(x):
    return re.sub(r'[^a-z0-9]', '', str(x or '').lower())


def main():
    home=Path.home()/'Desktop/Spring 2026/Chapter 3 MANUAL RERUN'
    ap=argparse.ArgumentParser()
    ap.add_argument('--study',type=Path,default=home/'S4_N6039_AGGREGATES_STAGE3/TABLE_S4_HOUSEHOLD_SIZE_N6039.csv')
    ap.add_argument('--census',type=Path,default=home/'Demographid analysis/02_household_size_benchmark/Chapter3_STEP02A_household_size_benchmark.xlsx')
    ap.add_argument('--out',type=Path,default=home/'S4_N6039_CENSUS_COMPARISON_RECALC')
    a=ap.parse_args()
    if not a.study.is_file() or not a.census.is_file():
        raise SystemExit('STOP: Missing aggregate survey counts or Census reference workbook')
    with a.study.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if not rows or 'N' not in rows[0]:
        raise SystemExit('STOP: Study distribution lacks N')
    cat_field='Category' if 'Category' in rows[0] else 'Size_category'
    study={r[cat_field].strip():int(r['N']) for r in rows if r[cat_field].strip()!='MISSING'}
    groups=['1','2','3','4','5','6','7+']
    if set(study)!=set(groups) or sum(study.values())!=6013:
        raise SystemExit('STOP: Expected seven groups and 6,013 valid study observations')
    book=load_workbook(a.census,read_only=True,data_only=True)
    try:
        ws=book['01_category_comparison']
        iterator=ws.iter_rows(values_only=True)
        header=[key(v) for v in next(iterator)]
        ci=header.index('census2011n')
        si=header.index('householdsize')
        census={str(row[si]).strip():int(row[ci]) for row in iterator if row[si] is not None}
    finally: book.close()
    if set(census)!=set(groups) or sum(census.values())!=18462231:
        raise SystemExit('STOP: Original Census Tamil Nadu HH-01 data mismatch')
    n=6013; nc=18462231
    comparisons=[]
    for group in groups:
        p=study[group]/n; q=census[group]/nc
        comparisons.append((group,study[group],census[group],100*p,100*q,100*(p-q)))
    tvd=sum(abs(r[5])/100 for r in comparisons)/2
    overlap=1-tvd
    if not(0.78<overlap<0.80):
        raise SystemExit('STOP: Census overlap outside expected range')
    a.out.mkdir(parents=True,exist_ok=True)
    f=a.out/'household_size_vs_Census2011_N6039.csv'
    if f.exists():raise SystemExit('STOP: Existing CSV preserved; choose another --out folder')
    with f.open('w',newline='',encoding='utf-8') as g:
        w=csv.writer(g)
        w.writerow(['Household_size','Study_N','Census_2011_N','Study_percent','Census_2011_percent','Difference_percentage_points'])
        w.writerows(comparisons)
    report={'analytical_N':6039,'valid_household_size_N':n,'Census_TN_households_N':nc,
            'total_variation_distance':tvd,'distribution_overlap':overlap,
            'distribution_overlap_percent':round(overlap*100,2)}
    (a.out/'household_size_overlap_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS | Census household-size overlap {overlap*100:.2f}%; TVD={tvd:.6f}')

if __name__=='__main__':main()
