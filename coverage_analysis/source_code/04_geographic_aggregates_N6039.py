#!/usr/bin/env python3
"""Recompute descriptive geographic diagnostics from a PRIVATE PIN-level input.

Input: Stage4 private 'S4_GEOGRAPHIC_FIGURE_INPUT_PRIVATE_N6039.csv'.
This source is not bundled for privacy. Outputs are aggregate-only.
No geographic plots, respondent data or raw/PIN-level outputs are exported.
"""
from pathlib import Path
from collections import Counter
import csv
import json
import math
import argparse


def read(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def calc_q(arr,q):
    v=sorted(arr)
    pos=(len(v)-1)*q
    a=math.floor(pos); b=math.ceil(pos)
    return v[a] if a==b else v[a]*(b-pos)+v[b]*(pos-a)


def haversine(a,b):
    lat1,lon1=map(math.radians,a);lat2,lon2=map(math.radians,b)
    dlat=lat2-lat1;dlon=lon2-lon1
    h=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0088*2*math.asin(min(1,math.sqrt(h)))


def main():
    parser=argparse.ArgumentParser()
    default=(Path.home()/'WSQS_PRIVATE_COVERAGE_N6039_NOT_FOR_PUBLIC'/'STAGE4'
             /'S4_GEOGRAPHIC_FIGURE_INPUT_PRIVATE_N6039.csv')
    parser.add_argument('--private-pin-input',type=Path,default=default)
    parser.add_argument('--out-dir',type=Path,default=Path.cwd()/'S4_GEO_AGGREGATE_INDEPENDENT_RECALC')
    args=parser.parse_args()
    if not args.private_pin_input.is_file():
        raise SystemExit('STOP: Private validated PIN-level input unavailable')
    rows=read(args.private_pin_input)
    expected={'PIN','District','Latitude','Longitude','Settlement_context','Population_density','Household_N'}
    if not rows or not expected.issubset(rows[0]):
        raise SystemExit('STOP: Stage4 geographic source columns differ')
    if len(rows)!=629 or len({r['PIN'] for r in rows})!=629:
        raise SystemExit('STOP: Expected 629 distinct matched Tamil Nadu PINs')
    n=sum(int(r['Household_N']) for r in rows)
    if n!=5778:raise SystemExit('STOP: Expected 5778 spatially matched households')
    district=Counter(); households=Counter(); pins=Counter(); coords=[]; density=[]
    for r in rows:
        count=int(r['Household_N'])
        district[r['District']]+=count
        households[r['Settlement_context']]+=count
        pins[r['Settlement_context']]+=1
        xy=(float(r['Latitude']),float(r['Longitude']))
        if not(7<=xy[0]<=15 and 75<=xy[1]<=83):
            raise SystemExit('STOP: Unexpected coordinate bounds')
        coords.append(xy)
        value=float(r['Population_density'])
        if not math.isfinite(value) or value<0:
            raise SystemExit('STOP: Invalid density')
        density.append(value)
    if len(district)!=37 or sum(pins.values())!=629:
        raise SystemExit('STOP: Geographic class or district validation failed')
    top=district.most_common()
    effective=1/sum((x/n)**2 for x in district.values())
    maximum=max(haversine(a,b) for i,a in enumerate(coords) for b in coords[i+1:])
    agg={'spatial_household_N':n,'unique_PIN_locations':len(rows),
         'districts_represented':len(district),
         'top3_district_household_percent':100*sum(x for _,x in top[:3])/n,
         'top5_district_household_percent':100*sum(x for _,x in top[:5])/n,
         'effective_number_of_districts':effective,
         'maximum_PIN_reference_separation_km':maximum,
         'population_density_P10_per_km2':calc_q(density,.1),
         'population_density_median_per_km2':calc_q(density,.5),
         'population_density_P90_per_km2':calc_q(density,.9),
         'population_density_max_per_km2':max(density),
         'input_is_private_PIN_level':True,'outputs_are_aggregate_only':True}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    def output(name,header,entries):
        file=args.out_dir/name
        if file.exists():raise SystemExit('STOP: Preserving existing output '+str(file))
        with file.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(entries)
        return file
    output('district_distribution.csv',['District','N','Percent_of_5778'],
           [[k,v,round(100*v/n,5)] for k,v in top])
    output('settlement_distribution.csv',
           ['Settlement','Households','Household_percent','PIN_locations','PIN_percent'],
           [[k,households[k],round(100*households[k]/n,5),pins[k],
             round(100*pins[k]/629,5)] for k in sorted(households)])
    file=args.out_dir/'geographic_summary.json'
    if file.exists():raise SystemExit('STOP: Preserving existing audit '+str(file))
    file.write_text(json.dumps(agg,indent=2)+'\n',encoding='utf-8')
    print('PASS | independent geographic aggregate recalculation; 5778 households, 629 PINs')
    print('Output:',args.out_dir)
    print('No PIN codes, coordinates or respondent-level data exported.')

if __name__=='__main__':main()
