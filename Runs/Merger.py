import json
import os

SOURCE = os.getcwd() + "/Tests/NOAH/CEC2022/DATASET_1.json_"
ADD_JSON = os.getcwd() + "/Tests/NOAH/CEC2022/20260602093007__dim20_opti30_it100/dataset.json"

OUTFILENAME = 'DATASET_2'
OUTPUT_JSON = SOURCE.replace('DATASET_1.json', OUTFILENAME+'.json')
OUTPUT_STATS = SOURCE.replace('DATASET_1.json', 'STATS_'+OUTFILENAME+'.json')


ALGO = ['DE', 'SADE']

new_data = dict()
statistics_data = dict()

with open(SOURCE, "r") as f:
    source = json.load(f)

with open(ADD_JSON, "r") as f:
    add_json = json.load(f)

for function in source.keys():
    fn_data = source[function]
    new_data[function] = {}
    statistics_data[function] = {}

    for algorithm in fn_data.keys():
        new_data[function][algorithm] = source[function][algorithm].copy()
    print(function)
    for a in ALGO:
        new_data[function][a] = add_json[function][a].copy()
        print('\t', a)

    for algo in new_data[function].keys():
        scores = []
        optimizations = new_data[function][algo]['optimizations'].copy()
        for optimization in optimizations:
            scores.append(optimization['score'])

        statistics_data[function][algo] = {
            'scores': scores,
        }

    print(new_data[function].keys())

with open(OUTPUT_JSON+'_', "w", encoding="utf-8") as file:
    json.dump(new_data, file,  ensure_ascii=False)

with open(OUTPUT_STATS+'_', "w", encoding="utf-8") as file:
    json.dump(statistics_data, file,  ensure_ascii=False)

print(OUTPUT_JSON)
print(OUTPUT_STATS)
