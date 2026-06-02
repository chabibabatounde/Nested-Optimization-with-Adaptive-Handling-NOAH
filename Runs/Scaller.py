import json
import os
n = 30
scalable = []
INPUT_JSON = os.getcwd() + "/Tests/NOAH/CEC2022/DATA.json"

OUTPUT_JSON = INPUT_JSON.replace('new_dataset.json', 'Output.json')
Statistics_JSON = INPUT_JSON.replace('dataset.json', 'Statistics.json')

new_data = dict()
statistics_data = dict()

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

for function in data.keys():
    fn_data = data[function]

exit(data['F1']['GA'].keys())
for function in data.keys():
    new_data[function] = {}
    statistics_data[function] = {}

    for algorithm in fn_data.keys():
        new_data[function][algorithm] = None
        statistics_data[function][algorithm] = None
        optimizations = fn_data[algorithm]['optimizations'].copy()

        '''
        if algorithm in scalable:
            sorted_data = sorted(optimizations, key=lambda x: x['score']) # Reverse
            scaled_data = sorted_data.copy()
            scaled_data = scaled_data[-n:].copy()
        else:
            sorted_data = sorted(optimizations, key=lambda x: x['score'])  # Reverse
            scaled_data = sorted_data.copy()
            scaled_data = scaled_data[:n].copy()
        if algorithm in scalable:
            sorted_data = sorted(optimizations, key=lambda x: x['score'])  # Reverse
            scaled_data = sorted_data.copy()
            scaled_data = scaled_data[-n:].copy()
        else:
            if algorithm == 'NOAH':
                sorted_data = sorted(optimizations, key=lambda x: x['score'])  # Reverse
                scaled_data = sorted_data.copy()
                scaled_data = scaled_data[:n].copy()
            else:
                scaled_data = optimizations[:n].copy()
        random.shuffle(scaled_data)
        '''
        scaled_data = optimizations[:n].copy()
        index_min = min(range(len(scaled_data)), key=lambda i: scaled_data[i]['score'])
        scores = []
        for optimization in scaled_data:
            scores.append(optimization['score'])

        statistics_data[function][algorithm] = {
            'scores': scores,
        }

        new_data[function][algorithm] = {
            'optimizations': scaled_data,
            'best_index': index_min
        }

print()
with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
    json.dump(new_data, file, ensure_ascii=False)

with open(Statistics_JSON, "w", encoding="utf-8") as file:
    json.dump(statistics_data, file, indent=1, ensure_ascii=False)
print(OUTPUT_JSON)
