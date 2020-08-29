import json
from prettytable import PrettyTable

with open('slot_accuracy_dev.json') as f:
    dev = json.load(f)

for slot, value in dev.items():
    print(f'slot: {slot:10}')
    t = PrettyTable()
    t.field_names = ['', 'not in pred', 'in pred', 'correct pred']
    t.add_row(['not in label', value[0][0], value[0][1], '-'])
    t.add_row(['in lable', value[1][0], value[1][1], value[1][2]])
    print(t)
    print()


