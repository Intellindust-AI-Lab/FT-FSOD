# Reference: https://github.com/shenyunhang/APE/blob/main/datasets/tools/objects3652coco/fix_o365_names.py # noqa
import argparse
import copy
import json
from tqdm import tqdm
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ann',
        default='data/objects365v2/annotations/zhiyuan_objv2_train.json')
    parser.add_argument(
        '--fix_name_map',
        default='tools/dataset_converters/zhiyuan_objv2_train_names_fix.csv')
    args = parser.parse_args()

    new_names = {}
    old_names = {}
    with open(args.fix_name_map, 'r') as f:
        for line in f:
            tmp = line.strip().split(',')
            old_names[int(tmp[0])] = tmp[1]
            new_names[int(tmp[0])] = tmp[2]
    data = json.load(open(args.ann, 'r'))

    cat_info = copy.deepcopy(data['categories'])

    for x in tqdm(cat_info):
        if old_names[x['id']] != new_names[x['id']]:
            print('Renaming', x['id'], x['name'], ' to: ', new_names[x['id']])
            x['name'] = new_names[x['id']]

    data['categories'] = cat_info
    basename = os.path.basename(args.ann)
    out_name = basename[:-5] + '_fixname.json'
    out_name = os.path.join('./', out_name)
    print('Saving to', out_name)
    json.dump(data, open(out_name, 'w'))
