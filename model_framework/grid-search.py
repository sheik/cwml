#!/usr/bin/env python

import yaml
import itertools

with open('grid_search/grid.yaml') as f:
    grid = yaml.load(f, Loader=yaml.FullLoader)
    s = []
    for key, values in grid.items():
        v = []
        for value in values:
            v.append({key: value})
        s.append(v)
    for permutation in list(itertools.product(*s)):
        print("Create new file")
        for item in permutation:
            for label, value in item.items():
                print("Set {} = {}".format(label, value))


