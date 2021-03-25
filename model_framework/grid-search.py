#!/usr/bin/env python

import yaml
import itertools
from jinja2 import Template

with open('grid_search/grid.yaml') as f:
    grid = yaml.load(f, Loader=yaml.FullLoader)

with open('grid_search/grid-search-template.yaml.tpl') as fd:
    template = Template(fd.read())

s = []
for key, values in grid.items():
    s.append([(key, value) for value in values])

for permutation in list(itertools.product(*s)):
    print("Create new file")
    variables = {}
    for item in permutation:
        label, value = item
        variables[label] = value
    variables['data_dir'] = ''
    variables['test_dir'] = ''
    variables['model_dir'] = ''
    print(template.render(variables=variables))


