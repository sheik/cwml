#!/usr/bin/env python

import os
import yaml
import itertools
from jinja2 import Template
from multiprocessing import Pool

try:
    os.mkdir("/mnt/raid/grid-search")
except:
    pass

with open('grid_search/grid.yaml') as f:
    grid = yaml.load(f, Loader=yaml.FullLoader)

with open('grid_search/grid-search-template.yaml.tpl') as fd:
    template = Template(fd.read())

s = []
for key, values in grid.items():
    s.append([(key, value) for value in values])

commands = []
i = 0
for permutation in list(itertools.product(*s)):
    i += 1
    filename = "/mnt/raid/grid-search/grid-{}.yaml".format(i)
    variables = {}
    for item in permutation:
        label, value = item
        variables[label] = value
    variables['data_dir'] = 'grid-{}-data'.format(i)
    variables['test_dir'] = 'grid-{}-test'.format(i)
    variables['model_dir'] = 'grid-{}-model'.format(i)
    with open(filename, "w") as fd:
        fd.write(template.render(variables=variables))
        commands.append("./run.sh {} &> /mnt/raid/grid-output/grid-{}.log".format(filename, i))

with open("queue.sh", "w") as fp:
    fp.write("\n".join(commands))

