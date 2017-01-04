"""
Simulates the interactions between classical elements (fire, water, cloud, vegetation).
"""

import json
import time
from world import World
import os

dirname = os.path.dirname(os.path.realpath(__file__))

config = None
with open(dirname + '/config.json', 'r') as config_file:
    config = json.load(config_file)

ticks = config['ticks']
step = config['step']
debug = True if config['debug'] else False

world = World(config, debug=debug)

while True:
    j = 0
    for i in range(ticks):
        j += 1
        world.tick()
        if j == step:
            world.plot()
            world.plt.pause(0.0001)
            j = 0
        time.sleep(config['sleep'])

    reply = input('\nMenu:\n1) Continue simulation\n2) Reset\n3) Quit?\n\n(1/2/3) > ')
    if reply == '1':
        print('Continuing simulation')
        world.plt.cla()
    elif reply == '2':
        print('Starting new simulation')
        world.plt.close()
        world = World(config)
    else:
        print('Quitting')
        break
