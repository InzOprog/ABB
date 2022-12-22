from enum import Enum
from collections import namedtuple
import numpy as np

from sumolib import checkBinary
import traci
import optparse
import sys
import os


def safety():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    return sumoBinary



class Phase(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 1000


class SmartTraffic:
    def __init__(self, configfile="data/simplified.sumocfg", tripinfo="tripinfo.xml"):
        self.score = 0
        self.step_iteration = 0
        self.phase = Phase.LEFT
        self.configfile = configfile
        self.tripinfo = tripinfo
        traci.start([safety(), "-c", configfile, "--tripinfo-output", tripinfo])
        self.reset()
        self.alive = 0

    def reset(self):
        # init game state
        self.phase = Phase.LEFT
        self.score = 0
        self.step_iteration = 0
        traci.load(["--start", "-c", "data/simplified.sumocfg", "--tripinfo-output", "tripinfo.xml"])

    def get_state(self):  # TODO: Change to car count next to traffic lights recognition, passed car detection
        _state = [len(traci.edge.getLastStepVehicleIDs('1i')),
                  len(traci.edge.getLastStepVehicleIDs('2i')),
                  len(traci.edge.getLastStepVehicleIDs('3i')),
                  len(traci.edge.getLastStepVehicleIDs('4i')),
                  traci.edge.getWaitingTime('1i'),
                  traci.edge.getWaitingTime('2i'),
                  traci.edge.getWaitingTime('3i'),
                  traci.edge.getWaitingTime('4i'),
                  traci.trafficlight.getPhase("J2")]

        return _state

    def _take_step(self, amount=4):
        for _ in range(amount):
            traci.simulationStep()
            self.alive = traci.simulation.getMinExpectedNumber() > 0


    def _reward(self, _state, _new_state):
        # qLengths1 = _state[:4]
        # qLengths2 = _new_state[:4]
        #
        # q1 = np.average(qLengths1) * np.std(qLengths1)
        # q2 = np.average(qLengths2) * np.std(qLengths2)
        #
        # return (-1 * (q1 >= q2)) + (1 * (q1 < q2))  # return reward (1 || -1)
        reward = 0
        for i in range(8):
            if (_state[i]) >= (_new_state[i]):
                reward -= 1
            else:
                reward += 1
        return reward

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Phase.LEFT, Phase.UP, Phase.RIGHT, Phase.DOWN]
        idx = clock_wise.index(self.phase)

        if np.array_equal(action, [1, 0, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn l -> u -> r -> d
        elif np.array_equal(action, [0, 0, 1, 0]):
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn l -> d -> r -> u
        else:  # [0, 0, 0, 1]
            next_idx = (idx + 2) % 4
            new_dir = clock_wise[next_idx]  # left turn l -> d -> r -> u

        self.phase = new_dir

        traci.trafficlight.setPhase("J2", self.phase.value)
        self._take_step()
        return self.get_state()  # return new state

    def play_step(self, action):
        self.step_iteration += 1

        old_state = self.get_state()
        new_state = self._move(action)

        reward = self._reward(old_state, new_state)
        return reward