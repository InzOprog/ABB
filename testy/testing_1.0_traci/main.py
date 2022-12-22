#!/usr/bin/env python

import os
import sys
import optparse
import matplotlib.pyplot as plt
from itertools import cycle, islice


# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


from sumolib import checkBinary  # Checks for the binary in environ vars
import traci


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


# contains TraCI control loop
def _phase():
    return traci.trafficlight.getPhase('J2')


def run(_sumoBinary):
    step = 0
    out0 = []
    out1 = []
    out2 = []
    loop = [0, 1, 2, 3]
    #traci.load([_sumoBinary, "-c", "t1.sumocfg", "--tripinfo-output", "tripinfo.xml"])
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        print("← ", traci.edge.getWaitingTime('E0'))
        print("→ ", traci.edge.getWaitingTime('E1'))
        print("↑ ", traci.edge.getWaitingTime('E2'))
        # out0.append(len(traci.edge.getLastStepVehicleIDs('E0')))
        # print("→ ", traci.edge.getLastStepVehicleIDs('E1'))
        # out1.append(len(traci.edge.getLastStepVehicleIDs('E1')))
        # print("↑ ", traci.edge.getLastStepVehicleIDs('E2'))
        # out2.append(len(traci.edge.getLastStepVehicleIDs('E2')))
        # print('\n')

        #det_vehs = traci.inductionloop.getLastStepVehicleIDs("det_0")
        step += 1
        if step % 10 == 0:
            loop.append(loop.pop(0))
            traci.trafficlight.setPhase('J2', loop[0])


    traci.close()
    sys.stdout.flush()
    return out0, out1, out2


# main entry point
if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "t1.sumocfg", "--tripinfo-output", "tripinfo.xml"])
    o0, o1, o2 = run(sumoBinary)


