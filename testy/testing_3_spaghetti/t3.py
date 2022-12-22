import optparse
import os
import random
import sys

import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
from sumolib import checkBinary
import traci


# --------------------------------------------------------------------------------------------------------------------#
#                                                     SAFETY                                                          #
# --------------------------------------------------------------------------------------------------------------------#


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def getOptions():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


if getOptions().nogui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')


# --------------------------------------------------------------------------------------------------------------------#
#                                                     END                                                             #
# --------------------------------------------------------------------------------------------------------------------#


def getState():  # TODO: Change to car count next to traffic lights recognition, passed car detection
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


def makeMove(_state, _action):
    #if _action != 4:
    traci.trafficlight.setPhase("J2", _action)
    traci.simulationStep()
    traci.simulationStep()
    traci.simulationStep()
    traci.simulationStep()
    return getState()  # return new state


def getReward(_state, _new_state):
    # qLengths1 = _state[:4]
    # qLengths2 = _new_state[:4]
    #
    # q1 = np.average(qLengths1) * np.std(qLengths1)
    # q2 = np.average(qLengths2) * np.std(qLengths2)
    #
    # return (-1 * (q1 >= q2)) + (1 * (q1 < q2))  # return reward (1 || -1)
    _reward = 0
    for _i in range(8):
        if (_state[_i]) >= (_new_state[_i]):
            _reward -= 1
        else:
            _reward += 1
    return _reward


def build_model():
    num_hidden_units_lstm = 10
    num_actions = 4
    _model = Sequential()
    _model.add(LSTM(num_hidden_units_lstm, input_shape=(100, 9)))
    _model.add(Dense(num_actions, activation='linear'))
    opt = RMSprop(lr=0.00025)
    _model.compile(loss='mse', optimizer=opt)
    return _model


num_episode = 400
gamma = 0.99  # discount rate
epsilon = 1  # randomness
buffer = 100
model = build_model()
traci.start([sumoBinary, "-c", "data/simplified.sumocfg", "--tripinfo-output", "tripinfo.xml"])
traci.trafficlight.setPhase("J2", 0)
state = getState()
experience = []

for i in range(100):
    experience.append(state)

for episode in range(num_episode):
    print("Episode # ", episode)
    while traci.simulation.getMinExpectedNumber() > 0:
        # get predicted action value based on experience
        q_val = model.predict((np.array(experience)).reshape((1, 100, 9)))
        # if random(0;1] < 1  - later limit lowered to use experience instead
        if random.random() < epsilon:
            action = np.random.choice(4)
        else:
            action = np.argmax(q_val)

        # advance 4 steps and get new traffic light state
        new_state = makeMove(state, action)
        old_experience = experience
        experience.pop(0)
        experience.append(new_state)
        reward = getReward(state, new_state)
        oracle = np.zeros((1, 4))
        oracle[:] = q_val[:]
        oracle[0][action] = reward + gamma * np.max(model.predict((np.array(experience)).reshape((1, 100, 9))))
        model.fit((np.array(old_experience)).reshape((1, 100, 9)), oracle, verbose=1)
        state = new_state
    if epsilon > 0.1:
        epsilon -= (episode/(4*num_episode))  # TODO: Optimize randomness reduction and learning rate
    model.save('model_save_test.h5')  # TODO: Test save/ load functionality
    traci.load(["--start", "-c", "data/simplified.sumocfg", "--tripinfo-output", "tripinfo.xml"])

# TODO: List:
#       1) Fix disappearing cars
#       2) Make OOP
#       3) Ilość + czas (statystyka)
#       4)
