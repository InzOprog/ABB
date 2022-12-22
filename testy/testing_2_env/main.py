import os
import sys
import optparse
from collections import deque
import numpy as np
import random
from sumolib import checkBinary  # Checks for the binary in environ vars
import traci
import torch
from model import LinearDeepQNetwork, Trainer

# --------------------------------------------------------------------------------------------------------------------#
#                                                     DQN AGENT BEGIN                                                 #
# --------------------------------------------------------------------------------------------------------------------#

__MAX_MEMORY__ = 100_000
__BATCH_SIZE__ = 1000
__LR__ = 0.001
__EPOCHS__ = 200
__ACTIONS__ = [0, 1, 2, 3]
__TRANS__ = {'g': 0, 'G': 1, 'y': 2, 'Y': 3, 'r': 4, 'R': 5}


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount factor
        self.memory = deque(maxlen=__MAX_MEMORY__)  # popleft()
        self.model = LinearDeepQNetwork(9, 256, 4)
        self.trainer = Trainer(self.model, lr=__LR__, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > __BATCH_SIZE__:
            mini_sample = random.sample(self.memory, __BATCH_SIZE__)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            phase = random.choice(__ACTIONS__)
        else:
            prediction = self.model(torch.tensor(state, dtype=torch.float32))
            phase = torch.argmax(prediction).item()
        return phase


def getState():
    trafficLightState = traci.trafficlight.getPhase("J2")
    state = np.zeros(9, dtype=np.float32)
    state[0] = traci.edge.getWaitingTime('E0'),  # ← wait s
    state[1] = traci.edge.getWaitingTime('E1'),  # → wait s
    state[2] = traci.edge.getWaitingTime('E2'),  # ↑ wait s

    # state[0, 0] = traci.edge.getLastStepVehicleIDs('E0')  # ← cars
    # state[1, 0] = traci.edge.getLastStepVehicleIDs('E1')  # → cars
    # state[2, 0] = traci.edge.getLastStepVehicleIDs('E2')  # ↑ cars
    for n, e in enumerate(trafficLightState):
        state[n + 3] = __TRANS__[e]

    return state  # np.array[dtype=np.float32, __len__=9]


def nStep(step_amount):
    for _ in range(step_amount):
        traci.simulationStep()


def makeMove(action):
    traci.trafficlight.setPhase("J2", action)
    nStep(10)  # TODO: Play around with step amount
    newState = getState()
    return newState


def getReward(old_state, new_state):
    avgWaitTime1 = np.average(old_state[:4], axis=1)[1]
    avgWaitTime2 = np.average(new_state[:4], axis=1)[1]

    reward = 1
    reward -= (2 * (avgWaitTime1 >= avgWaitTime2))

    return reward


def train():
    # plot_scores = []
    # plot_mean_scores = []
    # total_score = 0
    # record = 0
    agent = Agent()
    # game = run()
    for i in range(__EPOCHS__):
        # get old state
        state = getState()

        # cars_passed = 0
        # cars_stopped = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            move = agent.get_action(state)
            new_state = makeMove(move)
            reward = getReward(state, new_state)

            # train short memory
            agent.train_short_memory(state, move, reward, new_state, done)

            # remember
            agent.remember(state, move, reward, new_state, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                print("Model saved for epoch # ", i)

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
    traci.load(["--start", "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])


# --------------------------------------------------------------------------------------------------------------------#
#                                                     DQN AGENT END                                                   #
# --------------------------------------------------------------------------------------------------------------------#

# main entry point
if __name__ == "__main__":
    # -----------------------------------------------------------------------------------------------------------------#
    #                                              SUMO BINARY CHECK BEGIN                                             #
    # -----------------------------------------------------------------------------------------------------------------#
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    # -----------------------------------------------------------------------------------------------------------------#
    #                                               SUMO BINARY CHECK END                                              #
    # -----------------------------------------------------------------------------------------------------------------#

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary,  "-c", "data/t1.sumocfg", "--tripinfo-output", "data/tripinfo.xml"])
    train()
    traci.close()
    sys.stdout.flush()
