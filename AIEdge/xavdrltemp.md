# NVidia Xavier NX Temp Warning using LED using Deep RL

## Monitor Temperature and glow LED if temp is too high

## Pre-requsities

- NVidia Xavier NX Dev Kit
- Bread Board
- LED Diode
- Cables to connect from GPIO to BreadBoard
- Using Pin 28, 29
- 28 is output pin
- 29 is ground pin

## Steps

1. Connect LED to BreadBoard
2. Connect pin 28 to a Pin in breadboard - i am using 35th row
3. Connect pin 29 to a Pin in breadboard - using 33rd row
4. LED Diode positive(+) is connected to 35th row and negative(-) is connected to 33rd row
5. Pin 1 is 3V power
6. Pin 9 is another ground pin
7. I am using Pin 1 and 9 to power a 3V DC motor
8. Now install the following packages
9. Create a python environment to install tensorflow and keras

```
python3 -m virtualenv -p python tfenv
source tfenv/bin/activate
```

```
pip3 install -U numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta setuptools testresources
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow==2.10.0
```

10. Now install other libraries

```
pip install json_tricks
pip install objdict
sudo pip install jetson-stats
pip install openai
pip install gym
sudo pip install Jetson.GPIO
pip install Jetson.GPIO
pip install matplotlib
```

11. Check and see if tensorflow 2.10 is installed

```
python
import tensorflow as tf
```

12. No errors with tensorflow 2.10, then proceed with the code
13. Let's create custom environment for temperature control
14. Then we create the Deep learning neural network architecture
15. Custom environment will be the new Reinforcement Learning environment for temperature control
16. Conditions = if greater than 40 degree C then turn on LED
17. other wise switch off LED
18. Here is the entire python code to run in AI Edge compute ie NVidia Xavier NX

```
import numpy as np
import gym
from gym import spaces
import logging
import numpy as np
import random
import numpy as np
from collections import deque
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam
import os
#from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from jtop import jtop
import RPi.GPIO as GPIO
import time


class CabinSimulatorEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    cabintemp = 65
    outsidetemp = 66
    surfacetemp = 70
    outlettemp =72
    action = 2

    def __init__(self):
        self.logger = logging.getLogger("Logger")
        self.step_logger = logging.getLogger("StepLogger")
        self.__version__ = "0.0.1"
        self.logger.info(f"CabinSimulatorEnv - Version {self.__version__}")
        # Define the action_space
        # Set temp 
        # action 
        # 1 = increase
        # 2 = decrease
        # = = do nothing
        n_actions = 1
        self.action_space = spaces.Discrete(n_actions)

        # Define the observation_space
        # First dimension is CabinTemp (50..78)
        # Second dimension is outside temp in the air (10...110)
        # Third dimension is surface temperature (10..100)
        # Fourth dimension is outlet temperature (50..100)
        # Fifth dimension is climate temperature (30..100)
        low = np.array([0, 400, -100])
        high = np.array([3, 3000, 100])
        self.observation_space = spaces.Box(low=0, high=3,
                                        shape=(1,), dtype=np.float32)
    
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        # Initialize the agent at the right of the grid
        self.agent_pos = 0
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32)

    def step(self, cabintemp , outsidetemp, surfacetemp,outlettemp):
        action = 0
        if cabintemp > 40 and cabintemp < 46:
            action = 1
        elif cabintemp > 46:
            action = 2
        else:
            action = 0
            # raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # print('cabintemp=', cabintemp, 'outsidetemp=', outsidetemp, 'surfacetemp=', surfacetemp, 'outlettemp=', outlettemp, 'Action=', action)
        # Account for the boundaries of the grid
        # self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        # done = bool(self.agent_pos == 0)
        done = False

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if action >= 1 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        # print(self.action, reward, done, info)

        return np.array([action]).astype(np.float32), reward, done, info
    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("." * self.agent_pos, end="")
        print("x", end="")
        # print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass
 
env = CabinSimulatorEnv()

aggr_ep = { 'obs': [], 'reward': [], 'step': [], 'cabintemp': []}

import random

aggr_ep = { 'obs': [], 'reward': [], 'step': [], 'cabintemp': []}

env = CabinSimulatorEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
obs = env.reset()
env.render()

cabintemp = random.randrange(50, 80, 3)
outsidetemp = random.randrange(10, 110, 3)
outlettemp = random.randrange(50, 90, 3)
surfacetemp = random.randrange(50, 100, 3)

# Pin Definitions
output_pin = 29  # BCM pin 18, BOARD pin 12

batch_size = 32
n_episodes = 100
output_dir = "model_output/cabintemp/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
 
    def _build_model(self):
        model = Sequential() 
        model.add(Dense(32, activation="relu",
                        input_dim=self.state_size))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse",
                     optimizer=Adam(lr=self.learning_rate))
        return model
 
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action,
                            reward, next_state, done))

    def train(self, batch_size):
         minibatch = random.sample(self.memory, batch_size)
         for state, action, reward, next_state, done in minibatch:
            target = reward # if done 
            if not done:
                target = (reward +
                          self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) 
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size) 
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def save(self, name): 
        self.model.save_weights(name)
        
    def get_qs(self, state, step):
        return self.model_predict(np.array(state).reshape(-1, *state.shape/255([0])))
 
def getcurrentstatus():
    cabintemp = random.randrange(50, 80, 3)
    outsidetemp = random.randrange(10, 110, 3)
    outlettemp = random.randrange(50, 90, 3)
    surfacetemp = random.randrange(50, 100, 3)

    return cabintemp , outsidetemp, surfacetemp,outlettemp

def main():
    agent = DQNAgent(state_size, action_size)
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)
    print(' GPIO info ', GPIO.JETSON_INFO)
    print(' GPIO info ', GPIO.VERSION)
    curr_value = GPIO.LOW
    state = 0
    for e in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False 
        
        time = 0
        while not done:
            #env.render()
            action = agent.act(state)
            #cabintemp = random.randrange(50, 80, 3)
            cabintemp = 0
            outsidetemp = random.randrange(10, 110, 3)
            outlettemp = random.randrange(50, 90, 3)
            surfacetemp = random.randrange(50, 100, 3)
            with jtop() as jetson:
                xavier_nx = jetson.stats
                GPU_temperature = xavier_nx['Temp GPU']
                CPU_temperature = xavier_nx['Temp CPU']
                Thermal_temperature = xavier_nx['Temp thermal']
                cabintemp = GPU_temperature
            # print(env)
            # obs, reward, done, info = env.step(cabintemp , outsidetemp, surfacetemp,outlettemp)
            next_state, reward, done, _ = env.step(cabintemp , outsidetemp, surfacetemp,outlettemp)
            #print(next_state)
            reward = reward if not done else 0
            next_state = np.reshape(next_state, [1, state_size]) 
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            done = True
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}, reward: {}, state: {}"
                .format(e, n_episodes-1, time, agent.epsilon, reward, next_state[0]))
            time += 1
            if next_state[0] >= 1:
                curr_value = GPIO.HIGH
                GPIO.output(output_pin, curr_value)
            else:
                curr_value = GPIO.LOW
                GPIO.output(output_pin, curr_value)
        if len(agent.memory) > batch_size:
            agent.train(batch_size) 
        if e % 50 == 0:
            agent.save(output_dir + "weights_"
                + "{:04d}".format(e) + ".hdf5")

if __name__ == "__main__":
   main()
```

19. Save the python file as rltemp1.py
20. Run the code21. 
22. use elevated mode to run the code to grab temperature information

```
sudo python3 rltemp1.py
```

23. Ignore the warning for now
24. Code should run episodes and print out the temperature information
25. Check the temperature LED glow when the NVIDIA Xavier NX board gets hot
26. Run the Active Training going
27. My sample code runs for 100 episodes