# pylint: skip-file
from math import radians
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from kinematic_bicycle_model.kinematic_model import KinematicBicycleModel
from kinematic_bicycle_model.sensor_model import SensorModel
from kinematic_bicycle_model.libs import SmoothRandom


# Parameters
max_steer = radians(33)
wheelbase = 2.96
velocity = 10
pos_scale = 1.0
interval = 200
dt = 1/50

kinematic_bicycle_model = KinematicBicycleModel(wheelbase, max_steer, dt)
sensor_model = SensorModel(pos_scale)
smooth_random = SmoothRandom(max_steer)
num_steps = int(interval/dt)
measurements = pd.DataFrame(np.nan, index=range(num_steps), columns=['X', 'Y', 'YAW'])
states = pd.DataFrame(np.nan, index=range(num_steps), columns=['X', 'Y', 'YAW'])

x, y, yaw = 0, 0, 0 
for i in range(num_steps):
    states.loc[i] = [x, y, yaw]
    measurements.loc[i] = sensor_model.update(x, y, yaw)
    wheel_angle = smooth_random.generate()
    x, y, yaw, velocity, _, _ = kinematic_bicycle_model.update(x, y, yaw, velocity, 0, wheel_angle)

measurements.to_csv('./data/measurements.csv')
states.to_csv('./data/states.csv')