from numpy import random
import numpy as np

class SmoothRandom:

    def __init__(self, limit):
        
        """
        Smooth Random Timeseries Generator

        At initialisation
        :param control_gain:                (float) time constant [1/s]
        :param softening_gain:              (float) softening gain [m/s]
        :param yaw_rate_gain:               (float) yaw rate gain [rad]
        :param steering_damp_gain:          (float) steering damp gain
        :param max_steer:                   (float) vehicle's steering limits [rad]
        :param wheelbase:                   (float) vehicle's wheelbase [m]
        :param path_x:                      (numpy.ndarray) list of x-coordinates along the path
        :param path_y:                      (numpy.ndarray) list of y-coordinates along the path
        :param path_yaw:                    (numpy.ndarray) list of discrete yaw values along the path
        :param dt:                          (float) discrete time period [s]

        At every time step
        :param x:                           (float) vehicle's x-coordinate [m]
        :param y:                           (float) vehicle's y-coordinate [m]
        :param yaw:                         (float) vehicle's heading [rad]
        :param target_velocity:             (float) vehicle's velocity [m/s]
        :param steering_angle:              (float) vehicle's steering angle [rad]

        :return limited_steering_angle:     (float) steering angle after imposing steering limits [rad]
        :return target_index:               (int) closest path index
        :return crosstrack_error:           (float) distance from closest path index [m]
        """

        # self.k = control_gain
        # self.k_soft = softening_gain
        # self.k_yaw_rate = yaw_rate_gain
        # self.k_damp_steer = steering_damp_gain
        # self.max_steer = max_steer
        # self.wheelbase = wheelbase

        # self.px = path_x
        # self.py = path_y
        # self.pyaw = path_yaw
        self.state = 0
        self.limit = limit

    def generate(self):
        x = random.normal()
        self.state = 0.9*self.state + 0.1*x
        self.state = np.clip(self.state, -self.limit, self.limit)
        return self.state

        # y = np.zeros_like(x)
        # for i in range(x.shape[0]-1):
        #     y[i+1] = y[i] + x[i]
        #     if y[i+1] > 2:
        #         y[i+1] = 2
        #     if y[i+1] < -2:
        #         y[i+1] = -2

        # return y
    