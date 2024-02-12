from numpy import random


class SensorModel:
    def __init__(self, pos_scale: float, yaw_scale: float=0):
        self.pos_scale = pos_scale
        self.yaw_scale = yaw_scale
    
    def update(self, x: float, y: float, yaw: float):
        meas_x = x + random.normal(scale=self.pos_scale)
        meas_y = y + random.normal(scale=self.pos_scale)
        meas_yaw = yaw + random.normal(scale=self.yaw_scale)
        return meas_x, meas_y, meas_yaw