# pylint: skip-file
from csv import reader
from dataclasses import dataclass
from math import radians
from numpy import random
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from kinematic_bicycle_model.kinematic_model import KinematicBicycleModel
from kinematic_bicycle_model.sensor_model import SensorModel
from models import StateEstimator
from kinematic_bicycle_model.libs import CarDescription, StanleyController, generate_cubic_spline
from models import CNN


class Simulation:

    def __init__(self):

        fps = 50.0

        self.dt = 1/fps
        self.map_size_x = 30
        self.map_size_y = 20
        self.frames = 600
        self.loop = False


class Path:

    def __init__(self):

        # Get path to waypoints.csv
        with open('./kinematic_bicycle_model/data/waypoints.csv', newline='') as f:
            rows = list(reader(f, delimiter=','))

        ds = 0.05
        x, y = [[float(i) for i in row] for row in zip(*rows[1:])]
        self.px, self.py, self.pyaw, _ = generate_cubic_spline(x, y, ds)


class Car:

    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, delta_time):

        # Model parameters
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.delta_time = delta_time
        self.time = 0.0
        self.velocity = 10.0
        self.wheel_angle = 0.0
        self.angular_velocity = 0.0
        max_steer = radians(33)
        wheelbase = 2.96
        pos_scale = 1.0

        # Acceleration parameters
        target_velocity = 10.0
        self.time_to_reach_target_velocity = 5.0
        self.required_acceleration = target_velocity / self.time_to_reach_target_velocity

        # Tracker parameters
        self.px = px
        self.py = py
        self.pyaw = pyaw
        self.k = 8.0
        self.ksoft = 1.0
        self.kyaw = 0.01
        self.ksteer = 0.0
        self.crosstrack_error = None
        self.target_id = None

        # State estimator parameters
        model_path = 'nn_model.pt'
        window_size = (100,2)

        # Description parameters
        self.colour = 'black'
        overall_length = 4.97
        overall_width = 1.964
        tyre_diameter = 0.4826
        tyre_width = 0.265
        axle_track = 1.7
        rear_overhang = 0.5 * (overall_length - wheelbase)

        self.tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, max_steer, wheelbase, self.px, self.py, self.pyaw)
        self.kinematic_bicycle_model = KinematicBicycleModel(wheelbase, max_steer, self.delta_time)
        self.sensor_model = SensorModel(pos_scale)
        self.state_estimator = StateEstimator(model_path, window_size)
        self.description = CarDescription(overall_length, overall_width, rear_overhang, tyre_diameter, tyre_width, axle_track, wheelbase)
        self.x_buffer = np.full(200, np.nan)
        self.y_buffer = np.full(200, np.nan)
        self.x_estimated = np.full(200, np.nan)
        self.y_estimated = np.full(200, np.nan)

    
    def get_required_acceleration(self):

        self.time += self.delta_time
        return self.required_acceleration
    

    def plot_car(self):
        
        return self.description.plot_car(self.x, self.y, self.yaw, self.wheel_angle)

    
    def drive(self):
        
        acceleration = 0
        self.wheel_angle, self.target_id, self.crosstrack_error = self.tracker.stanley_control(self.x, self.y, self.yaw, self.velocity, self.wheel_angle)
        self.x, self.y, self.yaw, self.velocity, _, _ = self.kinematic_bicycle_model.update(self.x, self.y, self.yaw, self.velocity, acceleration, self.wheel_angle)
        x_meas, y_meas, _ = self.sensor_model.update(self.x, self.y, self.yaw)
        self.x_buffer = np.roll(self.x_buffer, -1)
        self.x_buffer[-1] = x_meas
        self.y_buffer = np.roll(self.y_buffer, -1)
        self.y_buffer[-1] = y_meas

        self.x_estimated = np.roll(self.x_estimated, -1)
        self.y_estimated = np.roll(self.y_estimated, -1)


        self.x_estimated[-1], self.y_estimated[-1] = self.state_estimator.update([x_meas,y_meas])
        print(f"Cross-track term: {self.crosstrack_error}{' '*10}", end="\r")


@dataclass
class Fargs:
    ax: plt.Axes
    sim: Simulation
    path: Path
    car: Car
    car_outline: plt.Line2D
    front_right_wheel: plt.Line2D
    front_left_wheel: plt.Line2D
    rear_right_wheel: plt.Line2D
    rear_left_wheel: plt.Line2D
    rear_axle: plt.Line2D
    measured_position: plt.Line2D
    estimated_position: plt.Line2D
    annotation: plt.Annotation
    target: plt.Line2D
   

def animate(frame, fargs):

    ax                 = fargs.ax
    sim                = fargs.sim
    path               = fargs.path
    car                = fargs.car
    car_outline        = fargs.car_outline
    front_right_wheel  = fargs.front_right_wheel
    front_left_wheel   = fargs.front_left_wheel
    rear_right_wheel   = fargs.rear_right_wheel
    rear_left_wheel    = fargs.rear_left_wheel
    rear_axle          = fargs.rear_axle
    measured_position  = fargs.measured_position
    estimated_position = fargs.estimated_position
    annotation         = fargs.annotation
    target             = fargs.target

    # Camera tracks car
    ax.set_xlim(car.x - sim.map_size_x, car.x + sim.map_size_x)
    ax.set_ylim(car.y - sim.map_size_y, car.y + sim.map_size_y)

    # Drive and draw car
    car.drive()
    outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = car.plot_car()
    car_outline.set_data(*outline_plot)
    front_right_wheel.set_data(*fr_plot)
    rear_right_wheel.set_data(*rr_plot)
    front_left_wheel.set_data(*fl_plot)
    rear_left_wheel.set_data(*rl_plot)
    rear_axle.set_data(car.x, car.y)
    measured_position.set_data(car.x_buffer, car.y_buffer)
    estimated_position.set_data(car.x_estimated, car.y_estimated)

    # Show car's target
    target.set_data(path.px[car.target_id], path.py[car.target_id])

    # Annotate car's coordinate above car
    annotation.set_text(f'{car.x:.1f}, {car.y:.1f}')
    annotation.set_position((car.x, car.y + 5))

    plt.title(f'{sim.dt*frame:.2f}s', loc='right')
    plt.xlabel(f'Speed: {car.velocity:.2f} m/s', loc='left')
    # plt.savefig(f'image/visualisation_{frame:03}.png', dpi=300)

    return car_outline, front_right_wheel, rear_right_wheel, front_left_wheel, rear_left_wheel, rear_axle, target,


def main():
    
    sim  = Simulation()
    path = Path()
    car  = Car(path.px[0], path.py[0], path.pyaw[0], path.px, path.py, path.pyaw, sim.dt)

    interval = sim.dt * 1000

    fig = plt.figure(figsize =(12,12))
    ax = plt.axes()
    ax.set_aspect('equal')

    road = plt.Circle((0, 0), 50, color='lightgray', fill=False, linewidth=30)
    ax.add_patch(road)
    # ax.plot(path.px, path.py, '--', color='gold')

    empty               = ([], [])
    target,             = ax.plot(*empty, '+r')
    car_outline,        = ax.plot(*empty, color=car.colour)
    front_right_wheel,  = ax.plot(*empty, color=car.colour)
    rear_right_wheel,   = ax.plot(*empty, color=car.colour)
    front_left_wheel,   = ax.plot(*empty, color=car.colour)
    rear_left_wheel,    = ax.plot(*empty, color=car.colour)
    rear_axle,          = ax.plot(car.x, car.y, '+', color=car.colour, markersize=2)
    measured_position,  = ax.plot(car.x, car.y, '+', color='blue', markersize=3, label='Measured Position')
    estimated_position, = ax.plot(car.x, car.y, '-', color='orange', markersize=3, label='Estimated Position')
    ax.legend(loc='upper right')
    annotation          = ax.annotate(f'{car.x:.1f}, {car.y:.1f}', xy=(car.x, car.y + 5), color='black', annotation_clip=False)

    fargs = [Fargs(
        ax=ax,
        sim=sim,
        path=path,
        car=car,
        car_outline=car_outline,
        front_right_wheel=front_right_wheel,
        front_left_wheel=front_left_wheel,
        rear_right_wheel=rear_right_wheel,
        rear_left_wheel=rear_left_wheel,
        rear_axle=rear_axle,
        measured_position=measured_position,
        estimated_position=estimated_position,
        annotation=annotation,
        target=target
    )]

    anim = FuncAnimation(fig, animate, frames=sim.frames, init_func=lambda: None, fargs=fargs, interval=interval, repeat=sim.loop)
    anim.save('animation.gif', fps=50)
    
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
