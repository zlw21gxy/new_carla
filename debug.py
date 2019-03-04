#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import re
import weakref
try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import pygame
import random
import time
import subprocess
from carla import ColorConverter as cc
import math
import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple
# Default environment configuration
ENV_CONFIG = {
    "log_images": True,
    "enable_planner": True,
    "framestack": 2,  # note: only [1, 2] currently supported
    "convert_images_to_video": True,
    "early_terminate_on_collision": True,
    "verbose": True,
    "reward_function": "custom",
    "render_x_res": 800,
    "render_y_res": 600,
    "x_res": 800,
    "y_res": 800,
    "server_map": "/Game/Maps/Town02",
    "use_depth_camera": False,
    "discrete_actions": True,
    "squash_action_logits": False,
}

DISCRETE_ACTIONS = {
    # coast
    0: [0.0, 0.0],
    # turn left
    1: [0.0, -0.5],
    # turn right
    2: [0.0, 0.5],
    # forward
    3: [1.0, 0.0],
    # brake
    4: [-0.5, 0.0],
    # forward left
    5: [1.0, -0.5],
    # forward right
    6: [1.0, 0.5],
    # brake left
    7: [-0.5, -0.5],
    # brake right
    8: [-0.5, 0.5],
}

# Mapping from string repr to one-hot encoding index to feed to the model
# Some command we want give to agent
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "STOP": 1,
    "LANE_KEEP": 2,
    "TURN_RIGHT": 3,
    "TURN_LEFT": 4,
    "SURPASS": 5
}

class CarlaEnv(gym.Env):
    def __init__(self, config=ENV_CONFIG):
        self.config = config
        self.city = self.config["server_map"].split("/")[-1]
        if self.config["enable_planner"]:
            pass

        if config["discrete_actions"]:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
        else:
            self.action_space = Box(-1.0, 1.0, shape=(2, ), dtype=np.float32)
        if config["use_depth_camera"]:
            image_space = Box(
                -1.0,
                1.0,
                shape=(config["y_res"], config["x_res"],
                       1 * config["framestack"]),
                dtype=np.float32)
        else:
            image_space = Box(
                0,
                255,
                shape=(config["y_res"], config["x_res"],
                       3 * config["framestack"]),
                dtype=np.uint8)
        self.observation_space = Tuple(          # forward_speed, dist to goal
            [
                image_space,
                Discrete(len(COMMAND_ORDINAL)),  # next_command
                Box(-128.0, 128.0, shape=(2, ), dtype=np.float32)
            ])
        # environment config
        self._spec = lambda: None
        self._spec.id = "Carla_v0"
        # experiment config
        self.num_steps = 0
        self.total_reward = 0
        self.episode_id = None
        self.measurements_file = None
        self.weather = None
        # actors
        self.actor_list = []          # save actor list for destroying them after finish
        self.vehicle = None
        self.collision_sensor = None
        self.camera_rgb = None
        # states
        self._history_info = []       # info history
        self._history_collision = []  # collision history
        self._history_invasion = []   # invasion history
        self._image_depth = []        # save a list of depth image
        self._image_rgb = []          # save a list of rgb image
        # server
        self.server_port = 2000
        self.client = carla.Client("localhost", self.server_port)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()

    def restart(self):
        pass

    def reset(self):

        world = self.world
        bp_library = world.get_blueprint_library()

        # setup vehicle
        spawn_point = random.choice(world.get_map().get_spawn_points())
        bp_vehicle = bp_library.find('vehicle.lincoln.mkz2017')
        bp_vehicle.set_attribute('role_name', 'hero')
        self.vehicle = world.spawn_actor(bp_vehicle, spawn_point)
        self.actor_list.append(self.vehicle)

        # setup camera
        camera_transform = carla.Transform(carla.Location(x=0, z=2.4))
        bp_rgb = bp_library.find('sensor.camera.rgb')
        self.camera_rgb = world.spawn_actor(bp_rgb, camera_transform, attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.camera_rgb.listen(lambda image: self._parse_image(weak_self, image, carla.ColorConverter.Raw))
        self.actor_list.append(self.camera_rgb)

        # add collision sensors
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: self._parse_collision(weak_self, event))
        self.actor_list.append(self.collision_sensor)

        time.sleep(0.1)

        return self._image_rgb[-1]

    @staticmethod
    def _parse_image(weak_self, image, cc):
        self = weak_self()
        if not self:
            return
        image.convert(cc)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, -2:-5:-1]
        self._image_rgb.append(array)

    @staticmethod
    def _parse_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._history_collision.append((event.frame_number, intensity))
        if len(self._history) > 40:
            self._history_collision.pop(0)

    def step(self, action):

        def compute_reward(info):
            reward = 0.0
            reward += info["speed"]

            return reward

        done = False
        if self.config["discrete_actions"]:
            action = DISCRETE_ACTIONS[int(action)]

        throttle = float(np.clip(action[0], 0, 1))
        brake = float(np.abs(np.clip(action[0], -1, 0)))
        steer = float(np.clip(action[1], -1, 1))
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        # self.vehicle.apply_control(carla.VehicleControl(throttle=1, brake=0, steer=0))
        # get other measurement
        t = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        c = self.vehicle.get_vehicle_control()
        acceleration = self.vehicle.get_acceleration()

        info = {"speed": math.sqrt(v.x**2 + v.y**2 + v.z**2),  # m/s
                "acceleration": math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2),
                "location_x": t.location.x,
                "location_y": t.location.y,
                "Throttle": c.throttle,
                "Steer": c.steer,
                "Brake": c.brake}

        self._history_info.append(info)

        if len(self._history_info) > 16:
            self._history_info.pop(0)

        reward = compute_reward(info)
        if info["acceleration"] > 20:
            done = True

        return self._image_rgb[-1], reward, done, self._history_info[-1]





if __name__ == '__main__':

    env = CarlaEnv()
    obs = env.reset()
    print(obs.shape)
    # obs, reward, done, info = env.step(3)
    # print(reward)
    for i in range(1000):
        obs, reward, done, info = env.step(3)
        if i % 10 == 0:
            print(reward)

    for actor in env.actor_list:
        actor.destroy()
