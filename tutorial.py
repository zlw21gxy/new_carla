#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

import glob
import os
import sys
import re
import weakref
try:
    sys.path.append('/home/gu/Documents/carla94/PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg')
except IndexError:
    pass

import carla
import pygame
import random
import time
import numpy as np

def main():
    actor_list = []

    try:

        port = 2000
        client = carla.Client('localhost', port)
        client.set_timeout(2.0)


        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = random.choice(blueprint_library.filter('vehicle'))

        transform = random.choice(world.get_map().get_spawn_points())

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, transform)

        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True)

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        camera_bp = blueprint_library.find('sensor.camera.depth')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera2 = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera2)
        print('created %s' % camera2.type_id)

        cc = carla.ColorConverter.LogarithmicDepth
        # add lane invasion sensors
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        invasion_sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        invasion_sensor.listen(lambda event: _parse_invasion(event))
        actor_list.append(invasion_sensor)
        camera.listen(lambda image: _parse_image(image, cc))
        camera2.listen(lambda image: _parse_image(image, cc))



        map = world.get_map()
        waypoint = map.get_waypoint(vehicle.get_location())
        waypoint = random.choice(waypoint.next(12.0))
        print(waypoint.transform)


        time.sleep(2)

    finally:

        # print('destroying actors')
        for actor in actor_list:
            # print(actor)
            actor.destroy()
        # print('done.')


def _parse_image(image, cc):
    image.convert(cc)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))

    array = array[:, :, -2:-5:-1]

    # surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    # print(array.shape)
    print("parse_image")
#
def _parse_invasion(event):
    # text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
    # event
    print(str(event))


if __name__ == '__main__':
    main()
