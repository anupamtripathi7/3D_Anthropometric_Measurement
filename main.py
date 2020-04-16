# TODO: Practicum project
import camera
import numpy as np
import math
c = camera.Camera()
c.set_K_elements(1225.0, math.pi / 2, 1, 480, 384)

c.set_t(np.array([[-1.365061486465], [3.431608806127], [17.74182159488]]))
print(c.world_to_image(np.array([[0., 0., 0.]]).T))
print('heyy')
print("there")
print("there")