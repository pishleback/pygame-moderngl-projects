import pgbase
import numpy as np
import math
import pygame
import moderngl



class Window2D(pgbase.core.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center = np.array([0, 0])

        self.trans_mat = np.eye(2)
        self.aspect_vec = None

    def set_rect(self, rect):
        super().set_rect(rect)
        av = math.sqrt(self.width * self.height)
        self.aspect_vec = np.array([self.width / av, self.height / av])

    @property
    def cam_mat(self):
        return self.trans_mat * self.aspect_vec[None, :]

    def pygame_to_gl(self, pos):
        return np.array([2 * pos[0] / self.width - 1, 1 - 2 * pos[1] / self.height])
    def gl_to_pygame(self, pos):
        return np.array([self.width * (0.5 + 0.5 * pos[0]), self.height * (0.5 - 0.5 * pos[1])])

    def gl_to_world(self, pos):
        return self.cam_mat @ np.array(pos) + self.center
    def world_to_gl(self, pos):
        return np.linalg.inv(self.cam_mat) @ np.array(np.array(pos) - self.center)

    def pygame_to_world(self, pos):
        return self.gl_to_world(self.pygame_to_gl(pos))
    def world_to_pygame(self, pos):
        return self.gl_to_pygame(self.world_to_gl(pos))

    def set_uniforms(self, progs):
        cam_mat = self.cam_mat
        cam_mat_inv = np.linalg.inv(cam_mat)
        cam_mat = cam_mat.transpose()
        cam_mat_inv = cam_mat_inv.transpose()
        for prog in progs:
            try:
                prog["cam_center"].value = tuple(self.center)
            except KeyError:
                pass
            try:
                prog["cam_mat"].value = tuple(cam_mat.flatten())
            except KeyError:
                pass
            try:
                prog["cam_mat_inv"].value = tuple(cam_mat_inv.flatten())
            except KeyError:
                pass

    def event(self, event):
        super().event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in {4, 5}:
                mult = 1.2
                if event.button == 4:
                    mult = 1 / mult

                c1 = self.pygame_to_world(event.pos)
                self.trans_mat = self.trans_mat * mult
                c2 = self.pygame_to_world(event.pos)
                self.center = self.center + c1 - c2

##            if event.button == 1:
##                import math
##                a = 0.1
##                c = math.cos(a)
##                s = math.sin(a)
##                rot = np.array([[c, -s],
##                                [s, c]])
##                self.trans_mat = self.trans_mat @ rot




	
