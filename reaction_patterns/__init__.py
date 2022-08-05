import pygame
import moderngl
import numpy as np
import sys
import math
import imageio
import time
import pgbase
import os
import random






class Viewer(pgbase.canvas2d.Window2D):
    def __init__(self, *args, **kwargs):
        #given the current state in the form of a texture (bound to 0), render the next state to the output.
        self.step_prog = self.ctx.program(
            vertex_shader = """
                #version 430
                in vec2 unit_pos;
                out vec2 v_unit_pos;
                out vec2 v_tex_pos;

                void main() {
                    gl_Position = vec4(unit_pos, 0, 1);
                    v_unit_pos = unit_pos;
                    v_tex_pos = (unit_pos + 1) / 2;
                }
            """,
            fragment_shader = """
                #version 430
                in vec2 v_unit_pos;
                in vec2 v_tex_pos;

                uniform sampler2D old_state;
                uniform vec2 cam_center;
                uniform mat2 cam_mat;

                out vec4 f_colour;

                void main() {
                    // texture2D(old_state, v_tex_pos) + 0.01 * vec4(v_tex_pos, 0, 0);
                    
                    float A0;
                    float B0;
                    
                    float A1 = A0 + ;
                    float B1 = B0 + ;
                
                    f_colour = vec4(A0, A1, 0, 0);
                }
    
            """,
        )
        try: self.step_prog["old_state"].value = 0
        except KeyError: pass

        vertices = self.ctx.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]).astype('f4'))
        indices = self.ctx.buffer(np.array([[0, 1, 3], [0, 2, 3]]))
        self.step_vao = self.ctx.vertex_array(self.step_prog,
                                              [(vertices, "2f4", "unit_pos")],
                                              indices)
        
        self.tex1 = self.ctx.texture([1, 1], 4, dtype = "f4")
        self.fbo1 = self.ctx.framebuffer(self.tex1)
        self.tex2 = self.ctx.texture([1, 1], 4, dtype = "f4")
        self.fbo2 = self.ctx.framebuffer(self.tex2)
        
        super().__init__(*args, **kwargs)


    def set_rect(self, rect):
        super().set_rect(rect)
        self.tex1.release()
        self.fbo1.release()
        self.tex2.release()
        self.fbo2.release()

        self.tex1 = self.ctx.texture(rect[2:4], 4, dtype = "f4")
        self.fbo1 = self.ctx.framebuffer(self.tex1)
        self.tex2 = self.ctx.texture(rect[2:4], 4, dtype = "f4")
        self.fbo2 = self.ctx.framebuffer(self.tex2)
        
    def __del__(self):
        self.step_vao.release()

        self.tex1.release()
        self.fbo1.release()
        self.tex2.release()
        self.fbo2.release()


    def event(self, event):
        if event.type == pygame.KEYDOWN:
            print(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            print(event)
        super().event(event)
            
    def draw(self):
        super().set_uniforms([self.step_prog])

        for _ in range(1):
            self.tex1.use(0)
            self.fbo2.use()
            self.step_vao.render(moderngl.TRIANGLES, instances = 1)
            self.fbo1.use()
            pgbase.tools.render_tex(self.tex2)
        self.ctx.screen.use()
        self.ctx.clear(0, 0, 0, 0)
        pgbase.tools.render_tex(self.tex1)
        


def run():
    pgbase.core.Window.setup(size = [2000, 1600])
    pgbase.core.run_root(Viewer())
    pgbase.core.Window.quit()
    






















	












	
