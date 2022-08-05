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
                uniform vec4 rect;

                out vec4 f_colour;

                vec4 lap() {
                    float x = v_tex_pos.x;
                    float y = v_tex_pos.y;
                    
                    float dx = 1 / rect[2];
                    float dy = 1 / rect[3];
                    
                    vec4 ans = vec4(0, 0, 0, 0);
                    ans = ans - 3 * texture2D(old_state, vec2(x, y));
                    ans = ans + 0.5 * texture2D(old_state, vec2(x + dx, y));
                    ans = ans + 0.5 * texture2D(old_state, vec2(x - dx, y));
                    ans = ans + 0.5 * texture2D(old_state, vec2(x, y + dy));
                    ans = ans + 0.5 * texture2D(old_state, vec2(x, y - dy));
                    ans = ans + 0.25 * texture2D(old_state, vec2(x + dx, y + dy));
                    ans = ans + 0.25 * texture2D(old_state, vec2(x - dx, y + dy));
                    ans = ans + 0.25 * texture2D(old_state, vec2(x + dx, y - dy));
                    ans = ans + 0.25 * texture2D(old_state, vec2(x - dx, y - dy));
                    return ans;
                }

                void main() {
                    float DA = 1;
                    float DB = 0.5;
                    float f = 0.05;
                    float k = 0.062;
                    float dt = 0.1;
                    
                    vec4 lap_eval = lap();
                    float A = texture2D(old_state, v_tex_pos)[0];
                    float B = texture2D(old_state, v_tex_pos)[1];
                    float lap_A = lap_eval[0];
                    float lap_B = lap_eval[1];
                    
                    float A_new = A + dt * (DA * lap_A - A * B * B + f * (1 - A));
                    float B_new = B + dt * (DB * lap_B + A * B * B - (k + f) * B);

                    if (abs(v_unit_pos.x) < 0.02 && abs(v_unit_pos.y) < 0.02) {
                        B_new += 0.001;
                    }
                
                    f_colour = vec4(A_new, B_new, 0, 0);
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
        self.tex1.repeat_x = False
        self.tex1.repeat_y = False
        self.fbo1 = self.ctx.framebuffer(self.tex1)
        self.tex2 = self.ctx.texture([1, 1], 4, dtype = "f4")
        self.tex2.repeat_x = False
        self.tex2.repeat_y = False
        self.fbo2 = self.ctx.framebuffer(self.tex2)
        
        super().__init__(*args, **kwargs)


    def set_rect(self, rect):
        super().set_rect(rect)
        self.tex1.release()
        self.fbo1.release()
        self.tex2.release()
        self.fbo2.release()

        self.tex1 = self.ctx.texture(rect[2:4], 4, dtype = "f4")
        self.tex1.repeat_x = False
        self.tex1.repeat_y = False
        self.fbo1 = self.ctx.framebuffer(self.tex1)
        
        self.tex2 = self.ctx.texture(rect[2:4], 4, dtype = "f4")
        self.tex2.repeat_x = False
        self.tex2.repeat_y = False
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

        for _ in range(20):
            self.tex1.use(0)
            self.fbo2.use()
            self.step_vao.render(moderngl.TRIANGLES, instances = 1)
            self.fbo1.use()
            pgbase.tools.render_tex(self.tex2)
        #replace this with a nicer renderer
        self.ctx.screen.use()
        self.ctx.clear(0, 0, 0, 0)
        pgbase.tools.render_tex(self.tex1)
        


def run():
    pgbase.core.Window.setup(size = [1000, 1000])
    pgbase.core.run_root(Viewer())
    pgbase.core.Window.quit()
    






















	












	
