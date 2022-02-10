import pygame
import moderngl
import numpy as np
import sys
import math
import imageio
import time
import pgbase
import os







class MandelbrotBase(pgbase.Window2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prog = self.get_prog()

        vertices = self.ctx.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]).astype('f4'))
        indices = self.ctx.buffer(np.array([[0, 1, 3], [0, 2, 3]]))
        self.vao = self.ctx.vertex_array(self.prog,
                                         [(vertices, "2f4", "unit_pos")],
                                         indices)

        N = 1

        self.ignore_tex = self.ctx.texture([N * self.width, N * self.height], 4)
        self.ignore_fbo = self.ctx.framebuffer(self.ignore_tex)
        self.colour_tex = self.ctx.texture([N * self.width, N * self.height], 4)
        self.colour_fbo = self.ctx.framebuffer(self.colour_tex)

        self.bg_tex = self.ctx.texture([N * self.width, N * self.height], 4)
        self.bg_fbo = self.ctx.framebuffer(self.bg_tex)
        self.bg_poses = [[-1, -1], [1, 1]]
        self.bg_timeout = 3

        self.palette_tex = pgbase.load_tex(self.ctx, os.path.join("mandel", "water.jpg"))

        self.done_iter = 0
        self.targ_iter = 0

        self.last_darw_time = time.time()
        self.last_user_time = time.time()

        self.zoom_clear()

    def __del__(self):
        self.vao.release()
        
        self.ignore_tex.release()
        self.ignore_fbo.release()
        self.colour_tex.release()
        self.colour_fbo.release()
        
        self.bg_tex.release()
        self.palette_tex.release()

    def zoom_clear(self):
        self.done_iter = 0
        self.targ_iter = 10
        self.ignore_fbo.clear(0, 0, 0, 0)
        self.colour_fbo.clear(0, 0, 0, 0)

    def full_clear(self):
        self.zoom_clear()
        self.bg_fbo.clear(0, 0, 0, 0)

    def event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in {4, 5}:
                if time.time() - self.last_user_time > self.bg_timeout:
                    self.bg_poses = [self.gl_to_world([-1, -1]), self.gl_to_world([1, 1])]
                    self.bg_fbo.use()
                    ctx.clear(0, 0, 0, 0)
                    pgbase.render_tex(self.colour_tex)
                self.zoom_clear()
                self.last_user_time = time.time()
        super().event(event)
            
    def draw(self, ctx):
        super().set_uniforms([self.prog])
        if self.done_iter < self.targ_iter:
            self.prog["iter"].value = self.targ_iter
            
            self.colour_fbo.use()
            self.ignore_tex.use(0)
            self.palette_tex.use(1)
            self.vao.render(moderngl.TRIANGLES, instances = 1)

            self.ignore_fbo.use()
            pgbase.render_tex(self.colour_tex)

            self.done_iter = self.targ_iter

            t = time.time()
            if t - self.last_darw_time < 1:
                self.targ_iter = min(math.floor(1.1 * self.targ_iter), 10 ** 6)
            else:
                print(self.targ_iter)

            self.last_darw_time = t


        ctx.screen.use()
        ctx.enable_only(moderngl.BLEND)
        ctx.clear(0, 0, 0, 0)

        if time.time() - self.last_user_time < self.bg_timeout:
            p1 = self.world_to_gl(self.bg_poses[0])
            p2 = self.world_to_gl(self.bg_poses[1])
            pgbase.render_tex(self.bg_tex, p1, p2)

        pgbase.render_tex(self.colour_tex)




class Mandelbrot(MandelbrotBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_prog(self):
        prog = self.ctx.program(
            vertex_shader="""
                #version 430
                in vec2 unit_pos;
                out vec2 v_tex_pos;
                out vec2 v_unit_pos;

                void main() {
                    gl_Position = vec4(unit_pos, 0.0, 1.0);
                    v_tex_pos = 0.5 + 0.5 * unit_pos;
                    v_unit_pos = unit_pos;
                }
            """,
            fragment_shader="""
                #version 430
                in vec2 v_unit_pos;
                in vec2 v_tex_pos;
                out vec4 f_colour;

                uniform int iter;
                uniform sampler2D tex;
                uniform sampler2D palette;
                uniform dvec2 cam_center;
                uniform dmat2 cam_mat;

                dvec2 mult(dvec2 a, dvec2 b) {
                    return dvec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
                }

                void main() {
                    float colour_scale = 0.1;
                
                    float colour_scale_x = colour_scale;
                    float colour_offset_x = 0.1;
                    float colour_scale_y = colour_scale * (1 + sqrt(5)) / 2;
                    float colour_offset_y = 0.1;
                    
                    if (texture2D(tex, v_tex_pos).a > 0.5) {
                        discard;
                    } else {
                        int i;
                        dvec2 c = cam_mat * dvec2(v_unit_pos) + cam_center;
                        dvec2 z = dvec2(0, 0);
                        for(i = 0; i < iter; i++) {
                            z = mult(z, z) + c;

                            if ((z.x * z.x + z.y * z.y) > 8.0) break;
                        }

                        z = dvec2(z.x * z.x - z.y * z.y, 2 * z.y * z.x) + c;
                        z = dvec2(z.x * z.x - z.y * z.y, 2 * z.y * z.x) + c;
                        z = dvec2(z.x * z.x - z.y * z.y, 2 * z.y * z.x) + c;

                        if (i != iter) {
                            float logmod = 0.5 * log(float(z.x * z.x + z.y * z.y));
                            float j = 1 + i + 1 - log(logmod) / log(2);
                            if (j > 1) {
                                j = log(j);
                            } else {
                                j = 0;
                            }
                            float x = colour_scale_x * j + colour_offset_x;
                            float y = colour_scale_y * j + colour_offset_y;
                            f_colour = texture(palette, vec2(x, y));
                        } else {
                            f_colour = vec4(0.0, 0.0, 0.0, 0.0);
                        }
                    }
                }
    
            """,
        )
        prog["tex"].value = 0
        prog["palette"].value = 1
        return prog
        


class Julia(MandelbrotBase):
    def __init__(self, param, world_to_param, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_to_param = world_to_param
        self.param = param

    def event(self, event):
        super().event(event)
        if event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:
                self.param = self.world_to_param(event.pos)
                self.full_clear()

    def tick(self, td):
        self.prog["param"].value = tuple(self.param)




    


class JuliaSelect(Mandelbrot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def event(self, event):
##        class JuliaView(Julia):
##            def event(self, event):
##                super().event(event)
##                if event.type == pygame.MOUSEBUTTONUP:
##                    if event.button == 1:
##                        raise self.ExitException()
        
        super().event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                Julia(self.pygame_to_world(event.pos), self.pygame_to_world, self.screen, self.ctx).run()

        




class FuncWindow(pgbase.Window2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prog = self.get_prog()
        vertices = self.ctx.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]).astype('f4'))
        indices = self.ctx.buffer(np.array([[0, 1, 3], [0, 2, 3]]))
        self.vao = self.ctx.vertex_array(self.prog,
                                         [(vertices, "2f4", "unit_pos")],
                                         indices)

    def get_prog(self):
        prog = self.ctx.program(
            vertex_shader="""
                #version 430
                in vec2 unit_pos;
                out vec2 v_tex_pos;
                out vec2 v_unit_pos;

                uniform vec2 cam_center;
                uniform mat2 cam_mat;

                void main() {
                    gl_Position = vec4(unit_pos, 0.0, 1.0);
                    v_tex_pos = 0.5 + 0.5 * unit_pos;
                    v_unit_pos = cam_mat * unit_pos + cam_center;
                }
            """,
            fragment_shader="""
                #version 430
                in vec2 v_unit_pos;
                in vec2 v_tex_pos;
                out vec4 f_colour;

                float tau = 6.28318530718;
                
                vec2 mult(vec2 a, vec2 b) {
                    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
                }

                vec3 hsl2rgb(vec3 c){return mix(c.bbb,mix(clamp(vec3(-1,2,2)-abs(c.r-vec3(3,2,4))*vec3(-1,1,1),0.,1.),vec3(c.b>.5),abs(.5-c.b)*2.),c.g);}

                vec3 complex_to_rgb(vec2 z) {
                    return hsl2rgb(vec3(mod(atan(z.y, z.x), tau), 1, 0.5 + 0.3 * sin(6 * log(length(z)))));
                }

                void main() {
                    vec2 z = v_unit_pos;
                    vec2 w = mult(z, z) - z;
                    f_colour = vec4(complex_to_rgb(w), 1);
                }
            """,
        )
        return prog

    def draw(self, ctx):
        super().set_uniforms([self.prog])
        ctx.screen.use()
        self.vao.render(moderngl.TRIANGLES, instances = 1)






def run(size = [2000, 1600]):
    if size is None:
        screen = pygame.display.set_mode(flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(size, flags = pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.init()
    ctx = moderngl.create_context()
            
    boop = FuncWindow(screen, ctx)
    boop.run()

    pygame.quit()
    sys.exit()
    




	
