import pygame
import moderngl
import numpy as np
import sys
import math
import imageio
import time
import pgbase
import os





class MandelbrotBase(pgbase.canvas2d.Window2D):
    def __init__(self, *args, **kwargs):
        self.prog = self.get_prog()

        vertices = self.ctx.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]).astype('f4'))
        indices = self.ctx.buffer(np.array([[0, 1, 3], [0, 2, 3]]))
        self.vao = self.ctx.vertex_array(self.prog,
                                         [(vertices, "2f4", "unit_pos")],
                                         indices)

        self.render_squares = 3
        self.render_idx = 0
        self.res_mul = 2

        self.ignore_tex = self.ctx.texture([1, 1], 4)
        self.ignore_fbo = self.ctx.framebuffer(self.ignore_tex)
        self.colour_tex = self.ctx.texture([1, 1], 4)
        self.colour_fbo = self.ctx.framebuffer(self.colour_tex)

        self.bg_tex_info = [] #list of tuples [bg_tex, bg_fbo, p1, p2]
        self.bg_timeout = 1

        self.palette_tex = pgbase.tools.load_tex(self.ctx, os.path.join("mandel", "space.jpg"))

        self.targ_iter = 0

        self.last_darw_time = time.time()
        self.sq_draw_times = []
        
        self.last_user_time = time.time()

        self.zoom_clear()

        super().__init__(*args, **kwargs)

    def set_rect(self, rect):
        super().set_rect(rect)
        self.ignore_tex.release()
        self.ignore_fbo.release()
        self.colour_tex.release()
        self.colour_fbo.release()
        for bg_tex, bg_fbo, bgp1, bgp2 in self.bg_tex_info:
            bg_tex.release()
            bg_fbo.release()
        self.bg_tex_info = []

        N = self.res_mul
        self.ignore_tex = self.ctx.texture([N * self.width, N * self.height], 4)
        self.ignore_fbo = self.ctx.framebuffer(self.ignore_tex)
        self.colour_tex = self.ctx.texture([N * self.width, N * self.height], 4)
        self.colour_fbo = self.ctx.framebuffer(self.colour_tex)

        self.bg_tex_info = []

    def __del__(self):
        self.vao.release()
        self.ignore_tex.release()
        self.ignore_fbo.release()
        self.colour_tex.release()
        self.colour_fbo.release()
        for bg_tex, bg_fbo, bgp1, bgp2 in self.bg_tex_info:
            bg_tex.release()
            bg_fbo.release()
        self.palette_tex.release()
        

    def zoom_clear(self):
        self.targ_iter = 1
        self.render_idx = 0
        self.render_squares = 1
        self.ignore_fbo.clear(0, 0, 0, 0)
        self.colour_fbo.clear(0, 0, 0, 0)
        
    def full_clear(self):
        #clear screen and backgrounds
        self.zoom_clear()
        for bg_tex, bg_fbo, bgp1, bgp2 in self.bg_tex_info:
            bg_tex.release()
            bg_fbo.release()
        self.bg_tex_info = []

    def save_img(self):
        import glob
        paths = set(glob.glob(os.path.join("images", "img*.png")))
        i = 0
        while True:
            path = os.path.join("images", "img" + str(i) + ".png")
            if path in paths:
                i += 1
            else:
                break

        data = pgbase.tools.tex_to_np(self.colour_tex)
        r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
        gap = a == 0
        r = np.where(gap, 0, r)
        g = np.where(gap, 0, g)
        b = np.where(gap, 0, b)
        a = np.where(gap, 255, a)
        data = np.array([r, g, b, a]).transpose([1, 2, 0])
        imageio.imwrite(path, data)

    def event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                self.save_img()
                
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in {4, 5}:
                if time.time() - self.last_user_time > self.bg_timeout:
                    bgp1 = self.gl_to_world([-1, -1])
                    bgp2 = self.gl_to_world([1, 1])

                    if -4 < bgp1[0] < 4 and -4 < bgp1[1] < 4 and -4 < bgp2[0] < 4 and -4 < bgp2[1] < 4:
                        N = self.res_mul
                        bg_tex = self.ctx.texture([N * self.width, N * self.height], 4)
                        bg_fbo = self.ctx.framebuffer(bg_tex)
                        bg_fbo.use()

                        #remove background textured covered by the new one
                        rem_idxs = set([])
                        for idx, info in enumerate(self.bg_tex_info):
                            cur_bg_tex, cur_bg_fbo, cur_p1, cur_p2 = info
                            if bgp1[0] < cur_p1[0] < bgp2[0] and bgp1[1] < cur_p1[1] < bgp2[1]:
                                if bgp1[0] < cur_p2[0] < bgp2[0] and bgp1[1] < cur_p2[1] < bgp2[1]:
                                    rem_idxs.add(idx)
                        for idx in rem_idxs:
                            cur_bg_tex, cur_bg_fbo, cur_p1, cur_p2 = self.bg_tex_info[idx]
                            cur_bg_tex.release()
                            cur_bg_fbo.release()
                        self.bg_tex_info = [info for idx, info in enumerate(self.bg_tex_info) if not idx in rem_idxs]
                        
                        self.ctx.clear(0, 0, 0, 0)
                        pgbase.tools.render_tex(self.colour_tex)
                        self.bg_tex_info.append((bg_tex, bg_fbo, bgp1, bgp2))
                    else:
                        self.full_clear()

                    print("BG TEXS:", len(self.bg_tex_info))
                          
                self.zoom_clear()
                self.last_user_time = time.time()
        super().event(event)
            
    def draw(self):        
        super().set_uniforms([self.prog])

        if time.time() - self.last_user_time > 0.2:
            try:
                self.prog["iter"].value = self.targ_iter
            except KeyError:
                pass
            try:
                self.prog["squares"].value = self.render_squares
            except KeyError:
                pass
            try:
                self.prog["sq_pos"].value = divmod(self.render_idx, self.render_squares)
            except KeyError:
                pass
            
            self.colour_fbo.use()
            self.ignore_tex.use(0)
            self.palette_tex.use(1)
            self.vao.render(moderngl.TRIANGLES, instances = 1)

            self.ignore_fbo.use()
            pgbase.tools.render_tex(self.colour_tex)

            t = time.time()
            self.sq_draw_times.append(t - self.last_darw_time)
            self.last_darw_time = t

            if max(self.sq_draw_times) > 0.1:
                self.render_squares += 1
                self.sq_draw_times = []
                print(f"OOOOOF", self.targ_iter, self.render_squares)
            else:
                self.render_idx = (self.render_idx + 1) % (self.render_squares ** 2)
                if self.render_idx == 0:
                    self.targ_iter = min([math.ceil(2 * self.targ_iter), 10 ** 6])
                    self.sq_draw_times = []

        self.ctx.screen.use()
        self.ctx.enable_only(moderngl.BLEND)
        self.ctx.screen.clear(0.0, 0.0, 0.0, 0.0)

        if time.time() - self.last_user_time < self.bg_timeout:
            for bg_tex, bg_fbo, bgp1, bgp2 in self.bg_tex_info:
                p1 = self.world_to_gl(bgp1)
                p2 = self.world_to_gl(bgp2)
                pgbase.tools.render_tex(bg_tex, p1, p2)

        pgbase.tools.render_tex(self.colour_tex)





class Mandelbrot(MandelbrotBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_prog(self):
        prog = self.ctx.program(
            vertex_shader = """
                #version 430
                in vec2 unit_pos;
                out vec2 g_unit_pos;

                void main() {
                    g_unit_pos = unit_pos;
                }
            """,
            geometry_shader = """
            #version 430
            layout (triangles) in;
            layout (triangle_strip, max_vertices = 100) out;

            in vec2 g_unit_pos[3];

            out vec2 v_tex_pos;
            out vec2 v_unit_pos;

            uniform float squares;
            uniform vec2 sq_pos;

            void main() {            
                vec2 unit_pos;
                for (int i = 0; i < 3; i++) {
                    unit_pos = 2 * (0.5 + 0.5 * g_unit_pos[i] + sq_pos) / squares - 1;
                
                    gl_Position = vec4(unit_pos, 0.0, 1.0);
                    v_tex_pos = 0.5 + 0.5 * unit_pos;
                    v_unit_pos = unit_pos;
                    EmitVertex();
                }
                EndPrimitive();
            }
            """,
            fragment_shader = """
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
                    float colour_scale = 0.2;
                
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

        try:
            prog["tex"].value = 0
        except KeyError:
            pass
        try:
            prog["palette"].value = 1
        except KeyError:
            pass
        return prog
        


class Julia(MandelbrotBase):
    def __init__(self, param, param_gl_to_world, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_gl_to_world = param_gl_to_world
        self.param = param

    def event(self, event):
        super().event(event)
        if event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:
                self.param = self.param_gl_to_world(self.pygame_to_gl(event.pos))
                self.full_clear()

    def get_prog(self):
        prog = self.ctx.program(
            vertex_shader = """
                #version 430
                in vec2 unit_pos;
                out vec2 g_unit_pos;

                void main() {
                    g_unit_pos = unit_pos;
                }
            """,
            geometry_shader = """
            #version 430
            layout (triangles) in;
            layout (triangle_strip, max_vertices = 100) out;

            in vec2 g_unit_pos[3];

            out vec2 v_tex_pos;
            out vec2 v_unit_pos;

            uniform float squares;
            uniform vec2 sq_pos;

            void main() {            
                vec2 unit_pos;
                for (int i = 0; i < 3; i++) {
                    unit_pos = 2 * (0.5 + 0.5 * g_unit_pos[i] + sq_pos) / squares - 1;
                
                    gl_Position = vec4(unit_pos, 0.0, 1.0);
                    v_tex_pos = 0.5 + 0.5 * unit_pos;
                    v_unit_pos = unit_pos;
                    EmitVertex();
                }
                EndPrimitive();
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

                uniform vec2 param;

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
                        dvec2 z = cam_mat * dvec2(v_unit_pos) + cam_center;
                        dvec2 c = param;
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
        prog["param"].value = (-0.8, 0.156)
        return prog

    def tick(self, td):
        self.prog["param"].value = tuple(self.param)




    


class JuliaSelect(Mandelbrot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def event(self, event):
        class JuliaView(Julia):
            def __init__(self, bg_tex, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.mand_bg_tex = bg_tex
                
            def draw(self):
                super().draw()
                self.ctx.screen.use()
                self.ctx.enable_only(moderngl.BLEND)
                if pygame.mouse.get_pressed()[0]:
                    pgbase.tools.render_tex(self.mand_bg_tex, alpha = 0.3)
        
        super().event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
##                p1 = self.world_to_gl(self.bg_poses[0])
##                p2 = self.world_to_gl(self.bg_poses[1])
                pgbase.core.run(JuliaView(self.colour_tex,
                                          self.pygame_to_world(event.pos),
                                          self.gl_to_world))





def run():
    pgbase.core.Window.setup(size = None)
    pgbase.core.run(JuliaSelect())
##    pgbase.core.run(Mandelbrot())
    pygame.quit()
    sys.exit()
    










##
##class Mandelbrot2(pgbase.canvas2d.Window2D):
##    def __init__(self, *args, **kwargs):
##        super().__init__(*args, **kwargs)
##        self.prog = self.get_prog()
##
##        vertices = self.ctx.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]).astype('f4'))
##        indices = self.ctx.buffer(np.array([[0, 1, 3], [0, 2, 3]]))
##        self.vao = self.ctx.vertex_array(self.prog,
##                                         [(vertices, "2f4", "unit_pos")],
##                                         indices)
##
##        self.loop_tex = self.ctx.texture([self.width, self.height], 4, dtype = "f4")
##        self.loop_fbo = self.ctx.framebuffer(self.loop_tex)
##        self.colour_tex = self.ctx.texture([self.width, self.height], 4, dtype = "f4")
##        self.colour_fbo = self.ctx.framebuffer(self.colour_tex)
##
##        self.palette_tex = pgbase.tools.load_tex(self.ctx, os.path.join("mandel", "water.jpg"))
##        
##        self.clear()
##
##    def get_prog(self):
##        prog = self.ctx.program(
##            vertex_shader = """
##                #version 430
##                in vec2 unit_pos;
##                out vec2 v_tex_pos;
##                out vec2 v_unit_pos;
##
##                void main() {
##                    gl_Position = vec4(unit_pos, 0.0, 1.0);
##                    v_tex_pos = 0.5 + 0.5 * unit_pos;
##                    v_unit_pos = unit_pos;
##                }
##            """,
##            fragment_shader = """
##                #version 430
##                in vec2 v_unit_pos;
##                in vec2 v_tex_pos;
##                out vec4 f_colour;
##
##                uniform int iter;
##                uniform sampler2D tex;
##                uniform sampler2D palette;
##                uniform dvec2 cam_center;
##                uniform dmat2 cam_mat;
##
##                dvec2 mult(dvec2 a, dvec2 b) {
##                    return dvec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
##                }
##
##                void main() {
##                    float colour_scale = 0.1;
##                
##                    float colour_scale_x = colour_scale;
##                    float colour_offset_x = 0.1;
##                    float colour_scale_y = colour_scale * (1 + sqrt(5)) / 2;
##                    float colour_offset_y = 0.1;
##                    
##                    int i;
##                    dvec2 c = cam_mat * dvec2(v_unit_pos) + cam_center;
##                    f_colour = texture2D(tex, v_tex_pos);
##                    if (f_colour.a < 0.5) {
##                        dvec2 z = texture2D(tex, v_tex_pos).xy;
##                        for(i = int(f_colour.b); i < iter + f_colour.b; i++) {
##                            z = mult(z, z) + c;
##                            if ((z.x * z.x + z.y * z.y) > 8.0) {
##                                z = dvec2(z.x * z.x - z.y * z.y, 2 * z.y * z.x) + c;
##                                z = dvec2(z.x * z.x - z.y * z.y, 2 * z.y * z.x) + c;
##                                z = dvec2(z.x * z.x - z.y * z.y, 2 * z.y * z.x) + c;
##                                
##                                float logmod = 0.5 * log(float(z.x * z.x + z.y * z.y));
##                                float j = 1 + i + 1 - log(logmod) / log(2);
##                                if (j > 1) {
##                                    j = log(j);
##                                } else {
##                                    j = 0;
##                                }
##                                float x = colour_scale_x * j + colour_offset_x;
##                                float y = colour_scale_y * j + colour_offset_y;
##                                f_colour = vec4(texture(palette, vec2(x, y)).rgb, 1);
##                            }
##                        }
##                        if (f_colour.a < 0.5) {
##                            f_colour = vec4(z, float(i), 0);
##                        }
##                    }
##                }
##    
##            """,
##        )
##        prog["tex"].value = 0
##        prog["palette"].value = 1
##        return prog
##
##    def __del__(self):
##        self.vao.release()
##        
##        self.loop_tex.release()
##        self.loop_fbo.release()
##        self.colour_tex.release()
##        self.colour_fbo.release()
##        
##        self.palette_tex.release()
##
##    def clear(self):
##        self.loop_fbo.clear(0, 0, 0, 0)
##        self.colour_fbo.clear(0, 0, 0, 0)
##
##    def event(self, event):
##        if event.type == pygame.MOUSEBUTTONDOWN:
##            if event.button in {4, 5}:
##                self.clear()
##        super().event(event)
##            
##    def draw(self, ctx):
##        super().set_uniforms([self.prog])
##
##        self.prog["iter"].value = 1
##            
##        self.colour_fbo.use()
##        self.loop_tex.use(0)
##        self.palette_tex.use(1)
##        self.vao.render(moderngl.TRIANGLES, instances = 1)
##
##        self.loop_fbo.use()
##        pgbase.tools.render_tex(self.colour_tex)
##
##        ctx.screen.use()
##        ctx.clear(0, 0, 0, 0)
##        pgbase.tools.render_tex(self.colour_tex)
##
##
##









import pygame
import moderngl
import numpy as np
import sys
import math
import imageio
import time
import pgbase
import os





class BgInfo():
    def __init__(self, ctx, tex, size, p1, p2, done_iter):          
        self.tex = ctx.texture(size, 4)
        self.fbo = ctx.framebuffer(self.tex)
        self.fbo.use()
        ctx.clear(0, 0, 0, 0)
        pgbase.tools.render_tex(tex)
        self.p1 = p1
        self.p2 = p2
        self.done_iter = done_iter #how many itterations

    def render(self, world_to_gl):
        p1 = world_to_gl(self.p1)
        p2 = world_to_gl(self.p2)
        pgbase.tools.render_tex(self.tex, p1, p2)
        
    def __del__(self):
        self.tex.release()
        self.fbo.release()



class MandelbrotBase(pgbase.canvas2d.Window2D):
    def __init__(self, *args, **kwargs):
        self.prog = self.get_prog()

        vertices = self.ctx.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]).astype('f4'))
        indices = self.ctx.buffer(np.array([[0, 1, 3], [0, 2, 3]]))
        self.vao = self.ctx.vertex_array(self.prog,
                                         [(vertices, "2f4", "unit_pos")],
                                         indices)

        self.render_squares = 3
        self.render_idx = 0
        self.res_mul = 2

        self.ignore_tex = self.ctx.texture([1, 1], 4)
        self.ignore_fbo = self.ctx.framebuffer(self.ignore_tex)
        self.colour_tex = self.ctx.texture([1, 1], 4)
        self.colour_fbo = self.ctx.framebuffer(self.colour_tex)

        self.bgs = []
        self.bg_timeout = 1

        self.palette_tex = pgbase.tools.load_tex(self.ctx, os.path.join("mandel", "space.jpg"))

        self.targ_iter = 1
        self.done_iter = 0

        self.last_darw_time = time.time()
        self.sq_draw_times = []
        
        self.last_user_time = time.time()

        self.zoom_clear()

        super().__init__(*args, **kwargs)

    def set_rect(self, rect):
        super().set_rect(rect)
        self.ignore_tex.release()
        self.ignore_fbo.release()
        self.colour_tex.release()
        self.colour_fbo.release()
        self.bgs = []

        N = self.res_mul
        self.ignore_tex = self.ctx.texture([N * self.width, N * self.height], 4)
        self.ignore_fbo = self.ctx.framebuffer(self.ignore_tex)
        self.colour_tex = self.ctx.texture([N * self.width, N * self.height], 4)
        self.colour_fbo = self.ctx.framebuffer(self.colour_tex)
        
    def __del__(self):
        self.vao.release()
        
        self.ignore_tex.release()
        self.ignore_fbo.release()
        self.colour_tex.release()
        self.colour_fbo.release()
        
        self.palette_tex.release()
        

    def zoom_clear(self):
        self.targ_iter = 1
        self.done_iter = 0
        self.render_idx = 0
        self.render_squares = 1
        self.ignore_fbo.clear(0, 0, 0, 0)
        self.colour_fbo.clear(0, 0, 0, 0)
        

    def full_clear(self):
        self.zoom_clear()
        self.bgs = []

    def save_img(self):
        import glob
        paths = set(glob.glob(os.path.join("images", "img*.png")))
        i = 0
        while True:
            path = os.path.join("images", "img" + str(i) + ".png")
            if path in paths:
                i += 1
            else:
                break

        data = pgbase.tools.tex_to_np(self.colour_tex)
        r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
        gap = a == 0
        r = np.where(gap, 0, r)
        g = np.where(gap, 0, g)
        b = np.where(gap, 0, b)
        a = np.where(gap, 255, a)
        data = np.array([r, g, b, a]).transpose([1, 2, 0])
        imageio.imwrite(path, data)

    def event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                self.save_img()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in {4, 5}:
                if time.time() - self.last_user_time > self.bg_timeout:
                    p1 = self.gl_to_world([-1, -1])
                    p2 = self.gl_to_world([1, 1])
                    N = self.res_mul
                    def dodel(bgp1, bgp2):
                        if p1[0] < bgp1[0] < p2[0] and p1[1] < bgp1[1] < p2[1]:
                            if p1[0] < bgp2[0] < p2[0] and p1[1] < bgp2[1] < p2[1]:
                                return True
                        return False
                    self.bgs = [bg for bg in self.bgs if not dodel(bg.p1, bg.p2)]
                    if p1[0] ** 2 + p1[1] ** 2 < 100 and p2[0] ** 2 + p2[1] ** 2 < 100:
                        self.bgs.append(BgInfo(self.ctx, self.colour_tex, [N * self.width, N * self.height], p1, p2, self.done_iter))
                    print("BG COUNT", len(self.bgs))
                    
                self.zoom_clear()
                self.last_user_time = time.time()
        super().event(event)
            
    def draw(self):
        super().set_uniforms([self.prog])

        if time.time() - self.last_user_time > 0.2:
            self.prog["iter"].value = self.targ_iter
            self.prog["squares"].value = self.render_squares
            self.prog["sq_pos"].value = divmod(self.render_idx, self.render_squares)
            
            self.colour_fbo.use()
            self.ignore_tex.use(0)
            self.palette_tex.use(1)
            self.vao.render(moderngl.TRIANGLES, instances = 1)

            self.ignore_fbo.use()
            pgbase.tools.render_tex(self.colour_tex)

            t = time.time()
            self.sq_draw_times.append(t - self.last_darw_time)
            self.last_darw_time = t

            if max(self.sq_draw_times) > 0.2:
                self.render_squares += 1
                self.sq_draw_times = []
                print(f"OOOOOF", self.targ_iter, self.render_squares)
            else:
                self.render_idx = (self.render_idx + 1) % (self.render_squares ** 2)
                if self.render_idx == 0:
                    self.done_iter = self.targ_iter
                    self.targ_iter = min([math.ceil(2 * self.targ_iter), 10 ** 6])
                    self.sq_draw_times = []


        self.ctx.screen.use()
        self.ctx.enable_only(moderngl.BLEND)
        self.ctx.clear(0, 0, 0, 0)

        for bg in self.bgs:
            if bg.done_iter > self.done_iter:
                bg.render(self.world_to_gl)

        pgbase.tools.render_tex(self.colour_tex)





class Mandelbrot(MandelbrotBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_prog(self):
        prog = self.ctx.program(
            vertex_shader = """
                #version 430
                in vec2 unit_pos;
                out vec2 g_unit_pos;

                void main() {
                    g_unit_pos = unit_pos;
                }
            """,
            geometry_shader = """
            #version 430
            layout (triangles) in;
            layout (triangle_strip, max_vertices = 100) out;

            in vec2 g_unit_pos[3];

            out vec2 v_tex_pos;
            out vec2 v_unit_pos;

            uniform float squares;
            uniform vec2 sq_pos;

            void main() {            
                vec2 unit_pos;
                for (int i = 0; i < 3; i++) {
                    unit_pos = 2 * (0.5 + 0.5 * g_unit_pos[i] + sq_pos) / squares - 1;
                
                    gl_Position = vec4(unit_pos, 0.0, 1.0);
                    v_tex_pos = 0.5 + 0.5 * unit_pos;
                    v_unit_pos = unit_pos;
                    EmitVertex();
                }
                EndPrimitive();
            }
            """,
            fragment_shader = """
                #version 430
                in vec2 v_unit_pos;
                in vec2 v_tex_pos;
                out vec4 f_colour;

                uniform int iter;
                uniform sampler2D tex;
                uniform sampler2D palette;
                uniform dvec2 cam_center;
                uniform dmat2 cam_mat;
                uniform vec2 colour_offset;
                uniform vec2 colour_scale;

                dvec2 mult(dvec2 a, dvec2 b) {
                    return dvec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
                }

                void main() {
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
                            float x = colour_scale.x * j + colour_offset.x;
                            float y = colour_scale.y * j + colour_offset.y;
                            f_colour = vec4(texture(palette, vec2(x, y)).xyz, 1);
                        } else {
                            f_colour = vec4(0.0, 0.0, 0.0, 0.0);
                        }
                    }
                }
    
            """,
        )
        import random
        prog["colour_offset"].value = (random.uniform(0, 1), random.uniform(0, 1))
        prog["colour_scale"].value = (random.uniform(0.1, 0.3), random.uniform(0.1, 0.3))
        prog["tex"].value = 0
        prog["palette"].value = 1
        return prog
        


class Julia(MandelbrotBase):
    def __init__(self, param, param_gl_to_world, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_gl_to_world = param_gl_to_world
        self.param = param

    def event(self, event):
        super().event(event)
        if event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:
                self.param = self.param_gl_to_world(self.pygame_to_gl(event.pos))
                self.full_clear()

    def get_prog(self):
        prog = self.ctx.program(
            vertex_shader = """
                #version 430
                in vec2 unit_pos;
                out vec2 g_unit_pos;

                void main() {
                    g_unit_pos = unit_pos;
                }
            """,
            geometry_shader = """
            #version 430
            layout (triangles) in;
            layout (triangle_strip, max_vertices = 100) out;

            in vec2 g_unit_pos[3];

            out vec2 v_tex_pos;
            out vec2 v_unit_pos;

            uniform float squares;
            uniform vec2 sq_pos;

            void main() {            
                vec2 unit_pos;
                for (int i = 0; i < 3; i++) {
                    unit_pos = 2 * (0.5 + 0.5 * g_unit_pos[i] + sq_pos) / squares - 1;
                
                    gl_Position = vec4(unit_pos, 0.0, 1.0);
                    v_tex_pos = 0.5 + 0.5 * unit_pos;
                    v_unit_pos = unit_pos;
                    EmitVertex();
                }
                EndPrimitive();
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
                uniform vec2 colour_offset;
                uniform vec2 colour_scale;

                uniform vec2 param;

                dvec2 mult(dvec2 a, dvec2 b) {
                    return dvec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
                }

                void main() {
                    if (texture2D(tex, v_tex_pos).a > 0.5) {
                        discard;
                    } else {
                        int i;
                        dvec2 z = cam_mat * dvec2(v_unit_pos) + cam_center;
                        dvec2 c = param;
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
                            float x = colour_scale.x * j + colour_offset.x;
                            float y = colour_scale.y * j + colour_offset.y;
                            f_colour = vec4(texture(palette, vec2(x, y)).xyz, 1);
                        } else {
                            f_colour = vec4(0.0, 0.0, 0.0, 0.0);
                        }
                    }
                }
    
            """,
        )
        import random
        prog["colour_offset"].value = (random.uniform(0, 1), random.uniform(0, 1))
        prog["colour_scale"].value = (random.uniform(0.1, 0.3), random.uniform(0.1, 0.3))
        prog["tex"].value = 0
        prog["palette"].value = 1
        prog["param"].value = (-0.8, 0.156)
        return prog

    def tick(self, td):
        self.prog["param"].value = tuple(self.param)




    


class JuliaSelect(Mandelbrot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def event(self, event):
        class JuliaView(Julia):
            def __init__(self, bg_tex, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.mand_bg_tex = bg_tex
                
            def draw(self):
                super().draw()
                self.ctx.screen.use()
                self.ctx.enable_only(moderngl.BLEND)
                if pygame.mouse.get_pressed()[0]:
                    pgbase.tools.render_tex(self.mand_bg_tex, alpha = 0.3)
        
        super().event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
##                p1 = self.world_to_gl(self.bg_poses[0])
##                p2 = self.world_to_gl(self.bg_poses[1])
                pgbase.core.run(JuliaView(self.colour_tex,
                          self.pygame_to_world(event.pos),
                          self.gl_to_world))

        




def run():
    pgbase.core.Window.setup(size = [1000, 1000])
    pgbase.core.run(JuliaSelect())
    pygame.quit()
    sys.exit()
    






















	












	
