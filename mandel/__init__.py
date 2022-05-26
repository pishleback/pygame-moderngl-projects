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


PIX_ITER_PER_SEC = 3 * 10 ** 9
MAX_ITER = 10 ** 6

class RenderBox():
    def __init__(self, total_size, rect):
        total_size = tuple(total_size)
        rect = tuple(rect)
        assert len(total_size) == 2
        assert len(rect) == 4
        for n in total_size:
            assert type(n) == int
        for n in rect:
            assert type(n) == int
        assert 0 <= rect[0] < total_size[0]
        assert 0 <= rect[1] < total_size[1]
        assert 0 <= rect[0] + rect[2] <= total_size[0]
        assert 0 <= rect[1] + rect[3] <= total_size[1]
        self.total_size = total_size
        self.rect = rect
        self.subs = None
        self.complete = False
        self.todo_count = rect[2] * rect[3]

    @property
    def sub_rects(self):
        w1 = self.rect[2] // 2
        w2 = self.rect[2] - w1
        h1 = self.rect[3] // 2
        h2 = self.rect[3] - h1
        yield [self.rect[0], self.rect[1], w1, h1]
        yield [self.rect[0] + w1, self.rect[1], w2, h1]
        yield [self.rect[0], self.rect[1] + h1, w1, h2]
        yield [self.rect[0] + w1, self.rect[1] + h1, w2, h2]

    def should_split(self, iters):
        tot_count = self.rect[2] * self.rect[3]
        done_count = tot_count - self.todo_count

        if done_count / tot_count > 0.9:
            return True
        
        secs = self.todo_count * iters / PIX_ITER_PER_SEC
        return secs > 0.2
        

    def reset(self):
        self.rect = [0, 0, self.total_size[0], self.total_size[1]]
        self.subs = None
        self.complete = False
        
    def update_complete(self, incomplete):
        todo_counts = [np.count_nonzero(incomplete[sr[0] : sr[0] + sr[2], sr[1] : sr[1] + sr[3]]) for sr in self.sub_rects]
        self.todo_count = sum(todo_counts)
        
        if self.subs is None:
            if self.todo_count == 0:
                self.complete = True
            elif any(c == 0 for c in todo_counts):
                self.split()
            
##            x = np.where(np.logical_and(self.rect[0] <= incomplete[0], incomplete[0] < self.rect[0] + self.rect[2]))
##            y = np.where(np.logical_and(self.rect[1] <= incomplete[1], incomplete[1] < self.rect[1] + self.rect[3]))
##            incompnum = len(np.intersect1d(x, y))
##            if incompnum == 0:
##                self.complete = True
####            elif incompnum < (self.rect[2] * self.rect[3]) // 16 and self.subs is None:
####                self.split()
        else:
            for sub in self.subs:
                sub.update_complete(incomplete)

    def get_leaves(self, targ_iter):
        if not self.complete:
            if self.subs is None or not self.should_split(targ_iter):
                yield self
            else:
                for sub in self.subs:
                    yield from sub.get_leaves(targ_iter)

    def split(self):
        assert self.subs is None
        self.subs = []
        for sub_rect in self.sub_rects:
            if sub_rect[2] != 0 and sub_rect[3] != 0:
                self.subs.append(RenderBox(self.total_size, sub_rect))

    def render(self, renderer, targ_iter):
        if self.should_split(targ_iter):
            self.split()
            return self.subs
        else:
            #print(self.rect[2] * self.rect[3], self.todo_count, targ_iter, max(1000, self.todo_count) * targ_iter / PIX_ITER_PER_SEC)
            renderer(self.rect, targ_iter)
            return []
            

##
##b = RenderBox([100, 100], [0, 0, 50, 50], 0)
##b.split()
##
##for a in b.get_leaves():
##    print(a.subs)
##    a.split()
##    print(a.subs)
##
##for a in b.get_leaves():
##    print(a.rect)
##
##
##input()




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
    def generate_program(self):
        self.palette_tex = pgbase.tools.load_tex(self.ctx, os.path.join("mandel", random.choice(["cfire.PNG", "cfire2.PNG", "space.jpg", "fire.jpg", "grr.jpg", "grrr.jpg", "lightning.jpg", "water.jpg", "stock.jpg", "pal2.jpg"])))


        self.prog = self.get_prog()

        vertices = self.ctx.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]).astype('f4'))
        indices = self.ctx.buffer(np.array([[0, 1, 3], [0, 2, 3]]))
        self.vao = self.ctx.vertex_array(self.prog,
                                         [(vertices, "2f4", "unit_pos")],
                                         indices)
    
    def __init__(self, *args, **kwargs):
        self.generate_program()

##        self.render_squares = 3
##        self.render_idx = 0
##        self.targ_iter = 1
##        self.done_iter = 0
##        self.last_darw_time = time.time()
##        self.sq_draw_times = []

        self.render_root = RenderBox([1, 1], [0, 0, 1, 1])
        self.render_leaves = []
        self.done_iter = 0
        
        self.res_mul = 2

        self.ignore_tex = self.ctx.texture([1, 1], 4)
        self.ignore_fbo = self.ctx.framebuffer(self.ignore_tex)
        self.colour_tex = self.ctx.texture([1, 1], 4)
        self.colour_fbo = self.ctx.framebuffer(self.colour_tex)

        self.bgs = []
        self.bg_timeout = 1
        
        self.last_user_time = time.time()

        self.zoom_clear()

        super().__init__(*args, **kwargs)

    @property
    def targ_iter(self):
        return max(16, min(MAX_ITER, self.done_iter * 4 + 1))

    @property
    def pixel_size(self):
        return [self.res_mul * self.width, self.res_mul * self.height]

    def set_rect(self, rect):
        super().set_rect(rect)
        self.ignore_tex.release()
        self.ignore_fbo.release()
        self.colour_tex.release()
        self.colour_fbo.release()
        self.bgs = []
        self.render_root = RenderBox(self.pixel_size, [0, 0, self.pixel_size[0], self.pixel_size[1]])

        self.ignore_tex = self.ctx.texture(self.pixel_size, 4)
        self.ignore_fbo = self.ctx.framebuffer(self.ignore_tex)
        self.colour_tex = self.ctx.texture(self.pixel_size, 4)
        self.colour_fbo = self.ctx.framebuffer(self.colour_tex)
        
    def __del__(self):
        self.vao.release()
        
        self.ignore_tex.release()
        self.ignore_fbo.release()
        self.colour_tex.release()
        self.colour_fbo.release()
        
        self.palette_tex.release()
        

    def zoom_clear(self):
##        self.targ_iter = 1
##        self.done_iter = 0
##        self.render_idx = 0
##        self.render_squares = 1
        self.render_root.reset()
        self.render_leaves = []
        self.done_iter = 0
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
        data = np.flip(data, axis = 0)
        r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
        gap = a == 0
        r = np.where(gap, 0, r)
        g = np.where(gap, 0, g)
        b = np.where(gap, 0, b)
        a = np.where(gap, 255, a)
        data = np.array([r, g, b, a]).transpose([1, 2, 0])
        imageio.imwrite(path, data)

    def get_incomplete(self):
        #return a list of pixels (in high res space) whose colour we havent yet determined
        ar = pgbase.tools.tex_to_np(self.colour_tex)
        return np.transpose(ar[:, :, 3] == 0, [1, 0])
##        y, x = np.where(ar[:, :, 3] == 0)
##        return x, y

    def event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                self.save_img()
            if event.key == pygame.K_c:
                self.generate_program()
                self.full_clear()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in {4, 5}:
                if time.time() - self.last_user_time > self.bg_timeout:
                    p1 = self.gl_to_world([-1, -1])
                    p2 = self.gl_to_world([1, 1])
                    def dodel(bgp1, bgp2):
                        if p1[0] < bgp1[0] < p2[0] and p1[1] < bgp1[1] < p2[1]:
                            if p1[0] < bgp2[0] < p2[0] and p1[1] < bgp2[1] < p2[1]:
                                return True
                        return False
                    self.bgs = [bg for bg in self.bgs if not dodel(bg.p1, bg.p2)]
                    if p1[0] ** 2 + p1[1] ** 2 < 100 and p2[0] ** 2 + p2[1] ** 2 < 100:
                        self.bgs.append(BgInfo(self.ctx, self.colour_tex, self.pixel_size, p1, p2, self.done_iter))
                    print("BG COUNT", len(self.bgs))
                    
                self.zoom_clear()
                self.last_user_time = time.time()
        super().event(event)
            
    def draw(self):
        super().set_uniforms([self.prog])

        if time.time() - self.last_user_time > 0.5:
            if len(self.render_leaves) == 0:
                self.render_root.update_complete(self.get_incomplete())
                self.done_iter = self.targ_iter
                self.render_leaves = list(self.render_root.get_leaves(self.targ_iter))
                print(len(self.render_leaves), self.targ_iter)
                            

            if len(self.render_leaves) != 0:
                leaf = self.render_leaves[-1]
                def renderer(rect, iterations):
                    self.prog["iter"].value = iterations
                    ps = self.pixel_size
                    self.prog["box_pos"].value = rect[0] / ps[0], rect[1] / ps[1]
                    self.prog["box_size"].value = rect[2] / ps[0], rect[3] / ps[1]
                    
                    self.colour_fbo.use()
                    self.ignore_tex.use(0)
                    self.palette_tex.use(1)
                    self.vao.render(moderngl.TRIANGLES, instances = 1)

                    self.ignore_fbo.use()
                    pgbase.tools.render_tex(self.colour_tex)
                todo = leaf.render(renderer, self.targ_iter)
                self.render_leaves.pop()
                self.render_leaves.extend(todo)


##        if time.time() - self.last_user_time > 0.2:
##            self.prog["iter"].value = self.targ_iter
##            self.prog["squares"].value = self.render_squares
##            self.prog["sq_pos"].value = divmod(self.render_idx, self.render_squares)
##            
##            self.colour_fbo.use()
##            self.ignore_tex.use(0)
##            self.palette_tex.use(1)
##            self.vao.render(moderngl.TRIANGLES, instances = 1)
##
##            self.ignore_fbo.use()
##            pgbase.tools.render_tex(self.colour_tex)
##
##            self.sq_draw_times.append(time.time() - self.last_darw_time)
##
##            if max(self.sq_draw_times) > 0.2:
##                self.render_squares += 1
##                self.sq_draw_times = []
##
####                T = self.res_mul ** 2 * self.width * self.height
####                IC = len(self.get_incomplete())
####                C = T - IC
####                P = C / T
##                
##                print(f"OOOOOF", self.targ_iter, self.render_squares)
##            else:
##                self.render_idx = (self.render_idx + 1) % (self.render_squares ** 2)
##                if self.render_idx == 0:
##                    self.done_iter = self.targ_iter
##                    self.targ_iter = min([math.ceil(2 * self.targ_iter), 10 ** 6])
##                    self.sq_draw_times = []
##
##            self.last_darw_time = time.time()

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

            uniform vec2 box_pos;
            uniform vec2 box_size;

            void main() {            
                vec2 unit_pos;
                for (int i = 0; i < 3; i++) {
                    unit_pos = 2 * (box_size * (0.5 + 0.5 * g_unit_pos[i]) + box_pos) - 1;
                
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

                        z = mult(z, z) + c;
                        z = mult(z, z) + c;
                        z = mult(z, z) + c;

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
                            //f_colour = vec4(0, 0, 0, 0);
                            //return;
                            dvec2 z0 = z;

                            for(i = 0; i < iter / 100; i++) {
                                z = mult(z, z) + c;
                                if (length(z - z0) < 0.000000000000001) {
                                    f_colour = vec4(0, 0, 0, 1);
                                    return;
                                }
                            }
                            f_colour = vec4(0, 0, 0, 0);
                        }
                    }
                }
    
            """,
        )
        prog["colour_offset"].value = (random.uniform(0, 1), random.uniform(0, 1))
        prog["colour_scale"].value = (random.uniform(0.02, 0.3), random.uniform(0.02, 0.3))
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

            uniform vec2 box_pos;
            uniform vec2 box_size;

            void main() {            
                vec2 unit_pos;
                for (int i = 0; i < 3; i++) {
                    unit_pos = 2 * (box_size * (0.5 + 0.5 * g_unit_pos[i]) + box_pos) - 1;
                
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
    






















	












	
