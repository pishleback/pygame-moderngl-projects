import pgbase
import pygame
import moderngl
import sys
import math
import numpy as np
import functools





class FlatModel(pgbase.canvas3d.Model):
    def __init__(self, sep):
        self.sep = sep
        self.sq_colours = [(1, 1, 1, 0.5)] * 64

    def set_sq_colours(self, sq_colours):
        assert len(sq_colours) == 64
        for colour in sq_colours:
            assert len(colour) == 4
        self.sq_colours = sq_colours
        
    def triangles(self):
        verticies = [[-4, self.sep, -4], [4, self.sep, -4], [-4, self.sep, 4], [4, self.sep, 4]]
        normals = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
        indices = [[0, 1, 3], [0, 2, 3]]
        return verticies, normals, indices

    def make_vao(self, prog):
        ctx = prog.ctx
        
        vertices, normals, indices = self.triangles()
        
        vertices = ctx.buffer(np.array(vertices).astype('f4'))
        normals = ctx.buffer(np.array(normals).astype('f4'))
        indices = ctx.buffer(np.array(indices))
        
        vao = ctx.vertex_array(prog,
                               [(vertices, "3f4", "vert"),
                                (normals, "3f4", "normal")],
                               indices)

        return vao, moderngl.TRIANGLES

    def make_renderer(self, ctx):            
        prog = ctx.program(
            vertex_shader = """
                #version 430
                uniform mat4 proj_mat;
                uniform mat4 view_mat;
                
                in vec3 vert;
                in vec3 normal;
                
                out vec3 v_normal;
                out vec4 v_pos_h;

                void main() {
                    v_normal = normal;
                    v_pos_h = vec4(vert, 1);
                    gl_Position = proj_mat * view_mat * vec4(vert, 1);
                }
            """,
            geometry_shader = None,
            fragment_shader = """
                #version 430
                in vec4 v_pos_h;
                in vec3 v_normal;
                uniform sampler2D peel_tex;
                uniform vec2 scr_size;
                uniform bool do_peel;
                uniform int depth;
                uniform vec3 cam_pos;
                out vec4 f_colour;
                uniform vec4[64] v_colour_arr;
                
                void main() {
                    if (v_pos_h.x * v_pos_h.x + v_pos_h.z * v_pos_h.z < 5) {
                        discard;
                    }
                    vec4 v_colour = v_colour_arr[int(floor(v_pos_h.x + 4) + 8 * floor(v_pos_h.z + 4))];
                    
                """ + pgbase.canvas3d.FRAG_MAIN + "}"
        )

        try: prog["v_colour_arr"].value = self.sq_colours
        except KeyError: pass

        vao, mode = self.make_vao(prog)

        class ModelRenderer(pgbase.canvas3d.Renderer):
            def __init__(self, prog):
                super().__init__([prog])
            def render(self):
                vao.render(mode, instances = 1)

        return ModelRenderer(prog)





class CircModel(pgbase.canvas3d.Model):
    def __init__(self, sep, inner_rad):
        import random
        self.sep = sep
        self.inner_rad = inner_rad
        self.sq_colours = [tuple(random.uniform(0.5, 1) for _ in range(4)) for _ in range(48)]

    def set_sq_colours(self, sq_colours):
        assert len(sq_colours) == 48
        for colour in sq_colours:
            assert len(colour) == 4
        self.sq_colours = sq_colours

    @functools.cache
    def triangles(self):
        round_samples = 48
        ang_samples = 24 #how many horezontal strips do we want
        outer_rad = math.sqrt(5)
        ellipse_rad = outer_rad - self.inner_rad
        gap_eps = 2

        verts = {}
        i = 0
        for t in range(-1, ang_samples + 2):
            for x in range(round_samples):
                if t == -1:
                    p = [outer_rad + gap_eps, -self.sep, 0]
                    n = [0, -1, 0]
                elif t == ang_samples + 1:
                    p = [outer_rad + gap_eps, self.sep, 0]
                    n = [0, 1, 0]
                else:
                    v_ang = math.pi * (t / ang_samples - 0.5)
                    p = [outer_rad - ellipse_rad * math.cos(v_ang), self.sep * math.sin(v_ang), 0]
                    n = [-math.cos(v_ang) / ellipse_rad, math.sin(v_ang) / self.sep, 0]
                h_ang = 2 * math.pi * x / round_samples
                p = [math.cos(h_ang) * p[0] + math.sin(h_ang) * p[2], p[1], -math.sin(h_ang) * p[0] + math.cos(h_ang) * p[2]]
                n = [math.cos(h_ang) * n[0] + math.sin(h_ang) * n[2], n[1], -math.sin(h_ang) * n[0] + math.cos(h_ang) * n[2]]
                key = (x, t)
                assert not key in verts
                verts[key] = i, p, n
                i += 1
        
        verticies = [None] * len(verts)
        normals = [None] * len(verts)
        for key, info in verts.items():
            idx, pos, norm = info
            verticies[idx] = pos
            normals[idx] = norm
                
        indices = []
        for t1 in range(-1, ang_samples + 1):
            t2 = t1 + 1
            for x1 in range(round_samples):
                x2 = (x1 + 1) % round_samples
                i0, i1, i2, i3 = verts[(x1, t1)][0], verts[(x2, t1)][0], verts[(x1, t2)][0], verts[(x2, t2)][0]
                indices.append([i0, i1, i3])
                indices.append([i0, i2, i3])
                

        #verticies = [[-4, self.sep, -4], [4, self.sep, -4], [-4, self.sep, 4], [4, self.sep, 4]]
        #normals = [[0, 1, 0]] * len(verticies)
        #indices = [[0, 1, 3], [0, 2, 3]]
        return verticies, normals, indices

    def make_vao(self, prog):
        ctx = prog.ctx
        
        vertices, normals, indices = self.triangles()
        
        vertices = ctx.buffer(np.array(vertices).astype('f4'))
        normals = ctx.buffer(np.array(normals).astype('f4'))
        indices = ctx.buffer(np.array(indices))
        
        vao = ctx.vertex_array(prog,
                               [(vertices, "3f4", "vert"),
                                (normals, "3f4", "normal")],
                               indices)

        return vao, moderngl.TRIANGLES

    def make_renderer(self, ctx):            
        prog = ctx.program(
            vertex_shader = """
                #version 430
                uniform mat4 proj_mat;
                uniform mat4 view_mat;
                
                in vec3 vert;
                in vec3 normal;
                
                out vec3 v_normal;
                out vec4 v_pos_h;

                void main() {
                    v_normal = normal;
                    v_pos_h = vec4(vert, 1);
                    gl_Position = proj_mat * view_mat * vec4(vert, 1);
                }
            """,
            geometry_shader = None,
            fragment_shader = """
                #version 430
                in vec4 v_pos_h;
                in vec3 v_normal;
                uniform sampler2D peel_tex;
                uniform vec2 scr_size;
                uniform bool do_peel;
                uniform int depth;
                uniform vec3 cam_pos;
                uniform float sep;
                uniform float mid;
                out vec4 f_colour;
                uniform vec4[48] v_colour_arr;

                const float tau = 6.28318530718;
                
                void main() {
                    if (v_pos_h.x * v_pos_h.x + v_pos_h.z * v_pos_h.z >= 5) {
                        discard;
                    }

                    float a;
                    float b;
                    float c;
                    a = 4 * mod(atan(v_pos_h.x, v_pos_h.z) / tau, 1);
                    b = a + 0.0397563521171529 * sin(tau * a);
                    b = 3 * b;
                    c = 3 * a;
                    float f = (v_pos_h.y / sep);
                    f = f * f;
                    float t = f * b + (1 - f) * c;

                    int layer;
                    if (v_pos_h.y < -mid) {
                        layer = 0;
                    } else if (v_pos_h.y < 0) {
                        layer = 1;
                    } else if (v_pos_h.y < mid) {
                        layer = 2;
                    } else {
                        layer = 3;
                    }

                    vec4 v_colour = v_colour_arr[int(floor(t)) + 12 * layer];
                    
                """ + pgbase.canvas3d.FRAG_MAIN + "}"
        )

        try: prog["sep"].value = self.sep
        except KeyError: pass
        try: prog["mid"].value = 0.6 * self.sep
        except KeyError: pass
        try: prog["v_colour_arr"].value = self.sq_colours
        except KeyError: pass

        vao, mode = self.make_vao(prog)

        class ModelRenderer(pgbase.canvas3d.Renderer):
            def __init__(self, prog):
                super().__init__([prog])
            def render(self):
                vao.render(mode, instances = 1)

        return ModelRenderer(prog)
    



class BoardView(pgbase.canvas3d.Window):
    LIGHT_SQ_COLOUR = (255, 206, 158)
    DARK_SQ_COLOUR = (209, 139, 71)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, peel_depth = 4)

        side_sep = 2.2

        lines = pgbase.canvas3d.UnitCylinder(12, False)
        r = 0.05
        s = side_sep / 2
        lines.add_line([-4, s, -4], [-4, s, 4], r, [1, 0, 0, 1])
        lines.add_line([-4, s, -4], [4, s, -4], r, [1, 0, 0, 1])
        lines.add_line([4, s, 4], [-4, s, 4], r, [1, 0, 0, 1])
        lines.add_line([4, s, 4], [4, s, -4], r, [1, 0, 0, 1])
        lines.add_line([-4, -s, -4], [-4, -s, 4], r, [0, 0, 1, 1])
        lines.add_line([-4, -s, -4], [4, -s, -4], r, [0, 0, 1, 1])
        lines.add_line([4, -s, 4], [-4, -s, 4], r, [0, 0, 1, 1])
        lines.add_line([4, -s, 4], [4, -s, -4], r, [0, 0, 1, 1])
        self.draw_model(lines)

        spheres = pgbase.canvas3d.UnitSphere(3)
        r = 0.075
        s = side_sep / 2
        spheres.add_sphere([-4, s, -4], r, [1, 0, 0, 1])
        spheres.add_sphere([4, s, -4], r, [1, 0, 0, 1])
        spheres.add_sphere([-4, s, 4], r, [1, 0, 0, 1])
        spheres.add_sphere([4, s, 4], r, [1, 0, 0, 1])
        spheres.add_sphere([-4, -s, -4], r, [0, 0, 1, 1])
        spheres.add_sphere([4, -s, -4], r, [0, 0, 1, 1])
        spheres.add_sphere([-4, -s, 4], r, [0, 0, 1, 1])
        spheres.add_sphere([4, -s, 4], r, [0, 0, 1, 1])
        self.draw_model(spheres)

        self.top_flat = FlatModel(side_sep / 2)
        self.bot_flat = FlatModel(-side_sep / 2)
        self.circ = CircModel(side_sep / 2, 1.5)

        self.draw_model(self.top_flat)
        self.draw_model(self.bot_flat)
        self.draw_model(self.circ)

        self.update_sq_colours()

    def update_sq_colours(self):
        import random

        WHITE = (1, 1, 1, 0.8)
        BLACK = (0, 0, 0, 0.8)
        
        self.top_flat.set_sq_colours(([WHITE, BLACK] * 4 + [BLACK, WHITE] * 4) * 4)
        self.bot_flat.set_sq_colours(([BLACK, WHITE] * 4 + [WHITE, BLACK] * 4) * 4)
        self.circ.set_sq_colours(([WHITE, BLACK] * 6 + [BLACK, WHITE] * 6) * 2)
        self.clear_renderer_cache()

    def tick(self, tps):
        super().tick(tps)

    def event(self, event):
        super().event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3:
                self.update_sq_colours()

##    def draw(self):
##
##        print(self.render)
##        
##        self.render.render()





def run():
    pgbase.core.Window.setup(size = [1000, 1000])
    pgbase.core.run(BoardView())
    pygame.quit()
    sys.exit()

























    
