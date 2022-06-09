import pgbase
import pygame
import moderngl
import sys
import math
import numpy as np
import functools





class Renderer():
    def __init__(self, progs = []):
        self.progs = progs
    def render(self):
        pass


class Model():
    def make_renderer(self, ctx):
        raise NotImplementedError()





FRAG_MAIN = """
vec3 v_pos = v_pos_h.xyz / v_pos_h.w;
vec3 cam_vec = normalize(v_pos - cam_pos);

vec3 light = normalize(vec3(1, -1, 1));
vec3 normal = normalize(v_normal);
float diff = abs(dot(light, normal));
float spec = pow(abs(dot(-cam_vec, reflect(light, normal))), 10);

float thicc = 1 / abs(dot(normal, cam_vec));
float alpha = 1 - max(0, pow(1 - v_colour.a, thicc)); //max(0, x) used to avoid weird flickering for some reason

spec *= alpha;

f_colour = vec4((0.7 + 0.3 * diff) * v_colour.rgb + spec * (1 - v_colour.rgb), alpha + spec * (1 - alpha));
if (do_peel) {
    if (gl_FragCoord.z <= texture2D(peel_tex, gl_FragCoord.xy / scr_size).x) {
        discard;
    }
}
"""



class BaseModel(Model):
    def __init__(self, vertex_shader, geometry_shader):
        self.vertex_shader = vertex_shader
        self.geometry_shader = geometry_shader
        self.num = None

    def make_vao(self, prog):
        raise NotImplementedError()
        
    def make_renderer(self, ctx):
        if self.num == 0:
            return Renderer()
            
        prog = ctx.program(
            vertex_shader = self.vertex_shader,
            geometry_shader = self.geometry_shader,
            fragment_shader = """
                #version 430
                in vec4 v_pos_h;
                in vec4 v_colour;
                in vec3 v_normal;
                uniform sampler2D peel_tex;
                uniform vec2 scr_size;
                uniform bool do_peel;
                uniform int depth;
                uniform vec3 cam_pos;
                out vec4 f_colour;
                void main() {""" + FRAG_MAIN + "}"
        )

        vao, mode, num_instances = self.make_vao(prog)

        class ModelRenderer(Renderer):
            def __init__(self, prog):
                super().__init__([prog])
            def render(self):
                vao.render(mode, instances = num_instances)

        return ModelRenderer(prog)
    



            



class SimpleModel(BaseModel):
    def __init__(self):
        super().__init__("""
            #version 430

            uniform mat4 proj_mat;
            uniform mat4 view_mat;
            in mat4 model_mat;
            in mat3 normal_mat;
            
            in vec4 blanket_colour;
            in float source;
            in vec3 vert;
            in vec3 normal;
            in vec4 colour;
            
            out vec4 v_colour;
            out vec3 v_normal;
            out vec4 v_pos_h;

            void main() {
                v_colour = (1 - source) * colour + source * blanket_colour;
                v_normal = normal_mat * normal;
                v_pos_h = model_mat * vec4(vert, 1);
                gl_Position = proj_mat * view_mat * model_mat * vec4(vert, 1);
            }
        """, None)
        self.model_mats = []
        self.colours = [] #list of blanket model colours
        self.sources = [] #list of floats for how much model and how much blanket colour to use
        self.num = 0
        
        #self.add_instance(np.eye(4, dtype = "f4"), [0, 0, 0, 1], 0)

    def add_instance(self, model_mat, colour = None, source = 1):
        if colour is None:
            colour = [0, 0, 0, 0]
            source = 0
        self.num += 1
        self.model_mats.append(model_mat)
        self.colours.append(colour)
        self.sources.append(source)

    def clear_instances(self):
        self.model_mats = []
        self.colours = []
        self.sources = []
        self.num = 0
    
    @functools.cache
    def triangles():
        raise NotImplementedError()

    def make_vao(self, prog):
        ctx = prog.ctx
        
        vertices, normals, colours, indices = self.triangles()
        #normals = [normalize(vec) for vec in normals]
        
        vertices = ctx.buffer(np.array(vertices).astype('f4'))
        normals = ctx.buffer(np.array(normals).astype('f4'))
        colours = ctx.buffer(np.array(colours).astype('f4'))
        indices = ctx.buffer(np.array(indices))
        
        model_mats = ctx.buffer(np.array(self.model_mats, dtype = "f4").transpose([0, 2, 1]).flatten())
        normal_mats = ctx.buffer(np.array([np.linalg.inv(np.transpose(mat[0:3, 0:3])) for mat in self.model_mats], dtype = "f4").transpose([0, 2, 1]).flatten())
        blanket_colours = ctx.buffer(np.array(self.colours, dtype = "f4").flatten())
        sources = ctx.buffer(np.array(self.sources, dtype = "f4").flatten())

        vao = ctx.vertex_array(prog,
                               [(vertices, "3f4", "vert"),
                                (normals, "3f4", "normal"),
                                (colours, "4f4", "colour"),
                                (model_mats, "16f4 /i", "model_mat"),
                                (normal_mats, "9f4 /i", "normal_mat"),
                                (blanket_colours, "4f4 /i", "blanket_colour"),
                                (sources, "f4 /i", "source")],
                               indices)

        return vao, moderngl.TRIANGLES, self.num
    

class UnitBox(SimpleModel):
    def __init__(self):
        super().__init__()
        
    @functools.cache
    def triangles(self):
        import random
        vertices = [[0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 1, 1],
                    [1, 0, 1],
                    [0, 0, 1]]
        indices = [0, 1, 3, 1, 3, 2, 3, 2, 5, 3, 5, 4, 6, 5, 2, 2, 6, 1, 4, 7, 0, 4, 0, 3, 7, 4, 5, 5, 7, 6, 7, 6, 1, 1, 0, 7]
        return vertices, [[x, y, z, 1] for x, y, z in vertices], indices





class UnitSphere(SimpleModel):
    def __init__(self, ref = 2):
        #ref - how refined should the sphere model be
        super().__init__()
        self.ref = ref

    def add_sphere(self, pos, radius, colour = None, source = 1):
        self.add_instance(translation(*pos) @ scale(radius, radius, radius), colour, source)
        
    @functools.cache
    def triangles(self):        
        vertices = [[1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1]]
        
        indices = [[0, 4, 3],
                   [4, 3, 1],
                   [1, 3, 5],
                   [5, 3, 0],
                   [0, 4, 2],
                   [4, 2, 1],
                   [1, 2, 5],
                   [5, 2, 0]]

        def refine(vertices, indices):
            pairs = set()
            for tri in indices:
                pairs.add(frozenset([tri[0], tri[1]]))
                pairs.add(frozenset([tri[0], tri[2]]))
                pairs.add(frozenset([tri[1], tri[2]]))

            next_vertices = list(vertices)
            new_verts = {}#[pair for pair in itertools.combinations(range(len(vertices)), 2)]
            offset = len(vertices)
            for idx, pair in enumerate(pairs):
                new_verts[pair] = idx + offset
                pair = tuple(pair)
                va = vertices[pair[0]]
                vb = vertices[pair[1]]
                vm = [0.5 * (va[i] + vb[i]) for i in [0, 1, 2]]
                norm = math.sqrt(vm[0] ** 2 + vm[1] ** 2 + vm[2] ** 2)
                vm = [vm[i] / norm for i in [0, 1, 2]]
                next_vertices.append(vm)
                
            next_indices = []
            for tri in indices:
                a = tri[0]
                b = new_verts[frozenset([tri[0], tri[1]])]
                c = new_verts[frozenset([tri[0], tri[2]])]
                d = tri[1]
                e = new_verts[frozenset([tri[1], tri[2]])]
                f = tri[2]
                next_indices.append([a, b, c])
                next_indices.append([b, d, e])
                next_indices.append([b, c, e])
                next_indices.append([c, e, f])
                
            return next_vertices, next_indices

        for _ in range(self.ref):
            vertices, indices = refine(vertices, indices)
        
        return vertices, vertices, [[x, y, z, 0.5] for x, y, z in vertices], indices



class TextureSphere(Model):
    def __init__(self, texcube, ref = 2):
        #ref - how refined should the sphere model be    
        super().__init__()
        self.ref = ref
        self.centers = []
        self.radii = []
        self.num = 0
        self.texcube = texcube

    def add_sphere(self, pos, radius):
        self.num += 1
        self.centers.append(pos)
        self.radii.append(radius)
        
    @functools.cache
    def triangles(self):        
        vertices = [[1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1]]
        
        indices = [[0, 4, 3],
                   [4, 3, 1],
                   [1, 3, 5],
                   [5, 3, 0],
                   [0, 4, 2],
                   [4, 2, 1],
                   [1, 2, 5],
                   [5, 2, 0]]

        def refine(vertices, indices):
            pairs = set()
            for tri in indices:
                pairs.add(frozenset([tri[0], tri[1]]))
                pairs.add(frozenset([tri[0], tri[2]]))
                pairs.add(frozenset([tri[1], tri[2]]))

            next_vertices = list(vertices)
            new_verts = {}#[pair for pair in itertools.combinations(range(len(vertices)), 2)]
            offset = len(vertices)
            for idx, pair in enumerate(pairs):
                new_verts[pair] = idx + offset
                pair = tuple(pair)
                va = vertices[pair[0]]
                vb = vertices[pair[1]]
                vm = [0.5 * (va[i] + vb[i]) for i in [0, 1, 2]]
                norm = math.sqrt(vm[0] ** 2 + vm[1] ** 2 + vm[2] ** 2)
                vm = [vm[i] / norm for i in [0, 1, 2]]
                next_vertices.append(vm)
                
            next_indices = []
            for tri in indices:
                a = tri[0]
                b = new_verts[frozenset([tri[0], tri[1]])]
                c = new_verts[frozenset([tri[0], tri[2]])]
                d = tri[1]
                e = new_verts[frozenset([tri[1], tri[2]])]
                f = tri[2]
                next_indices.append([a, b, c])
                next_indices.append([b, d, e])
                next_indices.append([b, c, e])
                next_indices.append([c, e, f])
                
            return next_vertices, next_indices

        for _ in range(self.ref):
            vertices, indices = refine(vertices, indices)
        
        return vertices, vertices, indices
        
    def clear_instances(self):
        self.centers = []
        self.radii = []
        self.num = 0
    
    def make_vao(self, prog):
        ctx = prog.ctx
        
        vertices, normals, indices = self.triangles()
        #normals = [normalize(vec) for vec in normals]
        
        vertices = ctx.buffer(np.array(vertices).astype('f4'))
        normals = ctx.buffer(np.array(normals).astype('f4'))
        indices = ctx.buffer(np.array(indices))
        
        centers = ctx.buffer(np.array(self.centers, dtype = "f4").flatten())
        radii = ctx.buffer(np.array(self.radii, dtype = "f4").flatten())

        vao = ctx.vertex_array(prog,
                               [(vertices, "3f4", "vert"),
                                (normals, "3f4", "normal"),
                                (centers, "3f4 /i", "center"),
                                (radii, "f4 /i", "radius")],
                               indices)

        return vao, moderngl.TRIANGLES, self.num

    def make_renderer(self, ctx):
        if self.num == 0:
            return Renderer()
            
        prog = ctx.program(
            vertex_shader = """
                #version 430

                uniform mat4 proj_mat;
                uniform mat4 view_mat;
                in vec3 center;
                in float radius;
                
                in vec3 vert;
                in vec3 normal;
                
                out vec3 v_normal;
                out vec4 v_pos_h;
                out vec3 v_center;

                void main() {
                    v_normal = normal;
                    v_center = center;
                    v_pos_h = vec4(radius * vert + center, 1);
                    gl_Position = proj_mat * view_mat * vec4(radius * vert + center, 1);
                }
                """,
            geometry_shader = None,
            fragment_shader = """
                #version 430
                in vec4 v_pos_h;
                in vec4 v_colour;
                in vec3 v_normal;
                in vec3 v_center;
                uniform sampler2D peel_tex;
                uniform vec2 scr_size;
                uniform bool do_peel;
                uniform int depth;
                uniform vec3 cam_pos;
                uniform samplerCube texcube;
                
                out vec4 f_colour;
                void main() {
                    vec3 v_pos = v_pos_h.xyz / v_pos_h.w;
                    vec3 cam_vec = normalize(v_pos - cam_pos);
                
                    vec3 light = normalize(vec3(1, -1, 1));
                    vec3 normal = normalize(v_normal);
                    float diff = abs(dot(light, normal));
                    float spec = pow(abs(dot(-cam_vec, reflect(light, normal))), 10);

                    float thicc = 1 / abs(dot(normal, cam_vec));
                    float alpha = 1 - max(0, pow(1 - v_colour.a, thicc)); //max(0, x) used to avoid weird flickering for some reason

                    spec *= alpha;

                    vec4 v_colour = vec4(texture(texcube, v_pos - v_center));
                    f_colour = vec4((0.7 + 0.3 * diff) * v_colour.rgb + spec * (1 - v_colour.rgb), alpha + spec * (1 - alpha));
                    
                    if (do_peel) {
                        if (gl_FragCoord.z <= texture2D(peel_tex, gl_FragCoord.xy / scr_size).x) {
                            discard;
                        }
                    }
                }"""
        )

        vao, mode, num_instances = self.make_vao(prog)

##        texcube = ctx.texture_cube([4096, 4096], 4, None, dtype = "f1")
##        for i, face in enumerate(self.faces):
##            texcube.write(i, face.read())

        texcube = self.texcube
        
        class ModelRenderer(Renderer):
            def __init__(self, prog):
                super().__init__([prog])
            def render(self):                
                try: prog["texcube"].value = 1
                except KeyError: pass
                texcube.use(1)
                vao.render(mode, instances = num_instances)

        return ModelRenderer(prog)




class UnitCylinder(SimpleModel):
    def __init__(self, steps, caps):
        super().__init__()
        assert type(steps) == int
        assert type(caps) == bool
        self.steps = steps
        self.caps = caps

    def add_line(self, a, b, rad, colour = None, source = 1):
        if np.linalg.norm([a[i] - b[i] for i in [0, 1, 2]]) < 0.000000001:
            return        
        v = [b[i] - a[i] for i in [0, 1, 2]]
        u = normalize(np.cross(v, [1, 0, 0]) if abs(np.dot(normalize(v), [1, 0, 0])) < 4 / 5 else np.cross(v, [0, 1, 0]))
        w = normalize(np.cross(v, u))
        mat = translation(*a) @ np.array([[u[0], v[0], w[0], 0], [u[1], v[1], w[1], 0], [u[2], v[2], w[2], 0], [0, 0, 0, 1]]) @ scale(rad, 0.5, rad) @ translation(0, 1, 0)
        self.add_instance(mat, colour, source)
        
    @functools.cache
    def triangles(self):
        import random

        n = self.steps
        circ = [[math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)] for i in range(n)]
        vertices = [[c[0], 1, c[1]] for c in circ] + [[c[0], -1, c[1]] for c in circ]
        normals = [[c[0], 0, c[1]] for c in circ] + [[c[0], 0, c[1]] for c in circ]
        indices = []
        for i in range(n):
            j = (i + 1) % n
            indices.append([i, j, i + n])
            indices.append([j, i + n, j + n])
        if self.caps:
            for i in range(n - 1):
                j = i + 1
                indices.append([0, i, j])
                indices.append([n, i + n, j + n])
        
        return vertices, normals, [[x, y, z, 1] for x, y, z in vertices], indices





class Triangle(SimpleModel):
    def __init__(self):
        super().__init__()

    def add_tri(self, a, b, c, colour = None, source = 1):
        ab = [b[k] - a[k] for k in [0, 1, 2]]
        ac = [c[k] - a[k] for k in [0, 1, 2]]
        
        mat = np.eye(4)
        mat[0:3, 0] = ab
        mat[0:3, 1] = ac
        mat[0:3, 2] = np.cross(ab, ac)
        self.add_instance(translation(*a) @ mat, colour, source)
        
    @functools.cache
    def triangles(self):
        import random
        vertices = [[0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0]]
        normals = [[0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]]
        indices = [0, 1, 2]
        return vertices, normals, [[x, y, z, 1] for x, y, z in vertices], indices










def render_tex(texture):
    ctx = texture.ctx
    prog = ctx.program(
        vertex_shader = """
            #version 430
            in vec2 vert;
            out vec2 tex_pos;
            void main() {
                tex_pos = 0.5 + 0.5 * vert;
                gl_Position = vec4(vert, 0, 1);
            }
        """,
        fragment_shader = """
            #version 430
            in vec2 tex_pos;
            uniform sampler2D tex;
            out vec4 f_colour;
            void main() {
                f_colour = texture2D(tex, tex_pos);
            }"""
    )
    
    vertices = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
    indices = [[0, 1, 3], [0, 2, 3]]
    
    vertices = ctx.buffer(np.array(vertices).astype('f4'))
    indices = ctx.buffer(np.array(indices))

    vao = ctx.vertex_array(prog,
                           [(vertices, "2f4", "vert")],
                           indices)

    texture.use(0)
    prog["tex"].value = 0
    vao.render(moderngl.TRIANGLES, instances = 1)






def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ZeroDivisionError()
    else:
        return v / np.linalg.norm(v)
    
def translation(tx, ty, tz):
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])

def scale(sx, sy, sz):
    return np.array([[sx, 0, 0, 0],
                     [0, sy, 0, 0],
                     [0, 0, sz, 0],
                     [0, 0, 0, 1]])
    
def rotate(u, angle):
    u = np.array(u)
    c = math.cos(angle)
    s = math.sin(angle)
    u = normalize(u)
    u_cross = np.array([[0, u[2], -u[1]],
                        [-u[2], 0, u[0]],
                        [u[1], -u[0], 0]])
    mat = np.eye(4)
    mat[0:3, 0:3] = np.eye(3) * c + s * u_cross + (1 - c) * np.outer(u, u)
    return mat
    
def persp(n, f, fov_y, a):
    c = 1 / math.tan(math.pi * fov_y / 360)
    return np.array([[c / a, 0, 0, 0],
                     [0, c, 0, 0],
                     [0, 0, -(n + f) / (n - f), 2 * n * f / (n - f)],
                     [0, 0, 1, 0]])


class Camera():
    def __init__(self, pos, vec, fov = 70, near = 0.01, far = 10000):
        super().__init__()
        assert len(pos) == 3
        assert len(vec) == 3
        self.pos = pos
        self.vec = vec
        self.fov = fov
        self.near = near
        self.far = far
        self.fly_speed = 3

    def theta(self):
        return math.atan2(-self.vec[0], self.vec[2])
    def phi(self):
        return math.atan2(self.vec[1], math.sqrt(self.vec[2] ** 2 + self.vec[0] ** 2))

    def get_cam_mat(self):
        theta = self.theta()
        phi = self.phi()
        return translation(self.pos[0], self.pos[1], self.pos[2]) @ rotate([0, 1, 0], theta) @ rotate([1, 0, 0], phi)
    def get_view_mat(self):
        return np.linalg.inv(self.get_cam_mat())
    def get_proj_mat(self, width, height):
        return persp(self.near, self.far, self.fov, width / height)
    

    def convert_inwards(self, width, height, x, y):
        side = math.sqrt(width * height)
        return self.center[0] + width * self.scale * (2 * x / width - 1) / side, self.center[1] + height * self.scale * (2 * y / height - 1) / side
    def convert_outwards(self, width, height, x, y):
        side = math.sqrt(width * height)
        return width * (side * (x - self.center[0]) / (width * self.scale) + 1) / 2, height * (side * (y - self.center[1]) / (height * self.scale) + 1) / 2


    def tick(self, dt):        
        T_vec = self.vec
        U_vec = [0, 1, 0]
        R_vec = normalize(np.cross(U_vec, T_vec))
        F_vec = normalize(np.cross(R_vec, U_vec))

        if pygame.key.get_pressed()[pygame.K_w]:
            pos = self.pos
            self.pos = [pos[i] + self.fly_speed * F_vec[i] * dt for i in [0, 1, 2]]
        if pygame.key.get_pressed()[pygame.K_s]:
            pos = self.pos
            self.pos = [pos[i] - self.fly_speed * F_vec[i] * dt for i in [0, 1, 2]]
            
        if pygame.key.get_pressed()[pygame.K_a]:
            pos = self.pos
            self.pos = [pos[i] - self.fly_speed * R_vec[i] * dt for i in [0, 1, 2]]
        if pygame.key.get_pressed()[pygame.K_d]:
            pos = self.pos
            self.pos = [pos[i] + self.fly_speed * R_vec[i] * dt for i in [0, 1, 2]]

        if pygame.key.get_pressed()[pygame.K_z]:
            pos = self.pos
            self.pos = [pos[i] - self.fly_speed * U_vec[i] * dt for i in [0, 1, 2]]
        if pygame.key.get_pressed()[pygame.K_SPACE]:
            pos = self.pos
            self.pos = [pos[i] + self.fly_speed * U_vec[i] * dt for i in [0, 1, 2]]

    def mousemove(self, dx, dy):
        theta = self.theta()
        phi = self.phi()
        
        theta -= 0.002 * dx
        phi -= 0.002 * dy

        hpi = math.pi / 2
        phi = min(max(phi, -hpi), hpi)

        self.vec = [-math.cos(phi) * math.sin(theta), math.sin(phi), math.cos(phi) * math.cos(theta)]








class Window(pgbase.core.Window):
    def __init__(self, *args, bg_colour = [1, 1, 1, 1], peel_depth = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera = Camera([0, 0, 0], [1, 0, 0])
        self.models = []

        self.peel_depth = peel_depth
        self.bg_colour = bg_colour

        self.has_focus = False

    def set_focus(self, focus):
        if focus:
            self.has_focus = True
            pygame.mouse.set_pos(self.rect[0] + 0.5 * self.rect[2], self.rect[1] + 0.5 * self.rect[3])
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
        else:
            self.has_focus = False
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)

    def set_rect(self, rect):
        super().set_rect(rect)
        self.clear_renderer_cache()

    def set_uniforms(self, progs, peel_tex, depth):
        width, height = self.rect[2:4]
        camera = self.camera
        
        proj_mat = camera.get_proj_mat(width, height)
        view_mat = camera.get_view_mat()
        vp_mat = proj_mat @ view_mat

        if not peel_tex is None:
            peel_tex.use(0)

        for prog in progs:
            try: prog["scr_size"].value = (width, height)
            except KeyError: pass
            try: prog["proj_mat"].value = tuple(proj_mat.transpose().flatten())
            except KeyError: pass
            try: prog["view_mat"].value = tuple(view_mat.transpose().flatten())
            except KeyError: pass
            try: prog["cam_pos"].value = tuple(camera.pos)
            except KeyError: pass
            try: prog["cam_vec"].value = tuple(camera.vec)
            except KeyError: pass
            try: prog["do_peel"].value = not peel_tex is None
            except KeyError: pass
            try: prog["peel_tex"].value = 0
            except KeyError: pass
            try: prog["depth"].value = depth
            except KeyError: pass

    @functools.cached_property
    def renderer(self):
        ctx = self.ctx
        
        model_renderers = []
        for model in self.models:
            renderer = model.make_renderer(ctx)
            if not isinstance(renderer, Renderer):
                raise Exception(f"{renderer} is not a Renderer")
            model_renderers.append(renderer)
            
        def renderer(camera, fbo, width, height, peel_tex, depth):
            fbo.use()
            self.set_uniforms(sum([renderer.progs for renderer in model_renderers], []), peel_tex, depth)
            for renderer in model_renderers:
                renderer.render()

        class PeelPartCache():
            def __init__(self):
                self.all_tex = []

            def __del__(self):
                for tex in self.all_tex:
                    tex.release()
                
            @functools.cache
            def __call__(self, n, width, height):
                peel_parts = []
                for i in range(n + 1):
                    colour_tex = ctx.texture([width, height], 4)
                    depth_tex = ctx.depth_texture([width, height])
                    fbo = ctx.framebuffer(colour_tex, depth_tex)
                    peel_parts.append((fbo, colour_tex, depth_tex))
                    self.all_tex.append(colour_tex)
                    self.all_tex.append(depth_tex)
                return peel_parts
            
        get_peel_parts = PeelPartCache()
            
        def peel_renderer(camera, fbo, width, height):
            n = self.peel_depth
            
            #setup - clear fbo
            fbo.use()
            ctx.clear(*self.bg_colour)
            if n == 0:
                ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND)
                renderer(camera, fbo, width, height, None, 0)
            else:
                peel_parts = get_peel_parts(n, width, height)
                ctx.enable_only(moderngl.DEPTH_TEST)
                tmp_fbo, _, peel_tex = peel_parts[0]
                tmp_fbo.use()
                ctx.clear(depth = -1)
                for depth in range(n):
                    tmp_fbo, _, depth_tex = peel_parts[depth + 1]
                    peel_tex.compare_func = ""
                    depth_tex.compare_func = "<="
                    tmp_fbo.use()
                    ctx.clear()
                    renderer(camera, tmp_fbo, width, height, peel_tex, depth)
                    peel_tex = depth_tex

                fbo.use()
                ctx.enable_only(moderngl.BLEND)
                for depth in reversed(range(n)):
                    _, colour_tex, _ = peel_parts[depth + 1]
                    render_tex(colour_tex)
                
        return peel_renderer

    def draw_model(self, model):
        assert isinstance(model, Model)
        self.models.append(model)
        self.clear_renderer_cache()

    def clear_renderer_cache(self):
        if hasattr(self, "renderer"):
            del self.renderer

    def draw(self):
        super().draw()
        self.renderer(self.camera, self.ctx.screen, self.rect[2], self.rect[3])
        
    def tick(self, dt):
        super().tick(dt)
        self.camera.tick(dt)
        
    def event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if pgbase.tools.in_rect(event.pos, self.rect):
                if self.has_focus:
                    if not pygame.mouse.get_pressed()[2]:
                        self.camera.mousemove(*event.rel)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.has_focus:
                    self.set_focus(False)
                    return

        super().event(event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if pgbase.tools.in_rect(event.pos, self.rect):
                if event.button == 1:
                    if not self.has_focus:
                        self.set_focus(True)
                if event.button in {4, 5}:
                    if not pygame.mouse.get_pressed()[2]:
                        mul = 0.9
                        if event.button == 4:
                            mul = 1 / mul
                        self.camera.fly_speed *= mul

                




        
def test():
    pgbase.core.Window.setup()

    window = Window()
    
    model = UnitCylinder(24, True)
    for _ in range(10):
        import random
        r = lambda : random.uniform(0, 1)
        model.add_line([10 * r(), 10 * r(), 10 * r()], [10 * r(), 10 * r(), 10 * r()], 1, [r(), r(), r(), r()])

    window.draw_model(model)
    
    pgbase.core.run(window)
    pygame.quit()
    sys.exit()


















