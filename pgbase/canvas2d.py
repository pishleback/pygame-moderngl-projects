import pgbase
import numpy as np
import math
import pygame
import moderngl
import shapely.geometry
import triangle





class ShapelyModel():
    class Polygon():
        def __init__(self, polygon, colour):
            assert type(polygon) == shapely.geometry.Polygon
            self.polygon = polygon
            self.colour = colour

        def ext(self):
            return tuple(self.polygon.exterior.coords[:-1])

        def ints(self):
            for linring in self.polygon.interiors:
                pt = shapely.geometry.Polygon(linring).representative_point()
                yield tuple(reversed(linring.coords[:-1])), (pt.x, pt.y)
            
            
    def __init__(self, ctx):
        self.ctx = ctx
        self.polygons = []
        self.prog = None
        self.vao = None
        self.update_vao()

    def clear(self):
        self.polygons =[]
        self.update_vao()

    def add_shape(self, shape, colour = (1, 0, 1, 1)):
        assert len(colour) == 4
        if type(shape) == shapely.geometry.Polygon:
            self.polygons.append(ShapelyModel.Polygon(shape, colour))
        elif type(shape) == shapely.geometry.MultiPolygon:
            for sub_shape in shape.geoms:
                self.add_shape(sub_shape, colour)
        else:
            raise NotImplementedError(f"drawing {type(shape)} is not implemented")

    def update_vao(self):
        import time
        def gen_triangles():
            vert_offset = 0
            all_verts = []
            all_tris = []
            all_colours = []

            last_t = time.time()

            count = 0
            for poly in self.polygons:
                count += 1
                if (time.time() - last_t) > 1:
                    print(str(round(100 * count / len(self.polygons), 2)) + "%")
                    last_t = time.time()
                pts = []
                segs = []
                holes = []
                idx_offset = 0

                ext = poly.ext()
                for idx, pt in enumerate(ext):
                    all_colours.append(poly.colour)
                    pts.append(pt)
                    segs.append((idx, (idx + 1) % len(ext)))
                
                for coords, hole in poly.ints():
                    holes.append(hole)
                    idx_offset = len(pts)
                    for idx, pt in enumerate(coords):
                        all_colours.append(poly.colour)
                        pts.append(pt)
                        segs.append((idx_offset + idx, idx_offset + (idx + 1) % len(coords)))

                if len(holes) == 0:
                    A = dict(vertices = np.array(pts), segments = np.array(segs))
                else:
                    A = dict(vertices = np.array(pts), segments = np.array(segs), holes = holes)
                B = triangle.triangulate(A, 'p')
                all_verts.append(B["vertices"])
                all_tris.append(B["triangles"] + vert_offset)
                vert_offset += B["vertices"].shape[0]

            return np.vstack([np.empty([0, 2])] + all_verts), np.vstack([np.empty([0, 3], dtype = "int32")] + all_tris), np.vstack([np.empty([0, 4])] + all_colours)

        verts, tris, colours = gen_triangles()
        assert len(verts) == len(colours)

        self.prog = self.ctx.program(
            vertex_shader = """
                #version 430
                in vec2 in_vert;
                in vec4 in_colour;
                out vec4 out_colour;
                out vec4 gl_Position;
                uniform vec2 cam_center;
                uniform mat2 cam_mat_inv;

                void main() {
                    gl_Position = vec4(cam_mat_inv * (in_vert - cam_center), 0, 1);
                    out_colour = in_colour;
                }
            """,
            fragment_shader = """
                #version 430
                in vec4 out_colour;
                out vec4 f_colour;

                void main() {
                    f_colour = out_colour;
                }
    
            """,
        )

        assert (empty := (verts.shape[0] == 0)) == (tris.shape[0] == 0) == (colours.shape[0] == 0)
        if empty:
            self.vao = self.ctx.vertex_array(self.prog, [])
        else:
            vertex_buf = self.ctx.buffer(np.array(verts).astype('f4'))
            colour_buf = self.ctx.buffer(np.array(colours).astype('f4'))
            triangles = self.ctx.buffer(np.array(tris))
            self.vao = self.ctx.vertex_array(self.prog,
                                             [(vertex_buf, "2f4", "in_vert"),
                                              (colour_buf, "4f4", "in_colour")],
                                             triangles)






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
            try: prog["cam_center"].value = tuple(self.center)
            except KeyError: pass
            try: prog["cam_mat"].value = tuple(cam_mat.flatten())
            except KeyError: pass
            try: prog["cam_mat_inv"].value = tuple(cam_mat_inv.flatten())
            except KeyError: pass
            try: prog["rect"].value = tuple(self.rect)
            except KeyError: pass

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


	
