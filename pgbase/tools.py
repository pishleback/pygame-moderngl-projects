import imageio
import numpy as np
import pygame
import moderngl

def render_tex(texture, p1 = [-1, -1], p2 = [1, 1], alpha = 1):
    ctx = texture.ctx
    prog = ctx.program(
        vertex_shader = """
            #version 430
            in vec2 vert;
            out vec2 tex_pos;

            uniform vec2 p1;
            uniform vec2 p2;
            
            void main() {
                tex_pos = vert;
                gl_Position = vec4(p1 + vert * (p2 - p1), 0, 1);
            }
        """,
        fragment_shader = """
            #version 430
            in vec2 tex_pos;
            uniform sampler2D tex;
            uniform float alpha;
            out vec4 f_colour;
            void main() {
                f_colour = texture2D(tex, tex_pos);
                f_colour.a = f_colour.a * alpha;
            }"""
    )
    
    vertices = [[0, 0], [1, 0], [0, 1], [1, 1]]
    indices = [[0, 1, 3], [0, 2, 3]]
    
    vertices = ctx.buffer(np.array(vertices).astype('f4'))
    indices = ctx.buffer(np.array(indices))

    vao = ctx.vertex_array(prog,
                           [(vertices, "2f4", "vert")],
                           indices)

    texture.use(0)
    prog["tex"].value = 0
    prog["p1"].value = tuple(p1)
    prog["p2"].value = tuple(p2)
    prog["alpha"].value = alpha
    vao.render(moderngl.TRIANGLES, instances = 1)



def load_tex(ctx, path):
    data = imageio.imread(path)
    return np_uint8_to_tex(ctx, data)


def pysurf_to_np_uint8(surf):
    colour = pygame.surfarray.pixels3d(surf)
    alpha = pygame.surfarray.pixels_alpha(surf).reshape([surf.get_width(), surf.get_height(), 1])
    assert colour.dtype == np.uint8
    assert alpha.dtype == np.uint8
    data = np.concatenate([colour, alpha], axis = 2)
    return data


def np_uint8_to_tex(ctx, data):
    data = np.array(data, dtype = np.uint8)
    assert data.shape[2] <= 4
    if data.shape[2] < 4:
        data = np.concatenate([data, 255 * np.ones((data.shape[0], data.shape[1], 4 - data.shape[2]), dtype = np.uint8)], axis = 2)
    h, w, d = data.shape
    return ctx.texture([w, h], d, np.flip(data, 0).flatten(), dtype = "f1")


def tex_to_np(tex):
    if tex.dtype == "f1":
        data = np.frombuffer(tex.read(), dtype = np.uint8).reshape((tex.height, tex.width, tex.components))
        return data
    else:
        raise NotImplementedError(f"dtype={tex.dtype} is not implemented")


def in_rect(pos, rect):
    return 0 <= pos[0] - rect[0] <= rect[2] and 0 <= pos[1] - rect[1] <= rect[2]

        




































    




