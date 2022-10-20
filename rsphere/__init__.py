import pgbase
import pygame
import sys
import os
import moderngl
import numpy as np





class Rsphere():
    def __init__(self):
        pass

    def get_texcube(self, ctx, dim):
        assert type(dim) == int and dim >= 1

        def gen_faces():
            for i in range(6):
                prog = ctx.program(
                    vertex_shader = """
                        #version 430
                        in vec2 vert;
                        out vec3 v_pos;

                        uniform uint face;

                        void main() {
                            //figure out our position on the sphere given the texture coordinate "vert" and the face index
                            if (face == 0) {
                                v_pos = vec3(1, -vert.y, -vert.x);
                            } else if (face == 1) {
                                v_pos = vec3(-1, -vert.y, vert.x);
                            } else if (face == 2) {
                                v_pos = vec3(vert.x, 1, vert.y);
                            } else if (face == 3) {
                                v_pos = vec3(vert.x, -1, -vert.y);
                            } else if (face == 4) {
                                v_pos = vec3(vert.x, -vert.y, 1);
                            } else {
                                v_pos = vec3(-vert.x, -vert.y, -1);
                            }
                            
                            gl_Position = vec4(vert, 0, 1);
                        }
                        """,
                    geometry_shader = None,
                    fragment_shader = """
                        #version 430
                        in vec3 v_pos;
                        out vec4 f_colour;

                        dvec2 mult(dvec2 a, dvec2 b) {
                            return dvec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
                        }

                        float get_j(vec2 start_pos) {
                            int iter = 100;
                            float colour_scale = 0.2;
                        
                            float colour_scale_x = colour_scale;
                            float colour_offset_x = 0.1;
                            float colour_scale_y = colour_scale * (1 + sqrt(5)) / 2;
                            float colour_offset_y = 0.1;
                            
                            int i;
                            dvec2 c = start_pos;
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
                                return j;
                            } else {
                                return 0;
                            }
                        }


                        vec2 mult(vec2 a, vec2 b) {
                            return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
                        }

                        vec3 hsl2rgb(vec3 c){return mix(c.bbb,mix(clamp(vec3(-1,2,2)-abs(c.r-vec3(3,2,4))*vec3(-1,1,1),0.,1.),vec3(c.b>.5),abs(.5-c.b)*2.),c.g);}

                        vec3 complex_to_rgb(vec2 z) {
                            float tau = 6.28318530718;
                            return hsl2rgb(vec3(mod(atan(z.y, z.x), tau), 1, 0.5 + 0.3 * sin(6 * log(length(z)))));
                        }

                        vec4 colour(vec2 z) {
                            //return vec4(get_j(z), 0, 0, 1);
                            return vec4(complex_to_rgb(mult(mult(z, mult(z, z)), mult(z, z)) - mult(z, vec2(0, 1)) + vec2(1, 1)), 1);
                        }
                        
                        void main() {
                            vec3 s_pos = normalize(v_pos);
                            s_pos.y = -s_pos.y;
                            
                            f_colour = colour(s_pos.xz / (1 - s_pos.y));
                        }


                        """
                )
                try: prog["face"] = i
                except KeyError: pass
                
                vertices, indices = [[-1, -1], [1, -1], [-1, 1], [1, 1]], [[0, 1, 2], [1, 2, 3]]
                vertices = ctx.buffer(np.array(vertices).astype('f4'))
                indices = ctx.buffer(np.array(indices))
                vao = ctx.vertex_array(prog,
                                       [(vertices, "2f4", "vert")],
                                       indices)

                tex = ctx.texture([dim, dim], 4)
                fbo = ctx.framebuffer(tex)
                fbo.use()
                ctx.clear()

                vao.render(moderngl.TRIANGLES, instances = 1)
                
                yield i, tex

        texcube = ctx.texture_cube([dim, dim], 4, None, dtype = "f1")
        for i, face in gen_faces():
            texcube.write(i, face.read())
            face.release()

        return texcube

        


class Window(pgbase.core.Window):
    def __init__(self):
        super().__init__()

    def tick(self, dt):
        print(dt)
        rsphere = Rsphere()
        texcube = rsphere.get_texcube(self.ctx, 512)
        print(texcube)
    def draw(self):
        pass
    def event(self, event):
        pass




def run():
    def draw(window):
##        model = pgbase.canvas3d.UnitSphere(ref = 4)
##        for x in range(-1, 2):
##            for y in range(-1, 2):
##                for z in range(-1, 2):
##                    model.add_sphere([x, y, z], 0.3, [1, 0, 0, 0.2])
##        window.draw_model(model)

        faces = [pgbase.tools.load_tex(window.ctx, os.path.join("rsphere", "water.jpg")) for _ in range(6)]

        rsphere = Rsphere()
        model = pgbase.canvas3d.TextureSphere(rsphere.get_texcube(window.ctx, 4096), ref = 3)
        
        model.add_sphere([0, 0, 0], 1)
        
        window.draw_model(model)

        

    
    pgbase.core.Window.setup([1600, 1000])

    window = pgbase.canvas3d.Window(peel_depth = 0)
    draw(window)

    pgbase.core.run_root(window)


























    

