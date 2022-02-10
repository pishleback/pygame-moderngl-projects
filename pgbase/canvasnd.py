import pgbase
import pygame
import numpy as np
import moderngl
import functools
import math




class SplitGlsl():
    def __init__(self, num):
        assert num >= 2
        self.num = num

    @property
    def len(self):
        return len(self.sizes)

    @functools.cached_property
    def sizes(self):
        n = self.num
        if n < 4:
            sizes = [n]
        elif n % 4 == 0:
            sizes = [4] * (n // 4)
        elif n % 4 == 1:
            sizes = [2, 3] + [4] * ((n - 5) // 4)
        elif n % 4 == 2:
            sizes = [2] + [4] * ((n - 2) // 4)
        elif n % 4 == 3:
            sizes = [3] + [4] * ((n - 3) // 4)
        return sizes

    @functools.cached_property
    def idxs(self):
        return [sum(self.sizes[:i]) for i in range(self.len + 1)]

    def split_name(self, name):
        return ["split_" + name + "_" + str(i) for i in range(self.len)]

    def split_data(self, data, axis):
        data = np.array(data)
        assert data.shape[axis] == self.num
        perm = list(range(len(data.shape)))
        perm[0] = axis
        perm[axis] = 0
        return [(data.transpose(perm)[self.idxs[i] : self.idxs[i] + self.sizes[i]]).transpose(perm) for i in range(self.len)]


def rot_axes(dim, a, b, t):
    s = math.sin(t)
    c = math.cos(t)
    m = np.eye(dim)
    m[a, a] = c
    m[b, a] = s
    m[a, b] = -s
    m[b, b] = c
    return m

        




class Node(pgbase.canvas3d.BaseModel):
    def __init__(self, dim):
        self.dim = dim
        self.pos_spliter = SplitGlsl(self.dim)
        split_name_nd3d_mat = self.pos_spliter.split_name("nd3d_mat")
        split_name_pos = self.pos_spliter.split_name("pos")
        super().__init__("""
            #version 430

            uniform vec4 nd3d_offset;
            """ + "\n".join([f"uniform mat{self.pos_spliter.sizes[i]}x4 {split_name_nd3d_mat[i]};" for i in range(self.pos_spliter.len)]) + """
            """ + "\n".join([f"in vec{self.pos_spliter.sizes[i]} {split_name_pos[i]};" for i in range(self.pos_spliter.len)]) + """
            
            in float radius;
            in vec4 colour;

            out vec4 g_pos;
            out float g_radius;
            out vec4 g_colour;
            
            void main() {
                g_radius = radius;
                g_colour = colour;
                g_pos = """ + " + ".join([f"{split_name_nd3d_mat[i]} * {split_name_pos[i]}" for i in range(self.pos_spliter.len)]) + """ + nd3d_offset;
                
            }
        """, """
            #version 430
            layout (points) in;
            layout (triangle_strip, max_vertices = 100) out;

            in vec4 g_pos[];
            in vec4 g_colour[];
            in float g_radius[];

            out vec4 v_pos_h;
            out vec4 v_colour;
            out vec3 v_normal;

            uniform mat4 proj_mat;
            uniform mat4 view_mat;

            mat4 vp_mat = proj_mat * view_mat;

            void emit_tri(vec3 base_pos, vec3 p1, vec3 p2, vec3 p3, float mult) {            
                v_colour = g_colour[0];
                
                v_normal = p1;
                v_pos_h = vec4(base_pos + mult * g_radius[0] * p1, 1);
                gl_Position = vp_mat * v_pos_h;
                EmitVertex();
                v_normal = p2;
                v_pos_h = vec4(base_pos + mult * g_radius[0] * p2, 1);
                gl_Position = vp_mat * v_pos_h;
                EmitVertex();
                v_normal = p3;
                v_pos_h = vec4(base_pos + mult * g_radius[0] * p3, 1);
                gl_Position = vp_mat * v_pos_h;
                EmitVertex();
                EndPrimitive();
            }

            void main() {
                vec4 base_pos = g_pos[0];
                if (base_pos.w > 0) {
                    vec3 verts[12] = vec3[12](vec3(0, 1, 0),
                                              vec3(0.8944271909999159, 0.4472135954999579, 0.0),
                                              vec3(0.27639320225002106, 0.4472135954999579, 0.8506508083520399),
                                              vec3(-0.7236067977499788, 0.4472135954999579, 0.5257311121191337),
                                              vec3(-0.7236067977499789, 0.4472135954999579, -0.5257311121191335),
                                              vec3(0.27639320225002084, 0.4472135954999579, -0.85065080835204),
                                              vec3(0.7236067977499789, -0.4472135954999579, 0.5257311121191336),
                                              vec3(-0.27639320225002095, -0.4472135954999579, 0.85065080835204),
                                              vec3(-0.8944271909999159, -0.4472135954999579, 1.0953573965284052e-16),
                                              vec3(-0.2763932022500211, -0.4472135954999579, -0.8506508083520399),
                                              vec3(0.7236067977499788, -0.4472135954999579, -0.5257311121191338),
                                              vec3(0, -1, 0));

                    int idxs[60] = int[60](0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 1,
                                           11, 6, 7, 11, 7, 8, 11, 8, 9, 11, 9, 10, 11, 10, 6,
                                           1, 2, 6, 2, 3, 7, 3, 4, 8, 4, 5, 9, 5, 1, 10,
                                           2, 6, 7, 3, 7, 8, 4, 8, 9, 5, 9, 10, 1, 10, 6);

                    vec3 p1;
                    vec3 p2;
                    vec3 p3;
                    for (int k = 0; k < 20; k ++) {
                        p1 = verts[idxs[3 * k + 0]];
                        p2 = verts[idxs[3 * k + 1]];
                        p3 = verts[idxs[3 * k + 2]];
                        emit_tri(base_pos.xyz / base_pos.w, p1, p2, p3, 1 / base_pos.w);
                    }
                }
            }
        """)

        self.poses = []
        self.colours = []
        self.radii = []
        self.num = 0

    def make_vao(self, prog):
        ctx = prog.ctx

        pos_buffers = []
        split_data = self.pos_spliter.split_data(self.poses, 1)
        split_name = self.pos_spliter.split_name("pos")
        for i in range(self.pos_spliter.len):
            buffer = ctx.buffer(split_data[i].astype('f4'))
            pos_buffers.append((buffer, f"{self.pos_spliter.sizes[i]}f4", split_name[i]))
                
        radii = ctx.buffer(np.array(self.radii).astype('f4'))
        colours = ctx.buffer(np.array(self.colours).astype('f4'))
        indices = ctx.buffer(np.array(range(self.num)))
        
        vao = ctx.vertex_array(prog,
                               pos_buffers + [(radii, "f4", "radius"),
                                              (colours, "4f4", "colour")],
                               indices)

        return vao, moderngl.POINTS, 1

    def add_node(self, pos, colour, radius):
        assert len(pos) == self.dim
        assert len(colour) == 4
        
        self.poses.append(pos)
        self.colours.append(colour)
        self.radii.append(radius)
        self.num += 1
        





class Edge(pgbase.canvas3d.BaseModel):
    def __init__(self, dim):
        self.dim = dim
        self.pos_spliter = SplitGlsl(self.dim)
        split_name_nd3d_mat = self.pos_spliter.split_name("nd3d_mat")
        split_name_pos1 = self.pos_spliter.split_name("pos1")
        split_name_pos2 = self.pos_spliter.split_name("pos2")
        super().__init__("""
            #version 430

            uniform vec4 nd3d_offset;
            """ + "\n".join([f"uniform mat{self.pos_spliter.sizes[i]}x4 {split_name_nd3d_mat[i]};" for i in range(self.pos_spliter.len)]) + """
            """ + "\n".join([f"in vec{self.pos_spliter.sizes[i]} {split_name_pos1[i]};" for i in range(self.pos_spliter.len)]) + """
            """ + "\n".join([f"in vec{self.pos_spliter.sizes[i]} {split_name_pos2[i]};" for i in range(self.pos_spliter.len)]) + """
            
            in float radius;
            in vec4 colour;

            out vec4 g_pos1;
            out vec4 g_pos2;
            out float g_radius;
            out vec4 g_colour;
            
            void main() {
                g_radius = radius;
                g_colour = colour;
                g_pos1 = """ + " + ".join([f"{split_name_nd3d_mat[i]} * {split_name_pos1[i]}" for i in range(self.pos_spliter.len)]) + """ + nd3d_offset;
                g_pos2 = """ + " + ".join([f"{split_name_nd3d_mat[i]} * {split_name_pos2[i]}" for i in range(self.pos_spliter.len)]) + """ + nd3d_offset;
                
            }
        """, """
            #version 430
            layout (points) in;
            layout (triangle_strip, max_vertices = 100) out;

            in vec4 g_pos1[];
            in vec4 g_pos2[];
            in vec4 g_colour[];
            in float g_radius[];

            out vec4 v_pos_h;
            out vec4 v_colour;
            out vec3 v_normal;

            uniform mat4 proj_mat;
            uniform mat4 view_mat;

            mat4 vp_mat = proj_mat * view_mat;

            void emit_tri(vec4 start_pos_h, vec4 end_pos_h, vec3 perp1, vec3 perp2, float a, float b) {
                //start_mult = min(start_mult, 1);
                //end_mult = min(end_mult, 1);

                vec3 start_pos = start_pos_h.xyz / start_pos_h.w;
                vec3 end_pos = end_pos_h.xyz / end_pos_h.w;
                
                float ca = cos(a);
                float sa = sin(a);
                float cb = cos(b);
                float sb = sin(b);

                vec3 perp_a = ca * perp1 + sa * perp2;
                vec3 perp_b = cb * perp1 + sb * perp2;

                vec3 va = (end_pos + 1 * perp_a) - (start_pos + 1 * perp_a);
                vec3 vb = (end_pos + 1 * perp_b) - (start_pos + 1 * perp_b);

                vec3 normal_a = cross(va, cross(va, perp_a));
                vec3 normal_b = cross(vb, cross(vb, perp_b));

                vec4 p1 = start_pos_h + vec4(perp_a, 0);
                vec4 p2 = end_pos_h + vec4(perp_a, 0);
                vec4 p3 = start_pos_h + vec4(perp_b, 0);
                vec4 p4 = end_pos_h + vec4(perp_b, 0);

                v_colour = g_colour[0];

                v_normal = normal_a;
                v_pos_h = p1;
                gl_Position = vp_mat * p1;
                EmitVertex();
                v_normal = normal_a;
                v_pos_h = p2;
                gl_Position = vp_mat * p2;
                EmitVertex();
                v_normal = normal_b;
                v_pos_h = p3;
                gl_Position = vp_mat * p3;
                EmitVertex();
                EndPrimitive();

                v_normal = normal_b;
                v_pos_h = p4;
                gl_Position = vp_mat * p4;
                EmitVertex();
                v_normal = normal_a;
                v_pos_h = p2;
                gl_Position = vp_mat * p2;
                EmitVertex();
                v_normal = normal_b;
                v_pos_h = p3;
                gl_Position = vp_mat * p3;
                EmitVertex();
                EndPrimitive();
            }

            void main() {
                
                vec4 start_base_pos_h = g_pos1[0];
                vec4 end_base_pos_h = g_pos2[0];
                
                if (start_base_pos_h.w > 0 || end_base_pos_h.w > 0) {
                    if (start_base_pos_h.w < 0) {
                        float f = - start_base_pos_h.w / (end_base_pos_h.w - start_base_pos_h.w);
                        start_base_pos_h = start_base_pos_h + f * (end_base_pos_h - start_base_pos_h);
                        start_base_pos_h.w = 0.00001;
                    } else if (end_base_pos_h.w < 0) {
                        float f = - end_base_pos_h.w / (start_base_pos_h.w - end_base_pos_h.w);
                        end_base_pos_h = end_base_pos_h + f * (start_base_pos_h - end_base_pos_h);
                        end_base_pos_h.w = 0.00001;
                    }
                    
                    vec3 start_base_pos = start_base_pos_h.xyz / start_base_pos_h.w;
                    vec3 end_base_pos = end_base_pos_h.xyz / end_base_pos_h.w;

                    vec3 para = normalize(start_base_pos - end_base_pos);
                    vec3 other;
                    if (abs(dot(para, vec3(0, 1, 0))) > 0.5) {
                        other = vec3(1, 0, 0);
                    } else {
                        other = vec3(0, 1, 0);
                    }
                    vec3 perp1 = normalize(cross(para, other));
                    vec3 perp2 = cross(para, perp1);

                    perp1 = perp1 * g_radius[0];
                    perp2 = perp2 * g_radius[0];

                    float step = 2 * 3.1415926535 / 10;
                    for (int i = 0; i < 10; i++) {
                        emit_tri(start_base_pos_h, end_base_pos_h, perp1, perp2, i * step, (i + 1) * step);
                    }
                }
            }
        """)

        self.poses1 = []
        self.poses2 = []
        self.colours = []
        self.radii = []
        self.num = 0

    def make_vao(self, prog):
        ctx = prog.ctx

        pos1_buffers = []
        split_data = self.pos_spliter.split_data(self.poses1, 1)
        split_name = self.pos_spliter.split_name("pos1")
        for i in range(self.pos_spliter.len):
            buffer = ctx.buffer(split_data[i].astype('f4'))
            pos1_buffers.append((buffer, f"{self.pos_spliter.sizes[i]}f4", split_name[i]))

        pos2_buffers = []
        split_data = self.pos_spliter.split_data(self.poses2, 1)
        split_name = self.pos_spliter.split_name("pos2")
        for i in range(self.pos_spliter.len):
            buffer = ctx.buffer(split_data[i].astype('f4'))
            pos2_buffers.append((buffer, f"{self.pos_spliter.sizes[i]}f4", split_name[i]))
                
        radii = ctx.buffer(np.array(self.radii).astype('f4'))
        colours = ctx.buffer(np.array(self.colours).astype('f4'))
        indices = ctx.buffer(np.array(range(self.num)))
        
        vao = ctx.vertex_array(prog,
                               pos1_buffers + pos2_buffers + [(radii, "f4", "radius"),
                                              (colours, "4f4", "colour")],
                               indices)

        return vao, moderngl.POINTS, 1

    def add_edge(self, pos1, pos2, colour, radius):
        assert len(pos1) == self.dim
        assert len(pos2) == self.dim
        assert len(colour) == 4
        
        self.poses1.append(pos1)
        self.poses2.append(pos2)
        self.colours.append(colour)
        self.radii.append(radius)
        self.num += 1




class PencilEdge(pgbase.canvas3d.Model):
    def __init__(self, dim, dash_len):
        self.dim = dim
        self.pos_spliter = SplitGlsl(self.dim)
        super().__init__()

        self.dash_len = dash_len

        self.poses1 = []
        self.poses2 = []
        self.colours = []
        self.radii = []
        self.num = 0

    def make_vao(self, prog):
        ctx = prog.ctx

        pos1_buffers = []
        split_data = self.pos_spliter.split_data(self.poses1, 1)
        split_name = self.pos_spliter.split_name("pos1")
        for i in range(self.pos_spliter.len):
            buffer = ctx.buffer(split_data[i].astype('f4'))
            pos1_buffers.append((buffer, f"{self.pos_spliter.sizes[i]}f4", split_name[i]))

        pos2_buffers = []
        split_data = self.pos_spliter.split_data(self.poses2, 1)
        split_name = self.pos_spliter.split_name("pos2")
        for i in range(self.pos_spliter.len):
            buffer = ctx.buffer(split_data[i].astype('f4'))
            pos2_buffers.append((buffer, f"{self.pos_spliter.sizes[i]}f4", split_name[i]))
                
        radii = ctx.buffer(np.array(self.radii).astype('f4'))
        colours = ctx.buffer(np.array(self.colours).astype('f4'))
        indices = ctx.buffer(np.array(range(self.num)))
        
        vao = ctx.vertex_array(prog,
                               pos1_buffers + pos2_buffers + [(radii, "f4", "radius"),
                                              (colours, "4f4", "colour")],
                               indices)

        return vao, moderngl.POINTS, 1

    def make_renderer(self, ctx):
        if self.num == 0:
            return pgbase.canvas3d.Renderer()

        split_name_nd3d_mat = self.pos_spliter.split_name("nd3d_mat")
        split_name_pos1 = self.pos_spliter.split_name("pos1")
        split_name_pos2 = self.pos_spliter.split_name("pos2")
            
        prog = ctx.program(
            vertex_shader = """
                #version 430

                uniform vec4 nd3d_offset;
                """ + "\n".join([f"uniform mat{self.pos_spliter.sizes[i]}x4 {split_name_nd3d_mat[i]};" for i in range(self.pos_spliter.len)]) + """
                """ + "\n".join([f"in vec{self.pos_spliter.sizes[i]} {split_name_pos1[i]};" for i in range(self.pos_spliter.len)]) + """
                """ + "\n".join([f"in vec{self.pos_spliter.sizes[i]} {split_name_pos2[i]};" for i in range(self.pos_spliter.len)]) + """
                
                in float radius;
                in vec4 colour;

                out vec4 g_pos1;
                out vec4 g_pos2;
                out float g_radius;
                out vec4 g_colour;
                
                void main() {
                    g_radius = radius;
                    g_colour = colour;
                    g_pos1 = """ + " + ".join([f"{split_name_nd3d_mat[i]} * {split_name_pos1[i]}" for i in range(self.pos_spliter.len)]) + """ + nd3d_offset;
                    g_pos2 = """ + " + ".join([f"{split_name_nd3d_mat[i]} * {split_name_pos2[i]}" for i in range(self.pos_spliter.len)]) + """ + nd3d_offset;
                    
                }
            """,
            geometry_shader = """
                #version 430
                layout (points) in;
                layout (triangle_strip, max_vertices = 100) out;

                in vec4 g_pos1[];
                in vec4 g_pos2[];
                in vec4 g_colour[];
                in float g_radius[];

                out vec4 v_colour;
                out float dash;

                uniform mat4 proj_mat;
                uniform mat4 view_mat;
                uniform int depth;
                uniform vec3 cam_pos;
                uniform float dash_len;

                mat4 vp_mat = proj_mat * view_mat;

                void main() {
                    vec4 start_base_pos_h = g_pos1[0];
                    vec4 end_base_pos_h = g_pos2[0];
                    
                    if (start_base_pos_h.w > 0 || end_base_pos_h.w > 0) {
                        if (start_base_pos_h.w < 0) {
                            float f = -start_base_pos_h.w / (end_base_pos_h.w - start_base_pos_h.w);
                            start_base_pos_h = start_base_pos_h + f * (end_base_pos_h - start_base_pos_h);
                            start_base_pos_h.w = 0.00001;
                        } else if (end_base_pos_h.w < 0) {
                            float f = -end_base_pos_h.w / (start_base_pos_h.w - end_base_pos_h.w);
                            end_base_pos_h = end_base_pos_h + f * (start_base_pos_h - end_base_pos_h);
                            end_base_pos_h.w = 0.00001;
                        }
                        vec3 start_base_pos = start_base_pos_h.xyz / start_base_pos_h.w;
                        vec3 end_base_pos = end_base_pos_h.xyz / end_base_pos_h.w;

                        vec3 start_cam_vec = normalize(start_base_pos - cam_pos);
                        vec3 end_cam_vec = normalize(end_base_pos - cam_pos);

                        start_base_pos -= 3 * g_radius[0] * start_cam_vec;
                        end_base_pos -= 3 * g_radius[0] * end_cam_vec;
                        
                        vec3 para = g_radius[0] * normalize(end_base_pos - start_base_pos);
                        vec3 start_perp = (g_radius[0] ) * normalize(cross(para, start_cam_vec));
                        vec3 end_perp = (g_radius[0] ) * normalize(cross(para, end_cam_vec));

                        v_colour = g_colour[0];

                        vec4 vp_start_pos = vp_mat * vec4(start_base_pos, 1);
                        vec4 vp_end_pos = vp_mat * vec4(start_base_pos, 1);

                        float tot_dash = length(start_base_pos - end_base_pos) / dash_len;
                    
                        float start_dash = -tot_dash / 2;
                        float end_dash = tot_dash / 2;

                        dash = start_dash;
                        gl_Position = vp_mat * vec4(start_base_pos + start_perp, 1);
                        EmitVertex();
                        gl_Position = vp_mat * vec4(start_base_pos - start_perp, 1);
                        EmitVertex();
                        dash = end_dash;
                        gl_Position = vp_mat * vec4(end_base_pos + end_perp, 1);
                        EmitVertex();
                        EndPrimitive();
                        gl_Position = vp_mat * vec4(end_base_pos - end_perp, 1);
                        EmitVertex();
                        gl_Position = vp_mat * vec4(end_base_pos + end_perp, 1);
                        EmitVertex();
                        dash = start_dash;
                        gl_Position = vp_mat * vec4(start_base_pos - start_perp, 1);
                        EmitVertex();
                        EndPrimitive();

                        int tot_steps = 10;

                        vec3 vec_a;
                        vec3 vec_b;
                        float step = 3.1415926535 / tot_steps;

                        float dash_end_mult = tot_dash * g_radius[0] / length(start_base_pos - end_base_pos);

                        for (int i = 0; i < tot_steps; i++) {
                            dash = start_dash;
                            gl_Position = vp_mat * vec4(start_base_pos, 1);
                            EmitVertex();
                            dash = start_dash - sin(i * step) * dash_end_mult;
                            vec_a = -sin(i * step) * para + cos(i * step) * start_perp;
                            gl_Position = vp_mat * vec4(start_base_pos + vec_a, 1);
                            EmitVertex();
                            dash = start_dash - sin((i + 1) * step) * dash_end_mult;
                            vec_b = -sin((i + 1) * step) * para + cos((i + 1) * step) * start_perp;
                            gl_Position = vp_mat * vec4(start_base_pos + vec_b, 1);
                            EmitVertex();
                            EndPrimitive();
                        }

                        for (int i = 0; i < tot_steps; i++) {
                            dash = end_dash;
                            gl_Position = vp_mat * vec4(end_base_pos, 1);
                            EmitVertex();
                            dash = end_dash + sin(i * step) * dash_end_mult;
                            vec_a = sin(i * step) * para + cos(i * step) * end_perp;
                            gl_Position = vp_mat * vec4(end_base_pos + vec_a, 1);
                            EmitVertex();
                            dash = end_dash + sin((i + 1) * step) * dash_end_mult;
                            vec_b = sin((i + 1) * step) * para + cos((i + 1) * step) * end_perp;
                            gl_Position = vp_mat * vec4(end_base_pos + vec_b, 1);
                            EmitVertex();
                            EndPrimitive();
                        }
                        
                    }
                }
            """,
            fragment_shader = """
                #version 430
                in vec4 v_colour;
                in float dash;
                uniform sampler2D peel_tex;
                uniform vec2 scr_size;
                uniform bool do_peel;
                uniform int depth;
                out vec4 f_colour;
                void main() {
                    if (depth > 0) {
                        if (floor(mod(2 * dash - 0.5, 2)) == 0) {
                            discard;
                        }
                    }
                    f_colour = v_colour;
                    if (do_peel) {
                        if (gl_FragCoord.z <= texture2D(peel_tex, gl_FragCoord.xy / scr_size).x) {
                            discard;
                        }
                    }
                }"""
        )

        vao, mode, num_instances = self.make_vao(prog)

        dash_len = self.dash_len
        class ModelRenderer(pgbase.canvas3d.Renderer):
            def __init__(self):
                super().__init__([prog])
            def render(self):
                prog["dash_len"].value = dash_len
                vao.render(mode, instances = num_instances)

        return ModelRenderer()

    def add_edge(self, pos1, pos2, colour, radius):
        assert len(pos1) == self.dim
        assert len(pos2) == self.dim
        assert len(colour) == 4
        
        self.poses1.append(pos1)
        self.poses2.append(pos2)
        self.colours.append(colour)
        self.radii.append(radius)
        self.num += 1




class Triangle(pgbase.canvas3d.BaseModel):
    def __init__(self, dim):
        self.dim = dim
        self.pos_spliter = SplitGlsl(self.dim)
        split_name_nd3d_mat = self.pos_spliter.split_name("nd3d_mat")
        split_name_pos1 = self.pos_spliter.split_name("pos1")
        split_name_pos2 = self.pos_spliter.split_name("pos2")
        split_name_pos3 = self.pos_spliter.split_name("pos3")
        super().__init__("""
            #version 430

            uniform vec4 nd3d_offset;
            """ + "\n".join([f"uniform mat{self.pos_spliter.sizes[i]}x4 {split_name_nd3d_mat[i]};" for i in range(self.pos_spliter.len)]) + """
            """ + "\n".join([f"in vec{self.pos_spliter.sizes[i]} {split_name_pos1[i]};" for i in range(self.pos_spliter.len)]) + """
            """ + "\n".join([f"in vec{self.pos_spliter.sizes[i]} {split_name_pos2[i]};" for i in range(self.pos_spliter.len)]) + """
            """ + "\n".join([f"in vec{self.pos_spliter.sizes[i]} {split_name_pos3[i]};" for i in range(self.pos_spliter.len)]) + """
            
            in vec4 colour;

            out vec4 g_pos1;
            out vec4 g_pos2;
            out vec4 g_pos3;
            out vec4 g_colour;
            
            void main() {
                g_colour = colour;
                g_pos1 = """ + " + ".join([f"{split_name_nd3d_mat[i]} * {split_name_pos1[i]}" for i in range(self.pos_spliter.len)]) + """ + nd3d_offset;
                g_pos2 = """ + " + ".join([f"{split_name_nd3d_mat[i]} * {split_name_pos2[i]}" for i in range(self.pos_spliter.len)]) + """ + nd3d_offset;
                g_pos3 = """ + " + ".join([f"{split_name_nd3d_mat[i]} * {split_name_pos3[i]}" for i in range(self.pos_spliter.len)]) + """ + nd3d_offset;
                
            }
        """, """
            #version 430
            layout (points) in;
            layout (triangle_strip, max_vertices = 100) out;

            in vec4 g_pos1[];
            in vec4 g_pos2[];
            in vec4 g_pos3[];
            in vec4 g_colour[];

            out vec4 v_pos_h;
            out vec4 v_colour;
            out vec3 v_normal;

            uniform mat4 proj_mat;
            uniform mat4 view_mat;

            mat4 vp_mat = proj_mat * view_mat;

            void main() {
                vec4 pos1_h = g_pos1[0];
                vec4 pos2_h = g_pos2[0];
                vec4 pos3_h = g_pos3[0];

                int pos1_off = int(pos1_h.w < 0);
                int pos2_off = int(pos2_h.w < 0);
                int pos3_off = int(pos3_h.w < 0);

                int total_off = pos1_off + pos2_off + pos3_off;

                if (total_off == 0) {
                    vec3 pos1_ac = pos1_h.xyz / pos1_h.w;
                    vec3 pos2_ac = pos2_h.xyz / pos2_h.w;
                    vec3 pos3_ac = pos3_h.xyz / pos3_h.w;

                    v_normal = cross(pos2_ac - pos1_ac, pos3_ac - pos1_ac);
                    
                    v_colour = g_colour[0];

                    v_pos_h = pos1_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    v_pos_h = pos2_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    v_pos_h = pos3_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    EndPrimitive();
                    
                } else if (total_off == 1) {
                    vec4 on_pos1_h;
                    vec4 on_pos2_h;
                    vec4 off_pos_h;
                    if (pos1_off == 1) {
                        on_pos1_h = pos2_h;
                        on_pos2_h = pos3_h;
                        off_pos_h = pos1_h;
                    } else if (pos2_off == 1) {
                        on_pos1_h = pos1_h;
                        on_pos2_h = pos3_h;
                        off_pos_h = pos2_h;
                    } else {
                        on_pos1_h = pos1_h;
                        on_pos2_h = pos2_h;
                        off_pos_h = pos3_h;
                    }
                    
                    float f1 = - off_pos_h.w / (on_pos1_h.w - off_pos_h.w);
                    float f2 = - off_pos_h.w / (on_pos2_h.w - off_pos_h.w);
                    
                    vec4 off_on_pos1_h = off_pos_h + f1 * (on_pos1_h - off_pos_h);
                    vec4 off_on_pos2_h = off_pos_h + f2 * (on_pos2_h - off_pos_h);
                    
                    off_on_pos1_h.w = 0.00001;
                    off_on_pos2_h.w = 0.00001;

                    vec3 pos1_ac = on_pos1_h.xyz / on_pos1_h.w;
                    vec3 pos2_ac = on_pos2_h.xyz / on_pos2_h.w;
                    vec3 pos3_ac = off_on_pos1_h.xyz / off_on_pos1_h.w;
                    vec3 pos4_ac = off_on_pos2_h.xyz / off_on_pos2_h.w;

                    v_normal = cross(pos2_ac - pos1_ac, pos3_ac - pos1_ac);
                    v_colour = g_colour[0];

                    v_pos_h = on_pos1_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    
                    v_pos_h = on_pos2_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    
                    v_pos_h = off_on_pos1_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    EndPrimitive();


                    v_pos_h = on_pos2_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    
                    v_pos_h = off_on_pos1_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    
                    v_pos_h = off_on_pos2_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    EndPrimitive(); 
                } else if (total_off == 2) {
                    vec4 on_pos_h;
                    vec4 off_pos1_h;
                    vec4 off_pos2_h;
                    if (pos1_off == 0) {
                        off_pos1_h = pos2_h;
                        off_pos2_h = pos3_h;
                        on_pos_h = pos1_h;
                    } else if (pos2_off == 0) {
                        off_pos1_h = pos1_h;
                        off_pos2_h = pos3_h;
                        on_pos_h = pos2_h;
                    } else {
                        off_pos1_h = pos1_h;
                        off_pos2_h = pos2_h;
                        on_pos_h = pos3_h;
                    }
                    
                    float f1 = - off_pos1_h.w / (on_pos_h.w - off_pos1_h.w);
                    float f2 = - off_pos2_h.w / (on_pos_h.w - off_pos2_h.w);
                    
                    vec4 off_on_pos1_h = off_pos1_h + f1 * (on_pos_h - off_pos1_h);
                    vec4 off_on_pos2_h = off_pos2_h + f2 * (on_pos_h - off_pos2_h);
                    
                    off_on_pos1_h.w = 0.00001;
                    off_on_pos2_h.w = 0.00001;

                    vec3 pos1_ac = on_pos_h.xyz / on_pos_h.w;
                    vec3 pos2_ac = off_on_pos1_h.xyz / off_on_pos1_h.w;
                    vec3 pos3_ac = off_on_pos2_h.xyz / off_on_pos2_h.w;

                    v_normal = cross(pos2_ac - pos1_ac, pos3_ac - pos1_ac);
                    v_colour = g_colour[0];

                    v_pos_h = on_pos_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    
                    v_pos_h = off_on_pos1_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    
                    v_pos_h = off_on_pos2_h;
                    gl_Position = vp_mat * v_pos_h;
                    EmitVertex();
                    EndPrimitive();
                }
                
            }
        """)

        self.poses1 = []
        self.poses2 = []
        self.poses3 = []
        self.colours = []
        self.num = 0

    def make_vao(self, prog):
        ctx = prog.ctx

        pos1_buffers = []
        split_data = self.pos_spliter.split_data(self.poses1, 1)
        split_name = self.pos_spliter.split_name("pos1")
        for i in range(self.pos_spliter.len):
            buffer = ctx.buffer(split_data[i].astype('f4'))
            pos1_buffers.append((buffer, f"{self.pos_spliter.sizes[i]}f4", split_name[i]))

        pos2_buffers = []
        split_data = self.pos_spliter.split_data(self.poses2, 1)
        split_name = self.pos_spliter.split_name("pos2")
        for i in range(self.pos_spliter.len):
            buffer = ctx.buffer(split_data[i].astype('f4'))
            pos2_buffers.append((buffer, f"{self.pos_spliter.sizes[i]}f4", split_name[i]))

        pos3_buffers = []
        split_data = self.pos_spliter.split_data(self.poses3, 1)
        split_name = self.pos_spliter.split_name("pos3")
        for i in range(self.pos_spliter.len):
            buffer = ctx.buffer(split_data[i].astype('f4'))
            pos3_buffers.append((buffer, f"{self.pos_spliter.sizes[i]}f4", split_name[i]))
                
        colours = ctx.buffer(np.array(self.colours).astype('f4'))
        indices = ctx.buffer(np.array(range(self.num)))
        
        vao = ctx.vertex_array(prog,
                               pos1_buffers + pos2_buffers + pos3_buffers + [(colours, "4f4", "colour")],
                               indices)

        return vao, moderngl.POINTS, 1

    def add_tri(self, pos1, pos2, pos3, colour):
        assert len(pos1) == self.dim
        assert len(pos2) == self.dim
        assert len(pos3) == self.dim
        assert len(colour) == 4
        
        self.poses1.append(pos1)
        self.poses2.append(pos2)
        self.poses3.append(pos3)
        self.colours.append(colour)
        self.num += 1







class Window(pgbase.canvas3d.Window):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        
        self.node = Node(dim)
        self.edge = Edge(dim)
        self.pencil_edge = PencilEdge(dim, 0.2)
        self.triangle = Triangle(dim)

        self.draw_model(self.node)
        self.draw_model(self.edge)
        self.draw_model(self.pencil_edge)
        self.draw_model(self.triangle)

        self.nd_orthog = np.eye(self.dim)

        assert self.dim >= 3
        projection = np.zeros([4, dim + 1])
        projection[0, 0] = 1
        projection[1, 1] = 1
        projection[2, 2] = 1
        projection[3, self.dim] = 1
##        for i in range(3, self.dim):
##            projection[3, i] = 0.2
        self.projection = projection

        self.active_dim = 2

    def set_nd_orthog(self, nd_orthog):
        assert nd_orthog.shape == self.nd_orthog.shape
        self.nd_orthog = nd_orthog

    def set_projection(self, projection):
        assert projection.shape == self.projection.shape
        self.projection = projection

    @property
    def nd3d_mat(self):
        nd_orthog_h = np.eye(self.dim + 1)
        nd_orthog_h[0:self.dim, 0:self.dim] = self.nd_orthog
        return self.projection @ nd_orthog_h

    def set_uniforms(self, progs, peel_tex, depth):
        super().set_uniforms(progs, peel_tex, depth)

        nd3d_mat_splitter = SplitGlsl(self.dim)
        split_names = nd3d_mat_splitter.split_name("nd3d_mat")
        split_data = nd3d_mat_splitter.split_data(self.nd3d_mat[:, :-1], 1)

        for prog in progs:
            try: prog["nd3d_offset"].value = tuple(self.nd3d_mat[:, -1])
            except KeyError: pass
            for i in range(nd3d_mat_splitter.len):
                try: prog[split_names[i]].value = tuple(split_data[i].transpose().flatten())
                except KeyError: pass

    def draw_node(self, pos, colour, radius):
        self.node.add_node(pos, colour, radius)

    def draw_edge(self, pos1, pos2, colour, radius):
        self.edge.add_edge(pos1, pos2, colour, radius)

    def draw_pencil_edge(self, pos1, pos2, colour, radius):
        self.pencil_edge.add_edge(pos1, pos2, colour, radius)

    def draw_tri(self, pos1, pos2, pos3, colour):
        self.triangle.add_tri(pos1, pos2, pos3, colour)

    def event(self, event):
        super().event(event)

        if event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[2]:
            dx, dy = event.rel
            
            theta = self.camera.theta()
            phi = self.camera.phi()
            n = self.dim
            
            mat = np.eye(n)
            mat = mat @ rot_axes(n, 0, 2, theta)
            mat = mat @ rot_axes(n, 2, 1, phi)
            mat = mat @ rot_axes(n, 0, self.active_dim, 0.006 * dx)
            mat = mat @ rot_axes(n, self.active_dim, 1, 0.006 * dy)
            mat = mat @ rot_axes(n, 2, 1, -phi)
            mat = mat @ rot_axes(n, 0, 2, -theta)
            
            self.nd_orthog = mat @ self.nd_orthog

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if pygame.mouse.get_pressed()[2]:
                    self.active_dim += 1
                    if self.active_dim == self.dim:
                        self.active_dim = 2
                     
            if event.button == 2:
                if pygame.mouse.get_pressed()[2]:
                    self.projection[3, self.active_dim] = 0

        if event.type == pygame.MOUSEBUTTONDOWN:
            if pgbase.tools.in_rect(event.pos, self.rect):
                if pygame.mouse.get_pressed()[2]:
                    if event.button in {4, 5}:
                        dy = 1
                        if event.button == 4:
                            dy = -1
                        if self.active_dim != 2:
                            self.projection[3, self.active_dim] += dy / 100

    
Camera = pgbase.canvas3d.Camera


##class Window(pgbase.canvas3d.Window):
##    def __init__(self, size, canvas):
##        assert isinstance(canvas, Canvas)
##        super().__init__(size, canvas)
##        self.active_dim = 2
##
##    def on_mouse_press(self, x, y, button, modifiers):
##        super().on_mouse_press(x, y, button, modifiers)
##        if button == pyglet.window.mouse.LEFT:
##            if self.mouse[pyglet.window.mouse.RIGHT]:
##                self.active_dim += 1
##                if self.active_dim == self.canvas.dim:
##                    self.active_dim = 2
##                 
##        if button == pyglet.window.mouse.MIDDLE:
##            if self.mouse[pyglet.window.mouse.RIGHT]:
##                self.canvas.projection[3, self.active_dim] = 0
##
##    def on_mouse_scroll(self, x, y, dx, dy):
##        if self.mouse[pyglet.window.mouse.RIGHT]:
##            if self.active_dim != 2:
##                self.canvas.projection[3, self.active_dim] += dy / 100
##        else:
##            self.camera.fly_speed *= math.exp(dy / 10)



        
def test():
    import sys
    
    pgbase.core.Window.setup()

    window = Window(4)

    model = pgbase.canvas3d.UnitCylinder(24, True)
    for _ in range(10):
        import random
        r = lambda : random.uniform(0, 1)
        model.add_line([10 * r(), 10 * r(), 10 * r()], [10 * r(), 10 * r(), 10 * r()], 1, [r(), r(), r(), r()])

    window.draw_model(model)

    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                for w in [-1, 1]:
                    window.draw_node([x, y, z, w], [1, 1, 1, 1], 0.5)

    window.draw_node([0, 0, 0, 0], [1, 1, 1, 1], 0.5)
    
    pgbase.core.run(window)
    pygame.quit()
    sys.exit()
























