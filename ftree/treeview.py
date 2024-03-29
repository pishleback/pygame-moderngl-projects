from ftree import treedata
import pgbase
import pygame
import numpy as np
import moderngl
import sys
import shapely.geometry
import networkx as nx
import itertools
import scipy.optimize
import scipy.spatial
import math
import time
import random
import os




##from ftree import shapelayout
##def entity_format(entity):    
##    if type(entity) == treedata.Person:
##        sex = entity.get_sex()
##        if sex == "male":
##            colour = (0, 0.5, 1, 1)
##        elif sex == "female":
##            colour = (1, 0.5, 0.5, 1)
##        else:
##            colour = (1, 0.5, 0, 1)
##        name = " ".join(entity.get_first_names())
##        return shapelayout.height_frame(shapelayout.string((name if len(name) != 0 else "?"), (0, 0, 0, 1)), 0.9, colour)
##    elif type(entity) == treedata.Partnership:
##        return shapelayout.height_frame(shapelayout.letter("M", (0, 0, 0, 1)), 0.9, (0.7, 0.7, 0, 1))
##    else:
##        return shapelayout.height_frame(shapelayout.letter("?", (0, 0, 0, 1)), 0.9)










class TreeView(pgbase.canvas2d.Window2D):
    def __init__(self, tree, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = tree

        self.bg_prog = self.ctx.program(
            vertex_shader = """
                #version 430
                in vec2 unit_pos;
                out vec4 gl_Position;
                out vec2 v_pos;

                void main() {
                    gl_Position = vec4(unit_pos, 0.0, 1.0);
                    v_pos = vec2(unit_pos);
                }
            """,
            fragment_shader = """
                #version 430
                in vec2 v_pos;
                out vec4 f_colour;

                uniform vec2 cam_center;
                uniform mat2 cam_mat;

                void main() {
                    vec2 pos = cam_mat * v_pos + cam_center;
                    bool bx = mod(pos.x, 2) < 1;
                    bool by = mod(pos.y, 2) < 1;

                    if ((bx && by) || (!bx && !by)) {
                        f_colour = 0.98 * vec4(1, 1, 1, 0);
                    } else {
                        f_colour = 1 * vec4(1, 1, 1, 0);
                    }
                }
    
            """,
        )
        vertices = self.ctx.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]).astype('f4'))
        indices = self.ctx.buffer(np.array([[0, 1, 3], [0, 2, 3]]))
        self.bg_vao = self.ctx.vertex_array(self.bg_prog,
                                         [(vertices, "2f4", "unit_pos")],
                                         indices)


        self.entity_prog = self.ctx.program(
            vertex_shader = """
                #version 430
                in vec2 pos;
                out vec2 g_pos;

                in float border;
                out float g_border;
                
                in uint v_tex_idx;
                flat out uint g_tex_idx;

                void main() {
                    g_pos = pos;
                    g_tex_idx = v_tex_idx;
                    g_border = border;
                }
            """,
            geometry_shader = """
                #version 430
                layout (points) in;
                layout (triangle_strip, max_vertices = 100) out;
                in vec2 g_pos[];

                flat in uint g_tex_idx[];
                flat out uint f_tex_idx;

                in float g_border[];
                out float f_border;

                uniform vec2 cam_center;
                uniform mat2 cam_mat_inv;

                uniform float w;
                uniform float h;

                out vec2 f_tex_pos;

                void main() {
                    f_tex_idx = g_tex_idx[0];
                    f_border = g_border[0];
                    
                    vec2 pos;

                    float bw = w + 2 * g_border[0];
                    float bh = h + 2 * g_border[0];

                    f_tex_pos = vec2(-g_border[0] / w, -g_border[0] / h);
                    pos = g_pos[0] + vec2(-bw, -bh);
                    gl_Position = vec4(cam_mat_inv * (pos - cam_center), 0, 1);
                    EmitVertex();
                    f_tex_pos = vec2(1 + g_border[0] / w, -g_border[0] / h);
                    pos = g_pos[0] + vec2(bw, -bh);
                    gl_Position = vec4(cam_mat_inv * (pos - cam_center), 0, 1);
                    EmitVertex();
                    f_tex_pos = vec2(1 + g_border[0] / w, 1 + g_border[0] / h);
                    pos = g_pos[0] + vec2(bw, bh);
                    gl_Position = vec4(cam_mat_inv * (pos - cam_center), 0, 1);
                    EmitVertex();
                    EndPrimitive();

                    f_tex_pos = vec2(-g_border[0] / w, -g_border[0] / h);
                    pos = g_pos[0] + vec2(-bw, -bh);
                    gl_Position = vec4(cam_mat_inv * (pos - cam_center), 0, 1);
                    EmitVertex();
                    f_tex_pos = vec2(-g_border[0] / w, 1 + g_border[0] / h);
                    pos = g_pos[0] + vec2(-bw, bh);
                    gl_Position = vec4(cam_mat_inv * (pos - cam_center), 0, 1);
                    EmitVertex();
                    f_tex_pos = vec2(1 + g_border[0] / w, 1 + g_border[0] / h);
                    pos = g_pos[0] + vec2(bw, bh);
                    gl_Position = vec4(cam_mat_inv * (pos - cam_center), 0, 1);
                    EmitVertex();
                    EndPrimitive();
                }
            """,
            fragment_shader = """
                #version 430
                in vec2 f_tex_pos;
                in float f_border;
                out vec4 f_colour;
                flat in uint f_tex_idx;

                uniform sampler2DArray tex;

                void main() {
                    if (f_tex_pos.x < 0 || f_tex_pos.x > 1 || f_tex_pos.y < 0 || f_tex_pos.y > 1) {
                        f_colour = vec4(0, 0, 0, 1);
                    } else {
                        f_colour = texture(tex, vec3(f_tex_pos, f_tex_idx));
                    }
                }
    
            """,
        )
        self.entity_prog["tex"] = 0
        self.entity_prog["w"] = 0.4
        self.entity_prog["h"] = 0.6
        self.entity_vao = None

        self.edge_prog = self.ctx.program(
            vertex_shader = """
                #version 430
                in vec2 top_pos;
                in vec2 bot_pos;
                in vec4 colour;
                out vec2 g_top_pos;
                out vec2 g_bot_pos;
                out vec4 g_colour;

                void main() {
                    g_top_pos = top_pos;
                    g_bot_pos = bot_pos;
                    g_colour = colour;
                }
            """,
            geometry_shader = """
                #version 430
                layout (points) in;
                layout (triangle_strip, max_vertices = 128) out;
                in vec2 g_top_pos[];
                in vec2 g_bot_pos[];
                in vec4 g_colour[];
                out vec4 g_colour_out;

                uniform vec2 cam_center;
                uniform mat2 cam_mat_inv;

                uniform float width;

                vec4 pos_to_gl(vec2 pos) {
                    return vec4(cam_mat_inv * (pos - cam_center), 0, 1);
                }

                void emit_line(vec2 p1, vec2 p2) {
                    vec2 vec = p2 - p1;
                    vec2 norm = width * normalize(vec2(-vec.y, vec.x));
                    
                    gl_Position = pos_to_gl(p1 + norm);
                    EmitVertex();
                    gl_Position = pos_to_gl(p1 - norm);
                    EmitVertex();
                    gl_Position = pos_to_gl(p2 - norm);
                    EmitVertex();
                    EndPrimitive();

                    gl_Position = pos_to_gl(p1 + norm);
                    EmitVertex();
                    gl_Position = pos_to_gl(p2 + norm);
                    EmitVertex();
                    gl_Position = pos_to_gl(p2 - norm);
                    EmitVertex();
                    EndPrimitive();
                }

                void emit_joint(vec2 p1, vec2 p2, vec2 p3) {
                    vec2 vec12 = p2 - p1;
                    vec2 norm12 = width * normalize(vec2(-vec12.y, vec12.x));
                    vec2 vec23 = p3 - p2;
                    vec2 norm23 = width * normalize(vec2(-vec23.y, vec23.x));

                    gl_Position = pos_to_gl(p2);
                    EmitVertex();
                    gl_Position = pos_to_gl(p2 + norm12);
                    EmitVertex();
                    gl_Position = pos_to_gl(p2 + norm23);
                    EmitVertex();
                    EndPrimitive();

                    gl_Position = pos_to_gl(p2);
                    EmitVertex();
                    gl_Position = pos_to_gl(p2 - norm12);
                    EmitVertex();
                    gl_Position = pos_to_gl(p2 - norm23);
                    EmitVertex();
                    EndPrimitive();
                }

                float bez2(float f, float a, float b) {
                    return a + f * (b - a);
                }

                float bez3(float f, float a, float b, float c) {
                    return bez2(f, a + f * (b - a), b + f * (c - b));
                }

                float bez4(float f, float a, float b, float c, float d) {
                    return bez3(f, a + f * (b - a), b + f * (c - b), c + f * (d - c));
                }

                void main() {
                    g_colour_out = g_colour[0];
                    
                    const int n = 10;
                    const float pi = 3.1415926535897932384626433;
                    
                    vec2[n+2] pts;
                    
                    vec2 p0 = g_top_pos[0];
                    vec2 p1 = g_top_pos[0] - vec2(0, 0.5);
                    vec2 p2 = g_top_pos[0] - vec2(0, 1);
                    vec2 p3 = g_bot_pos[0] + vec2(0, 1);
                    vec2 p4 = g_bot_pos[0] + vec2(0, 0.5);
                    vec2 p5 = g_bot_pos[0];

                    pts[0] = p0;
                    pts[n+1] = p5;

                    for (int i = 0; i < n; i++) {
                        float f = float(i) / float(n - 1);
                        f = 0.5 * (sin(pi * (f - 0.5)) + 1);
                        pts[i+1] = vec2(bez4(f, p1.x, p2.x, p3.x, p4.x), bez4(f, p1.y, p2.y, p3.y, p4.y));
                    }

                    for (int i = 0; i < n+1; i++) {
                        emit_line(pts[i], pts[i+1]);
                    }

                    for (int i = 0; i < n; i++) {
                        emit_joint(pts[i], pts[i+1], pts[i+2]);
                    }
                    
                }
            """,
            fragment_shader = """
                #version 430
                in vec4 g_colour_out;
                out vec4 f_colour;

                void main() {
                    f_colour = g_colour_out;
                }
    
            """,
        )
        self.edge_prog["width"] = 0.07
        self.edge_vao = None

        self.tex = self.ctx.texture_array([400, 600, 2048], 4, dtype = "f1")
        self.node_surf_data = {}
        self.nodes_waiting_for_surf_update = set()
        self.visible_nodes = {} #node -> texture index

##        self.shapes = pgbase.canvas2d.ShapelyModel(self.ctx)

        self.node_heights = {}
        self.node_widths = {}
        
        self.root = None
        self.node_widths = {}
        self.node_heights = {}
        self.more_to_see_nodes = set([])
        self.cycle_edges = set([])

        #variables for changing root node animation
        self.node_draw_positions = {}
        
        self.moving_nodes = []
        self.moving_nodes_old_positions = np.zeros([0, 2])
        self.moving_nodes_new_positions = np.zeros([0, 2])
        self.last_rootchange_time = time.time()
    
        self.G = self.tree.digraph() #the full graph
        self.T = nx.DiGraph() #a tree subgraph of G which we will draw as much of as possible

        root = max(self.G.nodes(), key = lambda x : len(nx.ancestors(self.G, x)))
        for _ in range(0):
            root = next(iter(G.predecessors(root)))

        self.click_pos = None

        self.set_root(root)

    def set_root(self, root):
        self.root = root
        self.update_node_positions()

    def update_node_positions(self):
        old_node_draw_positions = {n : p for n, p in self.node_draw_positions.items()}

        #set self.node_heights and self.T
        self.T = nx.DiGraph()
        self.T.add_node(self.root)
        self.node_heights = {self.root : 0}
        if self.root in old_node_draw_positions:
            self.node_heights[self.root] = old_node_draw_positions[self.root][1]
        
        g_edges = set(self.G.edges())
        for t_edge in nx.bfs_edges(self.G.to_undirected(), self.root):
            if t_edge in g_edges:
                g_edge = t_edge
                self.T.add_edge(g_edge[0], g_edge[1])
                self.node_heights[t_edge[1]] = self.node_heights[t_edge[0]] - 1
            else:
                g_edge = (t_edge[1], t_edge[0])
                assert g_edge in g_edges
                self.T.add_edge(g_edge[0], g_edge[1])
                self.node_heights[t_edge[1]] = self.node_heights[t_edge[0]] + 1
                
        #set self.node_widths and self.more_to_see_nodes
        
        #successors
        #predecessors
        #ancestors
        #descendants

        ordering_node_widths = {node : self.node_widths.get(node, random.uniform(0, 1)) for node in self.T.nodes}

        def match(node, rw1, rw2):
            assert node in rw1
            assert node in rw2
            for a in rw1:
                for b in rw2:
                    if a != node and b != node:
                        assert a != b
            off = rw1[node] - rw2[node]
            rw2m = {b : w + off for b, w in rw2.items()}
            return rw1 | rw2m

        def center(rw, c):
            if len(rw) == 0:
                return rw
            else:
                min_w = min(list(rw.values()))
                max_w = max(list(rw.values()))
                off = 0.5 * (max_w + min_w)
                return {a : w - off for a, w in rw.items()}

        def stack_pair(rw1, rw2, side):
            #side = 0: place rw1 to the left of rw2
            #side = 1: place rw2 to the right of rw1
            assert 0 <= side <= 1

            rw1_max = {}
            for a in rw1:
                h = self.node_heights[a]
                if not h in rw1_max:
                    rw1_max[h] = rw1[a]
                else:
                    rw1_max[h] = max(rw1_max[h], rw1[a])
            rw2_min = {}
            for b in rw2:
                h = self.node_heights[b]
                if not h in rw2_min:
                    rw2_min[h] = rw2[b]
                else:
                    rw2_min[h] = min(rw2_min[h], rw2[b])

            for h in rw1_max.keys() | rw2_min.keys():
                assert not math.isnan(rw2_min.get(h, math.inf) - rw1_max.get(h, -math.inf))

            m = min(rw2_min.get(h, math.inf) - rw1_max.get(h, -math.inf) - 1 for h in rw1_max.keys() | rw2_min.keys())
                    
            if m == math.inf:
                return center(rw1, 0) | center(rw2, 0)
            else:
                rw = {}
                for a, w in rw1.items():
                    rw[a] = w + m * (1 - side)
                for b, w in rw2.items():
                    rw[b] = w - m * side
                return rw

        def stack(rws):
            if len(rws) == 0:
                return {}
            elif len(rws) == 1:
                return rws[0]
            elif len(rws) == 2:
                return stack_pair(rws[0], rws[1], 0.5)
            else:
                k = len(rws) // 2
                return stack_pair(stack(rws[:k]), stack(rws[k:]), 0.5)



        
        def compute_widths_related(G, node):
            assert nx.is_tree(G.to_undirected())

            #decide on an order for things
            pred_lookup = {x : sorted(G.predecessors(x), key = lambda y : ordering_node_widths[y]) for x in G.nodes()}
            succ_lookup = {x : sorted(G.successors(x), key = lambda y : ordering_node_widths[y]) for x in G.nodes()}

            #heres how the algorithm works:
            #compute_upwards
            #    take a node and compute the positions of everything directly above it.
            #    directly above means all ancestors and all of their decendents lying inside the cone
            #compute_whole
            #    compute positions of all ancestors and _all_ of their decendents
            #    do this by using compute_upwards to find positions for all ancestors decendents inside the cone
            #    then go up the sides of the cone and place the remaining decendents fitted to each side of the cone

            def compute_widths_down_part(G, node, minh = -math.inf, excluded_succ = None):
                if excluded_succ is None:
                    excluded_succ = set([])
                assert self.node_heights[node] >= minh
                #add partnerships
                tops = [node]
                found_tops = set([node])
                for n in succ_lookup[node]:
                    if not n in excluded_succ:
                        if self.node_heights[n] >= minh:
                            for p in pred_lookup[n]:
                                if not p in found_tops:
                                    found_tops.add(p)
                                    if len(tops) % 2 == 0:
                                        tops = tops + [p]
                                    else:
                                        tops = [p] + tops
                tops = [{p : 0} for p in sorted(tops, key = lambda p : ordering_node_widths[p])]
                #add everything below
                bots = []
                for n in succ_lookup[node]:
                    if not n in excluded_succ:
                        if self.node_heights[n] >= minh:
                            bots.append(compute_widths_down_part(G, n, minh = minh))            
                return center(stack(tops), 0) | center(stack(bots), 0)

            def compute_upwards(G, node, block_left, block_right, minh_left, minh_right):
                assert block_left or block_right
                preds = pred_lookup[node]
                minh_mid = self.node_heights[node] + 1
                tops = []
                for i, n in enumerate(preds):
                    if i == 0 == len(preds) - 1:
                        tops.append(compute_upwards(G, n, block_left, block_right, minh_left, minh_right))
                    elif i == 0 and block_left:
                        tops.append(compute_upwards(G, n, True, False, minh_left, minh_mid))
                    elif i == len(preds) - 1 and block_right:
                        tops.append(compute_upwards(G, n, False, True, minh_mid, minh_right))
                    else:
                        if i == 0:
                            tops.append(compute_whole(G, n, compute_widths_down_part(G, n, minh = minh_left, excluded_succ = {node}), minh_left = minh_left, minh_right = minh_mid))
                        elif i == len(preds) - 1:
                            tops.append(compute_whole(G, n, compute_widths_down_part(G, n, minh = minh_right, excluded_succ = {node}), minh_left = minh_mid, minh_right = minh_right))
                        else:
                            tops.append(compute_whole(G, n, compute_widths_down_part(G, n, minh = minh_mid, excluded_succ = {node}), minh_left = minh_mid, minh_right = minh_mid))

                top = center(stack(tops) , 0)
                if block_left and block_right:
                    return top | {node : 0}
                elif block_left:
                    ans = stack_pair({node : 0.0}, top, 0)
                    ans[node] = min(ans[node], 0.0)
                    return ans
                elif block_right:
                    ans = stack_pair(top, {node : 0.0}, 1)
                    ans[node] = max(ans[node], 0.0)
                    return ans
                else:
                    assert False #at least one of block_left/block_right should be True

            def compute_whole(G, node, base, minh_left, minh_right):
                assert node in base
                core = match(node, compute_upwards(G, node, True, True, minh_left, minh_right), base)

##                def yield_middle(x):
##                    ps = pred_lookup[x]
##                    if len(ps) == 1:
##                        yield x, ps[0]
##                        yield from yield_left_side(ps[0])

                def yield_left_side(x):
                    ps = pred_lookup[x]
                    if len(ps) >= 1:
                        yield x, ps[0]
                        yield from yield_left_side(ps[0])

                def yield_right_side(x):
                    ps = pred_lookup[x]
                    if len(ps) >= 1:
                        yield x, ps[-1]
                        yield from yield_right_side(ps[-1])

##                if (minh_both := minh_left) == minh_right:
##                    bottoms = []
##                    for a, b in yield_middle(node):
##                        for x in succ_lookup[b]:
##                            bottoms.append((x, compute_widths_down_part(G, x, minh = minh_both)))
##                    bottoms = sorted(bottoms, key = lambda pair : ordering_node_widths[pair[0]])
##                    bottoms = [piece for x, piece in bottoms]
##                    node = b #named becasue its at the top of the 1 wide stalk of the tree


                preds = pred_lookup[node]
                done = set([]) #so that we dont repeat descendants of things lying on the initial 1 wide stalk
                def gen_hang_left():   
                    for a, b in yield_left_side(node):                            
                        if not b in done:
                            for x in reversed(succ_lookup[b]):
                                if not x is a:
                                    yield compute_widths_down_part(G, x, minh = minh_left)
                            done.add(b)

                def gen_hang_right():              
                    for a, b in yield_right_side(node):
                        if not b in done:
                            for x in succ_lookup[b]:
                                if not x is a:
                                    yield compute_widths_down_part(G, x, minh = minh_right)
                            done.add(b)
                    
                if minh_left <= minh_right:
                    hang_left = list(gen_hang_left())
                    hang_right = list(gen_hang_right())
                else:
                    hang_right = list(gen_hang_right())
                    hang_left = list(gen_hang_left())
                    
                for hang in hang_left:
                    core = stack([hang, core])
                for hang in hang_right:
                    core = stack([core, hang])
                return core

            return center(compute_whole(G, node, compute_widths_down_part(G, node), -math.inf, -math.inf), 0)

        self.node_widths = compute_widths_related(self.T, self.root)
        
        self.more_to_see_nodes = set([])
        for node in self.node_widths:
            for x in itertools.chain(self.G.successors(node), self.G.predecessors(node)):
                if not x in self.node_widths:
                    self.more_to_see_nodes.add(node)
        
##        self.update_shapes_vao()

        self.moving_nodes = list(old_node_draw_positions.keys() & self.node_widths.keys())
        self.moving_nodes_old_positions = np.array([old_node_draw_positions[n] for n in self.moving_nodes])
        self.moving_nodes_new_positions = np.array([[self.node_widths[n], self.node_heights[n]] for n in self.moving_nodes])
        self.last_rootchange_time = time.time()
        
        self.visible_nodes = {ident : idx for idx, ident in enumerate(self.node_widths.keys() | self.moving_nodes)}
        self.nodes_waiting_for_surf_update = set(self.visible_nodes.keys())

    def get_surf_data(self, entity):
        if not entity in self.node_surf_data:
            if type(entity) == treedata.Person:
                surf = pygame.Surface([self.tex.width, self.tex.height], flags = pygame.SRCALPHA)
                
                sex = entity.get_sex()
                if sex is None:
                    bg_colour = [128, 128, 128, 255]
                elif sex == "male":
                    bg_colour = [64, 160, 255, 255]
                elif sex == "female":
                    bg_colour = [255, 64, 255, 255]
                else:
                    bg_colour = [112, 64, 255, 255]
                surf.fill(bg_colour)
                pgbase.surftools.write(surf, entity.name(), [0, 0, 1, 0.2], [0, 0, 0, 255])

                event_strs = entity.get_event_strings(["birth", "death"])

                y = 0.2
                for event_str in event_strs:
                    pgbase.surftools.write(surf, event_str, [0, y, 1, 0.1], [0, 0, 0, 255])
                    y += 0.1
                
            else:
                surf = pygame.Surface([self.tex.width, self.tex.height], flags = pygame.SRCALPHA)
                surf.fill([0, 0, 0, 0])
                pygame.draw.circle(surf, [0, 0, 0, 255], [surf.get_width() / 2, surf.get_height() / 2], surf.get_width() / 3)

            data = pygame.image.tostring(surf, "RGBA", 1)
            self.node_surf_data[entity] = data

        return self.node_surf_data[entity]
    


    def node_pos(self, node):
        #currently requires the x position to be preserved so that the interactive stuff works
        #can add a inverse node pos function if this needs to be changed in the future
        pos = self.node_draw_positions[node]
        return [pos[0], 2 * pos[1]]

    def update_vaos(self):
        import random
        
        H = self.G.subgraph(self.node_draw_positions.keys())

        nodes = list(H.nodes)        
        buffer_vertices = self.ctx.buffer(np.array([self.node_pos(node) for node in nodes]).astype('f4'))
        tex_idx = self.ctx.buffer(np.array([self.visible_nodes[node] for node in nodes]))
        buffer_indices = self.ctx.buffer(np.array(range(len(nodes))))
        def get_border(node):
            if isinstance(self.tree.entity_lookup[node], treedata.Person):
                if node in self.more_to_see_nodes:
                    return 0.03
                else:
                    return 0.02
            else:
                return 0.0
        borders = self.ctx.buffer(np.array([get_border(node) for node in nodes]).astype('f4'))
        
        self.entity_vao = self.ctx.vertex_array(self.entity_prog,
                                                [(buffer_vertices, "2f4", "pos"),
                                                 (tex_idx, "u4", "v_tex_idx"),
                                                 (borders, "f4", "border")],
                                                buffer_indices)

        colour_tree_edge = [0, 0, 0, 1]
        colour_loop_edge = [1, 0, 1, 1]
        tree_edges = set(self.T.edges())

        edges = list(H.edges())
        buffer_top_vertices = self.ctx.buffer(np.array([self.node_pos(edge[0]) for edge in edges]).astype('f4'))
        buffer_bot_vertices = self.ctx.buffer(np.array([self.node_pos(edge[1]) for edge in edges]).astype('f4'))
        buffer_colours = self.ctx.buffer(np.array([(colour_tree_edge if edge in tree_edges else colour_loop_edge) for edge in edges]).astype('f4'))
        indices = self.ctx.buffer(np.array(range(len(edges))))
        self.edge_vao = self.ctx.vertex_array(self.edge_prog,
                                              [(buffer_top_vertices, "2f4", "top_pos"),
                                               (buffer_bot_vertices, "2f4", "bot_pos"),
                                               (buffer_colours, "4f4", "colour")],
                                              indices)


    def set_rect(self, rect):
        super().set_rect(rect)

    def tick(self, dt):
        d = 0.75
        if time.time() < self.last_rootchange_time + d and len(self.moving_nodes) != 0:
            f = (time.time() - self.last_rootchange_time) / d
            f = 0.5 * (math.cos(math.pi * (f - 1)) + 1)
            poses = f * self.moving_nodes_new_positions + (1 - f) * self.moving_nodes_old_positions
            self.node_draw_positions = {n : poses[i] for i, n in enumerate(self.moving_nodes)}
        else:
            self.node_draw_positions = {n : [self.node_widths[n], self.node_heights[n]] for n in self.node_widths}
        self.update_vaos()

        start_time = time.time()
        while time.time() - start_time < dt and len(self.nodes_waiting_for_surf_update) != 0:
            #generate some surfaces starting with the ones closest the center of the screen
            def distsq(a, b):
                return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            
            ident = min(self.nodes_waiting_for_surf_update, key = lambda ident : distsq(self.node_pos(ident), self.center)if ident in self.node_draw_positions else math.inf)
            self.nodes_waiting_for_surf_update.remove(ident)
            
            layer = self.visible_nodes[ident]
            self.tex.write(self.get_surf_data(self.tree.entity_lookup[ident]), viewport = (0, 0, layer, self.tex.width, self.tex.height, 1))
            
        
    def event(self, event):
        def dist(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        CLICK_DIST = 30
        
        super().event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.click_pos = event.pos
            
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if not self.click_pos is None:
                    if dist(self.click_pos, event.pos) < CLICK_DIST:
                        pos = self.pygame_to_world(self.click_pos)
                        node = min(self.node_draw_positions.keys(), key = lambda node : dist(pos, self.node_pos(node)))
                        node_pos = self.node_pos(node)
                        scr_node_pos = self.world_to_pygame(node_pos)
                        if dist(self.click_pos, scr_node_pos) < CLICK_DIST:
                            self.set_root(node)
                            self.center = self.center - np.array(node_pos) + np.array(self.node_pos(node))
                            if not pgbase.tools.in_rect(self.world_to_pygame([0, 0]), self.rect):
                                self.center[0] = 0

                    else:
                        pos = self.pygame_to_world(self.click_pos)
                        node = min(self.node_draw_positions.keys(), key = lambda node : dist(pos, self.node_pos(node)))
                        node_pos = self.node_pos(node)
                        scr_node_pos = self.world_to_pygame(node_pos)
                        if dist(self.click_pos, scr_node_pos) < CLICK_DIST:
                            #move node and re-calculate positions using the new position to determine the ordering
                            self.node_widths[node] = self.pygame_to_world(event.pos)[0]
                            self.set_root(self.root)
                        

                    self.click_pos = None
                    
            
    def draw(self):
        super().set_uniforms([self.bg_prog, self.entity_prog, self.edge_prog])
        
        self.ctx.screen.use()
        self.ctx.clear(1, 1, 1, 1)

        self.ctx.enable_only(moderngl.BLEND)
        
        self.bg_vao.render(moderngl.TRIANGLES, instances = 1)
        if not self.edge_vao is None:
            self.edge_vao.render(moderngl.POINTS, instances = 1)
        if not self.entity_vao is None:
            self.tex.use(0)
            self.entity_vao.render(moderngl.POINTS, instances = 1)





def run(tree):
    assert type(tree) == treedata.Tree
    pgbase.core.Window.setup(size = [1600, 1000])
    pgbase.core.run_root(TreeView(tree))































    
