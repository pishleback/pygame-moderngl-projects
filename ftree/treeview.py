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
from ftree import shapelayout




def entity_format(entity):    
    if type(entity) == treedata.Person:
        sex = entity.get_sex()
        if sex == "male":
            colour = (0, 0.5, 1, 1)
        elif sex == "female":
            colour = (1, 0.5, 0.5, 1)
        else:
            colour = (1, 0.5, 0, 1)
        name = " ".join(entity.get_first_names())
        return shapelayout.height_frame(shapelayout.string((name if len(name) != 0 else "?"), (0, 0, 0, 1)), 0.9, colour)
    elif type(entity) == treedata.Partnership:
        return shapelayout.height_frame(shapelayout.letter("M", (0, 0, 0, 1)), 0.9, (0.7, 0.7, 0, 1))
    else:
        return shapelayout.height_frame(shapelayout.letter("?", (0, 0, 0, 1)), 0.9)




class TreeView(pgbase.canvas2d.Window2D):
    def __init__(self, tree, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = tree

        self.prog = self.ctx.program(
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
        self.vao = self.ctx.vertex_array(self.prog,
                                         [(vertices, "2f4", "unit_pos")],
                                         indices)


        import random
        self.shapes = pgbase.canvas2d.ShapelyModel(self.ctx)
        
        self.root = None
        self.node_widths = {}
        self.node_heights = {}
        self.more_to_see_nodes = set([])
        self.cycle_edges = set([])

        self.entity_fmts = {ident : entity_format(self.tree.entity_lookup[ident]) for ident in self.tree.entity_lookup}
        

        
##        G = nx.DiGraph()
##        G.add_edge(2, 1)
##        G.add_edge(3, 1)
##
##        G.add_edge(4, 2)
##        G.add_edge(5, 2)
##
##        G.add_edge(6, 3)
##        G.add_edge(7, 3)
##
##        G.add_edge(7, 8)
##        G.add_edge(7, 9)
##        G.add_edge(9, 13)
##
##        G.add_edge(6, 10)
##        G.add_edge(6, 11)
##        G.add_edge(11, 12)
##
##        G.add_edge(14, 6)
##        G.add_edge(14, 15)
##        G.add_edge(15, 16)
##
##        G.add_edge(15, 1)
##
##        self.G = G
    
        self.G = self.tree.digraph()
        self.T = nx.DiGraph()

        root = max(self.G.nodes(), key = lambda x : len(nx.ancestors(self.G, x)))
        for _ in range(0):
            root = next(iter(G.predecessors(root)))
##        root = list(G.predecessors(root))[1]
##        root = max(G.nodes(), key = lambda x : len(nx.descendants(G, x)))

        self.set_root(root)

    def set_root(self, root):
        self.root = root
        self.update_node_heights()
        self.update_node_widths()

    def update_node_heights(self):
        #set self.node_heights and self.T
        self.T = nx.DiGraph()
        self.T.add_node(self.root)
        self.node_heights = {self.root : 0}
        
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
                

    def update_node_widths(self):
        #set self.node_widths and self.more_to_see_nodes
        
        #successors
        #predecessors
        #ancestors
        #descendants

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


        def compute_widths_down_part(G, node, minh = -math.inf, excluded_succ = None):
            if excluded_succ is None:
                excluded_succ = set([])
            assert self.node_heights[node] >= minh
            tops = [{node : 0}]
            found_tops = set([node])
            for n in G.successors(node):
                if not n in excluded_succ:
                    if self.node_heights[n] >= minh:
                        for p in G.predecessors(n):
                            if not p in found_tops:
                                found_tops.add(p)
                                if len(tops) % 2 == 0:
                                    tops = tops + [{p : 0}]
                                else:
                                    tops = [{p : 0}] + tops
            bots = []
            for n in G.successors(node):
                if not n in excluded_succ:
                    if self.node_heights[n] >= minh:
                        bots.append(compute_widths_down_part(G, n, minh = minh))            
            return center(stack(tops), 0) | center(stack(bots), 0)

        
        def compute_widths_related(G, node):
##            #remove some edges so that G is a tree
##            T = nx.bfs_tree(G.to_undirected(), root)
##            G = G.edge_subgraph(itertools.chain(T.edges(), [(e[1], e[0]) for e in T.edges()]))
            assert nx.is_tree(G.to_undirected())

            #decide on an order for things
            pred_lookup = {x : list(G.predecessors(x)) for x in G.nodes()}
            succ_lookup = {x : list(G.successors(x)) for x in G.nodes()}

            #heres how the algorithm works:
            #compute_upwards
            #    take a node and compute the positions of everything directly above it.
            #    directly above means all ancestors and all of their decendents lying inside the cone
            #compute_whole
            #    compute positions of all ancestors and _all_ of their decendents
            #    do this by using compute_upwards to find positions for all ancestors decendents inside the cone
            #    then go up the sides of the cone and place the remaining decendents fitted to each side of the cone

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

                preds = pred_lookup[node]

                done = set([]) #so that we dont repeat descendants of things lying on the initial 1 wide stalk
                
                def gen_hang_left():   
                    for a, b in yield_left_side(node):
                        if not b in done:
                            for x in succ_lookup[b]:
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

##        def compute_widths_up(G, node):
##            return {node : 0} | center(stack([compute_widths_up(G, n) for n in G.predecessors(node)]), 0)
##
##        def compute_widths_down(G, node):
##            return {node : 0} | center(stack([compute_widths_down(G, n) for n in G.successors(node)]), 0)

        self.node_widths = compute_widths_related(self.T, self.root)
        
        self.more_to_see_nodes = set([])
        for node in self.node_widths:
            for x in itertools.chain(self.G.successors(node), self.G.predecessors(node)):
                if not x in self.node_widths:
                    self.more_to_see_nodes.add(node)
        
        self.update_shapes_vao()

    def node_pos(self, node):
        return [self.node_widths[node], 2 * self.node_heights[node]]

    def update_shapes_vao(self):
        G = self.G.subgraph(self.node_widths.keys())

##        node_widths = {n : w + random.uniform(-0.1, 0.1) for n, w in node_widths.items()}
##        node_heights = {n : h + random.uniform(-0.1, 0.1) for n, h in node_heights.items()}
        
##        for x in G.nodes():
##            if type(self.tree.entity_lookup[x]) == treedata.Partnership:
##                adj = list(G.successors(x)) + list(G.predecessors(x))
##                w0, w1 = min(node_widths[y] for y in adj), max(node_widths[y] for y in adj)
##                h = node_heights[x]
##                colour = (1, 0, 0, 1)                
##                self.shapes.add_shape(shapely.geometry.LineString([[w0, h], [w1, h]]).buffer(0.05), colour)
##                for a in adj:
##                    w = node_widths[a]
##                    self.shapes.add_shape(shapely.geometry.LineString([[w, node_heights[a]], [w, h]]).buffer(0.05), colour)
##        for x in G.nodes():
##            if type(self.tree.entity_lookup[x]) == treedata.Person:
##                colour = (0, 1, 1, 1)                
##                self.shapes.add_shape(shapely.geometry.Point([node_widths[x], node_heights[x]]).buffer(0.2), colour)

        self.shapes.clear()

        tree_edges = set(self.T.edges())

        for x, y in G.edges():
            x_pos = self.node_pos(x)
            y_pos = self.node_pos(y)
            
            p0 = [x_pos[0], x_pos[1]]
            p1 = [x_pos[0], x_pos[1] - 0.5]
            p2 = [x_pos[0], x_pos[1] - 1]
            p3 = [y_pos[0], y_pos[1] + 1]
            p4 = [y_pos[0], y_pos[1] + 0.5]
            p5 = [y_pos[0], y_pos[1]]

            def bez(pts, f):
                if len(pts) == 1:
                    return pts[0]
                else:
                    sub_pts = []
                    for i in range(len(pts) - 1):
                        p1 = pts[i]
                        p2 = pts[i + 1]
                        sub_pts.append([p1[j] + f * (p2[j] - p1[j]) for j in [0, 1]])
                    return bez(sub_pts, f)

            def gen_f():
                f = 0.0
                while f < 1:
                    yield f
                    f += 0.01 + 0.1 * (math.cos(math.pi * (2 * f - 1)) + 1) / 2
                yield 1.0

            if (x, y) in tree_edges:
                colour = (0, 0, 0, 1)
            else:
                colour = (0.5, 0.5, 0.5, 1)
            
            self.shapes.add_shape(shapely.geometry.LineString([p0] + [bez([p1, p2, p3, p4], f) for f in gen_f()] + [p5]).buffer(0.1), colour)

        for x in G.nodes():
            if x in self.more_to_see_nodes:
                colour = (0, 0, 0, 1)
                self.shapes.add_shape(shapely.geometry.Point(self.node_pos(x)).buffer(0.5), colour)
            
        for x in G.nodes():
            if x == self.root:
                rad = 0.45
                colour = (1, 0, 0, 1)
                self.shapes.add_shape(shapely.geometry.Point(self.node_pos(x)).buffer(rad), colour)
                
            elif (type(self.tree.entity_lookup[x]) in {treedata.Person, treedata.Partnership } if x in self.tree.entity_lookup else False):                
                fmt = shapelayout.position_frame(self.entity_fmts[x], self.node_pos(x))
                for geom, colour in fmt.gen_shapes():
                    self.shapes.add_shape(geom, colour)

                    
##            elif (type(self.tree.entity_lookup[x]) == treedata.Partnership if x in self.tree.entity_lookup else False):
##                rad = 0.15
##                colour = (0, 0, 0, 1)
##                self.shapes.add_shape(shapely.geometry.Point(self.node_pos(x)).buffer(rad), colour)
            
        self.shapes.update_vao()

    def set_rect(self, rect):
        super().set_rect(rect)
        
    def event(self, event):
        def dist(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        
        super().event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                pos = self.pygame_to_world(event.pos)                    
                node = min(self.node_widths.keys(), key = lambda node : dist(pos, self.node_pos(node)))
                node_pos = self.node_pos(node)
                scr_node_pos = self.world_to_pygame(node_pos)
                
                if dist(event.pos, scr_node_pos) < 50:
                    self.set_root(node)
                    self.center = self.center - np.array(node_pos) + np.array(self.node_pos(node))
                    
            
    def draw(self):
        super().set_uniforms([self.prog, self.shapes.prog])
        
        self.ctx.screen.use()
        self.ctx.clear(1, 0, 1, 1)
        self.vao.render(moderngl.TRIANGLES, instances = 1)
        self.shapes.vao.render(moderngl.TRIANGLES, instances = 1)





def run(tree):
    assert type(tree) == treedata.Tree

    pgbase.core.Window.setup(size = [2000, 1600])
    pgbase.core.run(TreeView(tree))
    pygame.quit()
    sys.exit()































    
