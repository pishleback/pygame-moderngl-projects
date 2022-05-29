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
                        f_colour = 0.05 * vec4(1, 1, 1, 0);
                    } else {
                        f_colour = 0.1 * vec4(1, 1, 1, 0);
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
    

        G = self.tree.digraph()
        

        oriented_cycles = [list(zip(nodes,(nodes[1:] + nodes[:1]))) for nodes in nx.cycle_basis(G.to_undirected())]
        edges = list(G.edges())
        #extend some edges such that we get a well defined (up to adding a constant) height function on nodes
        if len(oriented_cycles) == 0:
            A_eq = None
            b_eq = None
        else:
            A_eq = np.array([[(1 if edge in oc else (-1 if (edge[1], edge[0]) in oc else 0)) for edge in edges] for oc in oriented_cycles])
            b_eq = np.zeros(len(oriented_cycles))
        opt_result = scipy.optimize.linprog(np.ones(len(edges)),
                                      A_eq = A_eq,
                                      b_eq = b_eq,
                                      bounds = [[1, None]] * len(edges))
        edge_lengths = {edges[i] : opt_result.x[i] for i in range(len(edges))}
        #smooth out the extensions along paths
        #for example, we could have .--1--.--1--.--4--. which can be smoothed to .--2--.--2--.--2--.
        #NOT IMPLEMENTED YET

        #convert edge lengths to a height function
        start = next(iter(G.nodes()))
        node_heights = {start : 0.0}
        for edge in nx.dfs_edges(G.to_undirected(), start):
            if edge in edge_lengths:
                dh = -edge_lengths[edge]
            else:
                dh = edge_lengths[(edge[1], edge[0])]
            node_heights[edge[1]] = node_heights[edge[0]] + dh
            
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
            

        def stack(rws):
            def stack_pair(rw1, rw2):
                m = math.inf
                for a, b in itertools.product(rw1.keys(), rw2.keys()):
##                    assert a != b
                    if abs(node_heights[a] - node_heights[b]) < 0.99:
                        m = min(m, rw2[b] - rw1[a] - 1)
                if m is math.inf:
                    return center(rw1, 0) | center(rw2, 0)
                else:
                    rw = {}
                    for a, w in rw1.items():
                        rw[a] = w
                    for b, w in rw2.items():
                        rw[b] = w - m
                    return rw
            if len(rws) == 0:
                return {}
            elif len(rws) == 1:
                return rws[0]
            elif len(rws) == 2:
                return stack_pair(rws[0], rws[1])
            else:
                k = len(rws) // 2
                return stack_pair(stack(rws[:k]), stack(rws[k:]))

        def compute_widths_down_part(G, node, minh = -math.inf):
            if node_heights[node] < minh:
                return {}
            tops = [{node : 0}]
            found_tops = set([node])
            for n in G.successors(node):
                if node_heights[n] >= minh:
                    for p in G.predecessors(n):
                        if not p in found_tops:
                            found_tops.add(p)
                            tops.append({p : 0})
            bots = []
            for n in G.successors(node):
                if node_heights[n] >= minh:
                    bots.append(compute_widths_down_part(G, n, minh = minh))            
            return center(stack(tops), 0) | center(stack(bots), 0)


##        def compute_widths_related_thin(G, node, minh_left = -math.inf, minh_right = -math.inf):
##            return {node : 0} | center(stack([compute_widths_up(G, n) for n in G.predecessors(node)]), 0)
        

        def compute_widths_related_thick_init(G, node):
            pred_lookup = {x : list(G.predecessors(x)) for x in G.nodes()}
            
            def compute_widths_related_thick(G, node, base, minh_left, minh_right):            
                assert node in base
                preds = pred_lookup[node]
                minh_mid = node_heights[node] + 0.99
                tops = []
                for i, n in enumerate(preds):
                    if i == 0:
                        tops.append(compute_widths_related_thick(G, n, {n : 0}, minh_left = math.inf, minh_right = minh_mid))
                    elif i == len(preds) - 1:
                        tops.append(compute_widths_related_thick(G, n, {n : 0}, minh_left = minh_mid, minh_right = math.inf))
                    else:
                        tops.append(compute_widths_related_thick(G, n, {n : 0}, minh_left = minh_mid, minh_right = minh_mid))

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

                def everything_above(x):
                    H = G.copy()
                    H.remove_node(x)
                    above = set([x])
                    for a in G.predecessors(x):
                        above |= set(nx.node_connected_component(H.to_undirected(), a))
                    return above

                mid = match(node, base, center(stack(tops), 0) | {node : 0})
                    
                if len(preds) >= 1:
                    for a, b in yield_left_side(node):
                        hang_left = []
                        for x in G.successors(b):
                            if not x is a:
                                hang_left.append(compute_widths_down_part(G, x, minh = minh_left))
                        mid = stack(hang_left + [mid])
                        
                    if len(preds) >= 2:
                        for a, b in yield_right_side(node):
                            hang_right = []
                            for x in G.successors(b):
                                if not x is a:
                                    hang_right.append(compute_widths_down_part(G, x, minh = minh_right))
                            mid = stack([mid] + hang_right)

                return mid

            return compute_widths_related_thick(G, node, compute_widths_down_part(G, node), -math.inf, -math.inf)


                  
        def compute_widths_up(G, node):
            return {node : 0} | center(stack([compute_widths_up(G, n) for n in G.predecessors(node)]), 0)

        def compute_widths_down(G, node):
            return {node : 0} | center(stack([compute_widths_down(G, n) for n in G.successors(node)]), 0)
            

        root = max(G.nodes(), key = lambda x : len(nx.ancestors(G, x)))
        for _ in range(6):
            root = next(iter(G.predecessors(root)))
##        root = max(G.nodes(), key = lambda x : len(nx.descendants(G, x)))

        T = nx.bfs_tree(G.to_undirected(), root)
        G = G.edge_subgraph(itertools.chain(T.edges(), [(e[1], e[0]) for e in T.edges()]))
        
        node_widths = compute_widths_related_thick_init(G, root)

        print(node_widths)









        

##        def rel_widths_downwards(G, node, complete = None, minh = -math.inf):
##            if complete is None:
##                complete = set()
##            complete.add(node)
##            next_nodes = list(x for x in G.successors(node) if node_heights[x] > minh and not x in complete)
##            rel_widths = {node : 0.0}
##            if len(next_nodes) == 0:
##                return rel_widths
##            else:
##                n = len(next_nodes)
##                sub_rel_widths = []
##                for i, p in enumerate(next_nodes):
##                    srw = rel_widths_downwards(G, p, complete = complete, minh = minh)
##                    complete |= srw.keys()
##                    sub_rel_widths.append(srw)
##                all_sub_rel_widths = sub_rel_widths[0]
##                for i in range(1, n):
##                    move_by = -math.inf
##                    for x in all_sub_rel_widths:
##                        for y in sub_rel_widths[i]:
##                            if abs(node_heights[x] - node_heights[y]) < 1:
##                                move_by = max(move_by, 1 + all_sub_rel_widths[x] - sub_rel_widths[i][y])
##                    for n, w in sub_rel_widths[i].items():
##                        all_sub_rel_widths[n] = w + move_by
##                offset = 0.5 * (max(all_sub_rel_widths[p] for p in next_nodes) + min(all_sub_rel_widths[p] for p in next_nodes))
##                all_sub_rel_widths = {p : w - offset for p, w in all_sub_rel_widths.items()} 
##                return rel_widths | all_sub_rel_widths
##
##        def rel_widths_upwards(G, node, base, complete = None, minh_left = -math.inf, minh_right = -math.inf):
##            assert node in base
##                
##            if complete is None:
##                complete = set(base.keys())
##            complete.add(node)
##            
##            #rel_widths_downwards(G, node, minh = min(minh_left, minh_right))
##
##            rel_widths = {x : w for x, w in base.items()}
##
##            next_nodes = list(x for x in G.successors(node) if not x in complete)
##            if len(next_nodes) != 0:
##                n = len(next_nodes)
##                sub_rel_widths = []
##                for i, p in enumerate(next_nodes):
##                    minh_mid = node_heights[node] + 1
##                    if i == 0:
##                        sub_minh_left = minh_left
##                        sub_minh_right = minh_mid
##                    elif i == len(next_nodes) - 1:
##                        sub_minh_left = minh_mid
##                        sub_minh_right = minh_right
##                    else:
##                        sub_minh_left = minh_mid
##                        sub_minh_right = minh_mid
##                    srw = rel_widths_downwards(G, p, complete = complete, minh = minh_left)
##                    complete |= srw.keys()
##                    sub_rel_widths.append(srw)
##                all_sub_rel_widths = sub_rel_widths[0]
##                for i in range(1, n):
##                    move_by = -math.inf
##                    for x in all_sub_rel_widths:
##                        for y in sub_rel_widths[i]:
##                            if abs(node_heights[x] - node_heights[y]) < 1:
##                                move_by = max(move_by, 1 + all_sub_rel_widths[x] - sub_rel_widths[i][y])
##                    for n, w in sub_rel_widths[i].items():
##                        all_sub_rel_widths[n] = w + move_by
##                offset = max(all_sub_rel_widths[p] for p in next_nodes) + 1
##                all_sub_rel_widths = {p : w - offset for p, w in all_sub_rel_widths.items()}
##                rel_widths |= all_sub_rel_widths
##
##
##            next_nodes = list(x for x in G.predecessors(node) if not x in complete)
##            if len(next_nodes) != 0:
##                n = len(next_nodes)
##                sub_rel_widths = []
##                for i, p in enumerate(next_nodes):
##                    minh_mid = node_heights[node] + 1
##                    if i == 0:
##                        sub_minh_left = minh_left
##                        sub_minh_right = minh_mid
##                    elif i == len(next_nodes) - 1:
##                        sub_minh_left = minh_mid
##                        sub_minh_right = minh_right
##                    else:
##                        sub_minh_left = minh_mid
##                        sub_minh_right = minh_mid
##                    srw = rel_widths_upwards(G, p, complete = complete, minh_left = sub_minh_left, minh_right = sub_minh_right)
##                    complete |= srw.keys()
##                    sub_rel_widths.append(srw)
##                all_sub_rel_widths = sub_rel_widths[0]
##                for i in range(1, n):
##                    move_by = -math.inf
##                    for x in all_sub_rel_widths:
##                        for y in sub_rel_widths[i]:
##                            if abs(node_heights[x] - node_heights[y]) < 1:
##                                move_by = max(move_by, 1 + all_sub_rel_widths[x] - sub_rel_widths[i][y])
##                    for n, w in sub_rel_widths[i].items():
##                        all_sub_rel_widths[n] = w + move_by
##                offset = 0.5 * (max(all_sub_rel_widths[p] for p in next_nodes) + min(all_sub_rel_widths[p] for p in next_nodes))
##                all_sub_rel_widths = {p : w - offset for p, w in all_sub_rel_widths.items()}
##                rel_widths |= all_sub_rel_widths
##
##
####            rel_widths[node] = 0.0
##
####            close_to_node = [x for x in rel_widths if abs(node_heights[x] - node_heights[node]) < 0.9]
####            if len(close_to_node) == 0:
####                rel_widths[node] = 0.0
####            else:
####                left_pos = -min(rel_widths[x] - 1 for x in close_to_node)
####                right_pos = max(rel_widths[x] + 1 for x in close_to_node)
####                if left_pos < right_pos:
####                    rel_widths[node] = -left_pos
####                else:
####                    rel_widths[node] = right_pos
##                
##            return rel_widths

            

####        root = max(G.nodes(), key = lambda x : len(nx.ancestors(G, x)))
####        node_widths = rel_widths_upwards(G, root, {root : 0.0})
##
####        root = max(G.nodes(), key = lambda x : len(nx.descendants(G, x)))
####        node_widths = rel_widths_downwards(G, root)



        

        G = G.subgraph(node_widths.keys())

        node_widths = {n : w + random.uniform(-0.1, 0.1) for n, w in node_widths.items()}
        node_heights = {n : h + random.uniform(-0.1, 0.1) for n, h in node_heights.items()}
        
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

        for x, y in G.edges():
            p1 = [node_widths[x], node_heights[x]]
            p2 = [node_widths[y], node_heights[y]]
            colour = (0, 0, 1, 1)
            self.shapes.add_shape(shapely.geometry.LineString([p1, p2]).buffer(0.05), colour)
        for x in G.nodes():
            colour = (1, 1, 0, 1)
##            if type(self.tree.entity_lookup[x]) == treedata.Person:
##                colour = (0, 1, 0, 1)
##            else:
##                colour = (1, 1, 0, 1)                
            self.shapes.add_shape(shapely.geometry.Point([node_widths[x], node_heights[x]]).buffer(0.2), colour)
            
        self.shapes.update_vao()

    def set_rect(self, rect):
        super().set_rect(rect)
        
    def event(self, event):
        super().event(event)
            
    def draw(self):
        super().set_uniforms([self.prog, self.shapes.prog])
        
        self.ctx.screen.use()
        self.ctx.clear(1, 0, 1, 1)
        self.vao.render(moderngl.TRIANGLES, instances = 1)
        self.shapes.vao.render(moderngl.TRIANGLES, instances = 1)





def run(tree):
    assert type(tree) == treedata.Tree

    pgbase.core.Window.setup(size = [1600, 1000])
    pgbase.core.run(TreeView(tree))
    pygame.quit()
    sys.exit()
