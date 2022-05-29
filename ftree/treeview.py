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
##
##        G.add_edge(14, 6)
##        G.add_edge(14, 15)
##        G.add_edge(15, 16)
    

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

        def stack_pair(rw1, rw2, side):
            #side = 0: place rw1 to the left of rw2
            #side = 1: place rw2 to the right of rw1
            assert 0 <= side <= 1
            m = math.inf
            for a, b in itertools.product(rw1.keys(), rw2.keys()):
                assert a != b
                if abs(node_heights[a] - node_heights[b]) < 0.99:
                    m = min(m, rw2[b] - rw1[a] - 1)
            if m is math.inf:
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
            if node_heights[node] < minh:
                return {}
            tops = [{node : 0}]
            found_tops = set([node])
            for n in G.successors(node):
                if not n in excluded_succ:
                    if node_heights[n] >= minh:
                        for p in G.predecessors(n):
                            if not p in found_tops:
                                found_tops.add(p)
                                if len(tops) % 2 == 1:
                                    tops = tops + [{p : 0}]
                                else:
                                    tops = [{p : 0}] + tops
            bots = []
            for n in G.successors(node):
                if not n in excluded_succ:
                    if node_heights[n] >= minh:
                        bots.append(compute_widths_down_part(G, n, minh = minh))            
            return center(stack(tops), 0) | center(stack(bots), 0)

        

        def compute_widths_related(G, node):
            #remove some edges so that G is a tree
            T = nx.bfs_tree(G.to_undirected(), root)
            G = G.edge_subgraph(itertools.chain(T.edges(), [(e[1], e[0]) for e in T.edges()]))

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
                minh_mid = node_heights[node] + 0.99
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
                

                print(node, minh_left, minh_right)
                
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

            return compute_whole(G, node, compute_widths_down_part(G, node), -math.inf, -math.inf)


                  
        def compute_widths_up(G, node):
            return {node : 0} | center(stack([compute_widths_up(G, n) for n in G.predecessors(node)]), 0)

        def compute_widths_down(G, node):
            return {node : 0} | center(stack([compute_widths_down(G, n) for n in G.successors(node)]), 0)
            

        root = max(G.nodes(), key = lambda x : len(nx.ancestors(G, x)))
        for _ in range(0):
            root = next(iter(G.predecessors(root)))
##        root = list(G.predecessors(root))[1]
##        root = max(G.nodes(), key = lambda x : len(nx.descendants(G, x)))
        
        node_widths = compute_widths_related(G, root)

        print(node_widths)




        G = G.subgraph(node_widths.keys())

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

        node_heights = {n : 2 * h for n, h in node_heights.items()}

        for x, y in G.edges():
            p0 = [node_widths[x], node_heights[x]]
            p1 = [node_widths[x], node_heights[x] - 0.25]
            p2 = [node_widths[x], node_heights[x] - 1]
            p3 = [node_widths[y], node_heights[y] + 1]
            p4 = [node_widths[y], node_heights[y] + 0.25]
            p5 = [node_widths[y], node_heights[y]]

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
                    
            
            colour = (0, 0.5, 1, 1)
            self.shapes.add_shape(shapely.geometry.LineString([p0] + [bez([p1, p2, p3, p4], f) for f in gen_f()] + [p5]).buffer(0.05), colour)
        for x in G.nodes():
            colour = (1, 1, 0, 1)
            if type(self.tree.entity_lookup[x]) == treedata.Partnership:
                colour = (0, 0.5, 1, 1)
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
