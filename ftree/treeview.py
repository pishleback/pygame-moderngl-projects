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
##        G.add_edge(1, 2)
##        G.add_edge(2, 3)
##        G.add_edge(3, 4)
##        G.add_edge(4, 5)
##        G.add_edge(5, 6)
##        G.add_edge(1, 7)
##        G.add_edge(7, 8)
##        G.add_edge(8, 6)
##        G.add_edge(1, 6)

        G = self.tree.digraph()
        

        oriented_cycles = [list(zip(nodes,(nodes[1:] + nodes[:1]))) for nodes in nx.cycle_basis(G.to_undirected())]
        edges = list(G.edges())
        #extend some edges such that we get a well defined (up to adding a constant) height function on nodes
        opt_result = scipy.optimize.linprog(np.ones(len(edges)),
                                      A_eq = np.array([[(1 if edge in oc else (-1 if (edge[1], edge[0]) in oc else 0)) for edge in edges] for oc in oriented_cycles]),
                                      b_eq = np.zeros(len(oriented_cycles)),
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
        node_widths = {}
            
        root = max(G.nodes(), key = lambda x : len(nx.ancestors(G, x)))
##        above = [root] + list(nx.ancestors(G, root))

        def rel_widths_above(node):
            pred = list(G.predecessors(node))
            if len(pred) == 0:
                return {node : 0.0}
            else:
                n = len(pred)
                sub_rel_widths_above = [rel_widths_above(p) for p in G.predecessors(node)]

                move_bys = []
                for i in range(n - 1):
                    move_by = -math.inf
                    for x in sub_rel_widths_above[i]:
                        for y in sub_rel_widths_above[i + 1]:
                            if abs(node_heights[x] - node_heights[y]) < 1:
                                move_by = max(move_by, 1 + sub_rel_widths_above[i][x] - sub_rel_widths_above[i + 1][y])
                    move_bys.append(move_by)
                                
                sub_rel_widths_above = [{n : w + sum(move_bys[:i]) for n, w in sub_rel_widths_above[i].items()} for i in range(n)]
                rel_widths = {}
                for i in range(n):
                    for n, w in sub_rel_widths_above[i].items():
                        rel_widths[n] = w
                rel_widths[node] = 0.5 * (max(rel_widths[p] for p in pred) + min(rel_widths[p] for p in pred))
                return rel_widths

        rel_widths = rel_widths_above(root)
        print(rel_widths)

        node_widths = rel_widths

##        layers = []
##        h = node_heights[root]
##        to_collect = set(above)
##        while len(to_collect) != 0:
##            layer = set(x for x in to_collect if node_heights[x] < h + 0.5)
##            for x in layer:
##                to_collect.remove(x)
##            layers.append(layer)
##            h += 1
##
##        print(layers)
##        
##        largest_idx = max(range(len(layers)), key = lambda i : len(layers[i]))
##        largest_layer = layers[largest_idx]
##        sorted_largest_layer = [next(iter(largest_layer))]
##        while len(sorted_largest_layer) != len(largest_layer):
##            x = min([x for x in largest_layer if not x in sorted_largest_layer],
##                    key = lambda y : nx.shortest_path_length(G.to_undirected(), sorted_largest_layer[-1], y))
##            sorted_largest_layer.append(x)
##
##        
####        sorted_largest_layer = sorted(largest_layer, key = lambda y : nx.shortest_path_length(G.to_undirected(), x, y))
##
##
##        
##        for i, x in enumerate(sorted_largest_layer):
##            node_widths[x] = i
##
##        for i in range(largest_idx + 1, len(layers)):
##            edges = list(nx.edge_boundary(G.to_undirected(), layers[i - 1], layers[i]))
##
##            layer = list(layers[i])
##            layer_idx_lookup = {x : idx for idx, x in enumerate(layer)}
##
##            def fun(widths):
##                tot = 0
##                for x, y in edges:
##                    xw, xh = node_widths[x], node_heights[x]
##                    yw, yh = widths[layer_idx_lookup[y]], node_heights[y]
##                    tot += (xw - yw) ** 2 + (xh - yh) ** 2                    
##                return tot
##
##            def min_w(widths):
##                if len(widths) <= 1:
##                    return 2
##                nums = []
##                for i in range(len(widths)):
##                    for j in range(i + 1, len(widths)):
##                        nums.append(abs(widths[i] - widths[j]))
##                return min(nums)
##
##            opt = scipy.optimize.minimize(fun, [random.uniform(-1, 1) for _ in range(len(layer))])
##            opt = scipy.optimize.minimize(fun, opt.x + (np.random.random([len(layer)]) - 0.5), constraints = [scipy.optimize.NonlinearConstraint(min_w, lb = 1, ub = np.inf)])
##            for i, x in enumerate(layer):
##                node_widths[x] = opt.x[i]
##
##
##        for i in reversed(range(0, largest_idx)):
##            edges = list(nx.edge_boundary(G.to_undirected(), layers[i], layers[i + 1]))
##            edges = [(e[1], e[0]) for e in edges]
##
##            layer = list(layers[i])
##            layer_idx_lookup = {x : idx for idx, x in enumerate(layer)}
##
##            def fun(widths):
##                tot = 0
##                for x, y in edges:
##                    xw, xh = node_widths[x], node_heights[x]
##                    yw, yh = widths[layer_idx_lookup[y]], node_heights[y]
##                    tot += (xw - yw) ** 2 + (xh - yh) ** 2                    
##                return tot
##
##            def min_w(widths):
##                if len(widths) <= 1:
##                    return 2
##                nums = []
##                for i in range(len(widths)):
##                    for j in range(i + 1, len(widths)):
##                        nums.append(abs(widths[i] - widths[j]))
##                return min(nums)
##
##            opt = scipy.optimize.minimize(fun, [random.uniform(-1, 1) for _ in range(len(layer))])
##            opt = scipy.optimize.minimize(fun, opt.x + (np.random.random([len(layer)]) - 0.5), constraints = [scipy.optimize.NonlinearConstraint(min_w, lb = 1, ub = np.inf)])
##            for i, x in enumerate(layer):
##                node_widths[x] = opt.x[i]

##        print(largest_idx)
##            
##            
##
##        input()
##
##        def gen_layers():
##            h = node_heights[root]
##            max_h = max(node_heights[a] for a in above)
##            while h < max_h:
##                yield h, [a for a in above if abs(node_heights[a] - h) < 0.5]
##                h += 0.3
##
##        h, layer = max(gen_layers(), key = lambda h_lay : len(h_lay[1]))
##        x = next(iter(layer))
##        sorted_layer = sorted(layer, key = lambda y : nx.shortest_path_length(G.to_undirected(), x, y))
##        for i, x in enumerate(sorted_layer):
##            node_widths[x] = i
##
##        while True:
##            edges = G.reverse().edges(layer)
##            next_layer = list(set(edge[1] for edge in edges))
##            if len(next_layer) == 0:
##                break
##            next_layer_idx = {x : idx for idx, x in enumerate(next_layer)}
##
##            def fun(widths):
##                tot = 0
##                for x, y in edges:
##                    xw, xh = node_widths[x], node_heights[x]
##                    yw, yh = widths[next_layer_idx[y]], node_heights[y]
##                    tot += (xw - yw) ** 2 + (xh - yh) ** 2
##                return tot
##
##            def min_w(widths):
##                nums = [0]
##                for i in range(len(widths)):
##                    for j in range(i + 1, len(widths)):
##                        nums.append(abs(widths[i] - widths[j]))
##                return min(nums)
##            
##            opt = scipy.optimize.minimize(fun, [random.uniform(-1, 1) for _ in range(len(next_layer))], constraints = [scipy.optimize.NonlinearConstraint(min_w, lb = 1, ub = np.inf)])
##            
##            for i in range(len(next_layer)):
##                node_widths[next_layer[i]] = opt.x[i]
##            layer = next_layer
##
##        layer = sorted_layer
##        while True:
##            edges = G.edges(layer)
##            next_layer = list(set(edge[1] for edge in edges))
##            if len(next_layer) == 0:
##                break
##            next_layer_idx = {x : idx for idx, x in enumerate(next_layer)}
##
##            def fun(widths):
##                tot = 0
##                for x, y in edges:
##                    xw, xh = node_widths[x], node_heights[x]
##                    yw, yh = widths[next_layer_idx[y]], node_heights[y]
##                    tot += (xw - yw) ** 2 + (xh - yh) ** 2
##                return tot
##
##            def min_w(widths):
##                nums = [0]
##                for i in range(len(widths)):
##                    for j in range(i + 1, len(widths)):
##                        nums.append(abs(widths[i] - widths[j]))
##                return min(nums)
##            
##            opt = scipy.optimize.minimize(fun, [random.uniform(-1, 1) for _ in range(len(next_layer))], constraints = [scipy.optimize.NonlinearConstraint(min_w, lb = 1, ub = np.inf)])
##            
##            for i in range(len(next_layer)):
##                node_widths[next_layer[i]] = opt.x[i]
##            layer = next_layer
            
                

        G = G.subgraph(node_widths.keys())
        
        for x in G.nodes():
            if type(self.tree.entity_lookup[x]) == treedata.Partnership:
                adj = list(G.successors(x)) + list(G.predecessors(x))
                w0, w1 = min(node_widths[y] for y in adj), max(node_widths[y] for y in adj)
                h = node_heights[x]
                colour = (1, 0, 0, 1)                
                self.shapes.add_shape(shapely.geometry.LineString([[w0, h], [w1, h]]).buffer(0.05), colour)
                for a in adj:
                    w = node_widths[a]
                    self.shapes.add_shape(shapely.geometry.LineString([[w, node_heights[a]], [w, h]]).buffer(0.05), colour)
            
        for x in G.nodes():
            if type(self.tree.entity_lookup[x]) == treedata.Person:
                colour = (0, 1, 1, 1)                
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

    pgbase.core.Window.setup(size = [1000, 1000])
    pgbase.core.run(TreeView(tree))
    pygame.quit()
    sys.exit()
