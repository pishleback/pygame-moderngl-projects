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

        
        G = nx.DiGraph()
        G.add_edge(2, 1)
        G.add_edge(3, 1)
        G.add_edge(7, 2)
        G.add_edge(8, 2)
        G.add_edge(2, 9)
        G.add_edge(1, 4)
        G.add_edge(1, 5)
        G.add_edge(1, 6)

##        G = self.tree.digraph()
        

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
        
        def rel_widths_below(G, node, limit = math.inf, ignore = None, flip_depth = 0):
            if ignore is None:
                ignore = set()
            next_nodes = list(x for x in G.successors(node) if not x in ignore)
            if len(next_nodes) == 0 or limit == 0:
                return {node : 0.0}
            else:
                n = len(next_nodes)
                sub_rel_widths = []
                sub_ignore = set(ignore) | set([node])
                for p in next_nodes:
                    srw = rel_widths_below(G, p, limit = limit - 1, ignore = sub_ignore, flip_depth = flip_depth)
                    if flip_depth != 0:
                        srw |= rel_widths_above(G, p, limit = 1, ignore = sub_ignore, flip_depth = flip_depth - 1)
                        #srw |= rel_widths_above(G, p, limit = limit - 1, ignore = sub_ignore, flip_depth = flip_depth - 1)
                    sub_rel_widths.append(srw)
                    for x in srw:
                        sub_ignore.add(x)
                rel_widths = sub_rel_widths[0]
                for i in range(1, n):
                    move_by = -math.inf
                    for x in rel_widths:
                        for y in sub_rel_widths[i]:
                            if abs(node_heights[x] - node_heights[y]) < 1:
                                move_by = max(move_by, 1 + rel_widths[x] - sub_rel_widths[i][y])
                    for n, w in sub_rel_widths[i].items():
                        rel_widths[n] = w + move_by
                if node == 2:
                    print(next_nodes)
                rel_widths[node] = 0.5 * (max(rel_widths[p] for p in next_nodes) + min(rel_widths[p] for p in next_nodes))
                return rel_widths
            
        def rel_widths_above(G, node, limit = math.inf, ignore = None, flip_depth = 0):
            return rel_widths_below(G.reverse(), node, limit = limit, ignore = ignore, flip_depth = flip_depth)

        root = max(G.nodes(), key = lambda x : len(nx.ancestors(G, x)))
        node_widths = rel_widths_above(G, root, flip_depth = 1)

        G = G.subgraph(node_widths.keys())
        
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
##            
##        for x in G.nodes():
##            if type(self.tree.entity_lookup[x]) == treedata.Person:
##                colour = (0, 1, 1, 1)                
##                self.shapes.add_shape(shapely.geometry.Point([node_widths[x], node_heights[x]]).buffer(0.2), colour)

        node_widths = {n : w + random.uniform(-0.1, 0.1) for n, w in node_widths.items()}
        node_heights = {n : h + random.uniform(-0.1, 0.1) for n, h in node_heights.items()}

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
