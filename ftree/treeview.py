from ftree import treedata
import pgbase
import pygame
import numpy as np
import moderngl
import sys
import shapely.geometry
import networkx as nx



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
                        f_colour = vec4(0.7, 0.7, 0.7, 0);
                    } else {
                        f_colour = vec4(0.8, 0.8, 0.8, 0);
                    }
                }
    
            """,
        )
        vertices = self.ctx.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]).astype('f4'))
        indices = self.ctx.buffer(np.array([[0, 1, 3], [0, 2, 3]]))
        self.vao = self.ctx.vertex_array(self.prog,
                                         [(vertices, "2f4", "unit_pos")],
                                         indices)


        import math
        import random
        self.shapes = pgbase.canvas2d.ShapelyModel(self.ctx)
##        for x in range(0, 1000):
##            self.shapes.add_shape(shapely.geometry.Point(math.sqrt(x) * math.cos(x), math.sqrt(x) * math.sin(x)).buffer(1), (math.sin(x), math.cos(x), 0.5, 1))
##        self.shapes.update_vao()


        G = self.tree.digraph()

        H = G.to_undirected()

        print(H)

        for boop in nx.k_edge_subgraphs(H, 3):
            if len(boop) >= 2:
                print(boop)


        input()
        
        entity_heights = {x : 0 for x in G.nodes() if type(self.tree.entity_lookup[x]) == treedata.Person}

        #ancestors
        #descendants
        #predecessors
        #successors


        x = max([x for x in G.nodes() if type(self.tree.entity_lookup[x]) == treedata.Person], key = lambda x : len(nx.ancestors(G, x)) + len(nx.descendants(G, x)))
        upwards = set([x])
        to_check = set([x])
        while len(to_check) != 0:
            new_to_check = set()
            for x in to_check:
                for y in G.predecessors(x):
                    upwards.add(y)
                    new_to_check.add(y)
                    entity_heights[y] = entity_heights[x] + 1
            to_check = new_to_check

##        maximal_upwards = set(x for x in upwards if len(list(G.predecessors(x))) == 0)
##
##        class Component():
##            def __init__(self, x):
##                self.graph = nx.DiGraph()
##                self.graph.add_node(x)
##                self.to_check = set(x)
##            
##
##        components = [Component(x) for x in upwards if len(list(G.predecessors(x))) == 0]            

        

        G = G.subgraph(upwards)

        
            

        



        entity_positions = {}
        for x in G.nodes:
            entity_positions[x] = [random.uniform(-10, 10), entity_heights[x]]
                
            
        for x, y in G.edges:               
            self.shapes.add_shape(shapely.geometry.LineString([entity_positions[x], entity_positions[y]]).buffer(0.05), (0, 0, 1, 1))

        for x in G.nodes():
##            print(source, root)
##            if x in maximal_upwards:
##                colour = (0, 1, 0, 1)
##            else:
            colour = (1, 1, 0, 1)                
            self.shapes.add_shape(shapely.geometry.Point(entity_positions[x]).buffer(0.2), colour)
            
        self.shapes.update_vao()

##        for x in nx.topological_generations(G):
##            print("  ".join([str(self.tree.entity_lookup[ident]) for ident in x]))

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
