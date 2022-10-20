import math
import numpy as np
import copy
import itertools
import colorsys
import random
import sympy


def coset_table_perms(table):
    from sympy.combinatorics import Permutation
    #table is a coset table from sympy
    #coset i is taken to coset j by generator g iff table[i][2 * g] == j
    num = len(table) #number of cosets
    n_gens = len(table[0]) // 2
    for row in table:
        assert len(row) == 2 * n_gens
    return [Permutation([table[i][2 * j] for i in range(num)]) for j in range(n_gens)], num



def relf_mat(vec):
    vec = np.array([vec]).transpose()
    return np.eye(len(vec)) - 2 * vec @ vec.transpose()



class Polytope():
    def __init__(self, dim, orders):
        for key in orders:
            assert type(key) == frozenset
            assert len(key) == 2
            assert type(orders[key]) == int
            assert orders[key] >= 2
            for n in key:
                assert type(n) == int
                assert 0 <= n < dim
        orders = copy.copy(orders)
        for g1 in range(dim):
            for g2 in range(g1 + 1, dim):
                key = frozenset([g1, g2])
                if not key in orders:
                    orders[key] = 2

        #dim is the number of generators
        #orders is {frozenset({g1, g2}) : order} for each pair of gens g1, g2
        self.orders = orders
        self.dim = dim

        def reflrepr(n, orders):
            #find vectors representing normals to hyperplanes in which pairs of reflections have the order specified in orders
            #ie, find planes at given angles to eachother
            dims = set([])
            for key in orders:
                assert type(key) == frozenset
                assert len(key) == 2
                for v in key:
                    assert type(v) == int
                    assert v >= 0
                    assert v < n
                for dim in key:
                    dims.add(dim)

            dots = {key : -abs(math.cos(math.pi / orders[key])) for key in orders}

            M = np.array([[1]])
            for d in range(1, n):
                np.array([dots[frozenset([i, d])] for i in range(0, d)])

                new_vec = np.zeros(d + 1)
                new_vec[0 : d] = np.linalg.inv(M) @ np.array([dots[frozenset([i, d])] for i in range(0, d)])
                new_vec[d] = math.sqrt(1 - sum(v ** 2 for v in new_vec[0 : d]))

                new_M = np.zeros([d + 1, d + 1])
                new_M[:d, :d] = M
                new_M[d, :] = new_vec

                M = new_M
            return [vec for vec in M] #return the rows of M as the list of vectors

        #for each gen, we get a coresponding mirror
        self.mirrors = reflrepr(self.dim, self.orders)

        from sympy.combinatorics.fp_groups import FpGroup
        from sympy.combinatorics.free_groups import free_group
        
        F = free_group(", ".join(f"g{i}" for i in range(self.dim)))[0]
        self.group_gens = F.generators
        rels = []
        for i in range(self.dim):
            rels.append(self.group_gens[i] ** 2)
        for pair in self.orders:
            i, j = tuple(pair)
            rels.append((self.group_gens[i] * self.group_gens[j]) ** self.orders[pair])            
        self.group = FpGroup(F, rels)

        self.phi = ( 1 + math.sqrt(5) ) / 2
        self.off = random.uniform(0, 1)


    def coset_info(self, stab_gens):
        from sympy.combinatorics import Permutation
        from sympy.combinatorics.perm_groups import PermutationGroup

        def product(elems):
            assert len(elems) >= 1
            if len(elems) == 1:
                return elems[0]
            else:
                return elems[0] * product(elems[1:])
                        
        T = self.group.coset_enumeration([product([self.group_gens[i] for i in gen_idxs]) for gen_idxs in stab_gens], max_cosets = 100000)
        T.standardize()
        perms, num_cosets = coset_table_perms(T.table)

        paths = {0 : ([], Permutation([], size = num_cosets))}
        to_check = {0}
        while len(to_check) != 0:
            new_to_check = set([])
            for gen_idx, gen in enumerate(perms):
                for x in list(paths.keys()):
                    y = gen(x)
                    if not y in paths:
                        paths[y] = (paths[x][0] + [gen_idx], paths[x][1] * gen)
                        new_to_check.add(y)
            to_check = new_to_check

        paths = [paths[x][0] for x in sorted(paths.keys())]

        assert len(paths) == num_cosets
        return num_cosets, perms, paths


    def coset_graph(self, stab_gens):
        from graphvis import examples
        n, perms, paths = self.coset_info(stab_gens) #how the generators act on the stabilizer cosets
        return examples.permutations(perms)


    def polytope_graph(self, active):
        assert len(active) == self.dim
        for actv in active:
            assert type(actv) == bool
        
        from graphvis import structure
        from sympy.combinatorics.perm_groups import PermutationGroup

        class CosetGraph(structure.Graph):
            @staticmethod
            def node_surfs(ctx, ident):
                return None

        #enumerate elements of the full group
        n, perms, paths = self.coset_info([])
        #construct subgroup whose cosets are the graph verticies
        H = PermutationGroup([perm for idx, perm in enumerate(perms) if not active[idx]])
        H_orbits = [frozenset(orb) for orb in H.orbits()]
        for x in range(n):
            if not any(x in orb for orb in H_orbits):
                H_orbits.append(frozenset([x]))

        graph = CosetGraph()
        for orb in H_orbits:
            graph.add_node(orb)
        edges = set([])
        for orb1 in H_orbits:
            for orb2 in H_orbits:
                for idx, actv in enumerate(active):
                    if actv:
                        if any(perms[idx](x) == y for x, y in itertools.product(orb1, orb2)):
                            edges.add((frozenset([orb1, orb2]), idx))

        phi = ( 1 + math.sqrt(5) ) / 2
        off = random.uniform(0, 1)
        for pair, idx in edges:
            orb1, orb2 = tuple(pair)
            graph.add_edge(orb1, orb2, list(colorsys.hls_to_rgb(off + idx / phi, 0.5, 1)) + [1])

        return graph

    def path_to_mat(self, path):
        gen_refl = [relf_mat(mirror) for mirror in self.mirrors]
        mat = np.eye(self.dim)
        for idx in path:
            mat = gen_refl[idx] @ mat
        return mat

    def refl_mats(self, stab_gens):
        n, perms, paths = self.coset_info(stab_gens)
        return [self.path_to_mat(path) for path in paths]

    def get_geoms(self, coeffs, radius = None):
        assert len(coeffs) == self.dim
        position = np.linalg.solve(np.array(self.mirrors), coeffs)
        if not radius is None:
            position = radius * position / np.linalg.norm(position)
        refl_poses = [list(relf_mat(mirror) @ position) for mirror in self.mirrors]

        verticies = []
        edges = []
        triangles = []

        for mat in self.refl_mats([[idx] for idx, coeff in enumerate(coeffs) if coeff == 0]):
            verticies.append(list(mat @ position))

        for idx in range(self.dim):
            if coeffs[idx] != 0:
                edge_pos = refl_poses[idx]#relf_mat(self.mirrors[idx]) @ position
                stab_gens = [[idx]]
                for idx2 in range(self.dim):
                    if idx != idx2:
                        if coeffs[idx2] == 0:
                            if self.orders[frozenset([idx, idx2])] == 2:
                                stab_gens.append([idx2])
                _, _, paths = self.coset_info(stab_gens)
                n, perms, _ = self.coset_info([[idx]])
                for path in paths:
                    mat = self.path_to_mat(path)
                    a = list(mat @ position)
                    b = list(mat @ edge_pos)
                    e = 0
                    for x in path:
                        e = perms[x](e)
                    edges.append([a, b, idx, e])

        face_cosets = {}
        for i, j in itertools.combinations(range(self.dim), 2):
            face_cosets[frozenset({i, j})] = self.coset_info([[i], [j]])
        
        for i in range(self.dim):
            if coeffs[i] != 0:
                edge_pos = refl_poses[i]#relf_mat(self.mirrors[idx]) @ position
                for j in range(self.dim):
                    if i != j:
                        if (coeffs[i] != 0 and coeffs[j] != 0) if (self.orders[frozenset([i, j])] == 2) else (coeffs[i] != 0 or coeffs[j] != 0):
                            stabs = [[i]] + [[k] for k in [k for k in range(self.dim) if (not k in {i, j} and coeffs[k] == 0)] if (self.orders[frozenset([i, k])] == 2 and self.orders[frozenset([j, k])] == 2)]
                            _, _, paths = self.coset_info(stabs)
                            n, perms, _ = face_cosets[frozenset({i, j})]
                            
                            for path in paths:
                                mat = self.path_to_mat(path)
                                q, r = np.linalg.qr(np.array([self.mirrors[i], self.mirrors[j]]).transpose(), mode = "complete")
                                nullsp = q[:, 2:]
                                #base_pos @ nullsp is the dot product of base pos in each nullspace direction
                                #nullsp @ (base_pos @ nullsp) is the nullsp vectors added up by these components
                                mid_pos = nullsp @ (position @ nullsp)

                                f = 0
                                for x in path:
                                    f = perms[x](f)
                                
                                triangles.append([mat @ position, mat @ edge_pos, mat @ mid_pos, i, j, f])
    
##        for i in range(self.dim):
##            if coeffs[i] != 0:
##                edge_pos = refl_poses[i]#relf_mat(self.mirrors[idx]) @ position
##                for j in range(self.dim):
##                    if i != j:
##                        if (coeffs[i] != 0 and coeffs[j] != 0) if (self.orders[frozenset([i, j])] == 2) else (coeffs[i] != 0 or coeffs[j] != 0):
##                            stabs = [[i]] + [[k] for k in [k for k in range(self.dim) if (not k in {i, j} and coeffs[k] == 0)] if (self.orders[frozenset([i, k])] == 2 and self.orders[frozenset([j, k])] == 2)]
##                            refls = self.refl_mats(stabs)
##                            for mat in refls:
##                                q, r = np.linalg.qr(np.array([self.mirrors[i], self.mirrors[j]]).transpose(), mode = "complete")
##                                nullsp = q[:, 2:]
##                                #base_pos @ nullsp is the dot product of base pos in each nullspace direction
##                                #nullsp @ (base_pos @ nullsp) is the nullsp vectors added up by these components
##                                mid_pos = nullsp @ (position @ nullsp)
##                                triangles.append([mat @ position, mat @ edge_pos, mat @ mid_pos, i, j])
                                
        return verticies, edges, triangles

    def get_colour(self, idx):
        return list(colorsys.hls_to_rgb(self.off + idx / self.phi, 0.5, 1))

    def draw_solid(self, canvas, coeffs, radius = None, node_rad = 0.3, edge_rad = 0.2, tri_alpha = 0.1):
        from pgbase import canvasnd
        from pgbase import canvasnd
        assert isinstance(canvas, canvasnd.Window)
        assert canvas.dim == self.dim
        
        verticies, edges, triangles = self.get_geoms(coeffs, radius)

        for pos in verticies:
            canvas.draw_node(pos, [0.4, 0.4, 0.4, 1], node_rad)

        
        for a, b, idx, e in edges:
            canvas.draw_edge(a, b, self.get_colour(idx) + [1], edge_rad)

        for a, b, c, i, j, f in triangles:
            i_col = self.get_colour(i)
            j_col = self.get_colour(j)
            colour = [0.5 * (i_col[k] + j_col[k]) for k in [0, 1, 2]]
            canvas.draw_tri(a, b, c, colour + [tri_alpha])

    def draw_pencil(self, canvas, coeffs, radius = None, edge_rad = 0.05, tri_alpha = 0.1):
        from pgbase import canvasnd
        from sympy.combinatorics import Permutation
        assert isinstance(canvas, canvasnd.Window)
        assert canvas.dim == self.dim
        
        verticies, edges, triangles = self.get_geoms(coeffs, radius)
            
        for a, b, idx, e in edges:
            a = np.array(a)
            b = np.array(b)

            pad = 2 * edge_rad
            vec = b - a
            v_len = np.linalg.norm(vec)
            if v_len < 2 * pad:
                continue
            vec = vec / v_len
            a = a + pad * vec
            b = b - pad * vec
            
            colour = self.get_colour(idx)
            canvas.draw_pencil_edge(a, b, colour + [1], edge_rad)

        colours = {frozenset(pair) : self.get_colour(idx) for idx, pair in enumerate(itertools.combinations(range(self.dim), 2))}
        for a, b, c, i, j, f in triangles:
            i_col = self.get_colour(i)
            j_col = self.get_colour(j)
            colour = [0.5 * (i_col[k] + j_col[k]) for k in [0, 1, 2]]
            colour = [0, 0, 0]
            canvas.draw_tri(a, b, c, colour + [tri_alpha])




def RegularPolytope(shape):
    orders = {frozenset([idx, idx + 1]) : order for idx, order in enumerate(shape)}
    dim = len(shape) + 1
    return Polytope(dim, orders)










def example():
    import sys
    import pgbase
    import pygame
    
    pgbase.core.Window.setup([1600, 1000])
    polytope = RegularPolytope([3, 3, 3])
    window = pgbase.canvasnd.Window(polytope.dim, peel_depth = 5)
    polytope.draw_pencil(window, [1, 1, 1, 1])
    pgbase.core.run_root(window)


























    

