import pgbase
import pygame
import moderngl
import sys
import math
import numpy as np
import functools
import dataclasses
import os
import random
from chessai import boardai
import time


BOARD_VERTEX_SHADER = """
#version 430
uniform mat4 proj_mat;
uniform mat4 view_mat;
uniform mat4 world_mat;
uniform mat3 world_normal_mat;

in vec3 vert;
in vec3 normal;

out vec3 v_normal;
out vec4 v_pos_h;
out vec4 v_pos_h_rel;

void main() {
    v_normal = world_normal_mat * normal;
    v_pos_h = world_mat * vec4(vert, 1);
    v_pos_h_rel = vec4(vert, 1);
    gl_Position = proj_mat * view_mat * world_mat * vec4(vert, 1);
}
"""


@dataclasses.dataclass(frozen = True)
class WormSq():
    def get_pos_vec(self, sep):
        raise NotImplementedError()
    def get_glsl_idxs(self):
        #return the index(s) of this square in the array of colours passed to glsl for rendering.
        #yield part, idx where part in {0, 1, 2} specifies whether the square is the top, bottom or wormhole
        #and where idx is the index in the array for that particular shader.
        raise NotImplementedError()
        return
        yield
    def mirror_x(self):
        raise NotImplementedError()
    def mirror_y(self):
        raise NotImplementedError()
    def mirror_z(self):
        raise NotImplementedError()
    def mirror_xy_tranpose(self):
        raise NotImplementedError()
    def rot180_x(self):
        raise NotImplementedError()

@dataclasses.dataclass(frozen = True)
class WormSqTop(WormSq):
    x : int
    y : int
    def __post_init__(self):
        assert type(self.x) == type(self.y) == int
        assert 0 <= self.x < 8
        assert 0 <= self.y < 8
        assert not (2 <= self.x < 6 and 2 <= self.y < 6)
    def get_pos_vec(self, sep, in_rad):
        off = 0.1
        p = [self.y - 3.5, sep, self.x - 3.5]
        if self.x in {3, 4} and self.y == 1:
            p[0] -= off
        elif self.x in {3, 4} and self.y == 6:
            p[0] += off
        elif self.y in {3, 4} and self.x == 1:
            p[2] -= off
        elif self.y in {3, 4} and self.x == 6:
            p[2] += off
        return p, [0, 1, 0]
    def get_glsl_idxs(self):
        yield 0, self.y + 8 * self.x
    def mirror_x(self):
        return WormSqTop(7 - self.x, self.y)
    def mirror_y(self):
        return WormSqTop(self.x, 7 - self.y)
    def mirror_z(self):
        return WormSqBot(self.x, self.y)
    def mirror_xy_tranpose(self):
        return WormSqTop(self.y, self.x)
    def rot180_x(self):
        return WormSqBot(self.x, 7 - self.y)
    
@dataclasses.dataclass(frozen = True)
class WormSqBot(WormSq):
    x : int
    y : int
    def __post_init__(self):
        assert type(self.x) == type(self.y) == int
        assert 0 <= self.x < 8
        assert 0 <= self.y < 8
        assert not (2 <= self.x < 6 and 2 <= self.y < 6)
    def get_pos_vec(self, sep, in_rad):
        off = 0.1
        p = [self.y - 3.5, -sep, self.x - 3.5]
        if self.x in {3, 4} and self.y == 1:
            p[0] -= off
        elif self.x in {3, 4} and self.y == 6:
            p[0] += off
        elif self.y in {3, 4} and self.x == 1:
            p[2] -= off
        elif self.y in {3, 4} and self.x == 6:
            p[2] += off
        return p, [0, -1, 0]
    def get_glsl_idxs(self):
        yield 1, self.y + 8 * self.x
    def mirror_x(self):
        return WormSqBot(7 - self.x, self.y)
    def mirror_y(self):
        return WormSqBot(self.x, 7 - self.y)
    def mirror_z(self):
        return WormSqTop(self.x, self.y)
    def mirror_xy_tranpose(self):
        return WormSqBot(self.y, self.x)
    def rot180_x(self):
        return WormSqTop(self.x, 7 - self.y)

@dataclasses.dataclass(frozen = True)
class WormSqHole(WormSq):
    layer : int
    rot : int #measured clockwise from the pentagon nearest to WormSqTop(0, 0)
    def __post_init__(self):
        assert type(self.layer) == type(self.rot) == int
        assert 0 <= self.layer < 4
        assert 0 <= self.rot < 12
    def get_pos_vec(self, sep, in_rad):
        outer_rad = math.sqrt(5)
        ellipse_rad = outer_rad - in_rad
        if self.rot % 3 == 0 and self.layer in {0, 3}:
            v_ang = [-3.2 * math.pi / 8, None, None, 3.2 * math.pi / 8][self.layer]
        else:
            v_ang = [-2.7 * math.pi / 8, -0.8 * math.pi / 8, 0.8 * math.pi / 8, 2.7 * math.pi / 8][self.layer]
        p = p = [outer_rad - ellipse_rad * math.cos(v_ang), sep * math.sin(v_ang), 0]
        n = [-math.cos(v_ang) / ellipse_rad, math.sin(v_ang) / sep, 0]
        h_ang = 2 * math.pi * (self.rot + 4.5) / 12
        p = [math.cos(h_ang) * p[0] + math.sin(h_ang) * p[2], p[1], -math.sin(h_ang) * p[0] + math.cos(h_ang) * p[2]]
        n = [math.cos(h_ang) * n[0] + math.sin(h_ang) * n[2], n[1], -math.sin(h_ang) * n[0] + math.cos(h_ang) * n[2]]
        return p, n
    def get_glsl_idxs(self):
        yield 2, self.rot + 12 * self.layer
        if self.rot % 3 == 0 and self.layer in {0, 3}:
            x, y = [(2, 2), (5, 2), (5, 5), (2, 5)][self.rot // 3]
            yield {3 : 0, 0 : 1}[self.layer], y + 8 * x
    def mirror_x(self):
        return WormSqHole(self.layer, (3 - self.rot) % 12)
    def mirror_y(self):
        return WormSqHole(self.layer, (9 - self.rot) % 12)
    def mirror_z(self):
        return WormSqHole(3 - self.layer, self.rot)
    def mirror_xy_tranpose(self):
        return WormSqHole(self.layer, (-self.rot) % 12)
    def rot180_x(self):
        return WormSqHole(3 - self.layer, (9 - self.rot) % 12)

ALL_SQ = []
for x in range(8):
    for y in range(8):
        if not (2 <= x < 6 and 2 <= y < 6):
            ALL_SQ.append(WormSqTop(x, y))
            ALL_SQ.append(WormSqBot(x, y))
for lay in range(4):
    for rot in range(12):
        ALL_SQ.append(WormSqHole(lay, rot))

@functools.cache
def sq_to_idx(sq):
    return ALL_SQ.index(sq)

def idx_to_sq(idx):
    return ALL_SQ[idx]



def symmetry_reduce(sq, other_sqs, reduced_opp):    
    assert isinstance(sq, WormSq)
    def do_mirror_z():
        if type(sq) == WormSqBot:
            return True
        if type(sq) == WormSqHole:
            if sq.layer in {0, 1}:
                return True
        return False
    if do_mirror_z():
        for sq_nb in symmetry_reduce(sq.mirror_z(), [s.mirror_z() for s in other_sqs], reduced_opp):
            yield sq_nb.mirror_z()
        return
    
    def do_mirror_y():
        assert type(sq) != WormSqBot
        if type(sq) == WormSqTop:
            if sq.y >= 4:
                return True
        else:
            assert type(sq) == WormSqHole
            if sq.rot in {5, 6, 7, 8, 9, 10}:
                return True
        return False
        
    if do_mirror_y():
        for sq_nb in symmetry_reduce(sq.mirror_y(), [s.mirror_y() for s in other_sqs], reduced_opp):
            yield sq_nb.mirror_y()
        return

    def do_mirror_x():
        assert type(sq) != WormSqBot
        if type(sq) == WormSqTop:
            assert sq.y < 4
            if sq.x >= 4:
                return True
        else:
            assert type(sq) == WormSqHole
            assert sq.rot in {11, 0, 1, 2, 3, 4}
            if sq.rot in {2, 3, 4}:
                return True
        return False
        
    if do_mirror_x():
        for sq_nb in symmetry_reduce(sq.mirror_x(), [s.mirror_x() for s in other_sqs], reduced_opp):
            yield sq_nb.mirror_x()
        return

    def do_mirror_xy_trans():
        assert type(sq) != WormSqBot
        if type(sq) == WormSqTop:
            assert sq.y < 4
            assert sq.x < 4
            if sq.x < sq.y:
                return True
        else:
            assert type(sq) == WormSqHole
            assert sq.rot in {11, 0, 1}
            if sq.rot == 11:
                return True
        return False
        
    if do_mirror_xy_trans():
        for sq_nb in symmetry_reduce(sq.mirror_xy_tranpose(), [s.mirror_xy_tranpose() for s in other_sqs], reduced_opp):
            yield sq_nb.mirror_xy_tranpose()
        return

    #we are left with a handful of squares to compute, namely
    #WormSqTop(x=0, y=0)
    #WormSqTop(x=1, y=0)
    #WormSqTop(x=2, y=0)
    #WormSqTop(x=3, y=0)
    #WormSqTop(x=1, y=1)
    #WormSqTop(x=2, y=1)
    #WormSqTop(x=3, y=1)
    #WormSqHole(layer=3, rot=0)
    #WormSqHole(layer=3, rot=1)
    #WormSqHole(layer=2, rot=0)
    #WormSqHole(layer=2, rot=1)
    #these should be handled by the reduced_opp function

    yield from reduced_opp(sq, *other_sqs)


def flat_nbs(idx):
    sq = idx_to_sq(idx)
    def reduced_opp(sq):
        if sq == WormSqTop(0, 0):
            yield WormSqTop(1, 0)
            yield WormSqTop(0, 1)
        elif sq == WormSqTop(1, 0):
            yield WormSqTop(0, 0)
            yield WormSqTop(1, 1)
            yield WormSqTop(2, 0)
        elif sq == WormSqTop(2, 0):
            yield WormSqTop(1, 0)
            yield WormSqTop(2, 1)
            yield WormSqTop(3, 0)
        elif sq == WormSqTop(3, 0):
            yield WormSqTop(2, 0)
            yield WormSqTop(3, 1)
            yield WormSqTop(4, 0)
        elif sq == WormSqTop(1, 1):
            yield WormSqTop(1, 0)
            yield WormSqTop(0, 1)
            yield WormSqTop(2, 1)
            yield WormSqTop(1, 2)
        elif sq == WormSqTop(2, 1):
            yield WormSqTop(1, 1)
            yield WormSqTop(2, 0)
            yield WormSqTop(3, 1)
            yield WormSqHole(3, 0)
        elif sq == WormSqTop(3, 1):
            yield WormSqTop(2, 1)
            yield WormSqTop(3, 0)
            yield WormSqTop(4, 1)
            yield WormSqHole(3, 1)
        elif sq == WormSqHole(3, 0):
            yield WormSqTop(1, 2)
            yield WormSqTop(2, 1)
            yield WormSqHole(3, 1)
            yield WormSqHole(2, 0)
            yield WormSqHole(3, 11)
        elif sq == WormSqHole(3, 1):
            yield WormSqTop(3, 1)
            yield WormSqHole(3, 0)
            yield WormSqHole(2, 1)
            yield WormSqHole(3, 2)
        elif sq == WormSqHole(2, 0):
            yield WormSqHole(3, 0)
            yield WormSqHole(1, 0)
            yield WormSqHole(2, 11)
            yield WormSqHole(2, 1)
        elif sq == WormSqHole(2, 1):
            yield WormSqHole(3, 1)
            yield WormSqHole(1, 1)
            yield WormSqHole(2, 0)
            yield WormSqHole(2, 2)
        else:
            assert False

    for sq_nb in symmetry_reduce(sq, [], reduced_opp):
        yield sq_to_idx(sq_nb)

def diag_nbs(idx):
    sq = idx_to_sq(idx)

    def reduced_opp(sq):
        if sq == WormSqTop(0, 0):
            yield WormSqTop(1, 1)
        elif sq == WormSqTop(1, 0):
            yield WormSqTop(0, 1)
            yield WormSqTop(2, 1)
        elif sq == WormSqTop(2, 0):
            yield WormSqTop(1, 1)
            yield WormSqTop(3, 1)
        elif sq == WormSqTop(3, 0):
            yield WormSqTop(2, 1)
            yield WormSqTop(4, 1)
        elif sq == WormSqTop(1, 1):
            yield WormSqTop(0, 0)
            yield WormSqTop(2, 0)
            yield WormSqTop(0, 2)
            yield WormSqHole(3, 0)
        elif sq == WormSqTop(2, 1):
            yield WormSqTop(1, 0)
            yield WormSqTop(3, 0)
            yield WormSqTop(1, 2)
            yield WormSqHole(3, 1)
        elif sq == WormSqTop(3, 1):
            yield WormSqTop(2, 0)
            yield WormSqTop(4, 0)
            yield WormSqHole(3, 0)
            yield WormSqHole(3, 2)
        elif sq == WormSqHole(3, 0):
            yield WormSqTop(1, 1)
            yield WormSqTop(1, 3)
            yield WormSqTop(3, 1)
            yield WormSqHole(2, 11)
            yield WormSqHole(2, 1)
        elif sq == WormSqHole(3, 1):
            yield WormSqTop(2, 1)
            yield WormSqTop(4, 1)
            yield WormSqHole(2, 0)
            yield WormSqHole(2, 2)
        elif sq == WormSqHole(2, 0):
            yield WormSqHole(3, 11)
            yield WormSqHole(3, 1)
            yield WormSqHole(1, 11)
            yield WormSqHole(1, 1)
        elif sq == WormSqHole(2, 1):
            yield WormSqHole(3, 0)
            yield WormSqHole(3, 2)
            yield WormSqHole(1, 0)
            yield WormSqHole(1, 2)
        else:
            assert False

    for sq_nb in symmetry_reduce(sq, [], reduced_opp):
        yield sq_to_idx(sq_nb)

def opp(i, j):
    sq_i = idx_to_sq(i)
    sq_j = idx_to_sq(j)

    def reduced_opp(sq_j, sq_i):
        if sq_j == WormSqTop(0, 0):
            pass
        elif sq_j == WormSqTop(1, 0):
            if sq_i == WormSqTop(0, 0):
                yield WormSqTop(2, 0)
            elif sq_i == WormSqTop(2, 0):
                yield WormSqTop(0, 0)
        elif sq_j == WormSqTop(2, 0):
            if sq_i == WormSqTop(1, 0):
                yield WormSqTop(3, 0)
            elif sq_i == WormSqTop(3, 0):
                yield WormSqTop(1, 0)
        elif sq_j == WormSqTop(3, 0):
            if sq_i == WormSqTop(2, 0):
                yield WormSqTop(4, 0)
            elif sq_i == WormSqTop(4, 0):
                yield WormSqTop(2, 0)
        elif sq_j == WormSqTop(1, 1):
            if sq_i == WormSqTop(0, 0):
                yield WormSqHole(3, 0)
            elif sq_i == WormSqTop(1, 0):
                yield WormSqTop(1, 2)
            elif sq_i == WormSqTop(2, 0):
                yield WormSqTop(0, 2)
            elif sq_i == WormSqTop(2, 1):
                yield WormSqTop(0, 1)
            elif sq_i == WormSqHole(3, 0):
                yield WormSqTop(0, 0)
            elif sq_i == WormSqTop(1, 2):
                yield WormSqTop(1, 0)
            elif sq_i == WormSqTop(0, 2):
                yield WormSqTop(2, 0)
            elif sq_i == WormSqTop(0, 1):
                yield WormSqTop(2, 1)
        elif sq_j == WormSqTop(2, 1):
            if sq_i == WormSqTop(1, 0):
                yield WormSqHole(3, 1)
            elif sq_i == WormSqTop(2, 0):
                yield WormSqHole(3, 0)
            elif sq_i == WormSqTop(3, 0):
                yield WormSqTop(1, 2)
            elif sq_i == WormSqTop(3, 1):
                yield WormSqTop(1, 1)
            elif sq_i == WormSqHole(3, 1):
                yield WormSqTop(1, 0)
            elif sq_i == WormSqHole(3, 0):
                yield WormSqTop(2, 0)
            elif sq_i == WormSqTop(1, 2):
                yield WormSqTop(3, 0)
            elif sq_i == WormSqTop(1, 1):
                yield WormSqTop(3, 1)
        elif sq_j == WormSqTop(3, 1):
            if sq_i == WormSqTop(2, 0):
                yield WormSqHole(3, 2)
            elif sq_i == WormSqTop(3, 0):
                yield WormSqHole(3, 1)
            elif sq_i == WormSqTop(4, 0):
                yield WormSqHole(3, 0)
            elif sq_i == WormSqTop(4, 1):
                yield WormSqTop(2, 1)
            elif sq_i == WormSqHole(3, 2):
                yield WormSqTop(2, 0)
            elif sq_i == WormSqHole(3, 1):
                yield WormSqTop(3, 0)
            elif sq_i == WormSqHole(3, 0):
                yield WormSqTop(4, 0)
            elif sq_i == WormSqTop(2, 1):
                yield WormSqTop(4, 1)
        elif sq_j == WormSqHole(3, 0):
            if sq_i == WormSqTop(1, 1):
                yield WormSqHole(2, 1)
                yield WormSqHole(2, 11)
            elif sq_i == WormSqTop(2, 1):
                yield WormSqHole(2, 0)
                yield WormSqHole(3, 11)
            elif sq_i == WormSqTop(3, 1):
                yield WormSqHole(2, 11)
                yield WormSqTop(1, 3)
            elif sq_i == WormSqHole(3, 1):
                yield WormSqHole(3, 11)
                yield WormSqTop(1, 2)
            elif sq_i == WormSqHole(2, 1):
                yield WormSqTop(1, 3)
                yield WormSqTop(1, 1)
            elif sq_i == WormSqHole(2, 0):
                yield WormSqTop(1, 2)
                yield WormSqTop(2, 1)
            elif sq_i == WormSqHole(2, 11):
                yield WormSqTop(1, 1)
                yield WormSqTop(3, 1)
            elif sq_i == WormSqHole(3, 11):
                yield WormSqTop(2, 1)
                yield WormSqHole(3, 1)
            elif sq_i == WormSqTop(1, 3):
                yield WormSqTop(3, 1)
                yield WormSqHole(2, 1)
            elif sq_i == WormSqTop(1, 2):
                yield WormSqHole(3, 1)
                yield WormSqHole(2, 0)
        elif sq_j == WormSqHole(3, 1):
            if sq_i == WormSqTop(2, 1):
                yield WormSqHole(2, 2)
            elif sq_i == WormSqTop(3, 1):
                yield WormSqHole(2, 1)
            elif sq_i == WormSqTop(4, 1):
                yield WormSqHole(2, 0)
            elif sq_i == WormSqHole(3, 2):
                yield WormSqHole(3, 0)
            elif sq_i == WormSqHole(2, 2):
                yield WormSqTop(2, 1)
            elif sq_i == WormSqHole(2, 1):
                yield WormSqTop(3, 1)
            elif sq_i == WormSqHole(2, 0):
                yield WormSqTop(4, 1)
            elif sq_i == WormSqHole(3, 0):
                yield WormSqHole(3, 2)
        elif sq_j == WormSqHole(2, 0):
            if sq_i == WormSqHole(3, 0):
                yield WormSqHole(1, 0)
            elif sq_i == WormSqHole(3, 1):
                yield WormSqHole(1, 11)
            elif sq_i == WormSqHole(2, 1):
                yield WormSqHole(2, 11)
            elif sq_i == WormSqHole(1, 1):
                yield WormSqHole(3, 11)
            elif sq_i == WormSqHole(1, 0):
                yield WormSqHole(3, 0)
            elif sq_i == WormSqHole(1, 11):
                yield WormSqHole(3, 1)
            elif sq_i == WormSqHole(2, 11):
                yield WormSqHole(2, 1)
            elif sq_i == WormSqHole(3, 11):
                yield WormSqHole(1, 1)
        elif sq_j == WormSqHole(2, 1):
            if sq_i == WormSqHole(3, 1):
                yield WormSqHole(1, 1)
            elif sq_i == WormSqHole(3, 2):
                yield WormSqHole(1, 0)
            elif sq_i == WormSqHole(2, 2):
                yield WormSqHole(2, 0)
            elif sq_i == WormSqHole(1, 2):
                yield WormSqHole(3, 0)
            elif sq_i == WormSqHole(1, 1):
                yield WormSqHole(3, 1)
            elif sq_i == WormSqHole(1, 0):
                yield WormSqHole(3, 2)
            elif sq_i == WormSqHole(2, 0):
                yield WormSqHole(2, 2)
            elif sq_i == WormSqHole(3, 0):
                yield WormSqHole(1, 2)
        else:
            assert False

    for sq_k in symmetry_reduce(sq_j, [sq_i], reduced_opp):
        yield sq_to_idx(sq_k)



def pawn_moves(team, idx):
    if team == 1:
        sq = idx_to_sq(idx)
        #need only implement team 1s pawn movements on the top half of the board (including half the wormhole)
        #the rest are dealt with via symmetry
        if type(sq) == WormSqTop:
            if sq.y == 7:
                return []
            elif sq.y == 1:
                if sq.x in {0, 1, 6, 7}:
                    return [[sq_to_idx(WormSqTop(sq.x, sq.y + 1)), [sq_to_idx(WormSqTop(sq.x, sq.y + 2))]]]
                elif sq.x in {3, 4}:
                    return [[sq_to_idx(WormSqHole(3, sq.x - 2)), [sq_to_idx(WormSqHole(2, sq.x - 2))]]]
                elif sq.x == 2:
                    return [[sq_to_idx(WormSqHole(3, 0)), [sq_to_idx(WormSqHole(2, 0)), sq_to_idx(WormSqHole(3, 11))]]]
                else:
                    assert sq.x == 5
                    return [[sq_to_idx(WormSqHole(3, 3)), [sq_to_idx(WormSqHole(2, 3)), sq_to_idx(WormSqHole(3, 4))]]]
            else:
                assert sq.y <= 6
                if sq.x in {0, 1, 6, 7} or sq.y in {0, 6}:
                    return [[sq_to_idx(WormSqTop(sq.x, sq.y + 1)), []]]
                elif sq.y == 1:
                    return [[sq_to_idx(WormSqHole(3, sq.x - 2)), []]]
                else:
                    return []
        elif type(sq) == WormSqHole:
            if sq.layer in {2, 3}:
                if sq.rot in {0, 3} and sq.layer == 3:
                    if sq.rot == 0:
                        return [[sq_to_idx(WormSqHole(3, 11)), []],
                                [sq_to_idx(WormSqHole(2, 0)), []]]
                    elif sq.rot == 3:
                        return [[sq_to_idx(WormSqHole(3, 4)), []],
                                [sq_to_idx(WormSqHole(2, 3)), []]]
                elif sq.rot in {0, 1, 2, 3}:
                    return [[sq_to_idx(WormSqHole(sq.layer - 1, sq.rot)), []]]
                elif sq.rot in {6, 7, 8, 9}:
                    if sq.layer == 2:
                        return [[sq_to_idx(WormSqHole(sq.layer + 1, sq.rot)), []]]
                    else:
                        return [[sq_to_idx(WormSqTop(11 - sq.rot, 6)), []]]
                elif sq.rot in {10, 11}:
                    return [[sq_to_idx(WormSqHole(sq.layer, sq.rot - 1)), []]]
                elif sq.rot in {4, 5}:
                    return [[sq_to_idx(WormSqHole(sq.layer, sq.rot + 1)), []]]
        moves = []
        for rot180_x_move in pawn_moves(1, sq_to_idx(idx_to_sq(idx).rot180_x())):
            moves.append([sq_to_idx(idx_to_sq(rot180_x_move[0]).rot180_x()), [sq_to_idx(idx_to_sq(rr).rot180_x()) for rr in rot180_x_move[1]]])
        return moves
    else:
        assert team == -1
        moves = []
        for mirror_z_move in pawn_moves(1, sq_to_idx(idx_to_sq(idx).mirror_z())):
            moves.append([sq_to_idx(idx_to_sq(mirror_z_move[0]).mirror_z()), [sq_to_idx(idx_to_sq(mm).mirror_z()) for mm in mirror_z_move[1]]])
        return moves 

def pawn_promotions(team):
    return
    yield



STARTING_LAYOUT = set()
for x in range(8):
    STARTING_LAYOUT.add(boardai.Pawn(sq_to_idx(WormSqTop(x, 1)), 1, False))
    STARTING_LAYOUT.add(boardai.Pawn(sq_to_idx(WormSqTop(x, 6)), -1, False))
    STARTING_LAYOUT.add(boardai.Pawn(sq_to_idx(WormSqBot(x, 6)), 1, False))
    STARTING_LAYOUT.add(boardai.Pawn(sq_to_idx(WormSqBot(x, 1)), -1, False))
STARTING_LAYOUT.add(boardai.Rook(sq_to_idx(WormSqTop(0, 0)), 1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx(WormSqTop(1, 0)), 1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx(WormSqTop(2, 0)), 1, False))
STARTING_LAYOUT.add(boardai.King(sq_to_idx(WormSqTop(3, 0)), 1, False))
STARTING_LAYOUT.add(boardai.Queen(sq_to_idx(WormSqTop(4, 0)), 1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx(WormSqTop(5, 0)), 1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx(WormSqTop(6, 0)), 1, False))
STARTING_LAYOUT.add(boardai.Rook(sq_to_idx(WormSqTop(7, 0)), 1, False))
STARTING_LAYOUT.add(boardai.Rook(sq_to_idx(WormSqTop(0, 7)), -1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx(WormSqTop(1, 7)), -1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx(WormSqTop(2, 7)), -1, False))
STARTING_LAYOUT.add(boardai.King(sq_to_idx(WormSqTop(3, 7)), -1, False))
STARTING_LAYOUT.add(boardai.Queen(sq_to_idx(WormSqTop(4, 7)), -1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx(WormSqTop(5, 7)), -1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx(WormSqTop(6, 7)), -1, False))
STARTING_LAYOUT.add(boardai.Rook(sq_to_idx(WormSqTop(7, 7)), -1, False))

STARTING_LAYOUT.add(boardai.Rook(sq_to_idx(WormSqBot(0, 7)), 1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx(WormSqBot(1, 7)), 1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx(WormSqBot(2, 7)), 1, False))
STARTING_LAYOUT.add(boardai.Prince(sq_to_idx(WormSqBot(3, 7)), 1, False))
STARTING_LAYOUT.add(boardai.Queen(sq_to_idx(WormSqBot(4, 7)), 1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx(WormSqBot(5, 7)), 1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx(WormSqBot(6, 7)), 1, False))
STARTING_LAYOUT.add(boardai.Rook(sq_to_idx(WormSqBot(7, 7)), 1, False))
STARTING_LAYOUT.add(boardai.Rook(sq_to_idx(WormSqBot(0, 0)), -1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx(WormSqBot(1, 0)), -1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx(WormSqBot(2, 0)), -1, False))
STARTING_LAYOUT.add(boardai.Prince(sq_to_idx(WormSqBot(3, 0)), -1, False))
STARTING_LAYOUT.add(boardai.Queen(sq_to_idx(WormSqBot(4, 0)), -1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx(WormSqBot(5, 0)), -1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx(WormSqBot(6, 0)), -1, False))
STARTING_LAYOUT.add(boardai.Rook(sq_to_idx(WormSqBot(7, 0)), -1, False))

BOARD_SIGNATURE = boardai.BoardSignature(len(ALL_SQ), flat_nbs, opp, diag_nbs, opp, pawn_moves, pawn_promotions, STARTING_LAYOUT)







    



class FlatModel(pgbase.canvas3d.Model):
    def __init__(self, sep):
        self.sep = sep
        self.sq_colours = [(1, 1, 1, 0.5)] * 64

    def set_sq_colours(self, sq_colours):
        assert len(sq_colours) == 64
        for colour in sq_colours:
            assert len(colour) == 4
        self.sq_colours = sq_colours
        
    def triangles(self):
        verticies = [[-4, self.sep, -4], [4, self.sep, -4], [-4, self.sep, 4], [4, self.sep, 4]]
        normals = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
        indices = [[0, 1, 3], [0, 2, 3]]
        return verticies, normals, indices
        
    def make_vao(self, prog, include_normals = True):
        ctx = prog.ctx
        
        vertices, normals, indices = self.triangles()

        atribs = []
        atribs.append((ctx.buffer(np.array(vertices).astype('f4')), "3f4", "vert"))
        if include_normals:
            atribs.append((ctx.buffer(np.array(normals).astype('f4')), "3f4", "normal"))
        indices = ctx.buffer(np.array(indices))
        
        vao = ctx.vertex_array(prog,
                               atribs,
                               indices)

        return vao, moderngl.TRIANGLES

    def make_renderer(self, ctx):            
        prog = ctx.program(
            vertex_shader = BOARD_VERTEX_SHADER,
            geometry_shader = None,
            fragment_shader = """
                #version 430
                in vec4 v_pos_h;
                in vec4 v_pos_h_rel;
                in vec3 v_normal;
                uniform sampler2D peel_tex;
                uniform vec2 scr_size;
                uniform bool do_peel;
                uniform int depth;
                uniform vec3 cam_pos;
                out vec4 f_colour;
                uniform vec4[64] v_colour_arr;
                
                void main() {
                    if (v_pos_h_rel.x * v_pos_h_rel.x + v_pos_h_rel.z * v_pos_h_rel.z < 5) {
                        discard;
                    }
                    //if (0.1 < mod(v_pos_h_rel.x, 1) && mod(v_pos_h_rel.x, 1) < 0.9 && 0.1 < mod(v_pos_h_rel.z, 1) && mod(v_pos_h_rel.z, 1) < 0.9 && sqrt(v_pos_h_rel.x * v_pos_h_rel.x + v_pos_h_rel.z * v_pos_h_rel.z) > sqrt(5) + 0.1) {
                    //    discard;
                    //}
                    vec4 v_colour = v_colour_arr[int(floor(v_pos_h_rel.x + 4) + 8 * floor(v_pos_h_rel.z + 4))];
                    
                """ + pgbase.canvas3d.FRAG_MAIN + "}"
        )

        try: prog["v_colour_arr"].value = self.sq_colours
        except KeyError: pass

        vao, mode = self.make_vao(prog)

        class ModelRenderer(pgbase.canvas3d.Renderer):
            def __init__(self, prog):
                super().__init__([prog])
            def render(self):
                vao.render(mode, instances = 1)

        return ModelRenderer(prog)

    @functools.cache
    def clickdet_render(self, ctx):
        prog = ctx.program(
            vertex_shader = BOARD_VERTEX_SHADER,
            geometry_shader = None,
            fragment_shader = """
                #version 430
                in vec4 v_pos_h;
                in vec4 v_pos_h_rel;
                in vec3 v_normal;
                uniform sampler2D peel_tex;
                uniform vec2 scr_size;
                uniform bool do_peel;
                uniform int depth;
                uniform vec3 cam_pos;
                out float f_colour;
                uniform bool[64] v_bool_arr;
                
                void main() {
                    if (v_pos_h_rel.x * v_pos_h_rel.x + v_pos_h_rel.z * v_pos_h_rel.z < 5) {
                        discard;
                    }
                    bool is_active = v_bool_arr[int(floor(v_pos_h_rel.x + 4) + 8 * floor(v_pos_h_rel.z + 4))];
                    if (is_active) {
                        f_colour = 1;
                    } else {
                        f_colour = 0;
                    }
                }
                """
        )

        vao, mode = self.make_vao(prog, include_normals = False)

        class Renderer():
            def __init__(self, prog):
                self.prog = prog
            def __call__(self, sq_bools):
                try: prog["v_bool_arr"].value = sq_bools
                except KeyError: pass
                vao.render(mode, instances = 1)
                
        return Renderer(prog)





class CircModel(pgbase.canvas3d.Model):
    def __init__(self, sep, inner_rad):
        import random
        self.sep = sep
        self.inner_rad = inner_rad
        self.sq_colours = [tuple(random.uniform(0.5, 1) for _ in range(4)) for _ in range(48)]

    def set_sq_colours(self, sq_colours):
        assert len(sq_colours) == 48
        for colour in sq_colours:
            assert len(colour) == 4
        self.sq_colours = sq_colours

    @functools.cache
    def triangles(self):
        round_samples = 48
        ang_samples = 24 #how many horezontal strips do we want
        outer_rad = math.sqrt(5)
        ellipse_rad = outer_rad - self.inner_rad
        gap_eps = 0.5

        verts = {}
        i = 0
        for t in range(-1, ang_samples + 2):
            for x in range(round_samples):
                if t == -1:
                    p = [outer_rad + gap_eps, -self.sep, 0]
                    n = [0, -1, 0]
                elif t == ang_samples + 1:
                    p = [outer_rad + gap_eps, self.sep, 0]
                    n = [0, 1, 0]
                else:
                    v_ang = math.pi * (t / ang_samples - 0.5)
                    p = [outer_rad - ellipse_rad * math.cos(v_ang), self.sep * math.sin(v_ang), 0]
                    n = [-math.cos(v_ang) / ellipse_rad, math.sin(v_ang) / self.sep, 0]
                h_ang = 2 * math.pi * x / round_samples
                p = [math.cos(h_ang) * p[0] + math.sin(h_ang) * p[2], p[1], -math.sin(h_ang) * p[0] + math.cos(h_ang) * p[2]]
                n = [math.cos(h_ang) * n[0] + math.sin(h_ang) * n[2], n[1], -math.sin(h_ang) * n[0] + math.cos(h_ang) * n[2]]
                key = (x, t)
                assert not key in verts
                verts[key] = i, p, n
                i += 1
        
        verticies = [None] * len(verts)
        normals = [None] * len(verts)
        for key, info in verts.items():
            idx, pos, norm = info
            verticies[idx] = pos
            normals[idx] = norm
                
        indices = []
        for t1 in range(-1, ang_samples + 1):
            t2 = t1 + 1
            for x1 in range(round_samples):
                x2 = (x1 + 1) % round_samples
                i0, i1, i2, i3 = verts[(x1, t1)][0], verts[(x2, t1)][0], verts[(x1, t2)][0], verts[(x2, t2)][0]
                indices.append([i0, i1, i3])
                indices.append([i0, i2, i3])
        
        return verticies, normals, indices

    def make_vao(self, prog, include_normals = True):
        ctx = prog.ctx
        
        vertices, normals, indices = self.triangles()

        atribs = []
        atribs.append((ctx.buffer(np.array(vertices).astype('f4')), "3f4", "vert"))
        if include_normals:
            atribs.append((ctx.buffer(np.array(normals).astype('f4')), "3f4", "normal"))
        indices = ctx.buffer(np.array(indices))
        
        vao = ctx.vertex_array(prog,
                               atribs,
                               indices)

        return vao, moderngl.TRIANGLES


    def set_uniforms(self, prog):
        try: prog["sep"].value = self.sep
        except KeyError: pass
        try: prog["mid"].value = 0.6 * self.sep
        except KeyError: pass

    def make_renderer(self, ctx):            
        prog = ctx.program(
            vertex_shader = BOARD_VERTEX_SHADER,
            geometry_shader = None,
            fragment_shader = """
                #version 430
                in vec4 v_pos_h;
                in vec4 v_pos_h_rel;
                in vec3 v_normal;
                uniform sampler2D peel_tex;
                uniform vec2 scr_size;
                uniform bool do_peel;
                uniform int depth;
                uniform vec3 cam_pos;
                uniform float sep;
                uniform float mid;
                out vec4 f_colour;
                uniform vec4[48] v_colour_arr;

                const float tau = 6.28318530718;
                
                void main() {
                    if (v_pos_h_rel.x * v_pos_h_rel.x + v_pos_h_rel.z * v_pos_h_rel.z > 5) {
                        discard;
                    }

                    float a;
                    float b;
                    float c;
                    a = 4 * mod(atan(v_pos_h_rel.x, v_pos_h_rel.z) / tau, 1);
                    b = a + 0.0397563521171529 * sin(tau * a);
                    b = 3 * b;
                    c = 3 * a;
                    float f = (v_pos_h_rel.y / sep);
                    f = f * f;
                    float t = f * b + (1 - f) * c;

                    int layer;
                    if (v_pos_h_rel.y < -mid) {
                        layer = 0;
                    } else if (v_pos_h_rel.y < 0) {
                        layer = 1;
                    } else if (v_pos_h_rel.y < mid) {
                        layer = 2;
                    } else {
                        layer = 3;
                    }

                    vec4 v_colour = v_colour_arr[int(mod(floor(t) + 5, 12)) + 12 * layer];
                    
                """ + pgbase.canvas3d.FRAG_MAIN + "}"
        )

        self.set_uniforms(prog)
        try: prog["v_colour_arr"].value = self.sq_colours
        except KeyError: pass

        vao, mode = self.make_vao(prog)

        class ModelRenderer(pgbase.canvas3d.Renderer):
            def __init__(self, prog):
                super().__init__([prog])
            def render(self):
                vao.render(mode, instances = 1)

        return ModelRenderer(prog)

    @functools.cache
    def clickdet_render(self, ctx):
        prog = ctx.program(
            vertex_shader = BOARD_VERTEX_SHADER,
            geometry_shader = None,
            fragment_shader = """
                #version 430
                in vec4 v_pos_h;
                in vec4 v_pos_h_rel;
                in vec3 v_normal;
                uniform sampler2D peel_tex;
                uniform vec2 scr_size;
                uniform bool do_peel;
                uniform int depth;
                uniform vec3 cam_pos;
                out float f_colour;
                uniform bool[48] v_bool_arr;
                uniform float sep;
                uniform float mid;

                const float tau = 6.28318530718;
                
                void main() {
                    if (v_pos_h_rel.x * v_pos_h_rel.x + v_pos_h_rel.z * v_pos_h_rel.z > 5) {
                        discard;
                    }

                    float a;
                    float b;
                    float c;
                    a = 4 * mod(atan(v_pos_h_rel.x, v_pos_h_rel.z) / tau, 1);
                    b = a + 0.0397563521171529 * sin(tau * a);
                    b = 3 * b;
                    c = 3 * a;
                    float f = (v_pos_h_rel.y / sep);
                    f = f * f;
                    float t = f * b + (1 - f) * c;

                    int layer;
                    if (v_pos_h_rel.y < -mid) {
                        layer = 0;
                    } else if (v_pos_h_rel.y < 0) {
                        layer = 1;
                    } else if (v_pos_h_rel.y < mid) {
                        layer = 2;
                    } else {
                        layer = 3;
                    }

                    bool is_active = v_bool_arr[int(mod(floor(t) + 5, 12)) + 12 * layer];

                    if (is_active) {
                        f_colour = 1;
                    } else {
                        f_colour = 0;
                    }
                }"""
        )

        self.set_uniforms(prog)

        vao, mode = self.make_vao(prog, include_normals = False)

        class Renderer():
            def __init__(self, prog):
                self.prog = prog
            def __call__(self, sq_bools):
                try: prog["v_bool_arr"].value = sq_bools
                except KeyError: pass
                vao.render(mode, instances = 1)
                
        return Renderer(prog)
    





class Camera(pgbase.canvas3d.FlipFlyCamera):
    def __init__(self):
        super().__init__([0, 0, 0], 0, 0)
        
    




class BoardView(pgbase.canvas3d.Window):
    LIGHT_SQ_COLOUR = (255, 206, 158)
    DARK_SQ_COLOUR = (209, 139, 71)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, peel_depth = 3, camera = Camera())
        self.side_sep = 2.2
        self.inner_rad = 1.5

        lines = pgbase.canvas3d.UnitCylinder(12, False)
        r = 0.05
        s = self.side_sep / 2
        lines.add_line([-4, s, -4], [-4, s, 4], r, [1, 0, 0, 1])
        lines.add_line([-4, s, -4], [4, s, -4], r, [1, 0, 0, 1])
        lines.add_line([4, s, 4], [-4, s, 4], r, [1, 0, 0, 1])
        lines.add_line([4, s, 4], [4, s, -4], r, [1, 0, 0, 1])
        lines.add_line([-4, -s, -4], [-4, -s, 4], r, [0, 0, 1, 1])
        lines.add_line([-4, -s, -4], [4, -s, -4], r, [0, 0, 1, 1])
        lines.add_line([4, -s, 4], [-4, -s, 4], r, [0, 0, 1, 1])
        lines.add_line([4, -s, 4], [4, -s, -4], r, [0, 0, 1, 1])
        self.draw_model(lines)

        spheres = pgbase.canvas3d.UnitSphere(3)
        r = 0.075
        s = self.side_sep / 2
        spheres.add_sphere([-4, s, -4], r, [1, 0, 0, 1])
        spheres.add_sphere([4, s, -4], r, [1, 0, 0, 1])
        spheres.add_sphere([-4, s, 4], r, [1, 0, 0, 1])
        spheres.add_sphere([4, s, 4], r, [1, 0, 0, 1])
        spheres.add_sphere([-4, -s, -4], r, [0, 0, 1, 1])
        spheres.add_sphere([4, -s, -4], r, [0, 0, 1, 1])
        spheres.add_sphere([-4, -s, 4], r, [0, 0, 1, 1])
        spheres.add_sphere([4, -s, 4], r, [0, 0, 1, 1])
        self.draw_model(spheres)

        self.top_flat = FlatModel(self.side_sep / 2)
        self.bot_flat = FlatModel(-self.side_sep / 2)
        self.circ = CircModel(self.side_sep / 2, self.inner_rad)

        self.top_clickdet_render = self.top_flat.clickdet_render(self.ctx)
        self.bot_clickdet_render = self.bot_flat.clickdet_render(self.ctx)
        self.circ_clickdet_render = self.circ.clickdet_render(self.ctx)

        self.draw_model(self.top_flat)
        self.draw_model(self.bot_flat)
        self.draw_model(self.circ)

        self.pawn = pgbase.canvas3d.StlModel(os.path.join("chessai", "wormhole", "pawn.obj"))
        self.rook = pgbase.canvas3d.StlModel(os.path.join("chessai", "wormhole", "rook.obj"))
        self.knight = pgbase.canvas3d.StlModel(os.path.join("chessai", "wormhole", "knight.obj"))
        self.bishop = pgbase.canvas3d.StlModel(os.path.join("chessai", "wormhole", "bishop.obj"))
        self.queen = pgbase.canvas3d.StlModel(os.path.join("chessai", "wormhole", "queen.obj"))
        self.king = pgbase.canvas3d.StlModel(os.path.join("chessai", "wormhole", "king.obj"))
        self.prince = pgbase.canvas3d.StlModel(os.path.join("chessai", "wormhole", "prince.obj"))
        self.piece_models = {boardai.Pawn : self.pawn,
                             boardai.Rook : self.rook,
                             boardai.Knight : self.knight,
                             boardai.Bishop : self.bishop,
                             boardai.Queen : self.queen,
                             boardai.King : self.king,
                             boardai.Prince : self.prince}

        for sq in ALL_SQ:
            if random.randint(0, 2) == 0:
                def normalize(v):
                    length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                    return [v[i] / length for i in range(3)]
                p, n = sq.get_pos_vec(self.side_sep / 2, self.inner_rad)
                m = np.cross(np.random.random(3), n)
                l = np.cross(n, m)
                n, m, l = normalize(n), normalize(m), normalize(l)
                T = np.array([[m[0], n[0], l[0], 0],
                              [m[1], n[1], l[1], 0],
                              [m[2], n[2], l[2], 0],
                              [0, 0, 0, 1]])
                M = pgbase.canvas3d.translation(p[0], p[1], p[2]) @ T @ pgbase.canvas3d.translation(0, 0.04, 0) @ pgbase.canvas3d.scale(0.03, 0.03, 0.03) @ pgbase.canvas3d.rotate([1, 0, 0], 0.5 * math.pi) @ pgbase.canvas3d.rotate([0, 0, 1], random.uniform(0, 2 * math.pi))
                random.choice([self.pawn, self.rook, self.knight, self.bishop, self.king, self.prince, self.queen]).add_instance(M, random.choice([[0.75, 0, 1, 1], [0.2, 0.7, 0.2, 1]]))

        for piece_type, model in self.piece_models.items():
            self.draw_model(model)

        self.board = None
        self.moves = []
        self.ai_player = None
        
        self.move_selected_chain = []
        self.last_move = None
        self.last_interact_time = time.time()

        self.set_board(BOARD_SIGNATURE.starting_board())


##        for _ in range(20):
##            self.make_move(self.ai_player.best_move)
                        

    @property
    def remaining_sel_moves(self):
        if len(self.move_select_chain) == 0:
            return [(move, move.select_points[0]) for move in self.moves]
        else:
            def is_rem(move):
                if len(move.select_points) > len(self.move_select_chain):
                    if all(move.select_points[i].idx == self.move_select_chain[i] for i in range(len(self.move_select_chain))):
                        return True
                return False
            return [(move, move.select_points[len(self.move_select_chain)]) for move in self.moves if is_rem(move)]


    def make_move(self, move, m_id = None):
        self.set_board(move.to_board, m_id = m_id)
        self.last_move = move

    def set_board(self, board, m_id = None):
        if not m_id is None:
            sub_move_score_info = self.ai_player.sub_move_score_info(m_id)
        else:
            sub_move_score_info = None
        print("new board", board.num, round(board.static_eval(), 2))
        self.board = board
        self.moves = tuple(board.get_moves())
        self.move_select_chain = []
        if len(self.moves) == 0:
           self.ai_player = None
        else:
            self.ai_player = boardai.AiPlayer(self.board, move_score_info = sub_move_score_info)

        self.last_interact_time = time.time()
        self.update_sq_colours()
        self.update_piece_models()
        
    def update_piece_models(self):
        import random
        
        for piece_type, model in self.piece_models.items():
            model.clear_instances()
        for piece in self.board.pieces:
            def normalize(v):
                length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                return [v[i] / length for i in range(3)]
            p, n = idx_to_sq(piece.idx).get_pos_vec(self.side_sep / 2, self.inner_rad)
            m = np.cross(np.random.random(3), n)
            l = np.cross(n, m)
            n, m, l = normalize(n), normalize(m), normalize(l)
            T = np.array([[m[0], n[0], l[0], 0],
                          [m[1], n[1], l[1], 0],
                          [m[2], n[2], l[2], 0],
                          [0, 0, 0, 1]])
            M = pgbase.canvas3d.translation(p[0], p[1], p[2]) @ T @ pgbase.canvas3d.translation(0, 0.04, 0) @ pgbase.canvas3d.scale(0.03, 0.03, 0.03) @ pgbase.canvas3d.rotate([1, 0, 0], 0.5 * math.pi) @ pgbase.canvas3d.rotate([0, 0, 1], random.uniform(0, 2 * math.pi))
            self.piece_models[type(piece)].add_instance(M, {1 : [0.75, 0, 1, 1], -1 : [0.2, 0.7, 0.2, 1]}[piece.team])

    def update_sq_colours(self):
        import random
          
        colour_info = {}

        if len(self.move_select_chain) != 0:
            sq = idx_to_sq(self.move_select_chain[0])
            colour_info[sq] = (0, 0.5, 1, 1)

        for move, msp in self.remaining_sel_moves:
            if not msp.kind == boardai.MoveSelectionPoint.INVIS:
                sq = idx_to_sq(msp.idx)
                colour_info[sq] = {boardai.MoveSelectionPoint.REGULAR : (0, 1, 0, 1), boardai.MoveSelectionPoint.CAPTURE : (1, 0, 0, 1), boardai.MoveSelectionPoint.SPECIAL : (1, 0, 0.5, 1)}[msp.kind]


        if not self.ai_player is None:   
            best_move = self.ai_player.best_move
            for msp in best_move.select_points:
                sq = idx_to_sq(msp.idx)
                colour_info[sq] = (1, 0.5, 0, 1)
        
        WHITE = (1, 1, 1, 0.6)
        BLACK = (0, 0, 0, 0.6)
        colours = [([BLACK, WHITE] * 4 + [WHITE, BLACK] * 4) * 4, ([WHITE, BLACK] * 4 + [BLACK, WHITE] * 4) * 4, ([WHITE, BLACK] * 6 + [BLACK, WHITE] * 6) * 2]
        for sq, colour in colour_info.items():
            for part, idx, in sq.get_glsl_idxs():
                colours[part][idx] = colour
            
        self.top_flat.set_sq_colours(colours[0])
        self.bot_flat.set_sq_colours(colours[1])
        self.circ.set_sq_colours(colours[2])
        self.clear_renderer_cache()

    def draw(self):
##        world_mat = np.eye(4)
##        world_mat[0:3, 0:3] = self.world_mat
##        self.set_uniforms([self.top_clickdet_render.prog, self.bot_clickdet_render.prog, self.circ_clickdet_render.prog])
        super().draw()

    def get_sq(self, pos):
        #calculate which (if any) square has been clicked
        #works by labeling each square with a positive integer using sq_to_idx and idx_to_sq
        #we think of these integers are 8 bit binary numbers
        #one render is one for each of the 8 bits
        #each render, only those squares with a 1s digit at that position is rendered
        #we check the colour of the square at the position of the mouse
        #using the resulting 8 results, we can figure out which (if any) square was clicked

        #perhaps this could be made better by only rendering the pixel where the mouse is? it is fairly fast and clean as is though so theres probably no need.

        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self.set_uniforms([self.top_clickdet_render.prog, self.bot_clickdet_render.prog, self.circ_clickdet_render.prog])

        def gen_bits():
            for bitidx in range(8):
                tex = self.ctx.texture([self.width, self.height], 1)
                depth_tex = self.ctx.depth_texture([self.width, self.height])
                fbo = self.ctx.framebuffer(tex, depth_tex)
                fbo.use()
                self.ctx.clear(0)

                checks = [[False] * 64, [False] * 64, [False] * 48]

                for sq in ALL_SQ:
                    check = (sq_to_idx(sq) + 1) & (2 ** bitidx) != 0 #the +1 here is so that 0 means no square was clicked
                    for part, i in sq.get_glsl_idxs():
                        checks[part][i] = check
                
                self.top_clickdet_render(checks[0])
                self.bot_clickdet_render(checks[1])
                self.circ_clickdet_render(checks[2])

                arr = pgbase.tools.tex_to_np(tex)
                arr = arr.transpose([1, 0, 2]).reshape([self.width, self.height])
                arr = np.flip(arr, 1)
                if arr[pos[0], pos[1]] > 128:
                    yield 1
                else:
                    yield 0

                fbo.release()
                tex.release()
                depth_tex.release()

        clicked_idx_plus_one = sum(c * 2 ** p for p, c in enumerate(gen_bits()))
        if clicked_idx_plus_one != 0: #some square was clicked
            clicked_idx = clicked_idx_plus_one - 1
            return idx_to_sq(clicked_idx)
        else:
            return None

    def tick(self, dt):
        super().tick(dt)

        #rotating the board
        if not pygame.mouse.get_pressed()[2]:
            f = dt * 10
            y1 = self.world_mat[0:3, 1]
            if abs(y1[1]) > math.sin(math.pi / 4):
                y2 = np.array([0, (1 if y1[1] > 0 else -1), 0])
                y = pgbase.canvas3d.normalize(f * y2 + (1 - f) * y1)
                x_off = self.world_mat[0:3, 0]
                z = np.cross(x_off, y)
                x = np.cross(y, z)
            else:                
                y2 = pgbase.canvas3d.normalize([y1[0], 0, y1[2]])
                y = pgbase.canvas3d.normalize(f * y2 + (1 - f) * y1)
                u = np.array([0, 1, 0])
                h = pgbase.canvas3d.normalize(np.cross(y, u))
                x1 = self.world_mat[0:3, 0]
                x_off = [np.dot(x1, h), np.dot(x1, u)]
                a = 2 * math.pi * math.floor((4 * math.atan2(x_off[1], x_off[0])) / (2 * math.pi) + 0.5) / 4
                x2 = math.cos(a) * h + math.sin(a) * u
                x = pgbase.canvas3d.normalize(f * x2 + (1 - f) * x1)
                z = np.cross(x, y)
            self.world_mat = np.array([x, y, z]).transpose()

##        if not self.ai_player is None and time.time() - self.last_interact_time > 0.1:
##            if not self.ai_player.best_move is None:
##                self.make_move(self.ai_player.best_move)
                
        if not self.ai_player is None and time.time() - self.last_interact_time > 0.75:
            if self.ai_player.tick():
                self.update_sq_colours()

    def focus_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if pgbase.tools.in_rect(event.pos, self.rect):
                if event.button == 1:
                    self.set_focus(not self.has_focus)
        return False


    def event(self, event):
        self.last_interact_time = time.time()
        super().event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if not self.has_focus:
                if event.button == 3:
                    sq = self.get_sq(event.pos)                    
                    
                    if sq is None:
                        self.move_select_chain = []
                    else:
                        click_idx = sq_to_idx(sq)
                        clicked_moves = [move for move, sel_info in self.remaining_sel_moves if sel_info.idx == click_idx]
                        for move in clicked_moves:
                            if [msp.idx for msp in move.select_points] == self.move_select_chain + [click_idx]:
                                self.make_move(move)
                                break
                        else:
                            if len(clicked_moves) == 0:
                                self.move_select_chain = []
                                clicked_moves = [move for move, sel_info in self.remaining_sel_moves if sel_info.idx == click_idx]
                                if len(clicked_moves) != 0:
                                    self.move_select_chain.append(click_idx)
                            else:
                                self.move_select_chain.append(click_idx)
                    self.update_sq_colours()

        if event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[2]:
                mat = np.eye(3)
                mat = mat @ pgbase.canvasnd.rot_axes(3, 0, 2, self.camera.theta)
                mat = mat @ pgbase.canvasnd.rot_axes(3, 2, 1, self.camera.phi)
                mat = mat @ pgbase.canvasnd.rot_axes(3, 0, 2, 0.01 * event.rel[0])
                mat = mat @ pgbase.canvasnd.rot_axes(3, 2, 1, 0.01 * event.rel[1])
                mat = mat @ pgbase.canvasnd.rot_axes(3, 2, 1, -self.camera.phi)
                mat = mat @ pgbase.canvasnd.rot_axes(3, 0, 2, -self.camera.theta)
                self.world_mat = mat @ self.world_mat

    def end(self):
        #terminate ai subprocesses
        if not self.ai_player is None:
            del self.ai_player
        self.ai_player = None

                




def run():
    pgbase.core.Window.setup(size = [1600, 1000])
    pgbase.core.run_root(BoardView())

























    
