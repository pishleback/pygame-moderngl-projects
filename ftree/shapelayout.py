import functools
import shapely
import math


def empty():
    return shapely.geometry.MultiPolygon([])
    
def box(a, b, c, d):
    return shapely.geometry.box(a, b, c, d)

def line(a, b, c, d):
    return shapely.geometry.LineString([[a, b], [c, d]])

def circle(x, y, r):
    return point(x, y).buffer(r).exterior

def point(x, y):
    return shapely.geometry.Point(x, y)

def bezier(ps):
    def bezier_sample(ps, f):
        if len(ps) == 1:
            return ps[0]
        else:
            sub_ps = []
            for i in range(len(ps) - 1):
                a = ps[i]
                b = ps[i + 1]
                sub_ps.append(tuple(a[i] + f * (b[i] - a[i]) for i in [0, 1]))
            return bezier_sample(sub_ps, f)

    def gen_points():
        N = 24
        for s in range(N + 1):
            yield bezier_sample(ps, s / N)

    ans = shapely.geometry.GeometryCollection([])
    points = tuple(gen_points())
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        ans |= line(*a, *b)
        
    return ans

def smoothline(ps):
    assert len(ps) >= 3
    def gen_bez_ctrl():
        for idx in range(len(ps) - 2):
            a = ps[idx]
            b = ps[idx + 1]
            c = ps[idx + 2]
            if idx != 0:
                a = [0.5 * (a[i] + b[i]) for i in [0, 1]]
            if idx != len(ps) - 3:
                c = [0.5 * (c[i] + b[i]) for i in [0, 1]]
            yield a, b, c

    ans = shapely.geometry.GeometryCollection([])
    for bez_ctrl in gen_bez_ctrl():
        ans |= bezier(bez_ctrl)
    return ans


@functools.cache
def letter_polygons(letter):
    assert type(letter) == str          

    ans = shapely.geometry.GeometryCollection([])
    width = 3

    if letter == " ":
        width = 1

    #ALPHABET
    
    elif letter == "A":
        h = 1/3
        ans |= line(0, 0, 1, 4)
        ans |= line(2, 0, 1, 4)
        ans |= line(h, 4 * h, 2 - h, 4 * h)
    elif letter == "B":
        ans |= circle(1, 3, 1)
        ans |= circle(1, 1, 1)
        ans -= box(0, 0, 1, 4)
        ans |= line(0, 0, 0, 4)
        ans |= line(0, 4, 1, 4)
        ans |= line(0, 2, 1, 2)
        ans |= line(0, 0, 1, 0)
    elif letter == "C":
        ans |= circle(2, 2, 2)
        ans -= box(2, 0, 4, 4)
    elif letter == "D":
        ans |= circle(0, 2, 2)
        ans -= box(-2, 0, 0, 4)
        ans |= line(0, 0, 0, 4)
    elif letter == "E":
        ans |= line(0, 0, 0, 4)
        ans |= line(0, 4, 2, 4)
        ans |= line(0, 2, 2, 2)
        ans |= line(0, 0, 2, 0)
    elif letter == "F":
        ans |= line(0, 0, 0, 4)
        ans |= line(0, 4, 2, 4)
        ans |= line(0, 2, 1, 2)
    elif letter == "G":
        ans |= circle(2, 2, 2)
        ans -= box(2, 0, 4, 4)
        ans |= line(2, 0, 2, 2)
        ans |= line(1, 2, 2, 2)
    elif letter == "H":
        ans |= line(0, 0, 0, 4)
        ans |= line(2, 0, 2, 4)
        ans |= line(0, 2, 2, 2)
    elif letter == "I":
        ans |= line(0, 0, 2, 0)
        ans |= line(0, 4, 2, 4)
        ans |= line(1, 0, 1, 4)
    elif letter == "J":
        ans |= circle(1, 1, 1)
        ans -= box(0, 1, 2, 2)
        ans |= line(0, 4, 2, 4)
        ans |= line(2, 1, 2, 4)
    elif letter == "K":
        ans |= line(0, 0, 0, 4)
        ans |= line(0, 2, 2, 0)
        ans |= line(0, 2, 2, 4)
    elif letter == "L":
        ans |= line(0, 0, 0, 4)
        ans |= line(0, 0, 2, 0)
    elif letter == "M":
        ans |= line(0, 0, 0, 4)
        ans |= line(2, 0, 2, 4)
        ans |= line(0, 4, 1, 2)
        ans |= line(1, 2, 2, 4)
    elif letter == "N":
        ans |= line(0, 0, 0, 4)
        ans |= line(2, 0, 2, 4)
        ans |= line(0, 4, 2, 0)
    elif letter == "O":
        ans |= circle(1, 1, 1)
        ans |= circle(1, 3, 1)
        ans -= box(0, 1, 2, 3)
        ans |= line(0, 1, 0, 3)
        ans |= line(2, 1, 2, 3)
    elif letter == "P":
        ans |= circle(1, 3, 1)
        ans -= box(0, 2, 1, 4)
        ans |= line(0, 0, 0, 4)
        ans |= line(0, 4, 1, 4)
        ans |= line(0, 2, 1, 2)
    elif letter == "Q":
        ans |= circle(1, 1, 1)
        ans |= circle(1, 3, 1)
        ans -= box(0, 1, 2, 3)
        ans |= line(0, 1, 0, 3)
        ans |= line(2, 1, 2, 3)
        ans |= line(1, 1, 2, 0)
    elif letter == "R":
        ans |= circle(1, 3, 1)
        ans -= box(0, 2, 1, 4)
        ans |= line(0, 0, 0, 4)
        ans |= line(0, 4, 1, 4)
        ans |= line(0, 2, 1, 2)
        ans |= line(1, 2, 2, 0)
    elif letter == "S":
        ans |= circle(1, 1, 1)
        ans |= circle(1, 3, 1)
        ans -= box(0, 1, 1, 2)
        ans -= box(1, 2, 2, 3)
    elif letter == "T":
        ans |= line(0, 4, 2, 4)
        ans |= line(1, 0, 1, 4)
    elif letter == "U":
        ans |= circle(1, 1, 1)
        ans -= box(0, 1, 2, 2)
        ans |= line(0, 1, 0, 4)
        ans |= line(2, 1, 2, 4)
    elif letter == "V":
        ans |= line(0, 4, 1, 0)
        ans |= line(1, 0, 2, 4)
    elif letter == "W":
        ans |= line(0, 4, 0, 0)
        ans |= line(2, 4, 2, 0)
        ans |= line(0, 0, 1, 2)
        ans |= line(1, 2, 2, 0)
    elif letter == "X":
        ans |= line(0, 0, 2, 4)
        ans |= line(0, 4, 2, 0)
    elif letter == "Y":
        ans |= line(0, 4, 1, 2)
        ans |= line(1, 2, 2, 4)
        ans |= line(1, 0, 1, 2)
    elif letter == "Z":
        ans |= line(0, 0, 2, 0)
        ans |= line(0, 4, 2, 4)
        ans |= line(0, 0, 2, 4)
    elif letter == "a":
        ans |= circle(1, 1, 1)
        ans |= line(2, 0, 2, 2)
    elif letter == "b":
        ans |= circle(1, 1, 1)
        ans |= line(0, 0, 0, 4)
    elif letter == "c":
        ans |= circle(1, 1, 1)
        ans -= box(1, 0, 2, 2)
        ans |= line(1, 2, 2, 2)
        ans |= line(1, 0, 2, 0)
    elif letter == "d":
        ans |= circle(1, 1, 1)
        ans |= line(2, 0, 2, 4)
    elif letter == "e":
        ans |= circle(1, 1, 1)
        ans -= box(1, 0, 2, 1)
        ans |= line(0, 1, 2, 1)
        ans |= line(1, 0, 2, 0)
    elif letter == "f":
        ans |= circle(0.5, -0.5, 0.5)
        ans |= circle(1.5, 2.5, 0.5)
        ans -= box(0, -0.5, 2, 2.5)
        ans |= line(1, -0.5, 1, 2.5)
        ans |= line(0, 1, 2, 1)
    elif letter == "g":
        ans |= circle(1, -1, 1)
        ans -= box(0, -1, 2, 0)
        ans |= circle(1, 1, 1)
        ans |= line(2, -1, 2, 2)
    elif letter == "h":
        ans |= circle(1, 1, 1)
        ans -= box(0, 0, 2, 1)
        ans |= line(0, 0, 0, 4)
        ans |= line(2, 0, 2, 1)
    elif letter == "i":
        ans |= line(0, 0, 0, 2)
        ans |= point(0, 3)
        width = 1
    elif letter == "j":
        ans |= circle(0.5, 0.5, 0.5)
        ans -= box(0, 0.5, 1, 1)
        ans |= line(1, 0.5, 1, 2)
        ans |= point(1, 3)
        width = 2
    elif letter == "k":
        ans |= line(0, 0, 0, 4)
        ans |= line(0, 1, 1.5, 0)
        ans |= line(0, 1, 1.5, 2)
        width = 2.5
    elif letter == "l":
        ans |= circle(1, 1, 1)
        ans &= box(0, 0, 1, 1)
        ans |= line(0, 1, 0, 3)
        width = 2
    elif letter == "m":
        ans |= circle(0.5, 1.5, 0.5)
        ans |= circle(1.5, 1.5, 0.5)
        ans -= box(0, 0, 2, 1.5)
        ans |= line(0, 0, 0, 2)
        ans |= line(1, 0, 1, 1.5)
        ans |= line(2, 0, 2, 1.5)
    elif letter == "n":
        ans |= circle(1, 1, 1)
        ans -= box(0, 0, 2, 1)
        ans |= line(0, 0, 0, 2)
        ans |= line(2, 0, 2, 1)
    elif letter == "o":
        ans |= circle(1, 1, 1)
    elif letter == "p":
        ans |= circle(1, 1, 1)
        ans |= line(0, 2, 0, -2)
    elif letter == "q":
        ans |= circle(1, 1, 1)
        ans |= line(2, 2, 2, -2)
        ans |= line(2, -2, 2.5, -1.5)
    elif letter == "r":
        ans |= circle(1, 1, 1)
        ans &= box(0, 1, 1, 2)
        ans |= line(0, 0, 0, 2)
        width = 2
    elif letter == "s":
        ans |= circle(0.5, 0.5, 0.5)
        ans |= circle(0.5, 1.5, 0.5)
        ans -= box(0, 0.25, 0.5, 1)
        ans -= box(0.5, 1, 1, 1.75)
        width = 2
    elif letter == "t":
        ans |= line(1, 1, 1, 3)
        ans |= line(0, 2, 2, 2)
        ans |= bezier([(2, 0), (1, 0), (1, 1)])
    elif letter == "u":
        ans |= circle(1, 1, 1)
        ans &= box(0, 0, 2, 1)
        ans |= line(0, 1, 0, 2)
        ans |= line(2, 0, 2, 2)
    elif letter == "v":
        ans |= line(1, 0, 0, 2)
        ans |= line(1, 0, 2, 2)
    elif letter == "w":
        ans |= circle(0.5, 0.5, 0.5)
        ans |= circle(1.5, 0.5, 0.5)
        ans &= box(0, 0, 2, 0.5)
        ans |= line(0, 0.5, 0, 2)
        ans |= line(1, 0.5, 1, 2)
        ans |= line(2, 0.5, 2, 2)
    elif letter == "x":
        ans |= line(0, 0, 2, 2)
        ans |= line(0, 2, 2, 0)
    elif letter == "y":
        ans |= circle(1, 1, 1)
        ans |= circle(1, -1, 1)
        ans -= box(0, 1, 2, 2)
        ans -= box(0, -1, 2, 0)
        ans |= line(2, -1, 2, 2)
        ans |= line(0, 1, 0, 2)
    elif letter == "z":
        ans |= line(0, 2, 2, 2)
        ans |= line(2, 2, 0, 0)
        ans |= line(0, 0, 2, 0)
        ans |= line(0.25, 1, 1.75, 1)

    #DIGITS
        
    elif letter == "0":
        ans |= circle(1, 1, 1)
        ans |= circle(1, 3, 1)
        ans -= box(0, 1, 2, 3)
        ans |= line(0, 1, 0, 3)
        ans |= line(2, 1, 2, 3)
        #ans |= line(0, 1.5, 2, 2.5) dash
    elif letter == "1":
        ans |= line(0, 0, 2, 0)
        ans |= line(1, 0, 1, 4)
        ans |= line(1, 4, 0, 3)
    elif letter == "2":
        ans |= circle(1, 3, 1)
        ans -= box(0, 1, 2, 12 / 5)
        ans -= box(0, 1, 1, 3)
        ans |= line(0, 0, 2, 0)
        ans |= line(0, 0, 9/5, 12/5)
    elif letter == "3":
        ans |= circle(1, 1, 1)
        ans |= circle(1, 3, 1)
        ans -= box(0, 0, 1 - 1 / math.sqrt(2), 4)
        ans -= box(0, 1, 1, 3)
        ans |= line(0.5, 2, 1, 2)
    elif letter == "4":
        ans |= line(1.5, 0, 1.5, 4)
        ans |= line(0, 1.5, 2, 1.5)
        ans |= line(0, 1.5, 1.5, 4)
    elif letter == "5":
        ans |= circle(1, 1, 1)
        ans -= box(0, 0, 1, 2)
        ans |= line(0, 0, 1, 0)
        ans |= line(0, 2, 1, 2)
        ans |= line(0, 4, 2, 4)
        ans |= line(0, 2, 0, 4)
    elif letter == "6":
        ans |= circle(1, 3, 1)
        ans &= box(0, 3, 1 + 1 / math.sqrt(2), 4)
        ans |= circle(1, 1, 1)
        ans |= line(0, 1, 0, 3)
    elif letter == "7":
        ans |= line(0, 0, 2, 4)
        ans |= line(0, 4, 2, 4)
    elif letter == "8":
        ans |= circle(1, 3, 1)
        ans |= circle(1, 1, 1)
    elif letter == "9":
        ans |= circle(1, 3, 1)
        ans |= line(2, 0, 2, 3)

    #ASCII SYMBOLS
        
    elif letter == "!":
        ans |= line(0, 4, 0, 1)
        ans |= point(0, 0)
        width = 1
    elif letter == "\"":
        ans |= line(0, 3, 0, 4)
        ans |= line(1, 3, 1, 4)
        width = 2
    elif letter == "#":
        ans |= line(0, 1, 2, 1)
        ans |= line(0, 3, 2, 3)
        ans |= line(0.5, 0, 0.5, 4)
        ans |= line(1.5, 0, 1.5, 4)
    elif letter == "$":
        ans |= circle(1, 1, 1)
        ans |= circle(1, 3, 1)
        ans -= box(0, 1, 1, 2)
        ans -= box(1, 2, 2, 3)
        ans |= line(1, -0.5, 1, 4.4)
    elif letter == "%":
        ans |= line(0, 0, 2, 4)
        ans |= circle(1.6, 0.5, 0.4)
        ans |= circle(0.4, 3.5, 0.4)
    elif letter == "&":
        ans |= circle(1, 1, 1)
        ans |= circle(1, 3, 1)
        ans -= box(0, 1, 2, 3)
        ans -= box(1, 0, 2, 1)
        ans |= line(0, 1, 2, 3)
        ans |= line(0, 3, 2, 0)
        ans |= line(1, 0, 2, 1)
    elif letter == "'":
        ans |= line(1.5, 3, 1.5, 4)
        width = 2
    elif letter == "(":
        ans |= circle(4, 2, math.sqrt(13))
        ans &= box(0, 0, 2, 4)
        width = 2
    elif letter == ")":
        ans |= circle(-3, 2, math.sqrt(13))
        ans &= box(0, 0, 2, 4)
        width = 2
    elif letter == "*":
        for i in [0, 1, 2, 3, 4]:
            a = 2 * math.pi * i / 5
            ans |= line(1, 2, 1 + math.sin(a), 2 + math.cos(a))
    elif letter == "+":
        ans |= line(0, 1, 2, 1)
        ans |= line(1, 0, 1, 2)
    elif letter == ",":
        ans |= line(0, 0, -0.25, -0.5)
        width = 1
    elif letter == "-":
        ans |= line(0, 1, 2, 1)
    elif letter == ".":
        ans |= point(0, 0)
        width = 1
    elif letter == "/":
        ans |= line(0, 0, 2, 4)
    elif letter == ":":
        ans |= point(0, 0)
        ans |= point(0, 2)
        width = 1
    elif letter == ";":
        ans |= line(0, 0, -0.25, -0.5)
        ans |= point(0, 2)
        width = 1
    elif letter == "<":
        ans |= line(0, 1, 2, 2)
        ans |= line(0, 1, 2, 0)
    elif letter == "=":
        ans |= line(0, 2, 2, 2)
        ans |= line(0, 0, 2, 0)
    elif letter == "!=": #ne, not equal
        ans |= line(0, 2, 2, 2)
        ans |= line(0, 0, 2, 0)
        ans |= line(0.5, -1, 1.5, 3)
    elif letter == ">":
        ans |= line(0, 2, 2, 1)
        ans |= line(0, 0, 2, 1)
    elif letter == "?":
        ans |= circle(1, 3, 1)
        ans -= box(0, 0, 1 + 1 / math.sqrt(2), 3)
        ans |= circle(2 + math.sqrt(2), 3 - 1 - math.sqrt(2), 1 + math.sqrt(2)) & box(0, 1, 1 + 1 / math.sqrt(2), 4)
        ans |= point(1, 0)
    elif letter == "[":
        ans |= line(0, 0, 0, 4)
        ans |= line(0, 0, 1, 0)
        ans |= line(0, 4, 1, 4)
        width = 2
    elif letter == "\\":
        ans |= line(0, 4, 2, 0)
    elif letter == "]":
        ans |= line(1, 0, 1, 4)
        ans |= line(0, 0, 1, 0)
        ans |= line(0, 4, 1, 4)
        width = 2
    elif letter == "^":
        ans |= line(0, 2, 1, 3)
        ans |= line(1, 3, 2, 2)
    elif letter == "_":
        ans |= line(0, 0, 2, 0)
    elif letter == "`":
        ans |= line(0.25, 3.5, -0.25, 4)
        width = 1
    elif letter == "{":
        ans |= circle(0, 1.5, 0.5)
        ans |= circle(1, 0.5, 0.5)
        ans |= circle(0, 2.5, 0.5)
        ans |= circle(1, 3.5, 0.5)
        ans &= box(0, 1.5, 1, 2.5) | box(0, 3.5, 1, 4) | box(0, 0, 1, 0.5)
        ans |= line(0.5, 2.5, 0.5, 3.5)
        ans |= line(0.5, 0.5, 0.5, 1.5)
        width = 2
    elif letter == "|":
        ans |= line(0, 0, 0, 4)
        width = 1
    elif letter == "}":
        ans |= circle(1, 1.5, 0.5)
        ans |= circle(0, 0.5, 0.5)
        ans |= circle(1, 2.5, 0.5)
        ans |= circle(0, 3.5, 0.5)
        ans &= box(1, 1.5, 0, 2.5) | box(1, 3.5, 0, 4) | box(1, 0, 0, 0.5)
        ans |= line(0.5, 2.5, 0.5, 3.5)
        ans |= line(0.5, 0.5, 0.5, 1.5)
        width = 2
    elif letter == "~":
        ans |= circle(0.5, 1.5, 1 / math.sqrt(2)) & box(0, 2, 1, 3)
        ans |= circle(1.5, 2.5, 1 / math.sqrt(2)) & box(1, 1, 2, 2)
        ans &= box(0, 2, 1, 3) | box(1, 1, 2, 2)

    #GREEK
        
    elif letter == "alpha":
        ans |= circle(1 - 1/8, 1, 1 - 1/8)
        ans |= line(1.75, 1, 2, 2)
        ans |= line(1.75, 1, 2, 0)
    elif letter == "beta":
        ans |= circle(1, 1, 1)
        ans |= circle(1, 3, 1)
        ans -= box(0, 1, 1, 3)
        ans |= line(0, -1, 0, 3)
        ans |= line(0, 2, 1, 2)
    elif letter == "gamma":
        ans |= circle(0.8, 1, 0.8)
        ans -= box(0, 0, 2, 0.8)
        ans -= box(0, 0, 1, 1.5)
        ans |= line(1, -1, 2, 2)
    elif letter == "Gamma":
        ans |= line(0.5, 0, 0.5, 4)
        ans |= line(0, 4, 2, 4)
        ans |= line(2, 3.5, 2, 4)
        ans |= line(0, 0, 1, 0)
    elif letter == "delta":
        ans |= circle(1, 1, 1)
        ans |= smoothline([(1 + 1 / math.sqrt(2) , 1 + 1 / math.sqrt(2)), (0, 4), (2, 4)])
    elif letter == "Delta":
        ans |= line(0, 0, 3, 0)
        ans |= line(0, 0, 1.5, 4)
        ans |= line(3, 0, 1.5, 4)
        width = 4
    elif letter == "epsilon":
        ans |= circle(1, 1, 1)
        ans -= box(1, 0, 2, 2)
        ans |= line(0, 1, 1, 1)
        width = 2
    elif letter == "varepsilon":
        ans |= circle(0.5, 1.5, 0.5)
        ans |= circle(0.5, 0.5, 0.5)
        ans -= box(0.5, 0, 1, 2)
        ans |= line(0.5, 0, 1, 0)
        ans |= line(0.5, 1, 1, 1)
        ans |= line(0.5, 2, 1, 2)
        width = 2
    elif letter == "zeta":
        ans |= circle(2, 2, 2)
        ans -= box(2, 0, 4, 4)
        ans |= circle(2, -0.5, 0.5) - box(0, -1, 2, 0)
        ans |= bezier([(2, 4), (3, 4), (2, 2.5), (0, 4)])
        width = 3.5
    elif letter == "eta":
        ans |= circle(1, 1, 1)
        ans -= box(0, 0, 2, 1)
        ans |= line(0, 0, 0, 2.5)
        ans |= line(2, -1.5, 2, 1)
    elif letter == "theta":
        ans |= smoothline([(1, 0), (0, 0), (0, 4), (1, 4)])
        ans |= smoothline([(1, 0), (2, 0), (2, 4), (1, 4)])
        ans |= line(0, 2, 2, 2)
    elif letter == "vartheta":
        ans |= smoothline([(0, 1.5), (0, 0), (2, 0), (2, 4), (0, 4), (0, 2), (2.25, 2)])
    elif letter == "Theta":
        ans |= smoothline([(1.5, 0), (0, 0), (0, 4), (1.5, 4)])
        ans |= smoothline([(1.5, 0), (3, 0), (3, 4), (1.5, 4)])
        ans |= line(1, 2, 2, 2)
        width = 4
    elif letter == "iota":
        ans |= circle(0.5, 0.5, 0.5)
        ans -= box(0, 0.5, 1, 1)
        ans |= line(0, 0.5, 0, 2.5)
        width = 2
    elif letter == "kappa":
        ans |= circle(1, 0, 1)
        ans -= box(0, -1, 2, 0)
        ans |= line(0, 0, 0, 3)
        ans |= bezier([(0, 0), (0, 2), (2, 2.5)])
    elif letter == "lambda":
        ans |= line(0, 4, 2, 0)
        ans |= line(0, 0, 1, 2)
    elif letter == "Lambda":
        ans |= line(0, 0, 1, 4)
        ans |= line(2, 0, 1, 4)
    elif letter == "mu":
        ans |= circle(1, 1, 1)
        ans &= box(0, 0, 2, 1)
        ans |= line(0, -1, 0, 2)
        ans |= line(2, 0, 2, 2)
    elif letter == "nu":
        ans |= line(0, 0, 0, 2)
        ans |= bezier([(0, 0), (2, 0), (2, 2)])
    elif letter == "xi":
        ans |= smoothline([(2, 2), (0, 2), (0, 0), (1.5, 0)])
        ans |= smoothline([(2, 2), (0, 2), (0, 4), (2, 4)])
        ans |= line(1, 2, 2, 2)
        ans |= bezier([(2, 4), (3, 4), (2, 3), (1, 4.5)])
        ans |= circle(1.5, -0.5, 0.5) - box(1, -1, 1.5, 0)
    elif letter == "pi":
        ans |= smoothline([(0, 1.8), (0, 2), (0.5, 2), (2, 2)])
        ans |= smoothline([(0.25, 0), (0.75, 0.5), (0.75, 2)])
        ans |= line(1.5, 0, 1.5, 2)
    elif letter == "Pi":
        ans |= line(0, 4, 3, 4)
        ans |= line(0.5, 0, 0.5, 4)
        ans |= line(2.5, 0, 2.5, 4)
        width = 4
    elif letter == "rho":
        ans |= circle(1, 1, 1)
        ans |= smoothline([(0, 1), (0, -1.5), (-0.5, -2)])
    elif letter == "varrho":
        raise NotImplementedError
    elif letter == "phi":
        ans |= circle(1, 1, 1)
        ans |= line(1, -1, 1, 3)
    elif letter == "varphi":
        ans |= circle(1, 1, 1) & box(0, 0, 2, 1)
        ans |= circle(1.5, 1.5, 0.5) & box(1, 1.5, 2, 2)
        ans |= line(0, 1, 0, 2)
        ans |= line(1, 1.5, 1, -1)
        ans |= line(2, 1, 2, 1.5)
    elif letter == "sigma":
        ans |= circle(1, 1, 1)
        ans |= line(1, 2, 2.25, 2)
    elif letter == "Sigma":
        ans |= line(0, 4, 3, 4)
        ans |= line(0, 4, 2, 2)
        ans |= line(0, 0, 2, 2)
        ans |= line(0, 0, 3, 0)
        width = 4
    elif letter == "tau":
        ans |= line(1, 0, 1, 1.9)
        ans |= smoothline([(0, 1.8), (0.5, 2), (2, 2)])
    elif letter == "upsilon":
        raise NotImplementedError
    elif letter == "Upsilon":
        raise NotImplementedError
    elif letter == "chi":
        ans |= line(0, -1, 2, 2)
        ans |= smoothline([(0, 2), (1, 2), (1, -1), (2, -1)])
    elif letter == "psi":
        ans |= smoothline([(0, 2), (0, 0), (2, 0), (2, 2)])
        ans |= line(1, -1, 1, 2)
    elif letter == "Psi":
        ans |= smoothline([(0, 3), (0, 1), (3, 1), (3, 3)])
        ans |= line(1.5, 0, 1.5, 4)
        ans |= line(0.5, 0, 2.5, 0)
        ans |= line(0.5, 4, 2.5, 4)
        width = 4
    elif letter == "omega":
        ans |= smoothline([(0, 2), (0, 0), (1, 0), (1, 2)])
        ans |= smoothline([(1, 2), (1, 0), (2, 0), (2, 2)])
    elif letter == "Omega":
        ans |= smoothline([(1, 0), (-1, 4), (4, 4), (2, 0)])
        ans |= line(0, 0, 1, 0)
        ans |= line(2, 0, 3, 0)

    #SYMBOLS (non ascii)
        
    elif letter == "times":
        ans |= line(0, 0, 2, 2)
        ans |= line(2, 0, 0, 2)
    elif letter == "integral":
        ans |= smoothline([(0.5, 0), (2, 0), (0, 4), (1.5, 4)])
        
    else:
        ans |= shapely.geometry.LinearRing([[0, 0], [2, 0], [2, 4], [0, 4]])
    return ans.buffer(0.25), width







class Frame():
    def __call__(self, fmt):
        raise NotImplementedError()



##class WidthFrame(Frame):
##    def __init__(self, center, width):
##        self.center = center
##        self.width = width
##    def __call__(self, fmt):
##        scale = self.width / fmt.width
##        fmt.set_scale(scale)
##        fmt.set_center(self.center)
##        fmt.set_sub_locations()
    

class HeightFrame(Frame):
    def __init__(self, center, height):
        self.center = center
        self.height = height
    def __call__(self, fmt):
        scale = self.height / fmt.height
        fmt.set_scale(scale)
        fmt.set_center(self.center)
        fmt.set_sub_locations()
      
##class ScaleFrame(Frame):
##    def __init__(self, center, scale):
##        self.center = center
##        self.scale = scale
##    def __call__(self, fmt):
##        fmt.set_scale(self.scale)
##        fmt.set_center(self.center)
##        fmt.set_sub_locations()




class Format():
    def __init__(self, subs, geom, center, bounds, colour):
        assert len(colour) == 4
        for sub in subs:
            assert isinstance(sub, Format)
        self.subs = subs
        self.geom = geom
        self.center = list(center)
        self.bounds = list(bounds)
        self.colour = colour
        assert len(self.center) == 2
        assert len(self.bounds) == 4

    @property
    def box_geom(self):
        a, b, c, d = self.min_x, self.min_y, self.max_x, self.max_y
        return (line(a, b, a, d) | line(a, b, c, b) | line(c, b, c, d) | line(a, d, c, d)).buffer(0.01) ^ point(*self.center).buffer(0.01)

##    @property
##    def geom(self):
##        return self._geom

    def gen_shapes(self):
        for sub_fmt in self.sub_fmts():
            yield sub_fmt.geom, sub_fmt.colour

    def sub_fmts(self):
        yield self
        for sub in self.subs:
            yield from sub.sub_fmts()

    @property
    def width(self):
        return self.max_x - self.min_x
    @property
    def height(self):
        return self.max_y - self.min_y

    @property
    def min_x(self):
        return self.bounds[0]
    @property
    def min_y(self):
        return self.bounds[1]
    @property
    def max_x(self):
        return self.bounds[2]
    @property
    def max_y(self):
        return self.bounds[3]

    @property
    def center_x_neg(self):
        return self.center[0] - self.min_x
    @property
    def center_x_pos(self):
        return self.max_x - self.center[0]
    @property
    def center_y_neg(self):
        return self.center[1] - self.min_y
    @property
    def center_y_pos(self):
        return self.max_y - self.center[1]

    def set_center(self, center):
        return self.move_by([center[i] - self.center[i] for i in [0, 1]])

    def move_by(self, off):
        off_x, off_y = off
        geom = shapely.affinity.translate(self.geom, off_x, off_y)
        subs = [sub.move_by(off) for sub in self.subs]
        center = [self.center[0] + off_x, self.center[1] + off_y]
        bounds = [self.bounds[0] + off_x, self.bounds[1] + off_y, self.bounds[2] + off_x, self.bounds[3] + off_y]
        return Format(subs, geom, center, bounds, self.colour)

    def set_scale(self, scale):
        assert scale >= 0
        subs = [sub.set_scale(scale) for sub in self.subs]
        geom = shapely.affinity.scale(self.geom, scale, scale, origin = (0, 0))
        center = [self.center[0] * scale, self.center[1] * scale]
        bounds = [self.bounds[0] * scale, self.bounds[1] * scale, self.bounds[2] * scale, self.bounds[3] * scale]
        return Format(subs, geom, center, bounds, self.colour)

    def set_sub_locations(self):
        raise NotImplementedError()



def height_frame(fmt, height, colour = None):
    scale = height / fmt.height
    fmt = fmt.set_scale(scale)
    if not colour is None:
        fmt = Format([fmt], shapely.geometry.Polygon([(fmt.min_x, fmt.min_y), (fmt.max_x, fmt.min_y), (fmt.max_x, fmt.max_y), (fmt.min_x, fmt.max_y)]), fmt.center, fmt.bounds, colour)
    return fmt



def position_frame(fmt, pos):
    fmt = fmt.set_center(pos)
    return fmt



def letter(string, colour):
    shape, width = letter_polygons(string)
    return Format([], shape, [width / 2 - 0.5, 2], [-0.5, -0.5, width - 0.5, 4.5], colour)





def row(subs):
    width = sum(sub.width for sub in subs)
    min_y = -max(sub.center_y_neg for sub in subs)
    max_y = max(sub.center_y_pos for sub in subs)

    def gen_subs():
        x_acc = 0
        for sub in subs:
            yield sub.set_center([x_acc + sub.center_x_neg, 0])
            x_acc += sub.width

    return Format(list(gen_subs()), empty(), [width / 2, 0], [0, min_y, width, max_y], (0, 0, 0, 0))




def string(letters, colour):
    return row([letter(let, colour) for let in letters])
        




##class Grid(Format):
##    def __init__(self, rows, pad):
##        self.rows = rows
##        
##        subs = []
##        self.r = len(self.rows)
##        self.c = len(self.rows[0])
##        for row in self.rows:
##            assert len(row) == self.c
##            for sub in row:
##                subs.append(sub)
##                sub.set_scale(1)
##
##        self.center_x_negs = [pad / 2 + max(self.rows[r][c].center_x_neg for r in range(self.r)) for c in range(self.c)]
##        self.center_x_poss = [pad / 2 + max(self.rows[r][c].center_x_pos for r in range(self.r)) for c in range(self.c)]
##        self.center_y_negs = [pad / 2 + max(self.rows[r][c].center_y_neg for c in range(self.c)) for r in range(self.r)]
##        self.center_y_poss = [pad / 2 + max(self.rows[r][c].center_y_pos for c in range(self.c)) for r in range(self.r)]
##
##        self.col_widths = [self.center_x_negs[c] + self.center_x_poss[c] for c in range(self.c)]
##        self.row_heights = [self.center_y_negs[r] + self.center_y_poss[r] for r in range(self.r)]
##
##        height = sum(self.row_heights)
##        width = sum(self.col_widths)
##        
##        min_x = -width / 2
##        min_y = -height / 2
##        max_x = width / 2
##        max_y = height / 2
##        
##        super().__init__(subs, empty(), 1, [0.5 * (min_x + max_x), 0.5 * (min_y + max_y)], [min_x, min_y, max_x, max_y])
##        
##    def set_sub_locations(self):        
##        row_heights = [h * self.scale for h in self.row_heights]
##        col_widths = [w * self.scale for w in self.col_widths]
##
##        print(row_heights)
##        print(col_widths)
##
##        for r in range(self.r):
##            for c in range(self.c):
##                sub = self.rows[r][c]
##                sub.set_scale(self.scale)
##                sub.set_center([self.min_x + sum(col_widths[:c]) + self.center_x_negs[c],
##                                self.min_y + sum(row_heights[:r]) + self.center_y_negs[r]])
##                sub.set_sub_locations()
##
##
##
##
##
##
##class Power(Format):
##    def __init__(self, base, power):
##        self.base = base
##        self.power = power
##        self.base.set_scale(1)
##        self.power.set_scale(0.7)
##        
##        super().__init__([self.base, self.power], empty(), 1, [0, 0],
##                         [-self.base.center_x_neg,
##                          -self.base.center_y_neg,
##                          self.base.center_x_pos + self.power.width,
##                          max(self.power.height, self.base.center_y_pos)])
##        
##    def set_sub_locations(self):
##        self.base.set_scale(self.scale)
##        self.power.set_scale(self.scale * 0.7)
##        
##        self.base.set_center(self.center)
##        self.power.set_center([self.center[0] + self.base.center_x_pos + self.power.center_x_neg,
##                               self.center[1] + self.power.center_y_neg])
##
##        self.base.set_sub_locations()
##        self.power.set_sub_locations()


            










































        
