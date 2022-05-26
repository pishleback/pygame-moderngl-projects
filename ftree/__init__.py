from ftree import treedata
from ftree import treeview
from ftree import myxml

def run():
    import os
    import itertools

##    for n in itertools.count():
##        path = "main" + str(n)
##        path = os.path.join("famillytree", "saves", path + ".txt")
##        if os.path.isfile(path):
##            continue
##        break
##    path = "main" + str(n - 1)
    
    path = os.path.join("ftree", "saves", "main0" + ".txt")
    print(path)

    with open(path, "r") as f:
        tree = treedata.Tree.Load(myxml.Tag.Load(f.read()))
    
##    tree = treedata.Tree()
##    ps = [treedata.Person("p1", []),
##          treedata.Person("p2", []),
##          treedata.Person("p3", []),
##          treedata.Person("p4", []),
##          treedata.Person("p5", [])]

    treeview.run(tree)

##    print(tree)
##    G = tree.digraph()
##
##    print(G)


    

##    tree2 = treedata.load_ged(os.path.join("famillytree", "myged.txt"))
##
##
    ##f = open(os.path.join("famillytree", "saves", "origsarahoof.txt"), "r")
    ##tree3 = famillytree.treedata.Tree.Load(myxml.Tag.Load(f.read()))
    ##f.close()


##    treeview.run([tree1, tree2])
