from ftree import treedata
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

    f = open(path, "r")
    tree = treedata.Tree.Load(myxml.Tag.Load(f.read()))
    f.close()

    print(tree)
    G = tree.digraph()

    print(G)


    

##    tree2 = treedata.load_ged(os.path.join("famillytree", "myged.txt"))
##
##
    ##f = open(os.path.join("famillytree", "saves", "origsarahoof.txt"), "r")
    ##tree3 = famillytree.treedata.Tree.Load(myxml.Tag.Load(f.read()))
    ##f.close()


##    treeview.run([tree1, tree2])
