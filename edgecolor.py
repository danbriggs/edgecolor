"""The edge coloring isomorphism problem
When are two 3-colorings of the edges of K_n isomorphic?
Only restriction is #red = #blue or #red = #blue + 1."""

#Make sure to create a folder 8/ before running for n=8 e.g. if outtofile is True.

from collections import Counter, defaultdict
from itertools import combinations
import operator as op
from functools import reduce
import os.path
import pickle
import copy

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def deg(i, edges):
    """Returns how many pairs in edges are adjacent to vertex i."""
    return sum([i in edge for edge in edges])

def freeze_dict(obj):
    dict_items = list(obj.items())
    return tuple([(k, v) for k, v in dict_items])

def unfreeze_dict(obj):
    return dict((k, v) for k, v in obj)


def get_spectrum(edgeseq):
    global n
    reds = edgeseq[::2]
    blues = edgeseq[1::2]
    vpairs = [] #For each vertex, how many it's joined to by red and blue edges
    for i in range(n):
        vpairs.append((deg(i,reds),deg(i,blues)))
    spectrum = Counter(vpairs)
    the_tuple = freeze_dict(spectrum)
    return tuple(reversed(sorted(the_tuple, key=lambda x: x[0][0]+x[0][1]/n)))
    #Dictionary order, from most reds on down

def restricted_spectrum(edgeseq,i):
    """Returns a pair of dictionaries.
    Each key of the 1st is a specpair of a vertex connecting to i by a red edge
    with value a sorted list of such vertices with said specpair.
    The 2nd dictionary is the same for blue."""
    reds = edgeseq[::2]
    blues = edgeseq[1::2]
    red_dict = {}
    blue_dict = {}
    for r in reds:
        if i==r[0]:
            j=r[1]
        elif i==r[1]:
            j=r[0]
        else:
            continue
        specpair = (deg(j,reds),deg(j,blues))
        if specpair in red_dict:
            red_dict[specpair].append(j)
        else:
            red_dict[specpair] = [j]
    for b in blues:
        if i==b[0]:
            j=b[1]
        elif i==b[1]:
            j=b[0]
        else:
            continue
        specpair = (deg(j,reds),deg(j,blues))
        if specpair in blue_dict:
            blue_dict[specpair].append(j)
        else:
            blue_dict[specpair] = [j]
    for k in red_dict:
        red_dict[k] = sorted(red_dict[k])
    for k in blue_dict:
        blue_dict[k] = sorted(blue_dict[k])
    return red_dict, blue_dict


class TaggedGraph():
    def __init__(self, edgeseq):
        global n
        self.n = n
        self.reds = edgeseq[::2]
        self.blues = edgeseq[1::2]
        self.spectra = []
        for i in range(n):
            self.spectra.append(restricted_spectrum(edgeseq,i))
    def equivalents(self, dictpair):
        """Returns a list of all vertices in self
        that have a pair of dictionaries equivalent to dictpair.
        This means same keys and that the values have the same length."""
        reddict = dictpair[0]
        bluedict = dictpair[1]
        retlist = []
        for i in range(len(self.spectra)):
            spectrum = self.spectra[i]
            myreddict = spectrum[0]
            mybluedict = spectrum[1]
            if len(reddict) != len(myreddict):
                continue
            wasbad = False
            for k in reddict:
                if k not in myreddict:
                    wasbad = True
                    break
                if len(myreddict[k]) != len(reddict[k]):
                    wasbad = True
                    break
            if wasbad:
                continue
            if len(bluedict) != len(mybluedict):
                continue
            for k in bluedict:
                if k not in mybluedict:
                    wasbad = True
                    break
                if len(mybluedict[k]) != len(bluedict[k]):
                    wasbad = True
                    break
            if not wasbad:
                retlist.append(i)
        return retlist
            
        
debugtg1 = None
debugtg2 = None
candidate_targets = []

def turnsinto(targetlist, graph1, graph2):
    #print("Checking to see whether",targetlist,"turns",graph1,"into",graph2)
    if len(graph1) != len(graph2):
        return False
    graph3 = []
    for e in graph1:
        if max(e[0],e[1])>=len(targetlist):
            print("Weird targetlist:",targetlist)
            print("with e=",e)
            print("graph1=",graph1)
            print("graph2=",graph2)
        graph3.append(tuple(sorted((targetlist[e[0]],targetlist[e[1]]))))
    #print("Now graph3 is",graph3)
    for e in graph3[::2]:
        if e not in graph2[::2]:
            return False
    for e in graph3[1::2]:
        if e not in graph2[1::2]:
            return False
    return True

def atleastonepermwith(candidate_targets,fixed_values,graph1,graph2):
    """Returns whether there is at least one permutation turning graph1 into graph2
    using a value from each list in candidate_targets with the first few
    vertices of graph1 being sent to the values in fixed_values, respectively."""
    global n
    #if fixed_values == []:
        #print("checking atleastonepermwith",candidate_targets,graph1,graph2)
        #input("Break point!")
    numfixed = len(fixed_values)
    if numfixed == len(candidate_targets):
        return turnsinto(fixed_values,graph1,graph2)
    for v in candidate_targets[numfixed]:
        if v not in fixed_values:
            extended_fixed_values = fixed_values + [v]
            if atleastonepermwith(candidate_targets,extended_fixed_values,graph1,graph2):
                return True
    #This is like a depth-first search, which is good when any ol' perm'll do,
    #as seems to often be the case with small graphs.
    return False
            
    

def is_isomorphic(graph1, graph2):
    """Determines whether graph1 and graph2 are isomorphic.
    We begin by tagging each vertex with its extended spectrum.
    It's a good idea to only pass graphs with the same spectrum in."""
    global debugtg1, debugtg2, candidate_targets
    global n
    
    tg1 = TaggedGraph(graph1)
    tg2 = TaggedGraph(graph2)
    debugtg1 = tg1
    debugtg2 = tg2
    
    #Now we see if there is a relabeling of tg1 that turns it into tg2.
    #We begin by seeing which vertices in tg2 have an equivalent spectrum pair
    #to vertex 0 in tg1.
    candidate_targets = []
    for i in range(n):
        tg1s = tg1.spectra[i]
        if len(tg1s[0]) == 0 and len(tg1s[1]) == 0:
            break #We're past connected vertices, by construction
        candidate_targets.append(tg2.equivalents(tg1s))
        #print("Vertex",i,"could perhaps be sent to vertices",candidate_targets[i])
        #input("Smaller break point in is_isomorphic")
    if any([len(x) == 0 for x in candidate_targets]):
        return False #There's a vertex with nowhere to send it
    if all([len(x) == 1 for x in candidate_targets]):
        targetlist = [x[0] for x in candidate_targets]
        if len(targetlist) != len(set(targetlist)): #If there are repeats
            return False #Multiple vertices would have to go to the same vertex
        return turnsinto(targetlist,graph1,graph2)
    else:
        truey = atleastonepermwith(candidate_targets,[],graph1,graph2)
        #if not truey:
            #print("Truth value:",truey)
            #input("Break point!")
        return truey

def do_sum_stuff(canonicals,k,from_file_note=""):
    global partialsummy, summy
    partialsummy = sum([len(canonicals[k][s]) for s in canonicals[k]])
    print("# with ",k,":",partialsummy,from_file_note)
    summy += partialsummy

def lacuna(edgeseq, move):
    """Determines whether move would needlessly jump over a vertex.
    Here we assume that edgeseq left no lacunae."""
    global n
    max_vertex = -1
    for e in edgeseq:
        if e[0] > max_vertex:
            max_vertex = e[0]
        if e[1] > max_vertex:
            max_vertex = e[1]
    #move[1] must be <= max_vertex + 1
    #or else move must == (max_vertex+1, max_vertex+2)
    #for there to be no lacuna.
    if move[1] > max_vertex + 1:
        return move != (max_vertex+1,max_vertex+2)
    return False
    

class Gamepos:
    def __init__(self, edgeseq):
        self.edgeseq = edgeseq
        self.rededges = edgeseq[0::2]
        self.blueedges = edgeseq[1::2]
        self.player = len(edgeseq)%2
    def nextmoves(self):
        return alledges.difference(set(self.edgeseq))
    def completesatriangle(self, move):
        if self.player == 0:
            theedgeseq = self.rededges
        elif self.player == 1:
            theedgeseq = self.blueedges
        for i in range(move[0]):
            if (i,move[0]) in theedgeseq and (i,move[1]) in theedgeseq:
                return True
        for i in range(move[0]+1,move[1]):
            if (move[0],i) in theedgeseq and (i,move[1]) in theedgeseq:
                return True
        for i in range(move[1]+1,n):
            if (move[0],i) in theedgeseq and (move[1],i) in theedgeseq:
                return True
        return False
    def winner(self, cheatdict = None):
        ne = self.nextmoves()
        if len(ne) == 0:
            return .5
        atleastonedraw = False
        for move in ne:
            if not self.completesatriangle(move):
                extendedseq = self.edgeseq+[move]
                if cheatdict is not None:
                    if lacuna(self.edgeseq,move): #We may disregard moves that jump over a vertex
                        continue #and we must for the way we implemented turnsinto
                    #WMA cheatdict is complete for one more edge.
                    #First, we get the spectrum of extendedseq for the key.
                    key = get_spectrum(extendedseq)
                    #print("key in winner():",key)
                    #print("extendedseq:",extendedseq)
                    #print("cheatdict[key]:",cheatdict[key])
                    match_found = False
                    for x in cheatdict[key]:
                        compseq = x[:-1]
                        result = x[-1]
                        if is_isomorphic(extendedseq,compseq):
                            #print("Found a match!")
                            match_found = True
                            w = result
                            break
                    if not match_found:
                        print("Error: no match found for",extendedseq)
                else:
                    child = Gamepos(extendedseq)
                    w = child.winner()
                if w == self.player:
                    return self.player
                elif w == .5:
                    atleastonedraw = True
        if atleastonedraw:
            return .5
        return 1-self.player
    def __str__(self):
        return str(self.edgeseq)

def annotate(n,k):
    """Creates an annotated version of the filename
    associated with n vertices and k edges colored red or blue.
    E.g., if n=7 and k=9, then 7/9n.txt is the input file
    and 7/9a.txt is the output file, where each game is
    extended by 0, .5, or 1 according as that game is a
    red win, a draw, or a blue win.

    If 7/10a.txt exists, it will be used to determine winners."""
    
    global avoid_triangles, avoid_triangles_string, loud
    if not avoid_triangles:
        print("Error: annotate() called without avoid_triangles set")
        return
    infilename = str(n)+'/'+str(k)+avoid_triangles_string+'.txt'
    if not os.path.isfile(infilename):
        print("Error: the file",infilename,"does not exist.")
        return
    auxfilename = str(n)+'/'+str(k+1)+'a.txt'
    cheatdict = None
    if os.path.isfile(auxfilename):
        with open(auxfilename,'rb') as fpaux:
            cheatdict = pickle.load(fpaux)
    with open(infilename, 'rb') as fp:
        ingraphdict = pickle.load(fp)
        graphdict = copy.deepcopy(ingraphdict)
        numgraphs = sum([len(graphdict[key]) for key in graphdict])
        print("Annotating",numgraphs,"graphs.")
        numgraphspassed = 0
        decilespassed = 0
        for key in graphdict:
            for i in range(len(graphdict[key])):
                g = graphdict[key][i]
                ga = Gamepos(g)
                winner = ga.winner(cheatdict = cheatdict)
                numgraphspassed += 1
                g_out = g + [winner]
                graphdict[key][i] = g_out
                newdecilespassed = numgraphspassed * 10 // numgraphs
                if newdecilespassed > decilespassed:
                    decilespassed = newdecilespassed
                    if loud:
                        print(str(decilespassed)+"0% complete...")
    outfilename = str(n)+'/'+str(k)+'a.txt'
    with open(outfilename,'wb') as fp:
        pickle.dump(graphdict, fp)  

n = int(input("Number of vertices?"))
alledges = set([e for e in combinations(range(n),2)])
displaystring = "Up to how many turns? Max = " + str(n) + " C 2 = " + str(ncr(n,2)) + ". "
m = int(input(displaystring))
canonicals = [{} for i in range(m+1)]
#Each dict has spectrums as keys & lists of non-isomorphic graphs as values.
emptygraph = []
emptyspectrum = get_spectrum(emptygraph)
canonicals[0][emptyspectrum] = [emptygraph]
summy = 0
partialsummy = 1 #initializing for progress bar
loud = True
infromfile = True
outtofile = True
avoid_triangles = True
do_annotate = True
if avoid_triangles:
    avoid_triangles_string = "n"
else:
    avoid_triangles_string = ""
for k in range(1, m+1):
    #Here is how we build canonicals[k].
    #We look at each key in canonicals[k-1].
    #input("Press enter to begin k="+str(k))
    filename = str(n)+'/'+str(k)+avoid_triangles_string+'.txt'
    if infromfile and os.path.isfile(filename):
        with open(filename, 'rb') as fp:
            canonicals[k] = pickle.load(fp)
            do_sum_stuff(canonicals,k,"from "+filename)
            if k==m and do_annotate: #Will only annotate on second pass & only for highest value
                annotate(n,k)
            continue
    numpassed = 0 #how many graphs of k-1 edges we've extended in all possible ways
    decilespassed = 0 #to display when we've passed each 10% mark
    for spectrum_tuple in canonicals[k-1]:
        #We look at each graph that has that spectrum.
        for graph in canonicals[k-1][spectrum_tuple]:
            #We extend this graph by all new edges that would make sense.
            #First, we determine the highest # vertex used in the graph.
            max_vertex = -1 #A hack to make the first edge come out (0,1)
            for e in graph:
                if max(e) > max_vertex:
                    max_vertex = max(e)
            #print("max_vertex",max_vertex)
            #Now, we try adding in every edge with max vertex up to max_vertex+1
            #or max_vertex+2 in the special case (max_vertex+1,max_vertex+2).
            newmax = min(max_vertex + 1, n - 1)
            candidate_edges = [e for e in combinations(range(newmax+1), 2)]
            if newmax < n - 1:
                candidate_edges.append((newmax, newmax + 1))
            numcands = len(candidate_edges)
            for i in reversed(range(numcands)):
                cand = candidate_edges[i]
                if cand in graph:
                    del candidate_edges[i]
            #print("candidate_edges",candidate_edges)
            for e in candidate_edges:
                newgraph = graph.copy()
                if avoid_triangles:
                    ga = Gamepos(newgraph)
                    if ga.completesatriangle(e):
                        continue
                newgraph.append(e)
                new_spectrum_tuple = get_spectrum(newgraph)
                if new_spectrum_tuple not in canonicals[k]:
                    canonicals[k][new_spectrum_tuple] = [newgraph]
                else:
                    #We have to see if newgraph is isomorphic to anything else
                    graphlist = canonicals[k][new_spectrum_tuple]
                    foundamatch = False
                    for graph_to_compare in graphlist:
                        if is_isomorphic(newgraph, graph_to_compare):
                            foundamatch = True
                            break
                    if not foundamatch:
                        canonicals[k][new_spectrum_tuple].append(newgraph)
            #input("Break point")
            numpassed += 1
            newdecilespassed = numpassed * 10 // partialsummy
            if newdecilespassed > decilespassed:
                decilespassed = newdecilespassed
                if loud:
                    print(str(decilespassed)+"0% complete...")
    do_sum_stuff(canonicals,k)
    if outtofile:
        with open(filename,'wb') as fp:
            pickle.dump(canonicals[k], fp)
print("Total up to and including",m,"edges:",summy,"(or ",summy+1,"including the empty graph).")

"""
with open('7/1a.txt','rb') as fp:
    dd = pickle.load(fp)
len(dd)
sum([len(dd[k]) for k in dd])
max([len(dd[k]) for k in dd])
keylist = list(dd)
key = keylist[0]
dd[key]
"""
