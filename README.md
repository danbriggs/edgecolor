Before running edgecolor.py, create an empty folder 7/ in the same directory.

Then when the program asks you Number of vertices? put 7.

When it asks you Up to how many turns? put 9.

The program will generate one graph from each isomorphism class of K₉s with five red edges, four blue edges, and the rest colorless, that have no complete triangle of either color.

On its way up, it will do the same for those with 1,...,8 colorful edges where the number of red edges is either equal to or one greater than the number of blue edges.

It will save the results, pickled, as the files 7/1n.txt through 7/9n.txt.

For each number of edges, it will try making essentially new graphs by adding each possible essentially distinct new edge of the current player's color to a graph with one fewer edge.

It will tell you when it's 10%, 20%, ... done going through the graphs with one fewer edge. The entire process should only take a couple minutes.

Now run the program again, and put 7 and 9 again.

It will not recalculate any of these, since it notices that the files already exist.

But it will create a new file 7/9a.txt which is an annotated version of 7/9n.txt, where each graph is annotated by whether it's a 0 i.e. an Alice win or a 1 i.e. a Bob win.

Now run the program again, and put 7 and 8.

It will make an annotated 7/8a.txt using 7/8n.txt for the list of graphs and 7/9a.txt as a "cheat" to quickly verify whether each extension by one edge is a win or a loss. It does have to try permutations of the vertices to find the isomorphic candidate, but I use its "spectrum" and "restricted spectrum" to make the search space and set of permutations to try very small.

The *spectrum* of a graph is a dictionary telling how many vertices have red degree 3 and blue degree 3, how many have red degree 3 and blue degree 2, etc.

The *restricted spectrum* of a graph from the viewpoint of vertex *i* is a pair of dictionaries, one telling the above for the vertices to which *i* is connected by a red edge, and the other telling the same for the vertices to which *i* is connected by a blue edge.

So, the only candidates for isomorphic graphs are graphs with the same spectrum, and thus I use spectrum as key in the main dictionary of graphs with k edges, canonicals[*k*]. The only candidates for a permutation that turns a graph into another are ones that send vertex i with a given restricted spectrum in the source graph to a vertex with the same restricted spectrum in the target graph. For low *n* such as 7, if there are any such permutations, one of the first couple tried usually proves to be an isomorphism, so the annotation runs rather swiftly.

Now run the program again, and put 7 and 7.

Now run the program again, and put 7 and 6.

...

Now run the program again, and put 7 and 1. If you've run it in IDLE, you can type the following into the shell:

	>>> with open('7/1a.txt','rb') as fp:
	>>>		dd = pickle.load(fp)
	>>>	len(dd)
	>>>	sum([len(dd[k]) for k in dd])
	>>>	max([len(dd[k]) for k in dd])
	>>>	keylist = list(dd)
	>>>	key = keylist[0]
	>>>	dd[key]

It will tell you the only graph with one edge, which is [(0,1)] up to isomorphism, extended by who wins in this case. So it will say [[(0, 1), 0]] if it's an Alice win and [[(0, 1), 1]] if it's a Bob win.

You can also run the same snippet with '7/4a.txt' instead to see if the winner is capable of throwing the game by move 4. But you will have to check each key, i.e., spectrum, in the keylist, separately, instead of just using keylist[0].