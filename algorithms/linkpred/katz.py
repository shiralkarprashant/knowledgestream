"""
Similarity by Katz (KZ) score, using the ensemble of all paths between two nodes.

Source: 'The Link-Prediction Problem for Social Networks' by D. Liben-Nowell & J.Kleinberg.
"""
import os
import sys
import numpy as np

from datastructures.relationalpath import RelationalPath
from algorithms.linkpred.pathenum import get_paths as c_get_paths

def katz(G, s, p, o, beta=0.05, k=3, linkpred=True):
	"""
	Returns Katz (KZ) score computed based on common neighbors of s and o.
	
	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	s, p, o: int
		Subject, Predicate and Object identifiers.
	beta: float
		Attenuation factor, typically 0.05.
	k: int
		Maximum length of any path, in terms of number of edges.
		e.g. 0 -> 1 -> 3 -> 5 is a path of length 3.
	linkpred: bool
		Whether or not to perform link prediction.
	
	Returns:
	--------
	score: float
		A score >= 0.
	"""
	# link prediction: second condition is to avoid unnecessary introduction 
	# of a zero in the if clause
	if linkpred and G[s, o, p] != 0: 
		G[s, o, p] = 0
	score = 0.
	for m in xrange(k + 1):
		if m == 0: # path of length 0 means nothing
			continue
		elif linkpred and m == 1: # no direct links allowed for link prediction
			continue
		paths = c_get_paths(G, s, p, o, length=m)
		score += (beta ** m) * len(paths)
	return score

def get_paths(G, s, p, o, length=3):
	"Returns all paths of length `length` starting at s and ending in o."
	path_stack = [[s]]
	relpath_stack = [[-1]]
	discoverd_paths = []
	while len(path_stack) > 0:
		# print 'Stack: {} {}'.format(path_stack, relpath_stack)
		curr_path = path_stack.pop()
		curr_relpath = relpath_stack.pop()
		node = curr_path[-1]
		# print 'Node: {}'.format(node)
		if len(curr_path) == length + 1:
			if node == o:
				# create a path
				path = RelationalPath(
					s, p, o, 0., length, curr_path, curr_relpath, np.ones(length+1)
				)
				discoverd_paths.append(path)
			continue
		relnbrs = G.get_neighbors(node)
		for i in xrange(relnbrs.shape[1]):
			rel, nbr = relnbrs[:, i]
			path_stack.append(curr_path + [nbr])
			relpath_stack.append(curr_relpath + [rel])
	return discoverd_paths

