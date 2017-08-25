"""
Similarity by Path Entropy (PE) index, using the ensemble of all paths between two nodes.

Source: 'Link prediction based on path entropy' by Z. Xu et al.

URL: http://www.sciencedirect.com/science/article/pii/S0378437116300899
"""
import os
import sys
import numpy as np

from datastructures.relationalpath import RelationalPath
from algorithms.linkpred.pathenum import get_paths as c_get_paths

link_entropies = dict()

path_entropies = dict()

def compute_link_entropy(G, a, b):
	"""
	Returns entropy of an edge/link (a, b). Eqn. (3) in the paper.
	"""
	ka = int(G.indeg_vec[a]) # degree of node 'a'
	kb = int(G.indeg_vec[b]) # degree of node 'b'
	m = G.csr.nnz / 2 # number of edges
	onevec = np.ones(kb)
	vec = np.arange(kb)
	num = ((m - ka) * onevec) - vec # product of (m-ka)-(i-1) in numerator
	deno = (m * onevec) - vec
	prob = 1. - np.prod(num / deno)
	link_ent = - np.log2(prob)
	return link_ent

def compute_path_entropy(G, path):
	"""
	Returns entropy of a path approximated by sum of its edge entropies.
	Eqn. (5) in the paper.
	"""
	path_ent = 0.
	for a, b in zip(path[:-1], path[1:]):
		if (a, b) in link_entropies:
			link_ent = link_entropies[(a, b)]
		else:
			link_ent = compute_link_entropy(G, a, b)
			# print 'Link ent: {} ->  {}'.format((a, b), link_ent)
			link_entropies[(a, b)] = link_ent # cache it.
		path_ent += link_ent
	return path_ent

def pathentropy(G, s, p, o, k=3, linkpred=True):
	"""
	Returns similarity score computed using path entropy. Eqn. (8) in the paper.
	
	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	s, p, o: int
		Subject, Predicate and Object identifiers.
	k: int
		Maximum length of any path, in terms of number of nodes.
		e.g. 0 -> 1 -> 3 is a path of length 3.
	linkpred: bool
		Whether or not to perform link prediction.
	
	Returns:
	--------
	score: float
		A score >= 0.
	"""
	global link_entropies, path_entropies
	link_entropies = dict()
	path_entropies = dict()

	# link prediction: second condition is to avoid unnecessary introduction 
	# of a zero in the if clause
	if linkpred and G[s, o, p] != 0: 
		G[s, o, p] = 0
	joint_ent = 0.
	for m in xrange(k + 1):
		if m == 0: # path of length 0 means nothing
			continue
		elif linkpred and m == 1: # no direct links allowed for link prediction
			continue
		paths = c_get_paths(G, s, p, o, length=m, maxpaths=200)

		# compute path entropies
		m_len_path_ent = 0.
		for pp in paths:
			path = tuple(pp.path)
			if path in path_entropies:
				path_ent += path_entropies[path]
			else:
				path_ent = compute_path_entropy(G, path)
				path_entropies[path] = path_ent
			m_len_path_ent += path_ent
		joint_ent += (1./ (m - 1)) * m_len_path_ent

	# subtract marginal entropy of link between s and o
	marginal_ent = compute_link_entropy(G, s, o)
	score = joint_ent - marginal_ent
	return score