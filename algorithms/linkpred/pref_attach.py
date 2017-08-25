"""
Similarity by Preferential Attachment (PA) score.

Source: 'The Link-Prediction Problem for Social Networks' by D. Liben-Nowell & J.Kleinberg.
"""
import os
import sys
import numpy as np

def preferential_attachment(G, s, p, o, linkpred=True):
	"""
	Returns Preferential Attachment (PA) score.
	
	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	s, p, o: int
		Subject, Predicate and Object identifiers.
	linkpred: bool
		Whether or not to perform link prediction.
	
	Returns:
	--------
	score: float
		A score >=0.
	"""
	# link prediction: second condition is to avoid unnecessary introduction 
	# of a zero in the if clause
	if linkpred and G[s, o, p] != 0: 
		G[s, o, p] = 0
	_, s_nbrs = G.get_neighbors(s)
	_, o_nbrs = G.get_neighbors(o)
	s_nbrs, o_nbrs = set(s_nbrs), set(o_nbrs)
	score = len(s_nbrs) * len(o_nbrs)
	return score