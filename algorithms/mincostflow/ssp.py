"""
Successive Shortest Path (SSP) algorithm for solving the minimum cost 
flow problem.

Reference: Ch 9 Network Flows by Ahuja, Magnanti and Orlin.
"""
import os
import sys
import numpy as np
import logging as log

from time import time
from os.path import expanduser, abspath, isfile, isdir, basename, splitext, \
	dirname, join, exists
from scipy.sparse import csr_matrix

from datastructures.flow import Flow
from algorithms.mincostflow.ssp_helper import compute_shortest_path_distances

# logging config: https://docs.python.org/2/library/logging.html#logrecord-attributes
log.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S', level=log.DEBUG)

def disable_logging(lvl):
	log.disable(lvl)

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float

def succ_shortest_path(G, cost_vec, s, p, t, linkpred=True, return_flow=True, npaths=-1):
	"""
	Returns the s-t min cost flow in graph G using Successive Shortest Path (SSP)
	algorithm. 

	* Note: Current implementation only works for undirected graphs.

	Parameters:
	----------
	G: rgraph
		A graph representing the knowledge graph.
	cost_vec: ndarray
		A vector of specifying cost to pass through each node. Shape: G.N
	s, p, t: int
		Source, predicate and target nodes. Predicate is used merely to create
		`flow.datastructures.flow.Flow' object when, not max-preflow, but max-flow
		is desired.
	npaths: int
		Threshold for maximum number of paths to compute.
	"""
	N, R = G.N, G.R 
	pi = np.zeros(N, dtype=_int)
	flow = dict() # eventual flow
	excess_at_t = 0
	ts = time()

	# link prediction: second condition is to avoid unnecessary introduction 
	# of a zero in the if clause
	if linkpred and G[s, t, p] != 0: 
		G[s, t, p] = 0

	# some prep
	outflow = sum(G.csr.data[G.csr.indptr[s]:G.csr.indptr[s+1]])

	# form cost matrix
	t1 = time()
	all_costs = np.round(cost_vec[G.targets]).astype(_int, copy=False) # cost per edge
	cost_mtx = csr_matrix((all_costs, G.csr.indices, G.csr.indptr), shape=(N, N * R))
	log.debug('Time for cost mtx: {:.2f}s'.format(time() - t1))

	# initial reduced cost
	reduced_cost = cost_mtx.data - pi[G.sources] + pi[G.targets]
	log.debug('Initial reduced cost: {}'.format(reduced_cost)) 

	# compute shortest path distances
	t1 = time()
	paths, relpaths, bncks, costs, flows = [], [], [], [], []
	path, rpath, bnck, is_permanent, distances = compute_shortest_path_distances(G, s, t, reduced_cost)
	itr = 1
	while path is not None:
		pathcost = cost_vec[path[1:-1]].sum()
		pathflow = bnck * (1./(1. + pathcost))
		log.debug('P{}: {} {}, (c, bnck, f): ({:.2f}, {:.2f}, {:.2f}), time: {:.2f}s'.format(
			itr, path, rpath, pathcost, bnck, pathflow, time() - t1
		))
		log.debug('Found: {}, dist: {}'.format(is_permanent, distances))
		paths.append(path.tolist())
		relpaths.append(rpath.tolist())
		bncks.append(bnck)
		costs.append(pathcost)
		flows.append(pathflow)
		t1 = time()
		itr += 1

		# register excess
		excess_at_t += pathflow
		if npaths != -1 and (1 + npaths) == itr:
			break

		# update pi vec
		permanent = (is_permanent == 1)
		pi[permanent] = pi[permanent] - distances[permanent] + distances[t]
		log.debug('Updated potentials: {}'.format(pi))

		# update residual graph and flow
		# 	* Decrement forward edge's res. cap by bottleneck
		#   * Change backward edge's res. cap. by bottleneck 
		#		[Note: In the case where we may have a directed edge (u, v) but not (v, u),
		#		a flow from u to v would result in a backward edge from v to u with capacity
		#		equal to the flow sent on (u, v). However, in the case where we may have both (u, v) 
		#		and (v, u) from the same predicate, a flow along (u, v) creates an additional capacity 
		#		along (v, u), which may result in a total capacity that exceeds 1. However, since the 
		#		capacity captures a notion of similarity between predicates, we want it to be less than or
		#		equal to 1. We therefore make an assumption in this work that there is only
		#		an edge in one direction. Accordingly, once one of the forward or backward edges is used, we 
		#		set the capacity of the other edge equal to the flow sent along the chosen edge, which is the bottleneck. 
		#		This assumption does however mean that it may lead to different results. 
		#		Current implementation thus makes this simplifying assumption (i.e., old (v, u) value 
		#		is overwritten if it exists), and can be improved in future work.]
		#	* Update cost matrix (NOTE: old actual cost is overwritten)
		# 	* Update flow
		pathlen = len(path) - 1
		for i in xrange(pathlen):
			u, v, r = path[i], path[i+1], rpath[i+1] # edge: (u, v, r)
			residual_cap = G.csr[u, N * r + v] # G[u, v, r]
			if residual_cap - bnck >= 0:
				G.csr[u, N * r + v] -= bnck # update G[u, v, r]
			G.csr[v, N * r + u] = bnck # G[v, u, r] 
			cost_mtx[v, N * r + u] = -cost_mtx[u, N * r + v]
			if return_flow:
				flow[(u, v, r)] = flow.get((u, v, r), 0.) + bnck

		# updated reduced costs
		reduced_cost = cost_mtx.data - pi[G.sources] + pi[G.targets]
		log.debug('Reduced cost: {}'.format(reduced_cost))
		path, rpath, bnck, is_permanent, distances = compute_shortest_path_distances(G, s, t, reduced_cost)
	tend = time()
	mincostflow = Flow(s, p, t, maxflow=outflow, annotation='SUCC-SHORTEST-PATH')
	mincostflow.flow = excess_at_t
	mincostflow.stream = {
		'sid': s, 'pid': p, 'oid': t, 'paths': paths, 'relpaths': relpaths, 
		'bottlenecks': bncks, 'costs': costs, 'flows': flows
	}
	if return_flow:
		mincostflow.edges = flow # a dict of (i, j, r) -> flow-val
	mincostflow.time = tend - ts
	return mincostflow
		