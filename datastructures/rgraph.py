import os
import sys
import numpy as np
import pandas as pd
import warnings
import cPickle

from os.path import exists, join, abspath, expanduser, basename, dirname, isdir
from time import time
from itertools import izip
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix, csc_matrix

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float

MAX_MAT_DIM_DISPLAY = 20

warnings.simplefilter(action = "ignore", category = FutureWarning)

class Graph(object):
	"""Graph data structure by wide CSR matrix."""
	def __init__(
		self, adj, shape, values=None, sym=True, save_csc=False, save_indeg_vec=True, 
		save_indeg_mat=False, display=True, reconstruct=False
	):
		if reconstruct:
			return
		if adj is None or adj.shape[1] < 2:
					raise Exception('Adjacency matrix of shape (*, 3) required.')
		if len(shape) != 3 and \
			not (shape[0] > 0 and shape[1] > 0 and shape[2] > 0):
			raise ValueError('Incorrect graph dimensions')
		self.N = shape[0] # no. of elements in mode 1 or mode 2
		self.R = shape[2] # no. of relations
		self.NR = self.N * self.R
		self.save_indeg_vec = save_indeg_vec # whether to cache node-based indegree
		self.save_indeg_mat = save_indeg_mat # whether to cache edge-based indegree
		self.sym = sym # symmetric or asymmetric
		self.display = display # whether to show intermediate print messages
		self.save_csc = save_csc # whether to also save wide CSC matrix

		# shape
		self.shape = (self.N, self.NR)

		# load graph
		self._create_graph(adj, values)

	def __str__(self):
		if self.N > MAX_MAT_DIM_DISPLAY or self.R > MAX_MAT_DIM_DISPLAY:
			print "Graph too large ({}) to print.".format(self.shape)
			return ""
		for k in xrange(self.R):
			print np.array_str(self.getslice(k).toarray()), '\n'
		return ''

	def __repr__(self):
		return self.__str__()

	def _check_bounds(self, i, j, k):
		"Check bounds for all three dimensions."
		if not (i < self.N and i >= 0):
			raise ValueError('Row index out of bounds.')
		elif not (j < self.N and j >= 0):
			raise ValueError('Column index out of bounds.')
		elif not (k < self.R and k >= 0):
			raise ValueError('Slice index out of bounds.')

	def _get_indices(self, i):
		return self.csr.indices[self.csr.indptr[i]:self.csr.indptr[i+1]]

	def __getitem__(self, idx):
		if len(idx) != 3: 
			raise ValueError('Incorrect number of indices specified.')
		i, j, k = idx
		self._check_bounds(i, j, k)
		return self.csr[i, self.N * k + j]

	def __setitem__(self, idx, val):
		if len(idx) != 3: 
			raise ValueError('Incorrect number of indices specified.')
		i, j, k = idx
		self._check_bounds(i, j, k)
		self.csr[i, self.N * k + j] = val
		
	def getrow(self, i, k, boundscheck=True):
		"""
		Returns indices of elements in i'th row under k'th relation.
		"""
		if boundscheck:
			self._check_bounds(i, 0, k)
		row = self._get_indices(i)
		row = row[np.logical_and(row >= k * self.N, row < (k+1) * self.N)]
		return row % self.N

	def get_neighbors(self, i, k=-1):
		"""
		Returns a 2 x M dimensional array, where the first row 
		indicates the relations of the neighbors located in the second row.
		If k = -1, neighbors across all relations are returned.
		"""
		if k == -1:
			self._check_bounds(i, 0, 0)
			nbrs = self._get_indices(i)
			rels = (nbrs - (nbrs % self.N)) / self.N
			nbrs = nbrs % self.N
		else:
			self._check_bounds(i, 0, k)
			nbrs = self.getrow(i, k, boundscheck=False)
			rels = k * np.ones(len(nbrs), dtype=_int)
		return np.asarray([rels, nbrs])

	def getslice(self, k):
		"Returns a csr_matrix representing k'th relation."
		self._check_bounds(0, 0, k)
		indptr = np.zeros(1+self.N, dtype=_int)
		incl = np.logical_and(
			self.csr.indices >= k * self.N, self.csr.indices < (k+1) * self.N
		)
		indices = self.csr.indices[incl] - self.N * k
		data = self.csr.data[incl]
		for i in xrange(self.N):
			indptr[i+1] = len(self.getrow(i, k, boundscheck=False))
		indptr = np.cumsum(indptr)
		return csr_matrix((data, indices, indptr), shape=(self.N, self.N))

	def _create_graph(self, d, values):
		"""
		Entry point for creation of the graph based on adjacency & values.
		"""
		# clean adjacency and values information
		ts = time() # start time
		d, vals = clean_adjacency(d, values, sym=self.sym, display=self.display)
		self.nnz = d.shape[0] # NNZ
		ii, jj, kk = d[:, 0], d[:, 1], d[:, 2]

		t1 = time()
		# indptr, indices, data vector
		indptr = np.cumsum(np.bincount(ii, minlength=self.N), dtype=_int)
		indptr = np.insert(indptr, 0, 0)
		indices = (self.N * kk) + jj
		data = vals
		self.csr = csr_matrix((data, indices, indptr), shape=self.shape)
		if self.display:
			print '==> wide-CSR matrix created: {:.4f} secs.'.format(time() - t1)

		# wide CSC matrix
		if self.save_csc: # **USE WITH CAUTION**: can be very memory-expensive
			t1 = time()
			self.csc = csc_matrix((data, (ii, indices)), shape=self.shape)
			if self.display:
				print '==> wide-CSC matrix created: {:.4f} secs.'.format(time() - t1)
		
		# indegree of a node (node-based), i.e. no. of neighboring nodes
		if self.save_indeg_vec:
			t1 = time()
			tmp_arr = unique_rows(d[:,:2], None) # ignore edge type
			self.indeg_vec = np.bincount(tmp_arr[:,1], minlength=self.N)
			if self.display:
				print '==> Indegree vector (node-based) [len: {}] created: {:.4f} secs.'.format(
					len(self.indeg_vec), time() - t1
				)

		print '==> Graph created: {}. Total time: {:.4f} secs.'.format(
			(self.N, self.N, self.R), time() - ts
		)
		print ''

	def save_graph(self, dirpath=os.curdir):
		"""
		Saves data structures representing this graph 
		at the specified directory. If the graph is undirected,
		i.e. sym = True, then all saved files are prefixed with 'undir_'.
		
		Saved data structures:
		* data: [undir_]data.npy
		* indptr: [undir_]indptr.npy
		* indices: [undir_]indices.npy
		* facemap: [undir_]facemap.pkl
		* indeg_vec: [undir_]indeg_vec.npy
		* indeg_mat: [undir_]indeg_mat.npy
		"""
		prefix = 'undir_' if self.sym else ''
		np.save(join(dirpath, '{}data.npy'.format(prefix)), self.csr.data)
		np.save(join(dirpath, '{}indptr.npy'.format(prefix)), self.csr.indptr)
		np.save(join(dirpath, '{}indices.npy'.format(prefix)), self.csr.indices)
		if self.save_csc:
			np.save(join(dirpath, '{}csc_data.npy'.format(prefix)), self.csc.data)
			np.save(join(dirpath, '{}csc_indptr.npy'.format(prefix)), self.csc.indptr)
			np.save(join(dirpath, '{}csc_indices.npy'.format(prefix)), self.csc.indices)
		if self.save_indeg_vec:
			np.save(join(dirpath, '{}indeg_vec.npy'.format(prefix)), self.indeg_vec)
		if self.display:
			print 'Saved graph data structures at: %s' % dirpath

	@classmethod
	def reconstruct(cls, dirpath, shape, adj=None, values=None, **kwargs):
		"""
		Recontructs graph from data structures saved on disk at dirpath,
		or creates a new graph using the input adjacency, shape and values.
		"""
		dirpath = abspath(expanduser(dirpath))
		if not isdir(dirpath) or not exists(dirpath):
			print '** Not a directory, or does not exist: %s' % dirpath
		if len(shape) != 3:
			raise Exception('Incorrect graph dimensions.')
		# reconstruct
		try:
			t1 = time()
			print 'Reconstructing graph from %s' % dirpath
			sys.stdout.flush()
			prefix = 'undir_' if kwargs.get('sym', True) else ''
			datafile = join(dirpath, '{}data.npy'.format(prefix))
			indicesfile = join(dirpath, '{}indices.npy'.format(prefix))
			indptrfile = join(dirpath, '{}indptr.npy'.format(prefix))
			indeg_vecfile =join(dirpath, '{}indeg_vec.npy'.format(prefix))
			data = np.load(datafile)
			print '=> Loaded: %s' % basename(datafile)
			indptr = np.load(indptrfile)
			print '=> Loaded: %s' % basename(indptrfile)
			indices = np.load(indicesfile)
			print '=> Loaded: %s' % basename(indicesfile)
			G = cls(None, None, reconstruct=True)
			G.N, G.R, G.NR = shape[0], shape[2], shape[0] * shape[2]
			G.shape = (G.N, G.NR)
			G.csr = csr_matrix((data, indices, indptr), shape=G.shape)
			# load CSC if available and desired
			if kwargs.get('save_csc', False):
				datafile = join(dirpath, '{}csc_data.npy'.format(prefix))
				indicesfile = join(dirpath, '{}csc_indices.npy'.format(prefix))
				indptrfile = join(dirpath, '{}csc_indptr.npy'.format(prefix))
				data = np.load(datafile)
				print '=> Loaded: %s' % basename(datafile)
				indptr = np.load(indptrfile)
				print '=> Loaded: %s' % basename(indptrfile)
				indices = np.load(indicesfile)
				print '=> Loaded: %s' % basename(indicesfile)
				G.csc = csc_matrix((data, indices, indptr), shape=G.shape)
			if kwargs.get('save_indeg_vec', True) and exists(indeg_vecfile):
				G.indeg_vec = np.load(indeg_vecfile)
				print '=> Loaded: %s' % basename(indeg_vecfile)
			print '=> Graph loaded: {:.2f} secs.\n'.format(time() - t1)
			return G
		except Exception, e:
			print '** Error reconstructing graph: ',
			exc_type, exc_obj, exc_tb = sys.exc_info()
			print 'Line: {}, exception type: {}, exception: {}'.format(exc_tb.tb_lineno, exc_type, exc_obj)
		print 'Trying to create graph using inputs..'
		if adj is None:
			raise Exception('Either directory, or (adj, shape) is required.')
		G = make_graph(adj, shape, values, kwargs)
		return G

# ======================= HELPER FUNCIONS =======================

def weighted_degree(arr, weight='logdegree'):
	"""Returns a weighted version of the array."""
	if weight == 'degree':
		arr = 1./(1 + arr)
	elif weight == 'logdegree':
		arr = 1./(1 + np.log(arr))
	else:
		raise ValueError('Unknown weight function.')
	return arr

def make_graph(adj, shape, values=None, **kwargs):
	"""
	Creates a graph out of the adjacency information.
	Values represent the data values. Taken as 1. if not provided.
	Other keyword arguments such as sym, etc. may be provided.
	"""
	G = Graph(
		adj, shape, values, 
		sym=kwargs.get('sym', True), 
		save_csc=kwargs.get('save_csc', False),
		save_indeg_vec=kwargs.get('save_indeg_vec', True), 
		save_indeg_mat=kwargs.get('save_indeg_mat', False), 
		display=kwargs.get('display', True)
	)
	return G

def search_idx_of_element(arr, start, end, element):
	"""
	Binary search for the index of a desired element in the input array.
	"""
	while start <= end:
		if arr[start] == element:
			return start
		if arr[end] == element:
			return end
		middle = (start + end) // 2
		if arr[middle] == element:
			return middle
		else:
			if element < arr[middle]:
				end = middle - 1
			else:
				start = middle + 1
	return -1

def symmetrize(d):
	"""
	The first and second column are made symmetric in the input array.
	"""
	ud = d.copy()
	tmp_arr = ud[:,0].copy()
	ud[:,0] = ud[:,1]
	ud[:,1] = tmp_arr
	d = np.vstack((d, ud))
	return d

def unique_rows(data, order):
	"""
	Returns new data matrix with unique rows from input data.
	The new matrix is sorted as per the order representing the
	columns of the input matrix.

	Parameters:
	-----------
	data: array_like
		An input matrix, 2-D array.

	order: array
		A sequence representing the column order to use for 
		sorting the output matrix.

	Returns:
	--------
	sorted_data: array_like
		Output matrix with duplicate rows removed.
		Same shape and type as the input matrix.
	"""
	row_mask = np.ones(data.shape[0], dtype=np.bool)
	if order is None:
		order = np.arange(data.shape[1])
	order = np.asarray(order, dtype=_int64)
	sorted_idx = np.lexsort(data[:, order].T)
	sorted_data =  data[sorted_idx, :]
	row_mask[1:] = np.any(np.diff(sorted_data, axis=0), axis=1)
	sorted_data = sorted_data[row_mask]
	return sorted_data

def clean_adjacency(d, values, sym=True, display=True):
	"""
	Makes input adjacency symmetric, eliminates duplicates
	and returns the same after sorting by sub, relation and object 
	in that order.
	"""
	if display:
		print 'Cleaning adjacency..'; sys.stdout.flush()
	ts = time()
	nrows = d.shape[0]
	ncols = d.shape[1]

	# values
	if values is None or len(values) == 0:
		values = np.ones(nrows, dtype=_float)
	else:
		values = np.asarray(values, dtype=_float)
	arr =  np.hstack((d, values.reshape(-1,1)))
	del d # remove d for now since it's additional unnecessary memory

	# symmetrize if necessary
	if sym:
		t1 = time()
		arr = symmetrize(arr)
		if display:
			print '==> Adjacency made symmetric: {:.4f} secs.'.format(time() - t1)
	elif display:
		print '==> Adjacency: {}'.format((nrows, ncols))

	# eliminate duplicates & re-order based on columns
	t1 = time()
	arr = unique_rows(arr, [1, 2, 0])
	if display:
		print '==> Unique rows extraction & re-ordering: {:.4f} secs.'.format(time() - t1)

	# if (np.max(arr[:, :2]) * np.max(arr[:, 2])) + np.max(arr[:, 1]) > np.iinfo(_int).max:
	# 		d = arr[:, :3].astype(_int64)
	# else:
	d = arr[:, :3].astype(_int64)
	vals = arr[:, 3].astype(_float)
	if display:
		print '==> Adjacency cleaned: {:.4f} secs.'.format(time() - ts)
	return d, vals