"""
A class to create instances of relational paths produced by closure algorithms.
"""
import numpy as np

ROUND_DIGITS = 15
rel_delim = " -{}-> "
node_delim = "[{}]"

class RelationalPath:
	def __init__(self, src, rel, target, score, pathlen, path, rel_path, caps):
		self.source = src
		self.relation = rel
		self.target = target
		self.score = score
		self.pathlen = pathlen
		self.path = path
		self.relational_path = rel_path
		self.weights = caps

	def __str__(self):
		terminal = ""
		terminal = self.pretty()
		return terminal

	def pretty(self, nodes=None, relations=None):
		"""
		Prints path information. 

		Note: Throws an exception if node and relation is 
		not found in the respective input dictionaries.

		Parameters:
		-----------
		nodes: dict
			A dictionary of node index and name pairs.

		relations: dict
			A dictionary of relation index and name pairs.
		"""
		terminal = ""
		print "SPO: [ {} {} {}], Score: {}, Path ({}):".format(
			self.source, self.relation, self.target, 
			round(self.score, ROUND_DIGITS), self.pathlen
		),
		for i in xrange(1 + self.pathlen):
			node = self.path[i]
			
			# print node
			node_name = str(node)
			if nodes is not None:
				node_name = nodes.get(node)
			terminal += node_delim.format(node_name)

			# print relation
			if i < self.pathlen:
				relation = self.relational_path[i+1]
				relation_name = str(relation)
				if relations is not None:
					relation_name = relations.get(relation)
				wt = ' ({})'.format(np.round(self.weights[i+1], 2))
				terminal += rel_delim.format(relation_name + wt)
		return terminal

