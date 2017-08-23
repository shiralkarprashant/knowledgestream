"""
A Flow object representing the maximum flow that can be pushed between a source s
and target t in a network.
"""
import os
import sys
import numpy as np

class Flow(object):
	"""A Flow object - list of RelationalPaths."""
	def __init__(self, src, pred, obj, maxflow=-1., annotation=''):
		self.flow = 0.
		self.paths = [] # list of relational paths.
		self.source = src
		self.predicate = pred
		self.target = obj
		self.maxflow = maxflow # total flow out of source.
		self.annotation = annotation # any notes, e.g. 'FF' for Ford-Fulkerson
		self.time = None # time taken to compute this flow

	def add_path(self, path, rev_sort=False):
		"Add the new path to the collection of flows, and keep them sorted."
		newflow = path.score + self.flow
		if self.maxflow != -1. and newflow > self.maxflow:
			raise Exception('Flow {} exceeds total flow out of source {}'.format(
				newflow, self.maxflow
			))
		self.flow += path.score
		self.paths.append(path)
		if rev_sort:
			self.paths = sorted(self.paths, key=lambda x: x.score, reverse=True)

	def __str__(self):
		time, maxflow, annotation = '', '', ''
		if self.maxflow != -1.:
			maxflow = 'TotOutOfS: {}'.format(self.maxflow)
		if self.annotation != '':
			annotation = '[{}] '.format(self.annotation)
		if self.time is not None:
			time = 'time: {:.3f}s'.format(self.time)
		print '\n=> {}SPO: {} | Flow: {} | {} | {}'.format(
			annotation, (self.source, self.predicate, self.target), 
			self.flow, maxflow, time
		)
		for i, path in enumerate(self.paths):
			print '  {}.'.format(i+1),
			print path.pretty()
		return ""

	def __repr__(self):
		return '<Flow object: {}>'.format(self.__str__())