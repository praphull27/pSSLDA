import pdb

import numpy as NP
import numpy.random as NPR

from pSSLDA import infer
import FastLDA 
import cProfile, pstats, StringIO
from guppy import hpy
import gc

gc.enable()

wordvec = []
docvec = []
vocabulary = []
W = 0
D = 0
T = 50

# with open('nyt_corpus_cleaned_for_lda.txt', 'rb') as file:
with open('test.txt', 'rb') as file:
	for line in file:
		for word in line.split():
			try:
				i = vocabulary.index(word)
			except ValueError:
				vocabulary.append(word)
				i = W
				W += 1
			wordvec.append(i)
			docvec.append(D)
		D += 1
		if D%1000000 == 0:
			print D
		else:
			if D%100000 == 0:
				print "."

print "\n--- Vectors created ---\n"

del(vocabulary)

(w, d) = (NP.array(wordvec, dtype = NP.int),
          NP.array(docvec, dtype = NP.int))

# Create parameters
alpha = NP.ones((1,T)) * 1
beta = NP.ones((T,W)) * 0.01

# How many parallel samplers do we wish to use?
P = 7

# Random number seed 
randseed = 194582

# Number of samples to take
numsamp = 1

gc.disable()

h = hpy()
print h.heap()

print "\n--- Starting LDA Inference ---\n"

pr = cProfile.Profile()
pr.enable()

# Do parallel inference
finalz = infer(w, d, alpha, beta, numsamp, randseed, P)

pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()

print "\n--- LDA Completed ---\n"
