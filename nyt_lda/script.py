import pdb

import numpy as NP
import numpy.random as NPR

from pSSLDA import infer
import FastLDA 
import cProfile, pstats, StringIO

pr1 = cProfile.Profile()
pr1.enable()

wordvec = []
docvec = []
vocabulary = []
W = 0
D = 0
T = 50

with open('nyt_corpus_cleaned_for_lda.txt', 'rb') as file:
# with open('test.txt', 'rb') as file:
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
# print wordvec
# print docvec
# print vocabulary
# print W
# print D

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

pr1.disable()
s1 = StringIO.StringIO()
sortby1 = 'cumulative'
ps1 = pstats.Stats(pr1, stream=s1).sort_stats(sortby1)
ps1.print_stats()
print s1.getvalue()



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
