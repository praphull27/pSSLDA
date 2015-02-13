import numpy as NP
from pSSLDA import infer
import cProfile, pstats, StringIO
from guppy import hpy

W = 354700
T = 50

print "\n--- Vectors creation started ---\n"

with open('wordvec.txt', 'rb') as file:
	wordvec = file.read()
	wordvec = wordvec.strip()
	wordvec = wordvec.split()
	wordvec = map(int, wordvec)

w = NP.array(wordvec, dtype = NP.int)
del(wordvec)

print "\n--- Word Vector Comepleted ---\n"

with open('docvec.txt', 'rb') as file:
	docvec = file.read()
	docvec = docvec.strip()
	docvec = docvec.split()
	docvec = map(int, docvec)

d = NP.array(docvec, dtype = NP.int)
del(docvec)

print "\n--- Vectors created ---\n"

# Create parameters
alpha = NP.ones((1,T)) * 1
beta = NP.ones((T,W)) * 0.01

# How many parallel samplers do we wish to use?
P = 7

# Random number seed 
randseed = 194582

# Number of samples to take
numsamp = 10

# h = hpy()
# print h.heap()

print "\n--- Starting LDA Inference ---\n"

pr = cProfile.Profile()
pr.enable()

# Do parallel inference
finalz = infer(w, d, alpha, beta, numsamp, randseed, P)

pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(.0000001)
print s.getvalue()

print "\n--- LDA Completed ---\n"
