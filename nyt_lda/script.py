import numpy as NP
from pSSLDA import infer
import cProfile, pstats, StringIO
from guppy import hpy
import time

print "Start date & time " + time.strftime("%c")

W = 354700
T = 50
wordvec = []
docvec = []
print "\n--- Vectors creation started ---\n"

with open('wordvec_line.txt', 'rb') as file:
	for line in file:
		wordvec.append(int(line.strip()))

w = NP.array(wordvec, dtype = NP.int)
del(wordvec)

print "\n--- Word Vector Comepleted ---\n"

with open('docvec_line.txt', 'rb') as file:
	for line in file:
		docvec.append(int(line.strip()))

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
print "End date & time " + time.strftime("%c")