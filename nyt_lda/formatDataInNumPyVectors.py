
wordvec = []
docvec = []
vocabulary = []
W = 0
D = 0

# with open('/Users/praphull/Dropbox/MS Project/Removed Empty lines/nyt_corpus_cleaned_for_lda.txt', 'rb') as file:
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


print wordvec
print docvec
print vocabulary
print W
print D