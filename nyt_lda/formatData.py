vocabulary = []
W = 0
D = 0
wordout = open('wordvec.txt', 'wb')
docout = open('docvec.txt', 'wb')

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
			wordout.write(str(i)+ ' ')
			docout.write(str(D)+ ' ')
		D += 1
		if D%10000 == 0:
			print D

print "\n--- Vectors created ---\n"
