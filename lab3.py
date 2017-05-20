import os, numpy, sys


##define functions here

def count(v, d):
	tf = {};
	idf ={}
	for w in v: 
		idf[w] = 0.0 #doc freq
	for di in d:
		f = open('./wsj/'+di) #access each doc through dir wsj
		tf[di] = {} #nested dict
		wdl = f.readline().split() 
		for w_in_di in wdl:
			if w_in_di in v:
				if not w_in_di in tf[di]: tf[di][w_in_di] = 0.0
				tf[di][w_in_di] += 1
		for w_in_di in set(wdl): 
			if w_in_di in v:
				idf[w_in_di] += 1
		f.close()
	return tf, idf

def tdm(tf, idf, v, d):
	m = []
	#fill out m: create a row vector for each word in v
	N = len(d)
	for w in v:
		row = []
		for di in d:
			score = tf[di].get(w,0.0) #term freq
			if idf[w] > 0: 
				score *= numpy.log(N/idf[w])	
			row.append(score)
		m.append(row)	
	m = numpy.matrix(m)
	return m

def cos(x,y): # cosine function for two vecotrs x and y
	"""Cosine similarity between x and y
		= dot(x,y) / (|x|*|y|)"""
	x_length = numpy.linalg.norm(x)
	y_length = numpy.linalg.norm(y)
	return numpy.dot(x,y) / (x_length * y_length)

if __name__ == '__main__':
	##use functions to solve the lab	
	# 1. Load the vocab file (vocab.select)
	vocab =  []
	f= open('vocab.select')
	for line in f:
		vocab.append(line.strip())
	f.close()
	vocab.sort()
	# 2. Load the docs (./wsj/*)
	#os.listdir('./wsj/') returns a list of names of all files in ./wsj/
	wsjdocs = os.listdir('./wsj/')
	wsjdocs.sort()
	# 3. Create term-doc matrix of tf-idf scores
	#two ways to do this:
	#3.1. Create one dict of term-freqs per doc:
	#dict[doc][wrd] = freq of word in doc
	#3.2. Create one dict of inverse-doc freqs:
	#dict[wrd] = number of docs that have wrd
	tf, idf = count(vocab, wsjdocs)
	C = tdm(tf, idf, vocab, wsjdocs)
	print C
	# 4. Find the top 10 tf-idf words on WSJ 0725
	ci = wsjdocs.index('WSJ_0725') # ci = column intex
	top10 =	C[:, ci].A1.argsort()[::-1][:10] #find index of object (: is entirety of row), A1 changes into 1 dimensional array arg sort reverses order and gives 10 words with highed tf-idf scores
	for wi in top10: print vocab[wi], C[wi, ci]
	# 5. Use the SVD to reduce the matrix to 100 dimensions	
	U, s, VT = numpy.linalg.svd(C, full_matrices = False) # returns matrix U, array of daigonals s and matrix VT
	S = numpy.diag(s) #changes array into proper array? Vector? 
	k = 100
	#S_k = numpy.matrix(numpy.diag(s_k))
	U_k = U[:, :k] #take the first k column s from U
	S_k = numpy.matrix(S[:k, :k]) # take first k rows & columns from S
	VT_k = VT[:k, :] # take first K columns from VT
	T_k = U_k*S_k #new term matrix (with k columns)
	D_k = S_k*VT_k #new document matrix (with k rows)
	# 6. Find top ten words similar to 'oil' (with cosine similarity)
	target_word = sys.argv[1]
		#6.1 find row matching 'oil'
	oil = T_k[vocab.index(target_word), :] #give index for oil
	score =[] #list of scores
	for i in range(len(vocab)): #len of matrix returns #of rows
		score.append( cos(T_k[i, :].A1, oil.A1) ) #take i'th ro, measure flattened row to compare to oil then append to score list
	oil_top10 = numpy.array(score).argsort()[::-1][1:11]#gives top ten indices aside from 1st which is oil
	print '#Top 10 words similar to oil'
	for i in oil_top10:
		print vocab[i], score[i]
	# 7. Use doc-matrix to find top 10 docs relevent (similar) to query 'oil price'
		#7.1 Create a document vector for query in the reduced space (D_k)
	query = sys.argv[2]
	q = numpy.zeros(len(vocab))
	for qw in query.split():
		q[vocab.index(qw)] = numpy.log(len(wsjdocs)/float(idf[qw]))
	q = numpy.matrix([q]).T	 # len(vocab) x 1
	q_k = S_k.I * U_k.T * q #of k rows in a single column
	#7.2 For each column in D_k, measure cos.sim wrt query
	doc_score = []
	for i in range(len(wsjdocs)):
		doc_score.append(cos(q_k.A1, D_k[:,i].A1))
	query_top10 = numpy.array(doc_score).argsort()[::-1][1:10]
	print '# Top 10 docs relevant to the query'
	for i in query_top10: print  wsjdocs[i], doc_score[i]
