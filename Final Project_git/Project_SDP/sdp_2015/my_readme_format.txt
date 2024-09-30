
Format of the sdp 2015 dataset
(Oepen et al. 2015 : shared task on broad coverage semantic dependency parsing - semeval 2015 Task 18)
https://aclanthology.org/S15-2153/

(I can't find the official documentation any more,
but I'll describe it briefly)

column 1 : id of token in sentence
column 2 : word form
column 3 : lemma
column 4 : part of speech
column 5 : whether the current token is a "root" of the graph (to connect as child of a dummy root)
column 6 : whether the current token is a predicate, meaning it has children nodes
column 7 : useless

The following columns give the dependents of each predicate:
the first predicate (in linear order of the sentence) has its dependents shown in column 8,
the second predicate  has its dependents shown in column 9, etc...

So e.g. below, the first predicate is "said", having "Rockwell" as verb_ARG1 dependent, and "calls" as verb_ARG2 dependent.
the second predicate is "the" (id 3), which has "agreement" as det_ARG1 dependent, etc...


#22000002
1	Rockwell	rockwell	NNP	-	-(6)	_(7)	verb_ARG1(8)	_	_	_	_	_	_	_	_	_	_
2	said	say	VBD	+	+	_	_	_	_	_	_	_	_	_	_	_	_
3	the	the	DT	-	+	_	_	_	_	_	_	_	_	_	_	_	_
4	agreement	agreement	NN	-	-	_	_	det_ARG1	verb_ARG1	_	_	_	_	_	_	_	_
5	calls	call	VBZ	-	+	_	verb_ARG2	_	_	_	_	_	_	_	_	_	_
6	for	for	IN	-	+	_	_	_	_	_	_	_	_	_	_	_	_
7	it	it	PRP	-	-	_	_	_	_	comp_ARG1	_	verb_ARG1	_	_	_	_	_
8	to	to	TO	-	+	_	_	_	_	_	_	_	_	_	_	_	_
9	supply	supply	VB	-	+	_	_	_	verb_ARG2	comp_ARG2	comp_ARG1	_	_	_	_	prep_ARG1	_
10	200	-NUMBER-	CD	-	+	_	_	_	_	_	_	_	_	_	_	_	_
11	additional	additional	JJ	-	+	_	_	_	_	_	_	_	_	_	_	_	_
12	so-called	so-called	JJ	-	+	_	_	_	_	_	_	_	_	_	_	_	_
13	shipsets	shipset	NNS	-	-	_	_	_	_	_	_	verb_ARG2	adj_ARG1	adj_ARG1	adj_ARG1	_	_
14	for	for	IN	-	+	_	_	_	_	_	_	_	_	_	_	_	_
15	the	the	DT	-	+	_	_	_	_	_	_	_	_	_	_	_	_
16	planes	plane	NNS	-	-	_	_	_	_	_	_	_	_	_	_	prep_ARG2	det_ARG1
17	.	_	.	-	-	_	_	_	_	_	_	_	_	_	_	_	_


