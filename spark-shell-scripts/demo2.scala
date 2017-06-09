// 3000 mots
// lDA k = 20
// SVD: k = 100
// 

qs.printTopDocsForTermQuery(List("money", "happiness"))

// SVD and computer
data.findDocsByTitle("computer")
qs.printTopDocsForDoc(76)
qs.printTopTermsForTerm(data.termIds.indexOf("computer"))

// LDA topics
topics.zipWithIndex.map(_.swap).mkString("\n")

qlTopTopicsForTerm("computer")

// Right to exist
docIds(970)
qlTopTopicsForDocument(970)
