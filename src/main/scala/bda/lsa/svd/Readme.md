# Singular Value Decomposition explained

SVD stands for "Singular Value Decomposition". It is a matrix factorization method, where a matrix `M` is decomposed as the product of three smaller matrixes. In this particular case, the matrix we'll decompose is the document-term-frequency matrix (TF-IDFs).


Let's consider `M` is a `m * n` matrix. `m` is the number of documents and `n` is the number of terms.
The factorization is generally represented as :

`M = U x S * V'`
`K` is a parameter of the decomposition : number of concepts (topics) to infer.
In our case the number of concepts represent the number of soft clusters of the model.

where :

	U  -->	m * K matrix, essentialy the link (document -> topic)
			each row represent a document
			each column represent a concept / topic

	S  -->	K * K diagonal matrix,
			Each element on the diagonal represent a concept (mathematicaly a column in V and a column in U).
			The magnitude of the value corresponds to the importance / relevance of the concept.

	V  -->	K * n matrix, essentialy the link (term -> topic)
			each row represents a term
			each column represent a concept / topic

## Compute the SVD model in Spark
To perform the singular value decomposition, we represent the matrix `M` as an RDD of row vectors (all of that in a `org.apache.spark.mllib.linalg.distributed.`RowMatrix[1] object). We use it's `computeSVD` method.The returned `org.apache.spark.mllib.linalg.`SingularValueDecomposition[2] object allows to retreive the coefficients matrix :

```
val u = svd.U
val s = svd.S
val v = svd.V
```

Notice :
 * The svd algorithm `computeSVD` method requires to iterate over the matrix multiple times. It is importand to cache the RDD before processing.
 * The V matrix is already transposed (should be called V' to be mathematicaly right)


## Querying the SVD model

### TopTermsByConcept
As it is mentioned above, the rows of the `V Matrix` rows represent the terms, and its columns represent the concept / topics. We can see this matrix as the representation of concepts, through terms that are relevant for this concept. In other words, it represents the `link between the terms and the concepts`. The value of each term represent its relevance for the concept.

To find the most relevant terms to a concepts, we have to extract the maximum values in the concept column (and limit the result to the number of desired terms). If we apply this query to each concept, we obtain the most relevant terms.

### TopDocumentByConcept
The V matrix is quite similar to the `U matrix` except that it represents the `link between the documents and the concepts`. The value of each document represent its relevance for the concept.
To find the most relevant documents for a concept, the query is similar than the TopTermsByConcept. Thus, we have to extract the maximum values in the concept column (and limit the result to the number of desired documents).

We can apply this query to each concept to obtain the most relevant `group of documents`.

### TopTermsByTerm
This request is a bit more tricky and requires a deeper understanding of LSA. LSA represents the term-term relation as the cosine similarity of their two columns, in the reconstructed M matrix (approximation of M obtained by multiplying U, S and V together). In fact, we don't need to compute the approximation of M. We can use some algebric properties instead : it's consine similarity is exactly the same for between the comlumns of the matrixes S and V that we already have.

What we have to do is to normalize to 1 each row in both matrix. Then multiply the normalized row corresponding to that term by the query term. The result is the vector of similarity term -> query term. If we apply this query for each terms (except the term itself), we'll obtain the `term to terms relevance`.
