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

# Compute the SVD model in Spark
To perform the singular value decomposition, we represent the matrix `M` as an RDD of row vectors (all of that in a `org.apache.spark.mllib.linalg.distributed.`RowMatrix[1] object). We use it's `computeSVD` method.The returned `org.apache.spark.mllib.linalg.`SingularValueDecomposition[2] object allows to retreive the coefficients matrix :

```
val u = svd.U
val s = svd.S
val v = svd.V
```

Notice :
 * The svd algorithm `computeSVD` method requires to iterate over the matrix multiple times. It is importand to cache the RDD before processing.
 * The V matrix is already transposed (should be called V' to be mathematicaly right)


# Querying the SVD model

asdas
