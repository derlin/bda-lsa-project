package bda.lsa.svd


import bda.lsa.Data
import breeze.linalg.{DenseMatrix => BDenseMatrix, SparseVector => BSparseVector}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vectors, Vector => mllib_Vector}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}

import scala.collection.Map

/**
  * This class implements useful methods to query and discover an SVD model.
  * It is inspired by the book Advanced Spark Programing v2, chapter 6, but has been modified to
  * work with [[org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix]] instead of
  * [[org.apache.spark.mllib.linalg.distributed.RowMatrix]]. This way, the document id is embedded in the matrix
  * instead of relying on a "zipWithUniqueId trick" that works only when we do everything
  * (preprocessing, modeling, querying) inside the same spark session.
  * <p>
  * context: BDA - Master MSE,
  * date: 18.05.17
  *
  * @author Lucy Linder [lucy.derlin@gmail.com]
  */
class SVDQueryEngine(val model: SingularValueDecomposition[IndexedRowMatrix, Matrix], val data: Data) {


  val VS: BDenseMatrix[Double] = multiplyByDiagonalMatrix(model.V, model.s)
  val normalizedVS: BDenseMatrix[Double] = rowsNormalized(VS)
  val US: IndexedRowMatrix = multiplyByDiagonalRowMatrix(model.U, model.s)
  val normalizedUS: IndexedRowMatrix = distributedRowsNormalized(US)

  val idTerms: Map[String, Int] = data.termIds.zipWithIndex.toMap

  /**
    * Finds the product of a dense matrix and a diagonal matrix represented by a vector.
    * Breeze doesn't support efficient diagonal representations, so multiply manually.
    */
  def multiplyByDiagonalMatrix(mat: Matrix, diag: mllib_Vector): BDenseMatrix[Double] = {
    val sArr = diag.toArray
    new BDenseMatrix[Double](mat.numRows, mat.numCols, mat.toArray)
      .mapPairs { case ((r, c), v) => v * sArr(c) }
  }

  /**
    * Finds the product of a distributed matrix and a diagonal matrix represented by a vector.
    */
  def multiplyByDiagonalRowMatrix(mat: IndexedRowMatrix, diag: mllib_Vector): IndexedRowMatrix = {
    val sArr = diag.toArray
    new IndexedRowMatrix(mat.rows.map { r =>
      val vecArr = r.vector.toArray
      val newArr = (0 until r.vector.size).toArray.map(i => vecArr(i) * sArr(i))
      IndexedRow(r.index, Vectors.dense(newArr))
    })
  }

  def describeTopicsWithWords(numTerms: Int = 10) = {
    model.V.transpose.rowIter.map {
      case v: mllib_Vector => v.toArray.
        zipWithIndex.
        sortBy(-_._1).
        take(numTerms).
        map(t => data.termIds(t._2)).
        mkString(", ")
    }.toArray
  }

  def topTermsForTopic(k: Int, numTerms: Int = 10) = {
    val termWeights = model.V.transpose.rowIter.slice(k, k + 1).toArray
    val sorted = termWeights(0).toArray.zipWithIndex.sortBy(-_._1)
    sorted.take(numTerms)
  }

  def topDocumentsForTopic(tid: Int, numDocs: Int = 10) = {
    model.U.rows.map(t => (t.vector.toArray(tid), t.index)).top(numDocs)
  }


  /**
    * Returns a matrix where each row is divided by its length.
    */
  def rowsNormalized(mat: BDenseMatrix[Double]): BDenseMatrix[Double] = {
    val newMat = new BDenseMatrix[Double](mat.rows, mat.cols)
    for (r <- 0 until mat.rows) {
      val length = math.sqrt((0 until mat.cols).map(c => mat(r, c) * mat(r, c)).sum)
      (0 until mat.cols).foreach(c => newMat.update(r, c, mat(r, c) / length))
    }
    newMat
  }

  /**
    * Returns a distributed matrix where each row is divided by its length.
    */
  def distributedRowsNormalized(mat: IndexedRowMatrix): IndexedRowMatrix = {
    new IndexedRowMatrix(mat.rows.map { r =>
      val array = r.vector.toArray
      val length = math.sqrt(array.map(x => x * x).sum)
      IndexedRow(r.index, Vectors.dense(array.map(_ / length)))
    })
  }

  /**
    * Finds docs relevant to a term. Returns the doc IDs and scores for the docs with the highest
    * relevance scores to the given term.
    */
  def topDocsForTerm(termId: Int): Seq[(Double, Long)] = {
    val rowArr = (0 until model.V.numCols).map(i => model.V(termId, i)).toArray
    val rowVec = Matrices.dense(rowArr.length, 1, rowArr)

    // Compute scores against every doc
    val docScores = US.multiply(rowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(t => (t.vector.toArray(0), t.index))
    allDocWeights.top(10)
  }

  /**
    * Finds terms relevant to a term. Returns the term IDs and scores for the terms with the highest
    * relevance scores to the given term.
    */
  def topTermsForTerm(termId: Int): Seq[(Double, Int)] = {
    // Look up the row in VS corresponding to the given term ID.
    val rowVec = normalizedVS(termId, ::).t

    // Compute scores against every term
    val termScores = (normalizedVS * rowVec).toArray.zipWithIndex

    // Find the terms with the highest scores
    termScores.sortBy(-_._1).take(10)
  }

  /**
    * Finds docs relevant to a doc. Returns the doc IDs and scores for the docs with the highest
    * relevance scores to the given doc.
    */
  def topDocsForDoc(docId: Long): Seq[(Double, Long)] = {
    // Look up the row in US corresponding to the given doc ID.
    val docRowArr = normalizedUS.rows.map(t => (t.index, t.vector)).lookup(docId).head.toArray
    val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)

    // Compute scores against every doc
    val docScores = normalizedUS.multiply(docRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(t => (t.vector.toArray(0), t.index))

    // Docs can end up with NaN score if their row in U is all zeros.  Filter these out.
    allDocWeights.filter(!_._1.isNaN).top(10)
  }

  /**
    * Builds a term query vector from a set of terms.
    */
  def termsToQueryVector(terms: Seq[String]): BSparseVector[Double] = {
    val indices = terms.map(idTerms(_)).toArray
    val values = indices.map(data.tfIdfs(_))
    new BSparseVector[Double](indices, values, idTerms.size)
  }

  /**
    * Finds docs relevant to a term query, represented as a vector with non-zero weights for the
    * terms in the query.
    */
  def topDocsForTermQuery(query: BSparseVector[Double]): Seq[(Double, Long)] = {
    val breezeV = new BDenseMatrix[Double](model.V.numRows, model.V.numCols, model.V.toArray)
    val termRowArr = (breezeV.t * query).toArray

    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    // Compute scores against every doc
    val docScores = US.multiply(termRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(t => (t.vector.toArray(0), t.index))
    allDocWeights.top(10)
  }

  def printTopTermsForTerm(term: Int): Unit = {
    val idWeights = topTermsForTerm(term)
    println(idWeights.map { case (score, id) => (data.termIds(id), score) }.mkString("\n"))
  }

  def printTopDocsForDoc(doc: Long): Unit = {
    val idWeights = topDocsForDoc(doc)
    println(idWeights.map { case (score, id) => (data.docTitle(id), score) }.mkString("\n"))
  }

  def printTopDocsForTerm(term: Int): Unit = {
    val idWeights = topDocsForTerm(term)
    println(idWeights.map { case (score, id) => (data.docTitle(id), score) }.mkString("\n"))
  }

  def printTopDocsForTermQuery(terms: Seq[String]): Unit = {
    val queryVec = termsToQueryVector(terms)
    val idWeights = topDocsForTermQuery(queryVec)
    println(idWeights.map { case (score, id) => (data.docTitle(id), score) }.mkString("\n"))
  }
}