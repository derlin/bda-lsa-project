package bda.lsa.svd


import bda.lsa.Data
import breeze.linalg.{DenseMatrix => BDenseMatrix, SparseVector => BSparseVector}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vectors, Vector => MLLibVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import scala.collection.Map

/**
  * date: 28.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
class SVDQueryEngine(val model: SingularValueDecomposition[RowMatrix, Matrix], val data: Data) {


  val VS: BDenseMatrix[Double] = multiplyByDiagonalMatrix(model.V, model.s)
  val normalizedVS: BDenseMatrix[Double] = rowsNormalized(VS)
  val US: RowMatrix = multiplyByDiagonalRowMatrix(model.U, model.s)
  val normalizedUS: RowMatrix = distributedRowsNormalized(US)

  val idTerms: Map[String, Int] = data.termIds.zipWithIndex.toMap
  val idDocs: Map[String, Long] = data.docIds.map(_.swap)

  /**
    * Finds the product of a dense matrix and a diagonal matrix represented by a vector.
    * Breeze doesn't support efficient diagonal representations, so multiply manually.
    */
  def multiplyByDiagonalMatrix(mat: Matrix, diag: MLLibVector): BDenseMatrix[Double] = {
    val sArr = diag.toArray
    new BDenseMatrix[Double](mat.numRows, mat.numCols, mat.toArray)
      .mapPairs { case ((r, c), v) => v * sArr(c) }
  }

  /**
    * Finds the product of a distributed matrix and a diagonal matrix represented by a vector.
    */
  def multiplyByDiagonalRowMatrix(mat: RowMatrix, diag: MLLibVector): RowMatrix = {
    val sArr = diag.toArray
    new RowMatrix(mat.rows.map { vec =>
      val vecArr = vec.toArray
      val newArr = (0 until vec.size).toArray.map(i => vecArr(i) * sArr(i))
      Vectors.dense(newArr)
    })
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
  def distributedRowsNormalized(mat: RowMatrix): RowMatrix = {
    new RowMatrix(mat.rows.map { vec =>
      val array = vec.toArray
      val length = math.sqrt(array.map(x => x * x).sum)
      Vectors.dense(array.map(_ / length))
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
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
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
    val docRowArr = normalizedUS.rows.zipWithUniqueId.map(_.swap).lookup(docId).head.toArray
    val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)

    // Compute scores against every doc
    val docScores = normalizedUS.multiply(docRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId

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
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  def printTopTermsForTerm(term: Int): Unit = {
    val idWeights = topTermsForTerm(term)
    println(idWeights.map { case (score, id) => (data.termIds(id), score) }.mkString(", "))
  }

  def printTopDocsForDoc(doc: Int): Unit = {
    val idWeights = topDocsForDoc(doc)
    println(idWeights.map { case (score, id) => (data.docIds(id), score) }.mkString(", "))
  }

  def printTopDocsForTerm(term: Int): Unit = {
    val idWeights = topDocsForTerm(term)
    println(idWeights.map { case (score, id) => (data.docIds(id), score) }.mkString(", "))
  }

  def printTopDocsForTermQuery(terms: Seq[String]): Unit = {
    val queryVec = termsToQueryVector(terms)
    val idWeights = topDocsForTermQuery(queryVec)
    println(idWeights.map { case (score, id) => (data.docIds(id), score) }.mkString(", "))
  }
}