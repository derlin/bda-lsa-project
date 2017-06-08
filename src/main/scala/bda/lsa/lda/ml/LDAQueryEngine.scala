package bda.lsa.lda.ml

import bda.lsa.Data
import org.apache.spark.ml.clustering.DistributedLDAModel
import org.apache.spark.ml.linalg.{Vector => ml_Vector, Vectors => ml_Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.collection.mutable.WrappedArray

/**
  * date: 19.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
class LDAQueryEngine(model: DistributedLDAModel, data: Data) {

  import data.spark.implicits._

  val docIdTitleRDD: RDD[(Long, String)] = data.dtm.select("id", "title").map {
    case Row(id: Long, title: String) => (id, title)
  }.rdd

  //  val termIdsRDD: RDD[(Int, String)] = data.spark.sparkContext.parallelize(data.termIds.zipWithIndex.map(_.swap))

  def describeTopicsWithWords(numWords: Int) = {
    model.
      describeTopics(numWords).
      select("termIndices").
      map { case Row(termIndices: Seq[Int]) => termIndices }.
      collect.
      map(_.map(data.termIds(_)).mkString(", ")) // doing this before throws NotSerializableException
  }


  def describeTopicsWithWordsAndStat(numWords: Int) = {
    model.
      describeTopics(numWords).
      select("termIndices", "termWeights").
      map { case Row(termIndices: Seq[Int], termWeights: Seq[Double]) =>
        termIndices.zip(termWeights)
      }.
      collect.
      map(_.map(t => (data.termIds(t._1), t._2)).mkString(", "))
  }


  def topTopicsForTerm(wid: Int) = {
    model.topicsMatrix.rowIter.
      drop(wid).next.toArray.
      zipWithIndex.
      sortBy(-_._1)
  }

// TODO other queries should be reimplemented to work with DataFrames
//
//  def topTopicsForDocument(id: Long, numTopics: Int = 10): Array[(Int, Double)] = {
//    model.topicsMatrix(numTopics).filter(_._1 == id).map(r => r._2 zip r._3).first.sortBy(-_._2)
//  }
//
//  def topDocumentsForTopic(tid: Int, numDocs: Int = 10) = {
//    val topDocs = model.topDocumentsPerTopic(numDocs)(tid)
//    data.spark.sparkContext.
//      parallelize(topDocs._1.zip(topDocs._2)).
//      join(docIdTitleRDD).
//      sortBy(_._2._1, ascending = false).
//      collect()
//  }


  //
  //  def topTopicsForWord_(wid: Int) = {
  //    // see https://gist.github.com/alex9311/774089d936eee505d7832c6df2eb597d
  //    val term = mllib_Vectors.sparse(data.termIds.length, Array(wid -> 1.0).toSeq)
  //    val topicDistrib = model.toLocal.topicDistribution(term).toArray.zipWithIndex.sortBy(-_._1)
  //    topicDistrib
  //  }
}