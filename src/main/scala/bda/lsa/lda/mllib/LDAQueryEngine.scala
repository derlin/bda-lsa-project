package bda.lsa.lda.mllib

import bda.lsa.Data
import org.apache.spark.mllib.clustering.DistributedLDAModel
import org.apache.spark.mllib.linalg.{Vector => mllib_Vector, Vectors => mllib_Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}

/**
  * date: 19.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
class LDAQueryEngine(model: DistributedLDAModel, data: Data) {

  val docIdTitleRDD: RDD[(Long, String)] = data.docIdsLookup.rdd.map {
    case Row(id: Long, title: String) => (id, title)
  }


  def describeTopicsWithWords(numWords: Int) = {
    model.
      describeTopics(numWords).
      map { topic => topic._1.map(data.termIds(_)) }
  }

  def topTopicsForDocument(id: Long, numTopics: Int = 10): Array[Int] = {
    model.topTopicsPerDocument(numTopics).filter(_._1 == id).map(_._2.toArray).first
  }

  def topDocumentsForTopic(tid: Int, numDocs: Int = 10) = {
    val topDocs = model.topDocumentsPerTopic(numDocs)(tid)
    data.spark.sparkContext.
      parallelize(topDocs._1.zipWithIndex).
      join(docIdTitleRDD).
      mapValues(_._2).
      collect()
  }

  def topTopicsForTerm(wid: Int) = {
    model.topicsMatrix.rowIter.
      drop(wid).next.toArray.
      zipWithIndex.
      sortBy(-_._1)
  }

  def topTopicsForWord_(wid: Int) = {
    // see https://gist.github.com/alex9311/774089d936eee505d7832c6df2eb597d
    val term = mllib_Vectors.sparse(data.termIds.length, Array((wid -> 1.0)).toSeq)
    val topicDistrib = model.toLocal.topicDistribution(term).toArray.zipWithIndex.sortBy(-_._1)
    topicDistrib
  }
}