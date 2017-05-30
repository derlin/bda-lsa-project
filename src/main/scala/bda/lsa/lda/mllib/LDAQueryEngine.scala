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

  import data.spark.implicits._
  val docIdTitleRDD: RDD[(Long, String)] = data.dtm.select("id", "title").map{
    case Row(id: Long, title: String) => (id, title)
  }.rdd


  def describeTopicsWithWords(numWords: Int) = {
    model.
      describeTopics(numWords).
      map { topic => topic._1.map(data.termIds(_)) }.
      map(_.mkString(", "))
  }

  def describeTopicsWithWordsAndStat(numWords: Int) = {
    model.
      describeTopics(numWords).
      map { topic => topic._1.map(data.termIds(_)).zip(topic._2) }.
      map(_.mkString(", "))
  }

  def topTopicsForDocument(id: Long, numTopics: Int = 10): Array[(Int, Double)] = {
    model.topTopicsPerDocument(numTopics).filter(_._1 == id).map(r => r._2 zip r._3).first.sortBy(-_._2)
  }

  def topDocumentsForTopic(tid: Int, numDocs: Int = 10) = {
    val topDocs = model.topDocumentsPerTopic(numDocs)(tid)
    data.spark.sparkContext.
      parallelize(topDocs._1.zip(topDocs._2)).
      join(docIdTitleRDD).
      sortBy(_._2._1, ascending = false).
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
    val term = mllib_Vectors.sparse(data.termIds.length, Array(wid -> 1.0).toSeq)
    val topicDistrib = model.toLocal.topicDistribution(term).toArray.zipWithIndex.sortBy(-_._1)
    topicDistrib
  }
}