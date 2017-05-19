package bda.lsa.lda.mllib

import org.apache.spark.mllib.clustering.DistributedLDAModel
import org.apache.spark.mllib.linalg.{Vector => MLLIBVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
  * date: 19.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
class LDAQueryEngine(spark: SparkSession, model: DistributedLDAModel, corpus: RDD[(Long, (MLLIBVector, String))], vocabulary: Array[String]) {

  import spark.implicits._

  val docsWithTitlesDF = // [id: bigint, title: string]
    corpus.
      map(x => (x._1, x._2._2)).
      toDF("id", "title").
      cache()

  val docsTitleLookup: RDD[(Long, String)] = docsWithTitlesDF.rdd.map(r => (r.getAs[Long](0), r.getAs[String](1)))


  def describeTopicsWithWords(numWords: Int) = {
    model.
      describeTopics(numWords).
      map { topic => topic._1.map(vocabulary(_)) }
  }

  def findDocs(search: String): Array[(Long, String)] =
    docsWithTitlesDF.
      select("id", "title").
      where($"title".like("%" + search + "%")).
      map(r => (r.getAs[Long](0), r.getAs[String](1))).
      collect()

  def topTopicsForDocument(id: Long, numTopics: Int = 10): Array[Int] = {
    model.topTopicsPerDocument(numTopics).filter(_._1 == id).map(_._2.toArray).first
  }

  def topDocumentsForTopic(tid: Int, numDocs: Int = 10) = {
    val topDocs = model.topDocumentsPerTopic(numDocs)(tid)
    spark.sparkContext.
      parallelize(topDocs._1.zipWithIndex).
      join(docsTitleLookup).
      mapValues(_._2).
      collect()
  }
}