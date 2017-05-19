package bda.lsa

import bda.lsa.preprocessing.AssembleDocumentTermMatrix
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA => MLLIB_LDA}
import org.apache.spark.rdd.RDD

/**
  * date: 18.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */



object MllibLDA extends App {

  val path = if (args.length > 0) args(0) else "wikidump.xml"
  val k = if (args.length > 1) args(1).toInt else 100
  val numTerms = if (args.length > 2) args(2).toInt else 20000

  val spark = SparkSession.builder().
    config("spark.serializer", classOf[KryoSerializer].getName).
    master("local[*]").
    getOrCreate()

  val (docTermMatrix, termIds, docIds, termIdfs): (DataFrame, Array[String], Map[Long, String], Array[Double]) = {
    val assembleMatrix = new AssembleDocumentTermMatrix(spark)
    import assembleMatrix._
    val docTexts: Dataset[(String, String)] = parseWikipediaDump(path)
    documentTermMatrix(docTexts, "src/main/resources/stopwords.txt", numTerms)
  }

  import spark.implicits._
  val corpus: RDD[(Long, (MLLibVector, String))] =
    docTermMatrix.
      select("tfidfVec", "title").
      map{ r => (Vectors.fromML(r.getAs[MLVector](0)), r.getAs[String](1)) }.
      rdd.
      zipWithIndex.
      map(_.swap)

  corpus.cache()

  val model: DistributedLDAModel =
    new MLLIB_LDA().
      setK(k).
      run(corpus.mapValues(_._1)).
      asInstanceOf[DistributedLDAModel]


}

class LDAModelQueryer(spark : SparkSession, model: DistributedLDAModel, corpus: RDD[(Long, (MLLibVector, String))], vocabulary: Array[String]) {
  import spark.implicits._

  val docsWithTitlesDF = // [id: bigint, title: string]
    corpus.
      map(x => (x._1, x._2._2)).
      toDF("id", "title").
      cache()

  val docsTitleLookup : RDD[(Long, String)] = docsWithTitlesDF.rdd.map(r => (r.getAs[Long](0), r.getAs[String](1)))


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

  def topDocumentsForTopic(tid: Int, numDocs : Int = 10) = {
    val topDocs = model.topDocumentsPerTopic(numDocs)(tid)
    spark.sparkContext.
      parallelize(topDocs._1.zipWithIndex).
      join(docsTitleLookup).
      mapValues(_._2).
      collect()
  }


}


class LDAModelQueryer_(spark : SparkSession, model: DistributedLDAModel, corpus: RDD[(Long, (MLLibVector, String))], vocabulary: Array[String]) {
  import spark.implicits._


  val maxWordsPerTopic = 10
  val maxDocsPerTopic = 10
  val maxTopicPerDoc = 10


  //  lazy val topicsWithWordsAndWeight: scala.collection.immutable.Map[Int,(String, Double)]  =
  //    model.
  //      describeTopics(10).
  //      zipWithIndex.
  //      map( t => (t._2, (t._1._1.map(vocabulary(_)).mkString("-"), t._1._2.sum))).
  //      toMap
  lazy val topicsPerWeight: Array[(Int, Double)] =
    model.
      describeTopics().
      map(_._2.sum).
      zipWithIndex.
      map(_.swap).
      sortBy(-_._2)

  lazy val docsWithTitlesDF = // [id: bigint, title: string]
    corpus.
      map(x => (x._1, x._2._2)).
      toDF("id", "title").
      cache()

  lazy val topDocsPerTopic =
    model.
      topDocumentsPerTopic(maxDocsPerTopic)


  def getDocsLike(search: String) = {
    docsWithTitlesDF.select("id", "title").where($"title".like(s"%${search}%"))
  }


  def getTopicsWithWords(numWords: Int): Array[Array[String]] =
    model.
      describeTopics(numWords).
      map(_._1.map(vocabulary(_))) //.mkString("-"))

  def topTermsInTopConcepts(numTerms: Int = 10, numTopics: Int = 10) = {
    val topicsWithWords = getTopicsWithWords(numTerms)
    topicsPerWeight.take(numTopics).map(t => (t._1, topicsWithWords(t._1)))
  }

}
