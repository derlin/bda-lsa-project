package bda.lsa.lda.ml

import bda.lsa.Data
import org.apache.spark.ml.clustering.DistributedLDAModel
import org.apache.spark.ml.linalg.{Vector => ml_Vector, Vectors => ml_Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import org.apache.spark.sql.functions._

/**
  * Class for interacting with an ML LDA model. Very useful in a spark shell. See the wiki on github for examples on how to use.
  * <p>
  * context: BDA - Master MSE,
  * date: 19.05.17
  *
  * @author Lucy Linder [lucy.derlin@gmail.com]
  */
class LDAQueryEngine(model: DistributedLDAModel, data: Data) {

  import data.spark.implicits._

  val docIdTitleRDD: RDD[(Long, String)] = data.dtm.select("id", "title").map {
    case Row(id: Long, title: String) => (id, title)
  }.rdd

  val topicDistUDF = udf((v: ml_Vector, i: Int) => v(i))

  /**
    * transformed is a dataframe with columns:
    *
    *  - `id` : the document id for the row
    *  - `title`: the document title
    *  - `terms`: the list of terms in this document
    *  - `termFreqs`: the frequency of each word in this document
    *  - `tfIdfVec`: the TF-IDF for this document
    *  - `topicDistribution`: LDA --> the weight/importance of this document for each topic.
    */
  val transformed = model.transform(data.dtm) // apply the model to the data
  transformed.cache()

  //  val termIdsRDD: RDD[(Int, String)] = data.spark.sparkContext.parallelize(data.termIds.zipWithIndex.map(_.swap))

  /**
    * The top words for each topic, as a string separated by comma
    *
    * @param numWords the number of top words
    * @return the array of top words as string
    */
  def describeTopicsWithWords(numWords: Int): Array[String] = {
    model.
      describeTopics(numWords).
      select("termIndices").
      map { case Row(termIndices: Seq[Int]) => termIndices }.
      collect.
      map(_.map(data.termIds(_)).mkString(", ")) // doing this before throws NotSerializableException
  }

  /**
    * Same as [[describeTopicsWithWords]], but also print the weight of each word for the topic
    *
    * @param numWords the number of top words
    * @return the array of top words as string
    */
  def describeTopicsWithWordsAndStat(numWords: Int): Array[String] = {
    model.
      describeTopics(numWords).
      select("termIndices", "termWeights").
      map { case Row(termIndices: Seq[Int], termWeights: Seq[Double]) =>
        termIndices.zip(termWeights)
      }.
      collect.
      map(_.map(t => (data.termIds(t._1), t._2)).mkString(", "))
  }

  /**
    * Get the top topics for a given document.
    *
    * @param id        the document id (see [[Data.docIds]])
    * @param numTopics the number of topics to return
    * @return an array of `(topicId, weight)`
    */
  def topTopicsForDocument(id: Long, numTopics: Int = 10): Array[(Int, Double)] = {
    val topicDist = transformed.
      select("topicDistribution").
      where($"id" === id).
      take(1).
      map({ case Row(vec: ml_Vector) => vec.toArray })
    
    if (topicDist.length == 0) null
    else topicDist.head.zipWithIndex.sortBy(-_._1).map(_.swap).take(numTopics)
  }

  /**
    * Get the top documents for a given topic.
    *
    * @param tid     the topic id
    * @param numDocs the number of documents to return
    * @return an array of `(id, title, weight)`
    */
  def topDocumentsForTopic(tid: Int, numDocs: Int = 10): Array[(Long, String, Double)] = {
    transformed.
      select("id", "title", "topicDistribution").
      orderBy(topicDistUDF($"topicDistribution", lit(tid))).
      take(numDocs).
      map { case Row(id: Long, title: String, v: ml_Vector) => (id, title, v(tid)) }
  }


  /**
    * Return the best topics for a given term
    *
    * @param wid the term id (see [[Data.termIds]])
    * @return an array of tuples `(topicId, weight)`
    */
  def topTopicsForTerm(wid: Int): Array[(Int, Double)] = {
    model.topicsMatrix.rowIter.
      drop(wid).next.toArray.
      zipWithIndex.
      sortBy(-_._1).
      map(_.swap)
  }

}