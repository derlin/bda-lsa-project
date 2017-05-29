package bda.lsa


import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
  * date: 21.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
case class Data(
                 spark: SparkSession,
                 dtm: DataFrame,
                 termIds: Array[String],
                 tfIdfs: Array[Double]) {

  import spark.implicits._
  lazy val termsLookup = termIds.zipWithIndex.toMap

  lazy val docIds: RDD[(Long, String)] = dtm.select("id", "title").map{
    case Row(id: Long, title: String) => (id, title)
  }.rdd

  def docTitle(id: Long) : String = {
    val res = docIds.lookup(id)
    if(res.isEmpty) "" else res.head
  }

  def findDocsByTitle(search: String, numDocs: Int = 15): Array[(Long, String)] = {
    import spark.implicits._
    import org.apache.spark.sql.functions._
    docIds.
      toDF("id", "title").
      where(lower($"title").like("%" + search + "%")).
      map { case Row(id: Long, title: String) => (id, title) }.
      take(numDocs)
  }

  def findDocsByBody(term: String, numDocs: Int = 10): Array[(Long, String, Double)] =
    findDocsByBody(termsLookup(term), numDocs)


  def findDocsByBody(termId: Int, numDocs: Int): Array[(Long, String, Double)] = {
    import org.apache.spark.sql.functions._
    import spark.implicits._
    dtm.select("title", "tfIdfVec", "id").
      map {
        case Row(title: String, v: org.apache.spark.ml.linalg.SparseVector, id: Long) => (id, title, v(termId))
      }.
      orderBy(desc("_2")).
      take(numDocs)
  }
}
