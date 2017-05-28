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
                 docIds:  Array[String],
                 tfIdfs: Array[Double]) {

  lazy val termsLookup = termIds.zipWithIndex.toMap

  lazy val docIdsLookup: Map[String, Long] = docIds.zipWithIndex.map(t => (t._1, t._2.toLong)).toMap
  lazy val docsDF: DataFrame =
    spark.sqlContext.createDataFrame(docIdsLookup.toSeq).toDF("title", "id")// TODO

  def findDocsByTitle(search: String, numDocs: Int = 15): Array[(Long, String)] = {
    import spark.implicits._
    import org.apache.spark.sql.functions._
    docsDF.
      select("id", "title").
      where(lower($"title").like("%" + search + "%")).
      map { case Row(id: Long, title: String) => (id, title) }.
      take(numDocs)
  }

  def findDocsByBody(term: String, numDocs: Int = 10): Array[(Long, String, Double)] =
    findDocsByBody(termsLookup(term), numDocs)


  def findDocsByBody(termId: Int, numDocs: Int): Array[(Long, String, Double)] = {
    import org.apache.spark.sql.functions._
    import spark.implicits._
    dtm.withColumn("id", monotonically_increasing_id).
      map {
        case Row(title: String, v: org.apache.spark.ml.linalg.SparseVector, id: Long) => (id, title, v(termId))
      }.
      orderBy(desc("_2")).
      take(numDocs)
  }
}
