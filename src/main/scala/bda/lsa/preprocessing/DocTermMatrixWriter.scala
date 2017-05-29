package bda.lsa.preprocessing

import org.apache.spark.sql.{Dataset, SparkSession}
import bda.lsa._
import org.apache.spark.{SparkConf, SparkContext}

/**
  * This class run [[bda.lsa.preprocessing.TextToIDF.preprocess]] and saves the result
  * as parquet file (see [[bda.lsa.saveData]]).
  * <p>
  * Command line arguments:
  *
  * - numTerms: the number of words in the dictionary (the terms with the max frequency are kept), default to 2000
  * - percent: the percentage of articles to keep, useful for sampling, default to -1 (i.e. no sampling)
  * - numDocs: use it in conjonction with percent to limit the number of documents to keep
  *
  * <p>
  * context: BDA - Master MSE,
  * date: 18.05.17
  *
  * @author Lucy Linder [lucy.derlin@gmail.com]
  */
object DocTermMatrixWriter extends App {
  val numTerms = if (args.length > 0) args(0).toInt else 2000
  val percent = if (args.length > 1) args(1).toFloat else -1
  val numDocs = if (args.length > 2) args(2).toInt else -1

  val spark = SparkSession.builder().appName("DocTermMatrixWriter").getOrCreate()

  import spark.implicits._

  println("loading docTexts.")
  val docTexts = spark.sqlContext.read.parquet(bda.lsa.wikidumpParquetPath).map {
    r => (r.getAs[String](0), r.getAs[String](1))
  }
  println("filtering docTexts.")
  var filtered = filterTitles(spark, docTexts)

  if (percent > 0f && percent < 1f) {
    println(s"selecting ${percent}% documents.")
    //val percent = (numDocs + 0.005) / filtered.count().toFloat
    filtered = filtered.randomSplit(Array(percent, 1 - percent))(0)
    if (numDocs > 0) {
      filtered = filtered.limit(numDocs)
    }
  } 

    println("preprocess using spark-nlp")
    val (docTermMatrix, termIds, docIds, termIdfs) = TextToIDF.preprocess(spark, filtered, numTerms)
    bda.lsa.saveData(Data(spark, docTermMatrix, termIds, docIds, termIdfs))


  /**
    * Remove articles with a title containing `List`, `Category` or a number.
    * @param spark  the spark context
    * @param docTexts the DataSet[(title: String, content: String)]
    * @return the filtered dataset
    */
  def filterTitles(spark: SparkSession, docTexts: Dataset[(String, String)]): Dataset[(String, String)] =  {
    import org.apache.spark.sql.functions._
    import spark.implicits._
    docTexts.
      where(not($"_1".rlike("(^(List)|(Category))|([0-9]+)"))). // remove Lists and articles with numbers in title
      where(length($"_1") > 1) // remove articles like "A"
  }

}
