package bda.lsa.preprocessing

import org.apache.spark.sql.{Dataset, SparkSession}
import bda.lsa._
import org.apache.spark.{SparkConf, SparkContext}

/**
  * date: 19.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
object DocTermMatrixWriter extends App {
  val numTerms = if (args.length > 0) args(0).toInt else 2000
  val percent = if (args.length > 1) args(1).toFloat else -1
  val numDocs = if (args.length > 2) args(2).toInt else -1
  val mode = if (args.length > 3 && args(3).startsWith("orig")) "orig" else "mine" // mine or orig

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

  if (mode == "orig") {
    println("preprocess using orig")
    val assembleMatrix = new AssembleDocumentTermMatrix(spark)
    val (docTermMatrix, termIds, docIds, termIdfs) = assembleMatrix.documentTermMatrix(filtered, bda.lsa.STOPWORDS_PATH, numTerms)
    bda.lsa.saveData(Data(spark, docTermMatrix, termIds, docIds, termIdfs))

  } else {
    println("preprocess using mine")
    val (docTermMatrix, termIds, docIds, termIdfs) = TextToIDF.preprocess(spark, filtered, numTerms)
    bda.lsa.saveData(Data(spark, docTermMatrix, termIds, docIds, termIdfs))
  }

  def filterTitles(spark: SparkSession, docTexts: Dataset[(String, String)]) = {
    import org.apache.spark.sql.functions._
    import spark.implicits._
    docTexts.
      where(not($"_1".rlike("(^(List)|(Category))|([0-9]+)"))). // remove Lists and articles with numbers in title
      where(length($"_1") > 1) // remove articles like "A"
  }

}
