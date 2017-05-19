package bda.lsa.preprocessing

import bda.lsa.preprocessing.XmlToParquetWriter.args
import org.apache.spark.sql.{Dataset, SparkSession}
import bda.lsa._

/**
  * date: 19.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
object DocTermMatrixWriter extends App {
  val numTerms = if (args.length > 0) args(0).toInt else 2000

  val spark = SparkSession.builder().master(properties.getProperty("spark.master")).getOrCreate()

  import spark.implicits._

  val docTexts = spark.sqlContext.read.parquet(bda.lsa.wikidumpParquetPath).map {
    r => (r.getAs[String](0), r.getAs[String](1))
  }

  val assembleMatrix = new AssembleDocumentTermMatrix(spark)
  val (docTermMatrix, termIds, docIds, termIdfs) = assembleMatrix.documentTermMatrix(docTexts, bda.lsa.STOPWORDS_PATH, numTerms)

  bda.lsa.saveData(spark, docTermMatrix, termIds, docIds)
}
