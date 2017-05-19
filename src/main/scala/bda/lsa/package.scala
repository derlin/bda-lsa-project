package bda


import org.apache.spark.mllib.linalg.{Vector => MLLIBVector}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.rdd.RDD

/**
  * date: 19.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
package object lsa {

  lazy val properties = {
    val prop = new java.util.Properties()
    prop.load(new java.io.FileInputStream("config.properties"))
    prop
  }


  val STOPWORDS_PATH = "src/main/resources/stopwords.txt"

  def baseDir = properties.getProperty("path.base")
  var wikidumpPath = properties.getProperty("path.wikidump")
  def wikidumpParquetPath = baseDir + "/wikidump-parquet"
  def wikidumpMatrixPath = baseDir + "/matrix"


  def docTermMatrixToRDD(docTermMatrix: DataFrame, featuresCol: String = "tfidfVec"): RDD[MLLIBVector] = {
    docTermMatrix.select(featuresCol).rdd.map { row =>
      Vectors.fromML(row.getAs[MLVector](featuresCol))
    }
  }

  def saveData(spark: SparkSession, docTermMatrix: DataFrame, termIds: Array[String], docIds: Map[Long, String]): Unit = {
    import spark.implicits._
    docTermMatrix.write.parquet(wikidumpMatrixPath + "/docTermMatrix")
    spark.sparkContext.parallelize(termIds, 1).toDF().write.parquet(wikidumpMatrixPath + "/termIds")
    spark.sparkContext.parallelize(docIds.toSeq).toDF.write.parquet(wikidumpMatrixPath + "/docIds")
  }

  def getData(spark: SparkSession): (DataFrame, Array[String], Map[Long, String]) = {
    import spark.implicits._
    val docTermMatrix = spark.sqlContext.read.parquet(wikidumpMatrixPath + "/docTermMatrix")
    val termIds = spark.sqlContext.read.parquet(wikidumpMatrixPath + "/termIds").map(_.getAs[String](0)).collect
    val docIds = spark.sqlContext.read.parquet(wikidumpMatrixPath + "/docIds").map {
      r => (r.getAs[Long](0), r.getAs[String](1))
    }.collect.toMap

    (docTermMatrix, termIds, docIds)
  }
}
