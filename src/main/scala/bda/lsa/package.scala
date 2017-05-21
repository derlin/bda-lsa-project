package bda


import org.apache.spark.mllib.linalg.{Vector => mllib_Vector}
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

  def wikidumpParquetPath = properties.getProperty("path.wikidump.parquet", baseDir + "/wikidump-parquet")

  def wikidumpMatrixPath = properties.getProperty("path.matrix", baseDir + "/matrix")


  def docTermMatrixToRDD(docTermMatrix: DataFrame, featuresCol: String = "tfidfVec"): RDD[mllib_Vector] = {
    docTermMatrix.select(featuresCol).rdd.map { row =>
      Vectors.fromML(row.getAs[MLVector](featuresCol))
    }
  }

  /**
    * Create an RDD corpus from the docTermMatrix. This is useful for SVD and MLLIB_LDA.
    *
    * @param spark         the current sparkSession
    * @param docTermMatrix the docTermMatrix dataset (see [[getData]])
    * @param featuresCol   (optional) the tfidfVec vectors column, default to `tfidfVec`
    * @param titleCol      (optional) the document title column, default to `title`
    * @return a PairRDD (id, (freqs-vector, title)): the key corresponds to the document's id (its row index in docTermMatrix),
    *         the value is a tuple made of the term frequencies (tfidfVec) as mllib.Vectors and the document's title.
    */
  def docTermMatrixToCorpusRDD(spark: SparkSession, docTermMatrix: DataFrame, featuresCol: String = "tfidfVec", titleCol: String = "title"):
  RDD[(Long, (mllib_Vector, String))] = {
    import spark.implicits._
    docTermMatrix.
      select("tfidfVec", "title").
      map { r => (Vectors.fromML(r.getAs[MLVector](0)), r.getAs[String](1)) }.
      rdd.
      zipWithIndex.
      map(_.swap)
  }

  /**
    * Save the data into a folder in parquet format.
    * The actual path is determined by the `path.base` property defined in `config.properties`.
    * Each parameter is saved in a subfolder, so in the end we get `path.base/docTermMatrix`, `path.base/termIds` and `path.base/docIds`.
    * To load back the data, see [[getData]]
    *
    * @param spark
    * @param docTermMatrix
    * @param termIds
    * @param docIds
    * @param idfs
    */
  def saveData(spark: SparkSession, docTermMatrix: DataFrame, termIds: Array[String], docIds: Map[Long, String], idfs: Array[Double]): Unit = {
    import spark.implicits._
    docTermMatrix.write.parquet(wikidumpMatrixPath + "/docTermMatrix")
    spark.sparkContext.parallelize(termIds, 1).toDF().write.parquet(wikidumpMatrixPath + "/termIds")
    spark.sparkContext.parallelize(docIds.toSeq).toDF.write.parquet(wikidumpMatrixPath + "/docIds")
    spark.sparkContext.parallelize(idfs).toDF.write.parquet(wikidumpMatrixPath + "/idfs")
    // TODO: don't save docIds, since we can get them back using
    // docTermMatrix.select("title").rdd.zipWithUniqueId.map(_.swap).collect.toMap
  }

  /**
    * Load the data previously saved by [[getData]]. The actual path used is determined by the  `path.base` property defined in `config.properties`.
    *
    * @param spark the current spark session
    * @return  a tuple3 made of:
    *          - `docTermMatrix`: a Dataset with two columns: `title:String` and `tfidfVec:ml.Vector`
    *          - `termIds`: the array of terms in the dictionary
    *          - `docIds`: a map mapping a document id with a document title
    *          - `idfs`: tf-idfs array, used by the original LSA Query Engine
    */
  def getData(spark: SparkSession): (DataFrame, Array[String], Map[Long, String], Array[Double]) = {
    import spark.implicits._
    val docTermMatrix = spark.sqlContext.read.parquet(wikidumpMatrixPath + "/docTermMatrix")
    val termIds = spark.sqlContext.read.parquet(wikidumpMatrixPath + "/termIds").map(_.getAs[String](0)).collect
    val docIds = spark.sqlContext.read.parquet(wikidumpMatrixPath + "/docIds").map {
      r => (r.getAs[Long](0), r.getAs[String](1))
    }.collect.toMap
    val idfs = spark.sqlContext.read.parquet(wikidumpMatrixPath + "/idfs").map(_.getAs[Double](0)).collect
    (docTermMatrix, termIds, docIds, idfs)
  }
}
