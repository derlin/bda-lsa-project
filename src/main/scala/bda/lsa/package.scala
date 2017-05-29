package bda


import org.apache.spark.mllib.linalg.{Vector => mllib_Vector, Vectors => mllib_Vectors}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.linalg.{Vector => ml_Vector}
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
      mllib_Vectors.fromML(row.getAs[ml_Vector](featuresCol))
    }
  }

  /**
    * Create an RDD corpus from the docTermMatrix. This is useful for SVD and MLLIB_LDA.
    *
    * @param spark the current sparkSession
    * @param data  the data
    * @return a PairRDD (id, (freqs-vector, title)): the key corresponds to the document's id (its row index in docTermMatrix),
    *         the value is a tuple made of the term frequencies (tfidfVec) as mllib.Vectors and the document's title.
    */
  def docTermMatrixToCorpusRDD(spark: SparkSession, data: Data):
  RDD[(Long, (mllib_Vector, String))] = {
    val idDocs = data.docIds.zipWithIndex.toMap
    import spark.implicits._
    data.dtm.
      select("tfidfVec", "title").
      map { case Row(v: ml_Vector, t: String) => (idDocs(t).toLong, (mllib_Vectors.fromML(v), t)) }.
      rdd
  }

  /**
    * Save the data into a folder in parquet format.
    * The actual path is determined by the `path.base` property defined in `config.properties`.
    * Each parameter is saved in a subfolder, so in the end we get `path.base/docTermMatrix`, `path.base/termIds` and `path.base/docIds`.
    * To load back the data, see [[getData]]
    *
    * @param data the data to save
    */
  def saveData(data: Data): Unit = {
    val ctx = data.spark.sparkContext
    import data.spark.implicits._
    data.dtm.write.parquet(wikidumpMatrixPath + "/docTermMatrix")
    ctx.parallelize(data.termIds, 1).toDF().write.parquet(wikidumpMatrixPath + "/termIds")
    ctx.parallelize(data.docIds.toSeq, 1).toDF.write.parquet(wikidumpMatrixPath + "/docIds")
    ctx.parallelize(data.tfIdfs, 1).toDF.write.parquet(wikidumpMatrixPath + "/idfs")
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
  def getData(spark: SparkSession): Data = {
    import spark.implicits._
    val docTermMatrix = spark.sqlContext.read.parquet(wikidumpMatrixPath + "/docTermMatrix")
    val termIds = spark.sqlContext.read.parquet(wikidumpMatrixPath + "/termIds").map(_.getAs[String](0)).collect
    val docIds = spark.sqlContext.read.parquet(wikidumpMatrixPath + "/docIds").map(_.getAs[String](0)).collect
    val idfs = spark.sqlContext.read.parquet(wikidumpMatrixPath + "/idfs").map(_.getAs[Double](0)).collect

    Data(spark, docTermMatrix, termIds, docIds, idfs)
  }

}
