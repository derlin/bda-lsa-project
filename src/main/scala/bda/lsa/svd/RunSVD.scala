package bda.lsa.svd

import bda.lsa.{baseDir, docTermMatrixToCorpusRDD, getData}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition => MLLIB_SingularValueDecomposition, Vector => MLLIBVector, Vectors => MLLIBVectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * date: 19.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
object RunSVD {
  val pathSuffix = "/svd"

  def main(args: Array[String]): Unit = {
    // parse arguments
    val k = if (args.length > 0) args(0).toInt else 100

    val spark = SparkSession.builder().
      config("spark.serializer", classOf[KryoSerializer].getName).
      master("local[*]").
      getOrCreate()

    val (docTermMatrix, termIds, docIds): (DataFrame, Array[String], Map[Long, String]) = getData(spark)
    val corpus: RDD[(Long, (MLLIBVector, String))] = docTermMatrixToCorpusRDD(spark, docTermMatrix)

    val vecRDD = corpus.map(_._2._1)
    vecRDD.cache()

    val mat = new RowMatrix(vecRDD)
    val model: MLLIB_SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(k, computeU = true)

    //  val U: RowMatrix = model.U // The U factor is a RowMatrix.
    //  val s: MLLIBVector = model.s // The singular values are stored in a local dense vector.
    //  val V: Matrix = model.V // The V factor is a local dense matrix.

    // save all
    saveModel(spark, model)
  }

  //--------------------------------

  /**
    * Save the model to disk (either locally or on hdfs depending on the spark mode)
    *
    * @param spark the current spark session
    * @param model  the model to save
    */
  def saveModel(spark: SparkSession, model: MLLIB_SingularValueDecomposition[RowMatrix, Matrix]) : Unit = {
    // U is distributed => use RDD utils
    model.U.rows.saveAsObjectFile(baseDir + pathSuffix + "/U")

    // s is a local vector => parallelize so it is also saved to hdfs in cluster mode
    spark.sparkContext.parallelize(model.s.toArray).saveAsObjectFile(baseDir + pathSuffix + "/s")

    // s is a local Matrix => parallelize so it is also saved to hdfs in cluster mode
    spark.sparkContext.parallelize(model.V.rowIter.toSeq).saveAsObjectFile(baseDir + pathSuffix + "/V")
  }

  /**
    * Load a model previously saved by [[saveModel()]].
    * @param spark the current spark session
    * @return the SVD model
    */
  def loadModel(spark: SparkSession): MLLIB_SingularValueDecomposition[RowMatrix, Matrix] = {

    val U: RowMatrix = new RowMatrix(spark.sparkContext.objectFile(baseDir + pathSuffix + "/U").asInstanceOf[RDD[MLLIBVector]])

    val s: MLLIBVector = MLLIBVectors.dense(spark.sparkContext.objectFile[Double](baseDir + pathSuffix + "/s").collect)

    // the matrix is more complex to recreate...
    val vs : Array[MLLIBVector] = spark.sparkContext.objectFile[MLLIBVector](baseDir + pathSuffix + "/V").collect
    val numRows = vs.length
    val numCols = vs(0).toArray.length
    val V: Matrix = org.apache.spark.mllib.linalg.Matrices.dense(numRows, numCols, vs.map(_.toArray).flatten)

    new MLLIB_SingularValueDecomposition[RowMatrix, Matrix](U, s, V)
  }

}
