package bda.lsa.svd

import bda.lsa.{Data, baseDir, docTermMatrixToCorpusRDD, getData}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition => mllib_SingularValueDecomposition, Vector => mllib_Vector, Vectors => mllib_Vectors}
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
      getOrCreate()

    val data: Data = getData(spark)
    val corpus: RDD[(Long, (mllib_Vector, String))] = docTermMatrixToCorpusRDD(spark, data)

    val vecRDD = corpus.sortByKey().map(_._2._1)
    vecRDD.cache()

    val mat = new RowMatrix(vecRDD)
    val model: mllib_SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(k, computeU = true)

    //  val U: RowMatrix = model.U // The U factor is a RowMatrix.
    //  val s: mllib_Vector = model.s // The singular values are stored in a local dense vector.
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
  def saveModel(spark: SparkSession, model: mllib_SingularValueDecomposition[RowMatrix, Matrix]) : Unit = {
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
  def loadModel(spark: SparkSession): mllib_SingularValueDecomposition[RowMatrix, Matrix] = {

    val U: RowMatrix = new RowMatrix(spark.sparkContext.objectFile(baseDir + pathSuffix + "/U").asInstanceOf[RDD[mllib_Vector]])

    val s: mllib_Vector = mllib_Vectors.dense(spark.sparkContext.objectFile[Double](baseDir + pathSuffix + "/s").collect)

    // the matrix is more complex to recreate...
    val vs : Array[mllib_Vector] = spark.sparkContext.objectFile[mllib_Vector](baseDir + pathSuffix + "/V").collect
    val numRows = vs.length
    val numCols = vs(0).toArray.length
    val V: Matrix = org.apache.spark.mllib.linalg.Matrices.dense(numRows, numCols, vs.flatMap(_.toArray))

    new mllib_SingularValueDecomposition[RowMatrix, Matrix](U, s, V)
  }

}
