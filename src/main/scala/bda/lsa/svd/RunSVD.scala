package bda.lsa.svd

import bda.lsa.{Data, baseDir, docTermMatrixToCorpusRDD, getData}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
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
    val model: mllib_SingularValueDecomposition[IndexedRowMatrix, Matrix] = run(spark, data, k)
    // save all
    saveModel(spark, model)
  }

  def run(spark: SparkSession, data: Data, k: Int): mllib_SingularValueDecomposition[IndexedRowMatrix, Matrix] = {
    
    val corpus: RDD[(Long, (mllib_Vector, String))] = docTermMatrixToCorpusRDD(spark, data)
    val vecRDD = corpus.map(t => IndexedRow(t._1, t._2._1))
    vecRDD.cache()

    val mat = new IndexedRowMatrix(vecRDD)
    val model: mllib_SingularValueDecomposition[IndexedRowMatrix, Matrix] = mat.computeSVD(k, computeU = true)

    //  val U: RowMatrix = model.U // The U factor is a RowMatrix.
    //  val s: mllib_Vector = model.s // The singular values are stored in a local dense vector.
    //  val V: Matrix = model.V // The V factor is a local dense matrix.
    model
  }

  //--------------------------------

  /**
    * Save the model to disk (either locally or on hdfs depending on the spark mode)
    *
    * @param spark the current spark session
    * @param model the model to save
    */
  def saveModel(spark: SparkSession, model: mllib_SingularValueDecomposition[IndexedRowMatrix, Matrix]): Unit = {
    // U is distributed => use RDD utils
    model.U.rows.zipWithIndex().map(_.swap).saveAsObjectFile(baseDir + pathSuffix + "/U")

    // s is a local vector => parallelize so it is also saved to hdfs in cluster mode
    spark.sparkContext.parallelize(model.s.toArray, 1).saveAsObjectFile(baseDir + pathSuffix + "/s")

    // s is a local Matrix => parallelize so it is also saved to hdfs in cluster mode
    spark.sparkContext.parallelize(model.V.rowIter.toSeq).zipWithIndex().map(_.swap).saveAsObjectFile(baseDir + pathSuffix + "/V")
  }

  /**
    * Load a model previously saved by [[saveModel()]].
    *
    * @param spark the current spark session
    * @return the SVD model
    */
  def loadModel(spark: SparkSession): mllib_SingularValueDecomposition[IndexedRowMatrix, Matrix] = {

    val Urdd = spark.sparkContext.objectFile[(Long, IndexedRow)](baseDir + pathSuffix + "/U")
    val U: IndexedRowMatrix = new IndexedRowMatrix(Urdd.sortByKey().map(_._2))

    val s: mllib_Vector = mllib_Vectors.dense(spark.sparkContext.objectFile[Double](baseDir + pathSuffix + "/s").collect)

    // the matrix is more complex to recreate...
    val vs: Array[mllib_Vector] = spark.sparkContext.objectFile[(Long, mllib_Vector)](baseDir + pathSuffix + "/V").
      sortByKey().map(_._2).collect
    val numRows = vs.length
    val numCols = vs(0).toArray.length
    val V: Matrix = org.apache.spark.mllib.linalg.Matrices.dense(numCols, numRows, vs.flatMap(_.toArray)).transpose

    new mllib_SingularValueDecomposition[IndexedRowMatrix, Matrix](U, s, V)
  }

}
