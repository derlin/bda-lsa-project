package bda.lsa.svd

import bda.lsa.{baseDir, docTermMatrixToCorpusRDD, getData}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vector => MLLIBVector}
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
    val pathSuffix = if (args.length > 1) args(1) else pathSuffix

    val spark = SparkSession.builder().
      config("spark.serializer", classOf[KryoSerializer].getName).
      master("local[*]").
      getOrCreate()

    val (docTermMatrix, termIds, docIds): (DataFrame, Array[String], Map[Long, String]) = getData(spark)
    val corpus: RDD[(Long, (MLLIBVector, String))] = docTermMatrixToCorpusRDD(spark, docTermMatrix)

    val vecRDD = corpus.map(_._2._1)
    vecRDD.cache()

    val mat = new RowMatrix(vecRDD)
    val model: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(k, computeU = true)

    //  val U: RowMatrix = model.U // The U factor is a RowMatrix.
    //  val s: MLLIBVector = model.s // The singular values are stored in a local dense vector.
    //  val V: Matrix = model.V // The V factor is a local dense matrix.

    // save all
    saveModel(model)
  }

  //--------------------------------

  /**
    * Save the model to disk
    * @param model
    * @return
    */
  def saveModel(model: SingularValueDecomposition[RowMatrix, Matrix]) = {
    // U is distributed => use RDD utils
    model.U.rows.saveAsObjectFile(baseDir + pathSuffix + "/U")

    // s is a local vector => serialize
    new java.io.ObjectOutputStream(new java.io.FileOutputStream(baseDir + pathSuffix + "/s")) {
      writeObject(model.s)
      close()
    }
    // V is a local Matrix => serialize
    new java.io.ObjectOutputStream(new java.io.FileOutputStream(baseDir + pathSuffix + "/V")) {
      writeObject(model.V)
      close()
    }
  }

  def loadModel(spark: SparkSession): SingularValueDecomposition[RowMatrix, Matrix] = {

    val U: RowMatrix = new RowMatrix(spark.sparkContext.objectFile(baseDir + "svd/U").asInstanceOf[RDD[MLLIBVector]])

    val s: MLLIBVector = new java.io.ObjectInputStream(new java.io.FileInputStream(baseDir + "/svd/s")) {
      val s = readObject().asInstanceOf[MLLIBVector]
      close()
    }.s

    val V: Matrix = new java.io.ObjectInputStream(new java.io.FileInputStream(baseDir + "/svd/V")) {
      val V = readObject().asInstanceOf[Matrix]
      close()
    }.V

    new SingularValueDecomposition[RowMatrix, Matrix](U, s, V)
  }

}
