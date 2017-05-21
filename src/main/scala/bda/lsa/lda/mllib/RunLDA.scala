package bda.lsa.lda.mllib

import bda.lsa._
import org.apache.spark.ml.linalg.{Vector => ml_Vector}
import org.apache.spark.mllib.clustering.{DistributedLDAModel => mllib_DistributedLDAModel, LDA => mllib_LDA}
import org.apache.spark.mllib.linalg.{Vector => mllib_Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * date: 18.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
object RunLDA {
  val pathSuffix = "/mllib-lda"

  def main(args: Array[String]): Unit = {
    // parse arguments
    val k = if (args.length > 0) args(0).toInt else 100

    val spark = SparkSession.builder().
      config("spark.serializer", classOf[KryoSerializer].getName).
      getOrCreate()
    
    val data: Data = getData(spark)
    val corpus: RDD[(Long, (mllib_Vector, String))] = docTermMatrixToCorpusRDD(spark, data.dtm)

    corpus.cache()

    val model: mllib_DistributedLDAModel =
      new mllib_LDA().
        setK(k).
        run(corpus.mapValues(_._1)).
        asInstanceOf[mllib_DistributedLDAModel]

    // to load it back, use val model = DistributedLDAModel.load(sc, "XXX/mllib-lda-model")
    saveModel(spark, model)
  }

  // -----------------

  def saveModel(spark: SparkSession, model: mllib_DistributedLDAModel) =
    model.save(spark.sparkContext, baseDir + pathSuffix)

  def loadModel(spark: SparkSession): mllib_DistributedLDAModel =
    mllib_DistributedLDAModel.load(spark.sparkContext, baseDir + pathSuffix)
}




