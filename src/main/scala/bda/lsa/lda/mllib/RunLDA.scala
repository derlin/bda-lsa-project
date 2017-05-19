package bda.lsa.lda.mllib

import bda.lsa._
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA => MLLIB_LDA}
import org.apache.spark.mllib.linalg.{Vector => MLLIBVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * date: 18.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
object RunLDA {
  val pathSuffix = "/mllib-lda-model"

  def main(args: Array[String]): Unit = {
    // parse arguments
    val k = if (args.length > 0) args(0).toInt else 100

    val spark = SparkSession.builder().
      config("spark.serializer", classOf[KryoSerializer].getName).
      master("local[*]").
      getOrCreate()

    val (docTermMatrix, termIds, docIds): (DataFrame, Array[String], Map[Long, String]) = getData(spark)
    val corpus: RDD[(Long, (MLLIBVector, String))] = docTermMatrixToCorpusRDD(spark, docTermMatrix)

    corpus.cache()

    val model: DistributedLDAModel =
      new MLLIB_LDA().
        setK(k).
        run(corpus.mapValues(_._1)).
        asInstanceOf[DistributedLDAModel]

    // to load it back, use val model = DistributedLDAModel.load(sc, "XXX/mllib-lda-model")
    saveModel(spark, model)
  }

  // -----------------

  def saveModel(spark: SparkSession, model: DistributedLDAModel) =
    model.save(spark.sparkContext, baseDir + pathSuffix)

  def loadModel(spark: SparkSession): DistributedLDAModel =
    DistributedLDAModel.load(spark.sparkContext, baseDir + pathSuffix)
}




