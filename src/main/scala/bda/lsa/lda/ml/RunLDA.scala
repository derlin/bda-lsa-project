package bda.lsa.lda.ml

import bda.lsa._

import org.apache.spark.ml.clustering.{DistributedLDAModel => ml_DistributedLDAModel, LDA => ml_LDA}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * date: 18.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
object RunLDA {
  val pathSuffix = "/ml-lda"

  def main(args: Array[String]): Unit = {
    // parse arguments
    val k = if (args.length > 0) args(0).toInt else 100

    val spark = SparkSession.builder().
      config("spark.serializer", classOf[KryoSerializer].getName).
      getOrCreate()

    val (docTermMatrix, termIds, docIds, idfs): (DataFrame, Array[String], Map[Long, String], Array[Double]) = getData(spark)

    val lda_model: ml_LDA =
      new ml_LDA().
        setK(k).
        setMaxIter(10).
        setFeaturesCol("tfidfVec")

    val model: ml_DistributedLDAModel = lda_model.fit(docTermMatrix).asInstanceOf[ml_DistributedLDAModel]

    // to load it back, use val model = DistributedLDAModel.load(sc, "XXX/mllib-lda-model")
    saveModel(spark, model)
  }

  // -----------------

  def saveModel(spark: SparkSession, model: ml_DistributedLDAModel) =
    model.save(baseDir + pathSuffix)

  def loadModel(spark: SparkSession): ml_DistributedLDAModel =
    ml_DistributedLDAModel.load(baseDir + pathSuffix)
}




