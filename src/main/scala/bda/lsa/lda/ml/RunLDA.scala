package bda.lsa.lda.ml

import bda.lsa._

import org.apache.spark.ml.clustering.{DistributedLDAModel => ml_DistributedLDAModel, LDA => ml_LDA}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession

/**
  * Class for creating a Latent Dirichlet allocation model using spark ml.
  * <p>
  * context: BDA - Master MSE,
  * date: 18.05.17
  * @author Lucy Linder [lucy.derlin@gmail.com]
  */
object RunLDA {
  val pathSuffix = "/ml-lda"

  /** the number of topics to infer */
  var k = 100
  /** max iterations during the model construction */
  var maxIterations = 100

  /**
    * Create the LDA model and save it to disk (see [[saveModel]]).
    *
    * @param args an array of String, in the order:
    *
    *  - k: the number of topics to infer, default to 100
    *  - maxIterations: default to 100
    */
  def main(args: Array[String]): Unit = {
    // parse arguments
    if (args.length > 0) k = args(0).toInt
    if (args.length > 1) maxIterations = args(1).toInt

    val spark = SparkSession.builder().
      appName("ml.RunLDA K=" + k).
      config("spark.serializer", classOf[KryoSerializer].getName).
      getOrCreate()

    val data = getData(spark)

    val lda_model: ml_LDA =
      new ml_LDA().
        setK(k).
        setOptimizeDocConcentration(true).
        setOptimizer("em").
        setMaxIter(10).
        setFeaturesCol("tfidfVec")

    val model: ml_DistributedLDAModel = lda_model.fit(data.dtm).asInstanceOf[ml_DistributedLDAModel]

    // to load it back, use val model = DistributedLDAModel.load(sc, "XXX/mllib-lda-model")
    saveModel(spark, model)
  }

  // -----------------

  def saveModel(spark: SparkSession, model: ml_DistributedLDAModel) =
    model.save(baseDir + pathSuffix)

  def loadModel(spark: SparkSession): ml_DistributedLDAModel =
    ml_DistributedLDAModel.load(baseDir + pathSuffix)
}




