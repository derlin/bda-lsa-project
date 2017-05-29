package bda.lsa.lda.mllib

import bda.lsa._
import org.apache.spark.ml.linalg.{Vector => ml_Vector}
import org.apache.spark.mllib.clustering.{DistributedLDAModel => mllib_DistributedLDAModel,
LDA => mllib_LDA,
EMLDAOptimizer => mllib_EMLDAOptimizer,
OnlineLDAOptimizer => mllib_OnlineLDAOptimizer}
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
  
  var k = 100
  var maxIterations = 20
  var alpha = (50.0 / k) + 1
  var beta = -1
  var optimizerAlgorithm = "em"

  def main(args: Array[String]): Unit = {
    // parse arguments
    if (args.length > 0) k = args(0).toInt
    if (args.length > 1) maxIterations = args(1).toInt
    if (args.length > 2) alpha = args(2).toInt
    // Typical value is (50 / nbTopics)
    if (args.length > 3) beta = args(3).toInt
    // Typical value is either 0.1 or (200 / vocabularySize)
    // We tried to use the 'online' algorithm, but the problem is that he's only able to run locally. Which means huge
    // performance cuts when running it, so we decided to not use it at all and disable that option.
    /*
     if (args.length > 4) optimizerAlgorithm = args(4).toLowerCase // either "em" or "online"
     */
    val optimizerAlgorithm = "em"

    val spark = SparkSession.builder().
      config("spark.serializer", classOf[KryoSerializer].getName).
      getOrCreate()

    val data: Data = getData(spark)
    val model: mllib_DistributedLDAModel = run(spark, data)

    // to load it back, use val model = DistributedLDAModel.load(sc, "XXX/mllib-lda-model")
    saveModel(spark, model)
  }

  def run(spark: SparkSession, data: Data): mllib_DistributedLDAModel = {

    val optimizer = optimizerAlgorithm.toLowerCase match {
      case "em" => new mllib_EMLDAOptimizer
      case "online" => new mllib_OnlineLDAOptimizer()
      case _ => throw new IllegalArgumentException(s"Only algorithms 'em', 'online' are supported, not ${optimizerAlgorithm}.")
    }
    
    val corpus: RDD[(Long, (mllib_Vector, String))] = docTermMatrixToCorpusRDD(spark, data)

    corpus.cache()

    val model: mllib_DistributedLDAModel =
      new mllib_LDA().
        setOptimizer(optimizer).
        setAlpha(alpha).
        setBeta(if (beta > 0) beta else (200.0 / data.termIds.length) + 1).
        setK(k).
        setMaxIterations(maxIterations).
        run(corpus.mapValues(_._1)).
        asInstanceOf[mllib_DistributedLDAModel]

    model
  }

  // -----------------

  def saveModel(spark: SparkSession, model: mllib_DistributedLDAModel) =
    model.save(spark.sparkContext, baseDir + pathSuffix)

  def loadModel(spark: SparkSession): mllib_DistributedLDAModel =
    mllib_DistributedLDAModel.load(spark.sparkContext, baseDir + pathSuffix)
}




