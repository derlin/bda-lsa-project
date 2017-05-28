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

  def main(args: Array[String]): Unit = {
    // parse arguments
    val k = if (args.length > 0) args(0).toInt else 100
    val maxIterations = if (args.length > 1) args(1).toInt else 20
    val alpha = if (args.length > 2) args(2).toInt else (50.0 / k) + 1
    // Typical value is (50 / nbTopics)
    val beta = if (args.length > 3) args(3).toInt else -1
    // Typical value is either 0.1 or (200 / vocabularySize)
    val optimizerAlgorithm = "em"

    // We tried to use the 'online' algorithm, but the problem is that he's only able to run locally. Which means huge
    // performance cuts when running it, so we decided to not use it at all and disable that option.
    /*
    val optimizerAlgorithm = if (args.length > 4) args(4).toLowerCase else "em" // Either "em" or "online"
     */

    val optimizer = optimizerAlgorithm.toLowerCase match {
      case "em" => new mllib_EMLDAOptimizer
      case "online" => new mllib_OnlineLDAOptimizer()
      case _ => throw new IllegalArgumentException(s"Only algorithms 'em', 'online' are supported, not ${optimizerAlgorithm}.")
    }

    val spark = SparkSession.builder().
      config("spark.serializer", classOf[KryoSerializer].getName).
      getOrCreate()

    val data: Data = getData(spark)
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

    // to load it back, use val model = DistributedLDAModel.load(sc, "XXX/mllib-lda-model")
    saveModel(spark, model)

  }

  // -----------------

  def saveModel(spark: SparkSession, model: mllib_DistributedLDAModel) =
    model.save(spark.sparkContext, baseDir + pathSuffix)

  def loadModel(spark: SparkSession): mllib_DistributedLDAModel =
    mllib_DistributedLDAModel.load(spark.sparkContext, baseDir + pathSuffix)
}




