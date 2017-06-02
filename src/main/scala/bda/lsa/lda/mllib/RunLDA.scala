package bda.lsa.lda.mllib

import bda.lsa._
import org.apache.spark.mllib.clustering.{
DistributedLDAModel => mllib_DistributedLDAModel,
LDA => mllib_LDA,
EMLDAOptimizer => mllib_EMLDAOptimizer,
OnlineLDAOptimizer => mllib_OnlineLDAOptimizer}
import org.apache.spark.mllib.linalg.{Vector => mllib_Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession

/**
  * Class for creating a Latent Dirichlet allocation model using spark mllib.
  * <p>
  * context: BDA - Master MSE,
  * date: 18.05.17
  * @author Lucy Linder [lucy.derlin@gmail.com]
  */
object RunLDA {
  /** where to save the model, relative to [[bda.lsa.baseDir]] */
  val pathSuffix = "/mllib-lda"


  /** the number of topics to infer */
  var k = 100
  /** max iterations during the model construction */
  var maxIterations = 100
  /** the prior on the per-document topic distributions
    * Typical value is (50 / nbTopics), with +1 if EM used  */
  var alpha = -1
  /** the prior on the per-topic word distribution
    * if set to -1, it will automatically be set by the library.
    * a good value is `200.0 / data.termIds.length` */
  var beta = -1
  /** the optimizer to use, "em" or "online"
    * Here, we force EM because online only runs locally */
  var optimizerAlgorithm = "em"

  /**
    * Create the LDA model and save it to disk (see [[saveModel]]).
    *
    * @param args an array of String, in the order:
    *
    *  - k: the number of topics to infer, default to 100
    *  - maxIterations: default to 100
    *  - alpha: the prior on the per-document topic distributions, default to (50/k)+1
    *  - beta: the prior on the per-topic word distribution, default to (200/vocabSize)
    */
  def main(args: Array[String]): Unit = {
    // parse arguments
    if (args.length > 0) k = args(0).toInt
    if (args.length > 1) maxIterations = args(1).toInt
    if (args.length > 2) alpha = args(2).toInt
    if (args.length > 3) beta = args(3).toInt

    // We tried to use the 'online' algorithm, but the problem is that he's only able to run locally. Which means huge
    // performance cuts when running it, so we decided to not use it at all and disable that option.
    /*
     if (args.length > 4) optimizerAlgorithm = args(4).toLowerCase // either "em" or "online"
     */
    val optimizerAlgorithm = "em"

    val spark = SparkSession.builder().
      appName("mllib.RunLDA K=" + k).
      config("spark.serializer", classOf[KryoSerializer].getName).
      getOrCreate()

    val data: Data = getData(spark)
    val model: mllib_DistributedLDAModel = run(spark, data)

    // to load it back, use
    // ```
    // val model = DistributedLDAModel.load(sc, "XXX/mllib-lda-model")
    // ```
    // or simply the `RunLDA.loadModel` method
    saveModel(spark, model)
  }

  /**
    * Create an LDA model using the parameters defined in this class.
    *
    * @param spark the spark context
    * @param data  the data to perform the LDA on
    * @return a distributed LDA model (see [[org.apache.spark.mllib.clustering.DistributedLDAModel]])
    */
  def run(spark: SparkSession, data: Data): mllib_DistributedLDAModel = {

    // create the optimizer
    val optimizer = optimizerAlgorithm.toLowerCase match {
      case "em" => new mllib_EMLDAOptimizer
      case "online" => new mllib_OnlineLDAOptimizer()
      case _ => throw new IllegalArgumentException(s"Only algorithms 'em', 'online' are supported, not ${optimizerAlgorithm}.")
    }

    // create the RDD and cache it since it is used in multiple passes
    val corpus: RDD[(Long, (mllib_Vector, String))] = docTermMatrixToCorpusRDD(spark, data)
    corpus.cache()

    val model: mllib_DistributedLDAModel =
      new mllib_LDA().
        setOptimizer(optimizer).
        setAlpha(alpha).
        setBeta(beta).
        setK(k).
        setMaxIterations(maxIterations).
        run(corpus.mapValues(_._1)).
        asInstanceOf[mllib_DistributedLDAModel]

    model
  }

  // -----------------

  /**
    * Save the model to disk, in [[bda.lsa.baseDir]]/[[pathSuffix]]. If the code is run on yarn, the default filesystem
    * is HDFS.
    * <p>
    * '''Note''': if the path already exists, an exception is thrown.
    *
    * @param spark the spark context
    * @param model the LDA model
    */
  def saveModel(spark: SparkSession, model: mllib_DistributedLDAModel) =
    model.save(spark.sparkContext, baseDir + pathSuffix)

  /**
    * Load the model from disk, from [[bda.lsa.baseDir]]/[[pathSuffix]]. If the code is run on yarn, the default filesystem
    * is HDFS.
    *
    * @param spark the spark context
    * @return the LDA model
    */
  def loadModel(spark: SparkSession): mllib_DistributedLDAModel =
    mllib_DistributedLDAModel.load(spark.sparkContext, baseDir + pathSuffix)
}




