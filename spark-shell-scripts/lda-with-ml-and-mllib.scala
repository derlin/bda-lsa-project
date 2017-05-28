import bda.lsa.book.AssembleDocumentTermMatrix

/**
  * date: 14.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */

// # Loading data

val path = "wikidump.xml"
val stopwordsFile = "src/main/resources/stopwords.txt"
val numTerms = 2000
val k = 100

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.Dataset

val assembleMatrix = new AssembleDocumentTermMatrix(spark)

import assembleMatrix._

val docTexts: Dataset[(String, String)] = parseWikipediaDump(path)
val (docTermMatrix, termIds, docIds, termIdfs) = documentTermMatrix(docTexts, stopwordsFile, numTerms)
docTermMatrix.cache()



// # Latent Dirichlet allocation (LDA) with mllib
// MLLIB is the old spark library for machine learning. It works with RDDs only.
// see https://spark.apache.org/docs/2.1.0/mllib-clustering.html#latent-dirichlet-allocation-lda
// and https://gist.github.com/jkbradley/ab8ae22a8282b2c8ce33

import org.apache.spark.mllib.clustering.LDA

import org.apache.spark.mllib.linalg.{Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
val vecRdd = docTermMatrix.select("tfidfVec").rdd.map { row =>
  Vectors.fromML(row.getAs[MLVector]("tfidfVec"))
}

val lda = new LDA().setK(k).setMaxIterations(10)
val ldaModel = lda.run(vecRdd.zipWithUniqueId.map(_.swap))

// Print topics, showing top-weighted terms for each topic.
val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 5)
topicIndices.take(5).foreach { case (terms, termWeights) =>
  println("TOPIC:")
  terms.zip(termWeights).foreach { case (term, weight) =>
    println(s"${termIds(term.toInt)}\t$weight")
  }
  println()
}

// # Latent Dirichlet allocation (LDA) ml
// ML is the new spark library for machine learning. It works with DataFrames.
// see https://spark.apache.org/docs/2.1.0/ml-clustering.html#latent-dirichlet-allocation-lda

import org.apache.spark.ml.clustering.{LDA => ML_LDA}

val ml_lda = new ML_LDA().setK(k).setMaxIter(10).setFeaturesCol("tfidfVec")
val model = ml_lda.fit(docTermMatrix)

val ll = model.logLikelihood(docTermMatrix)
val lp = model.logPerplexity(docTermMatrix)
println(s"The lower bound on the log likelihood of the entire corpus: $ll")
println(s"The upper bound bound on perplexity: $lp")

// Describe topics.
val topics = model.describeTopics(5)
println("The topics described by their top-weighted terms:")
topics.show(false)

// print topics
import scala.collection.mutable.WrappedArray

topics.take(5).foreach { row =>
  println("TOPIC:")
  // see http://stackoverflow.com/questions/38257630/how-to-iterate-scala-wrappedarray-spark
  val terms = row.getAs[WrappedArray[Int]](1)
  val termWeights = row.getAs[WrappedArray[Double]](2)
  terms.zip(termWeights).foreach { case (term, weight) =>
    println(s"${termIds(term.toInt)}\t$weight")
  }
  println()
}

// Shows the result.
val transformed = model.transform(docTermMatrix)
transformed.show(true)

