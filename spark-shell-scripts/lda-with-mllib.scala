/**
  * date: 17.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */

// https://s3.amazonaws.com/sparksummit-share/ml-ams-1.0.1/wiki-lda/scala/wiki-lda_answers.html
// http://lamastex.org/courses/ScalableDataScience/2016/S1/week7/14_ProbabilisticTopicModels/025_LDA_20NewsGroupsSmall.html
import bda.lsa.preprocessing.AssembleDocumentTermMatrix
import org.apache.spark.sql.Dataset
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA => MLLIB_LDA}
import org.apache.spark.rdd.RDD

// ## Load data

// use :load spark-shell-scripts/load-data-example.scala

// # Create RDD for use with LDA

val corpus = docTermMatrix.
  select("tfidfVec", "title").
  map(r => (Vectors.fromML(r.getAs[MLVector](0)), r.getAs[String](1))).
  rdd.
  zipWithIndex.
  map(_.swap)

corpus.cache()



// # Run and save model

val ldaModel = new MLLIB_LDA().setK(k).run(corpus.mapValues(_._1)).asInstanceOf[DistributedLDAModel]

// How to save and reload the model:

ldaModel.save(sc, "/tmp/LDA_MLLIB-01.model")


val ldaModel = DistributedLDAModel.load(sc, "/tmp/LDA_MLLIB-01.model")

// How to view the model:
import runtime.ScalaRunTime.stringOf

ldaModel.describeTopics(3).foreach(l => println(stringOf(l)))


// # View the topics

def describeTopicsWithWords(num: Int) = {
  val topicIndex = ldaModel.describeTopics(num)
  topicIndex.map { topic => topic._1.map(termIds(_)) }
}

def printTopicsWithWords(num: Int, termsSep: String = ", ") = {
  val topicIndex = ldaModel.describeTopics(num)
  describeTopicsWithWords(num).zipWithIndex.map {
    case (terms, i) => s"TOPIC $i: " + terms.mkString(termsSep)
  }.mkString("\n")
}

// returns a rdd of topics. num: max number of documents per topic
def topDocumentsWithTitle(num: Int) = {
  val topDocs = ldaModel.asInstanceOf[DistributedLDAModel].topDocumentsPerTopic(num)
  // topDocs: Array[(Array[Long], Array[Double])] => (ids, relevance)
  topDocs
    .map(topic => {
      val ids = topic._1
      val idsRDD = sc.parallelize(ids.zipWithIndex)
      // When called on datasets of type (K, V) and (K, W), join returns a dataset of (K, (V, W)) pairs
      // idsRDD = (id, index), corpus.mapValues = (id, title)
      // => (id, (index, title). With mapValues, we finally get (id, title)
      idsRDD.join(corpus.mapValues(_._2)).mapValues(_._2).collectAsMap
    })
}


printTopicsWithWords(10)


val topicCategories = describeTopicsWithWords(10).map(_.mkString("-"))

topDocumentsWithTitle(7).zip(topicCategories).foreach(x => {
  println(s"Next Topic: ${x._2}")
  x._1.foreach(println)
  println()
})

val idTitleDF = corpus.map(x => (x._1, x._2._2)).toDF("id", "title")
idTitleDF.select("title", "id").where($"title".like("%UEFA%"))


val topicDist = ldaModel.asInstanceOf[DistributedLDAModel].topicDistributions
// topicDist = RDD[(Long, org.apache.spark.mllib.linalg.Vector)] => (docId, relevance per topic)

def getTopicDistForID(id: Int) {
  println("Score\tTopic")
  topicDist
    .filter(_._1 == id)
    .map(_._2.toArray)
    .first // now we are working locally
    .zip(topicCategories)
    .sortWith(_._1 > _._1)
    .foreach(x => println(f"${x._1}%.3f\t${x._2}"))
}

getTopicDistForID(2880) //2880


def topDocumentsForTopic(tid: Int) = {
  val topDocs = ldaModel.asInstanceOf[DistributedLDAModel].topDocumentsPerTopic(10)(tid)
  val idsRDD = sc.parallelize(topDocs._1.zipWithIndex)
  idsRDD.join(corpus.mapValues(_._2)).mapValues(_._2).collectAsMap
}
ldaModel.asInstanceOf[DistributedLDAModel].topTopicsPerDocument(4)


// TODO http://stackoverflow.com/questions/38735413/graphx-visualization

import org.apache.spark.graphx.Graph

def toGexf[VD, ED](g: Graph[VD, ED]): String = {
  "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" +
    "<gexf xmlns=\"http://www.gexf.net/1.2draft\" version=\"1.2\">\n" +
    "  <graph mode=\"static\" defaultedgetype=\"directed\">\n" +
    "    <nodes>\n" +
    g.vertices.map(v => "      <node id=\"" + v._1 + "\" label=\"" +
      v._2 + "\" />\n").collect.mkString +
    "    </nodes>\n" +
    "    <edges>\n" +
    g.edges.map(e => "      <edge source=\"" + e.srcId +
      "\" target=\"" + e.dstId + "\" label=\"" + e.attr +
      "\" />\n").collect.mkString +
    "    </edges>\n" +
    "  </graph>\n" +
    "</gexf>"
}



def topTermsInTopicsToJSON(model: DistributedLDAModel, nbTerms: Int = 10, nbConcepts: Int = -1) = {
  var topicIndex = model.describeTopics(nbTerms)
  if (nbConcepts > 0) topicIndex = topicIndex.take(nbTerms)

  val tuplesRDD = sc.parallelize(topicIndex).zipWithIndex.flatMap {
    case (topic, i) =>
      topic._1.map(vocabulary(_)).zip(topic._2).map {
        case (a, b) => (a, b, i)
      }
  }

  val termsJson = tuplesRDD.
    toDF.
    withColumnRenamed("_1", "term").
    withColumnRenamed("_2", "probability").
    withColumnRenamed("_3", "topicId").
    toJSON.
    collect.
    mkString(", \n")

  s"[$termsJson]"
}

// categorize new document
//create test input, convert to term count, and get its topic distribution
val test_input = Seq("this is my test document")
val termsLookup = termIds.zipWithIndex.toMap
val test_document:RDD[(Long, mllib_Vector)] = sc.parallelize(test_input.map(doc=>doc.split("\\s"))).zipWithIndex.map{ case (tokens, id) =>
  val counts = new scala.collection.mutable.HashMap[Int, Double]()
  tokens.foreach { term =>
    if (termsLookup.contains(term)) {
      val idx = termsLookup(term)
      counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
    }
  }
  (id, mllib_Vectors.sparse(termIds.size, counts.toSeq))
}

val topicDistribution = ldaModel.toLocal.topicDistributions(test_document.first._2)
println("first topic distribution:"+topicDistribution.toArray.zipWithIndex.sortBy(-_._1).mkString(", "))


// same as getTopicDistForId, but print the topic's id as well
def getTopicDistForDoc(id: Int) {
  println("Score\tTopic")
  topicDist
    .filter(_._1 == id)
    .map(_._2.toArray)
    .first // now we are working locally
    .zip(topicCategories)
     .zipWithIndex
    .sortWith(_._1._1 > _._1._1)
    .foreach(x => println(f"${x._1._1}%.3f \t${x._2}:${x._1._2}"))
}