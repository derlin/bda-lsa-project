/**
  * date: 13.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */

// this script can be used in spark-shell. First, launch the shell with dependencies:
// ```
// spark-shell --jars target/scala-2.11/bda-project-lsa-assembly-1.0.jar
// ```
// Then, type the following:
// ```
// val path = "wikidump.xml"
// val stopwordsFile = "src/main/resources/stopwords.txt"
// val numTerms = 2000
// val k = 100
//
// :load spark-shell-scripts/load-data-example.scala
// ```
// you will then have the following variables at your disposal:
// * `docTermMatrix`: a DataFrame `[title: string, tfidfVec: vector]`
// * `termIds`: an array of words. Indexes are used in the `tfidfVec` vector of `docTermMatrix`
// * `docIds`: a map `Long => String` mapping a document id with its title
// * `termIdfs`: a array of Double corresponding to the relative frequency of a word in the corpus


/*
 val path = "wikidump.xml"
 val stopwordsFile = "src/main/resources/stopwords.txt"
 val numTerms = 2000
 val k = 100
 val path = "/shared/wikipedia/wikidump-mini.xml"
val path = "/shared/wikipedia/wikidump.xml"
  */

val numTerms = 1000
val k = 20
val path = "wikidump.xml"
val stopwordsFile = "src/main/resources/stopwords.txt"

import bda.lsa.AssembleDocumentTermMatrix
import org.apache.spark.sql.Dataset

val assembleMatrix = new AssembleDocumentTermMatrix(spark)

import assembleMatrix._

val docTexts: Dataset[(String, String)] = parseWikipediaDump(path)
val (docTermMatrix, termIds, docIds, termIdfs) = documentTermMatrix(docTexts, stopwordsFile, numTerms)
docTermMatrix.cache()
