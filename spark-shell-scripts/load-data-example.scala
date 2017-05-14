/**
  * date: 13.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */

// before use, set:
/*
 val path = "wikidump.xml"
 val stopwordsFile = "src/main/resources/stopwords.txt"
 val numTerms = 2000
 val k = 100
  */


import bda.lsa.AssembleDocumentTermMatrix
import org.apache.spark.sql.Dataset

val assembleMatrix = new AssembleDocumentTermMatrix(spark)

import assembleMatrix._

val docTexts: Dataset[(String, String)] = parseWikipediaDump(path)
val (docTermMatrix, termIds, docIds, termIdfs) = documentTermMatrix(docTexts, stopwordsFile, numTerms)
docTermMatrix.cache()
