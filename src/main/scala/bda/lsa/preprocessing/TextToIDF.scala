package bda.lsa.preprocessing


import org.apache.spark.ml.feature.{CountVectorizer, IDF, StopWordsRemover}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable


object TextToIDF {


  def preprocess(spark: SparkSession, docTexts: Dataset[(String, String)], numTerms: Int)
  : (DataFrame, Array[String], Array[String], Array[Double]) = {
    import spark.implicits._
    import org.apache.spark.sql.functions._
    import com.databricks.spark.corenlp.functions._

    // ensure we have the correct column names
    val input = docTexts.toDF("title", "doc")

    // tokenize + lemmatize
    val lemmas = input.select('title, lemma('doc).as('lemma))

    // remove stopwords + lowercase + remove words with length < 3 or with non-letters
    val remover = new StopWordsRemover().setInputCol("lemma").setOutputCol("words")
    val words = remover.transform(lemmas).select("title", "words").map {
      case Row(title: String, words: mutable.WrappedArray[String]) => (title, filterWords(words).map(_.toLowerCase))
    }.toDF("title", "terms")

    // remove articles with less than 2 relevant words
    val filtered = words.where(size($"terms") > 1)
    filtered.cache()

    println(s"filtered done. Num docs: ${filtered.count()}")
    // count terms
    val countVectorizer = new CountVectorizer().
      setMinDF(2).
      setMinTF(2).
      setInputCol("terms").
      setOutputCol("termFreqs").
      setVocabSize(numTerms)

    println("creating vocabulary model.")
    val vocabModel = countVectorizer.fit(filtered)
    println("fitting vocabulary model.")
    val docTermFreqs = vocabModel.transform(filtered)
    docTermFreqs.cache()

    // apply TF-IDF
    println("applying TF-IDF.")
    val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidfVec")
    val idfModel = idf.fit(docTermFreqs)
    val docTermMatrix = idfModel.transform(docTermFreqs).select("title", "tfidfVec")


    val docIds = docTermFreqs.rdd.map(_.getString(0)).collect

    println("matrix generated.")
    (docTermMatrix, vocabModel.vocabulary, docIds, idfModel.idf.toArray)
  }

  def filterWords(words: Seq[String]): Seq[String] =
    words.
      filter(w => w.forall(c => Character.isLetter(c))).
      filter(_.length > 2)


}