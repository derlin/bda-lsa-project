package bda.lsa.preprocessing


import org.apache.spark.ml.feature.{CountVectorizer, IDF, StopWordsRemover}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable

/**
  * Class for creating the docTermMatrix out of an RDD of (title,content) from wikipedia articles
  * (see [[bda.lsa.preprocessing.XmlToParquetWriter]]).
  *
  * Article content is preprocessed with the following steps:
  *
  * - tokenize/lemmatize using spark-corenlp [[https://github.com/databricks/spark-corenlp]]
  * - remove stopwords using [[org.apache.spark.ml.feature.StopWordsRemover]]
  * - remove terms with non-alpha characters or less than three chars
  * - remove articles with less than three terms in the body
  *
  * Once this is done, the TF-IDF is computed using [[org.apache.spark.ml.feature.CountVectorizer]] and
  * [[org.apache.spark.ml.feature.IDF]].
  *
  * <p>
  * context: BDA - Master MSE,
  * date: 18.05.17
  *
  * @author Lucy Linder [lucy.derlin@gmail.com]
  */
object TextToIDF {

  /**
    * preprocess the wikipedia articles and compute the TF-iDF
    * @param spark the spark context
    * @param docTexts an `RDD[(title: String, content: String)]`
    * @param numTerms the number of terms in the vocabulary. Only the most frequent `numTerms` terms are kept
    * @return see [[bda.lsa.Data]]
    */
  def preprocess(spark: SparkSession, docTexts: Dataset[(String, String)], numTerms: Int)
  : (DataFrame, Array[String], Array[Double]) = {
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
    val docTermMatrix = idfModel.transform(docTermFreqs)
      //withColumn("id",monotonically_increasing_id).
      //select("id", "title", "tfidfVec")

    val dtm = bda.lsa.addNiceRowId(spark, docTermMatrix) // add nice ids
    println("matrix generated.")
    (dtm, vocabModel.vocabulary, idfModel.idf.toArray)
  }

  /**
    * Remove words with non-alpha characters or with a length <= 2
    * @param words a sequence of terms
    * @return  a filtered sequence of terms
    */
  def filterWords(words: Seq[String]): Seq[String] =
    words.
      filter(w => w.forall(c => Character.isLetter(c))).
      filter(_.length > 2)


}