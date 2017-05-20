/**
  * date: 20.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */


import org.apache.spark.sql.functions._
import com.databricks.spark.corenlp.functions._
import org.apache.spark.ml.feature.{CountVectorizer, IDF, StopWordsRemover}
import org.apache.spark.sql.Row

val docTexts = spark.sqlContext.read.parquet(bda.lsa.wikidumpParquetPath).map {
  r => (r.getAs[String](0), r.getAs[String](1))
}.toDF("title", "doc")

val lemmas = docTexts.select('title, lemma('doc).as('lemma))

val remover = new StopWordsRemover().setInputCol("lemma").setOutputCol("words")
val words = remover.transform(lemmas).select("title", "words").map {
  case Row(title: String, words: mutable.WrappedArray[String]) => (title, words.filter(w => w.forall(c => Character.isLetter(c))).
    filter(_.length > 2).map(_.toLowerCase))
}.toDF("title", "terms")

val filtered = words.where(size($"terms") > 1)

val numTerms = 2000
val countVectorizer = new CountVectorizer().
  setInputCol("terms").
  setOutputCol("termFreqs").
  setVocabSize(numTerms)

val vocabModel = countVectorizer.fit(filtered)
val docTermFreqs = vocabModel.transform(filtered)
docTermFreqs.cache()

val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidfVec")
val idfModel = idf.fit(docTermFreqs)
val docTermMatrix = idfModel.transform(docTermFreqs).select("title", "tfidfVec")

//scala> vocabModel.vocabulary.filterNot(termIds.toSet)
//res29: Array[String] = Array(would, could, men, masters, formula, chi, cast)
//
//scala> termIds.filterNot(vocabModel.vocabulary.toSet)
//res30: Array[String] = Array(the, can, will, now, down, just, all)