/**
  * date: 28.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */


import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, IDF, StopWordsRemover}
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.mutable
import org.apache.spark.sql.functions._
import com.databricks.spark.corenlp.functions._


// # useful functions

// create a count user-defined function once and for all
val countTokens = udf { (words: Seq[String]) => words.length }


def sumAvg(df: DataFrame, countCol: String) = {
  (df.agg(sum(countCol)).first, df.agg(avg(countCol)).first)
}

def uniqueSumAvg(df: DataFrame, colName: String) = {
  import org.apache.spark.sql.functions._
  val sumed = df.select(colName).
    flatMap(r => r.getAs[mutable.WrappedArray[String]](0)).
    distinct().
    count()
  val avged = df.select(colName).
    map(r => r.getAs[mutable.WrappedArray[String]](0).distinct.length).
    agg(avg("value")).first

  (sumed, avged)
}

// # loading the data

val docTexts = spark.sqlContext.read.parquet(bda.lsa.wikidumpParquetPath).map {
  r => (r.getAs[String](0), r.getAs[String](1))
}
// ensure we have the correct column names
val input = docTexts.toDF("title", "doc")


// # tokenizing the articles

// use the SparkML tokenizer.
// _Note_: with this tokenizer, something like:
// > the 'wires' are red, what a shame.
// will become:
// > ["the", "'wires'", "are", "red", "what", "a", "shame"]
import org.apache.spark.ml.feature.Tokenizer

// tokenized is DF = [(doc: String, words: WrappedArray[String])]
val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("words")

// count the total number of "words" in our dataset
// and the average number of words per article
val tokenized = tokenizer.transform(input).select("title", "words").
  withColumn("count", countTokens(col("words")))

// sum of words in the dataset: 2'712'636, avg words per document: 1'784
sumAvg(tokenized, "count")

// sum of distinct in the dataset: 201'563, avg unique words per document: 694
uniqueSumAvg(tokenized, "words")


// # basic cleansing: remove stopwords

val remover1 = new StopWordsRemover().setInputCol("words").setOutputCol("wordsClean")
val cleaned = remover1.transform(tokenized).withColumn("count", countTokens(col("wordsClean")))

// after stopwords removal,
// sum of words in the dataset: 1'722'062, avg words per document: 1'133
sumAvg(cleaned, "count")

// after stopwords removal,
// sum of distinct in the dataset: 201'427, avg unique words per document: 638
uniqueSumAvg(cleaned, "wordsClean")


// # lematizing and stopwords removal

// tokenize + lemmatize  using nlp
// _Note_: with this tokenizer, something like:
// > the 'wires' are red, what a shame.
// will become:
// > ["the", "'", "wires, "'", "are", "red", ",", "what", "a", "shame", "."]
val lemmas = input.select('title, lemma('doc).as('lemma))
val remover = new StopWordsRemover().setInputCol("lemma").setOutputCol("words")
val words = remover.transform(lemmas).withColumn("count", countTokens(col("words")))

words.cache()

// after lemmatization + stopwords removal,
// sum of words in the dataset: 2'021'543, avg words per document: 1'329
sumAvg(words, "count")


// after lemmatization + stopwords removal,
// sum of distinct in the dataset: 109'995, avg unique words per document: 518
uniqueSumAvg(words, "words")


// # Filtering
// Finally, remove words with non-letter and words of length < 3

def filterWords(words: Seq[String]): Seq[String] =
  words.filter(w => w.forall(c => Character.isLetter(c))).filter(_.length > 2)

val wordsFiltered = words.select("title", "words").map {
  case Row(title: String, words: mutable.WrappedArray[String]) => (title, filterWords(words).map(_.toLowerCase))
}.toDF("title", "terms").
  withColumn("count", countTokens(col("terms")))


// after filtering,
// sum of words in the dataset: 1'458'090, avg words per document: 960
sumAvg(wordsFiltered, "count")


// after filtering,
// sum of distinct in the dataset: 70'638, avg unique words per document: 439
uniqueSumAvg(wordsFiltered, "terms")


// ------------------------------------------

// #  articles with few interesting terms

// 1519/1520 documents with more than 1 term
wordsFiltered.where(size($"terms") > 1).count()

// 11'000 terms for article "Development Communication"
wordsFiltered.select(max($"count")).first

// ------------------------------------------

// count terms
val countVectorizer = new CountVectorizer().setInputCol("terms").setOutputCol("termFreqs")
val vocabModel = countVectorizer.fit(wordsFiltered)
val docTermFreqs = vocabModel.transform(wordsFiltered)

docTermFreqs.cache()

// apply TF-IDF
println("applying TF-IDF.")
val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidfVec")
val idfModel = idf.fit(docTermFreqs)
val docTermMatrix = idfModel.transform(docTermFreqs).select("title", "tfidfVec")


