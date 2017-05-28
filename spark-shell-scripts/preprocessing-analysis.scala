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
val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidfVec")
val idfModel = idf.fit(docTermFreqs)
val docTermMatrix = idfModel.transform(docTermFreqs).select("title", "tfidfVec")


// get overall term frequencies
val termsRDD = wordsFiltered.select("terms").flatMap(r => r.getAs[mutable.WrappedArray[String]](0)).map(s => (s, 1)).rdd
termsRDD.reduceByKey(_ + _).sortBy(-_._2)

// We have some chinese terms in our vocabulary:
// > Array[(String, Int)] = Array((ﬂow,4), (ﬁrst,1), (ﬁnding,1), (ﬁeld,1), (카더라,1), (강남스타일,2), (ꜣbst,1), (黃婉卿,1), (馬蹄鞋,1), (香港電視專業人員協會,1), (香港華人西醫書まりこ現象,3), (電影電視工程師協會香港分會,1), (雪婆んご,1), (雑踏の中で,1), (長前臂,1), (鍾華亮,1), (醤油入れ,1), (過去ログ集,1), (週刊朝日,4), (週刊文春,2), (週刊平凡,3), (转账sty,1), (花盆鞋,1), (自発性の精神病理,1), (自律神経ダイエット,3), (羟考酮,1), (紀曉風,1), (禮樂制度,1), (現代妖怪,2), (猿似猴,1), (港人怒吼為李旺陽呼冤,1), (消防style,1), (活字探偵団,1, (本当にいる日本の,2), (本屋に行くと催すのはなぜ,2), (本屋と便意の謎,1), (本屋で急に便意を感じる,1), (本屋でトイレに行きたくなる,3), (本の雑誌,32), (書便派,1), (日本の都市伝説,1), 钗,1), (弁而釵,1), (廣播工程師協會香港分會,1), (孫中山,1), (好引氣也,1), (大而黑,1), (大國小器人權排名尾三,1...