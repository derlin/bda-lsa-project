// date: 12.05.17,
// author: Lucy Linder <lucy.derlin@gmail.com>

// # Introduction
// This notebook is my try to understand and redo the steps described in _Advanced Analytics With Spark_, 2nd edition, chapter 6.
// To execute this file, use:
// ```
// git clone https://github.com/sryza/aas.git
// cd aas/ch06-lsa
// mvn clean package
// spark-shell --jars target/ch06-lsa-2.0.0-jar-with-dependencies.jar
// :load spark-shell-from-book.scala
// ```

import java.util.Properties

import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.umd.cloud9.collection.XMLInputFormat
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io._
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition}
import org.apache.spark.sql.Dataset

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

// ## Getting data
// The `wikidump.xml` is at the root of the project (where you run the `spark-shell`)
// and contains a partial dump of wikipedia.
//
// To generate it:
//
//  1. go to https://en.wikipedia.org/wiki/Special:Export
//  2. add the _Commputer engineering_ category
//  3. press _Export_

val path = "wikidump.xml"
val stopwordsPath = "src/main/resources/stopwords.txt"

// ## Loading data
// First, load the xml dump and parse it into a DataSet of Tuples `(title, content)`. This uses
// utilities from cloud9 especially made for wikipedia dumps.
@transient val conf = new Configuration()
conf.set(XMLInputFormat.START_TAG_KEY, "<page>")
conf.set(XMLInputFormat.END_TAG_KEY, "</page>")
val kvs = spark.sparkContext.newAPIHadoopFile(path, classOf[XMLInputFormat], classOf[LongWritable], classOf[Text], conf)
val rawXmls = kvs.map(_._2.toString).toDS()

rawXmls.describe() //> org.apache.spark.sql.DataFrame = [summary: string, value: string]
//
// Running `rawXmls.take(1)`, you get:
//
//```
// res6: Array[String] =
// Array(<page>
//  <title>Sorting network</title>
//  <ns>0</ns>
//  <id>562061</id>
//  <revision>
//    <id>771651752</id>
//    <parentid>769140055</parentid>
//    <timestamp>2017-03-22T19:38:52Z</timestamp>
//    <contributor>
//      <ip>187.140.57.105</ip>
//    </contributor>
//    <comment>/* External links */</comment>
//    <model>wikitext</model>
//    <format>text/x-wiki</format>
//    <text xml:space="preserve" bytes="17682">[[File:SimpleSortingNetwork2.svg|thumb|250px|A simple sorting network consisting of four wires and five connectors]]
//// #
// ```

// ## Turning XML into plain text
// the Cloud9 project provides APIs that handle this entirely.
// In the project, the class `AssembleDocumentTermMatrix` can be used, but in spark-shell we just define the
// function "from scratch":
import edu.umd.cloud9.collection.wikipedia.language._
import edu.umd.cloud9.collection.wikipedia._

def wikiXmlToPlainText(pageXml: String): Option[(String, String)] = {
  // Wikipedia has updated their dumps slightly since Cloud9 was written,
  // so this hacky replacement is sometimes required to get parsing to work.
  val hackedPageXml = pageXml.replaceFirst(
    "<text xml:space=\"preserve\" bytes=\"\\d+\">",
    "<text xml:space=\"preserve\">")
  val page = new EnglishWikipediaPage()
  WikipediaPage.readPage(page, hackedPageXml)
  if (page.isEmpty) None // Better: if (page.isEmpty || !page.isArticle || page.isRedirect || page.getTitle.contains("(disambiguation)"))
  else Some((page.getTitle, page.getContent))
}
val docTexts = rawXmls.filter(_ != null).flatMap(wikiXmlToPlainText)


// The `docTexts` variable holds:
// ```
//scala> docTexts.describe()
//res10: org.apache.spark.sql.DataFrame = [summary: string, _1: string ... 1 more field]
//
//scala> docTexts.take(1)
//res11: Array[(String, String)] =
//Array((Sorting network,"Sorting network
//  In computer science, comparator networks are abstract devices built up of a fixed number ...
//scala>
// ```


// ## Cleansing Data
// Before anything else, we need to remove stopwords and stem the words in the wikipedia contents.
// We begin with some utility definitions:


import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._
import java.util.Properties

def createNLPPipeline(): StanfordCoreNLP = {
  val props = new Properties()
  props.put("annotators", "tokenize, ssplit, pos, lemma")
  new StanfordCoreNLP(props)
}

def isOnlyLetters(str: String): Boolean = {
  str.forall(c => Character.isLetter(c))
}

def plainTextToLemmas(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP)
: Seq[String] = {
  val doc = new Annotation(text)
  pipeline.annotate(doc)
  val lemmas = new ArrayBuffer[String]()
  val sentences = doc.get(classOf[SentencesAnnotation])
  for (sentence <- sentences.asScala;
       token <- sentence.get(classOf[TokensAnnotation]).asScala) {
    val lemma = token.get(classOf[LemmaAnnotation])
    if (lemma.length > 2 && !stopWords.contains(lemma) && isOnlyLetters(lemma)) {
      lemmas += lemma.toLowerCase
    }
  }
  lemmas
}

// Load the stopwords list:
// (don't forget to use `toSet` ! The book didn't mention it, but an `Iterator` is not serializable...)
val stopWords = scala.io.Source.fromFile(stopwordsPath).getLines.toSet // toSet !

// broadcast the stopwords list so that they are available to the whole cluster:
// (note: don't use the `value` at the end as the book said, since we use `.value` later in the code...)
val bStopWords = spark.sparkContext.broadcast(stopWords)

// finally, do the cleansing:
// Note that we use `mapPartitions` so that we only initialize the NLP pipeline object
// once per partition instead of once per document.
val terms: Dataset[(String, Seq[String])] = docTexts.mapPartitions { iter =>
  val pipeline = createNLPPipeline()
  iter.map { case (title, contents) => (title, plainTextToLemmas(contents, bStopWords.value, pipeline)) }
}

// Result:
// ```
//scala> terms.describe()
//res18: org.apache.spark.sql.DataFrame = [summary: string, _1: string]
//
//scala> terms.take(1)
//Adding annotator tokenize
//Adding annotator ssplit
//Adding annotator pos
//Adding annotator lemma
//res19: Array[(String, Seq[String])] = Array((Sorting network,WrappedArray(sort, network, computer, science, ...
//scala>
// ```
//
// => __terms is now a Dataset of sequences of terms, one per document__

// ## Computing term frequencies in documents (TF)
// The `spark.ml` package already contains utilities to compute both the term frequency per document and the global term frequencies.

// First, we need to convert the Dataset into a Dataframe. We also remove the documents with less than two terms:
val termsDF = terms.toDF("title", "terms")
val filtered = termsDF.where(size($"terms") > 1)

// `CountVectorizer` and `CountVectorizerModel` from the ml library aim to help convert a collection of text documents to vectors of token counts.
// As per the [documentation](https://spark.apache.org/docs/1.5.1/ml-features.html#countvectorizer):
// > During the fitting process, CountVectorizer will select the top vocabSize words ordered by term frequency across the corpus. An optional parameter “minDF” also affect the fitting process by specifying the minimum number (or fraction if < 1.0) of documents a term must appear in to be included in the vocabulary.
// >
// > CountVectorizerModel produces sparse representations for the documents over the vocabulary, which can then be passed to other algorithms
import org.apache.spark.ml.feature.CountVectorizer

val numTerms = 20000 // number of terms to keep (the most frequent ones)
val countVectorizer = new CountVectorizer().setInputCol("terms").setOutputCol("termFreqs").setVocabSize(numTerms)
val vocabModel = countVectorizer.fit(filtered) // extract the vocabulary from our DF and construct the model
val docTermFreqs = vocabModel.transform(filtered) // transform our DF / apply the model

// ```
//docTermFreqs.show()
//
//+--------------------+--------------------+--------------------+
//|               title|               terms|           termFreqs|
//+--------------------+--------------------+--------------------+
//|     Sorting network|[sort, network, c...|(4744,[1,4,5,7,8,...|
//|  Tyranny of numbers|[tyranny, number,...|(4744,[1,3,4,5,6,...|
//|Reflected-wave sw...|[switching, switc...|(4744,[0,1,3,4,5,...|
//|Category:Computer...|[category, comput...|(4744,[0,1,8,30,5...|
//+--------------------+--------------------+--------------------+
// ```

// It is a good idea to cache `docTermFreqs`, since we will use it more than once:
docTermFreqs.cache()

// ## Computing inverse document frequencies
// From the [documentation](https://spark.apache.org/docs/2.1.0/ml-features.html#tf-idf):
//
// > IDF is an Estimator which is fit on a dataset and produces an IDFModel. The IDFModel takes feature vectors (generally created from HashingTF or CountVectorizer) and scales each column. Intuitively, it down-weights columns which appear frequently in a corpus.

import org.apache.spark.ml.feature.IDF

// Once again, we first construct a model and then feed it with our data:
val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidfVec")
val idfModel = idf.fit(docTermFreqs)
val docTermMatrix = idfModel.transform(docTermFreqs) //.select("title", "tfidfVec")
// we get:
// ```
//+--------------------+--------------------+--------------------+--------------------+
//|               title|               terms|           termFreqs|            tfidfVec|
//+--------------------+--------------------+--------------------+--------------------+
//|     Sorting network|[sort, network, c...|(4744,[1,4,5,7,8,...|(4744,[1,4,5,7,8,...|
//|  Tyranny of numbers|[tyranny, number,...|(4744,[1,3,4,5,6,...|(4744,[1,3,4,5,6,...|
//|Reflected-wave sw...|[switching, switc...|(4744,[0,1,3,4,5,...|(4744,[0,1,3,4,5,...|
//|Category:Computer...|[category, comput...|(4744,[0,1,8,30,5...|(4744,[0,1,8,30,5...|
//|Category:Computer...|[category, comput...|(4744,[1,30,271],...|(4744,[1,30,271],...|
//+--------------------+--------------------+--------------------+--------------------+
// ```
// To ensure it worked, we can extract the actual term and tfidf frequencies for the first row:.
vocabModel.vocabulary.take(8).zip {
  docTermMatrix.first.getAs[org.apache.spark.ml.linalg.SparseVector](2).toArray.zip {
    docTermMatrix.first.getAs[org.apache.spark.ml.linalg.SparseVector](3).toArray
  }
}
// We get:
// ```
// Array((system,(0.0,0.0)), (computer,(3.0,0.6531704515346116)), (university,(0.0,0.0)),
// (engineering,(0.0,0.0)), (use,(13.0,5.556772192750215)), (design,(2.0,1.065609060969532)),
// (engineer,(0.0,0.0)), (can,(18.0,10.269807452417034)))
//```
// As we can see, _computer_ and _can_ have been clearly downsized..

// -------------------
// ## Notes about `SparseVector`
// When using `show` on our matrix, we see something like `4744,[1,4,5,7,8,`. This is a `SparseVector`.
// the `toString` methods shows 3 elements:
//
// 1. its size: `4744` => the size of the vocabulary (`vocabModel.vocabulary.length`)
// 2. a list of indices (non-zero values)
// 3. a list of values at those indices
//
// To extract it, we can do:
docTermMatrix.
  first. // we the first row
  getAs[org.apache.spark.ml.linalg.SparseVector](2). // convert column 2 (termFreqs) into a sparseVector
  toDense // convert it to a dense vector
// ```
// org.apache.spark.ml.linalg.DenseVector = [0.0,3.0,0.0,0.0,13.0,2.0,0.0,18.0,...
//```
//  So, here it is clearer: we have the frequency for each word in our vocabulary:
val f = docTermMatrix.first.getAs[org.apache.spark.ml.linalg.SparseVector](2).toArray
// to get the most frequent words:
val wordsFreqs = f.zipWithIndex.map {
  case (freq, idx) => (vocabModel.vocabulary(idx), freq)
}.sortBy(-_._2)
// which yields:
//```
//Array[(String, Double)] = [(network,53.0), (sort,46.0), (can,18.0), (comparator,18.0),
// (wire,16.0), (value,15.0), (use,13.0), (size,12.0), (number,11.0), (depth,11.0),
// (input,10.0), (construct,8.0), (principle,8.0), (construction,8.0), ...  ]
// ```

// -------------------



// ## Indexing documents and terms
// Now, we need to get rid of strings, since as we work with matrices and vectors, non-numeric columns are not welcomed.
// The terms are already indiced by our `vocabulary` in `vocabModel.vocabulary`, but we also need indices for documents:
val termIds = vocabModel.vocabulary
val docIds = docTermFreqs.rdd.map(_.getString(0)).
  zipWithUniqueId(). // title => (title, idx)
  map(_.swap). // (title, idx) => (idx, title)
  collect().toMap


// ## Converting Dataset to RDD
// as explained in the book:
// > At the time of this writing, the spark.ml package, which operates on DataFrames, does not include an implementation of SVD. However, the older spark.mllib, which operates on RDDs, does. This means that, to compute the SVD of our document­term matrix, we need to represent it as an RDD of Vectors.

import org.apache.spark.mllib.linalg.{Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}

val vecRdd = docTermMatrix.select("tfidfVec").rdd.map { row =>
  Vectors.fromML(row.getAs[MLVector]("tfidfVec"))
}

// ## Applying the SVD algorithm
// The computation requires `O(nk)` storage on the driver, O(n) storage for each task, and O(k) passes over the data.

import org.apache.spark.mllib.linalg.distributed.RowMatrix

// _important note_: The RDD should be cached in memory beforehand because the computation requires multiple passes over the data.
vecRdd.cache()
val mat = new RowMatrix(vecRdd)
val k = 1000
val svd = mat.computeSVD(k, computeU = true)

// Output:
//```
//svd: org.apache.spark.mllib.linalg.SingularValueDecomposition[org.apache.spark.mllib.linalg.distributed.RowMatrix,org.apache.spark.mllib.linalg.Matrix] =
//SingularValueDecomposition(org.apache.spark.mllib.linalg.distributed.RowMatrix@43b25917,[412.9399696525229,312.0056141219418,216.90564372238936,212.83663472536242,207.65338945876277,189.53356981493454,162.7343588223653,146.22785130517616,145.10273359697285,143.34274103603087,139.82952741627577,137.91556834483055,127.07933593098204,106.06799112848157,100.10579199997758,90.14373515703484,87.07893418664202,80.36514625902937,71.4908098993236,70.1172162163747,68.4914279950864,55.48323686515424,49.6425795878338,47.11294330746061,45.04494671224828,44.47124688168499,44.125219835730974,42.61106176312821,41.39339476611446,37.227726198323126,31.0...
//```
// ## Getting understandable results

def topTermsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int, numTerms: Int, termIds: Array[String]): Seq[Seq[(String, Double)]] = {
  val v = svd.V
  val topTerms = new ArrayBuffer[Seq[(String, Double)]]()
  val arr = v.toArray
  for (i <- 0 until numConcepts) {
    val offs = i * v.numRows
    val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
    val sorted = termWeights.sortBy(-_._1)
    topTerms += sorted.take(numTerms).map { case (score, id) => (termIds(id), score) }
  }
  topTerms
}

val topConceptTerms = topTermsInTopConcepts(svd, 4, 10, termIds)
topConceptTerms(0).map(_._1).mkString(", ")

// Result: _stub, motherboard, slang, neutrality, peer, browse, disciplined, quantifiable, computerized, tank_

val topTermsForConcepts = bda.lsa.RunLSA.topTermsInTopConcepts(svd, 1, 5, vocabModel.vocabulary)
topTermsForConcepts.foreach(s => {
  println("TOPIC:")
  s.foreach { case (term, weight) =>
    println(s"${term}\t$weight")
  }
  println()
})

// Result:
//```
//TOPIC:
//stub    -2.4294473245818626E-7
//motherboard -1.585645154750931E-6
//slang   -2.3079451799351905E-6
//neutrality  -2.3079451800045794E-6
//peer    -2.3079451800245288E-6
//```



