
/**
  * Created by kewin on 21.05.17.
  */

// To run this script, run spark-shell with the jar :
// ```
// spark-shell --jars target/scala-2.11/bda-project-lsa-assembly-1.0.jar
// ```

// Start by loading the necessary imports :
// ```
// :load spark-shell-scripts/imports.scala
// ```

// Then build the model
val model = lda.mllib.RunLDA.loadModel(spark);

// We then get the saved data and build the corpus with it
val (dtm, termIds, docIds, tfids) = bda.lsa.getData(spark);
val corpus = bda.lsa.docTermMatrixToCorpusRDD(spark, dtm);

// And finally, we create the QueryEngine
val q = new lda.mllib.LDAQueryEngine(spark, model, corpus, termIds);

// Now you can use the `q` to query the model with what you want. for example :
q.describeTopicsWithWords(5).map(_.mkString(",")).zipWithIndex.mkString("\n");