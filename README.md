# Table of Contents

- [About this repository](#about-this-repository)
  * [Structure](#structure)
- [Building and Running](#building-and-running)
  * [Building the jar](#building-the-jar)
  * [Project structure](#project-structure)
  * [Example of pipeline](#example-of-pipeline)
- [Models available](#models-available)
  * [SVD](#svd)
  * [LDA](#lda)
      - [ml.LDA](#mllda)
      - [mllib.LDA](#mlliblda)
  
 <small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>
 
# About this repository

_Context_: BDA (Big Data Analytics), Master MSE, june 2017.

_Authors_: Lucy Linder, Kewin Dousse, Davide Mazzolini, Christophe Blanquet.

This project is based on Chapter 6 of the book [_Advanced Analytics with Spark_](https://github.com/sryza/aas). It contains code and information on how to apply LSA techniques to the English Wikipedia articles corpus. 

## Structure

- the [source code](src/main/scala) is made of multiple classes thoroughly commented and documented with scaladoc
- the folder [spark-shell-scripts](spark-shell-scripts) contains scripts intended to be loaded into a spark-shell session. Once again, most of them are documented
- READMEs are scattered at each level in order to help you understand the repository structure
- the [wiki](https://github.com/derlin/bda-lsa-project/wiki) contains the results, notes, tips and tricks, how-to etc. This is a good place to start if you just want to see what we did.

# Building and Running

## Building the jar

This project uses sbt. To create the jar, use:
    
    export JAVA_OPTS="-Xms256m -Xmx4g"
    sbt assembly

The jar is now available under `target/scala-2.11/bda-project-lsa-assembly-1.0.jar`.

## Project structure

The project is made of multiple spark programs. Each program stores its output on disk, the actual location depending on the properties set in `config.properties`.
 
 A usual pipeline is:
 
 1. convert wikidump XML into plain text (`bda.lsa.preprocessing.XmlToParquetWriter`)
 2. create the vocabulary and the TF-IDF matrix (`bda.lsa.preprocessing.DocTermMatrixWriter`)
 3. create one of the models (`bda.lsa.svd.RunSVD`, `bda.lsa.lda.mllib.RunLDA`, `bda.lsa.lda.ml.RunLDA`)
 
 At this point, the model is persisted somewhere and you can load it inside a spark-shell to interact with it. The classes `bda.lsa.svd.SVDQueryEngine`, `bda.lsa.lda.mllib.LDAQueryEngine` and `bda.lsa.lda.ml.LDAQueryEngine` implement useful queries to analyse the models. 


## Example of pipeline

1. ensure you have a _wikidump_ somewhere to process.
2. create the jar: 

        export JAVA_OPTS="-Xms256m -Xmx4g"
        sbt assembly
        
3. create a `config.properties` and add the following:

        path.wikidump=wikidump-1500.xml # your xml 
        path.base=/tmp/spark-wiki/   # a base path to store the results
        
4. convert XML to text:  
 
        spark-submit --class bda.lsa.preprocessing.XmlToParquetWriter \
            target/scala-2.11/bda-project-lsa-assembly-1.0.jar
            
    This will save the DataFrame `[(title: String, content: String) ]` in `path.base/wikidump-parquet`.
    __important__: the job will fail if the aforementioned directory already exists !

5. create the TF-IDF matrix:

        spark-submit --class bda.lsa.preprocessing.DocTermMatrixWriter \
                target/scala-2.11/bda-project-lsa-assembly-1.0.jar  \
                <numTerms> [<percent> <numDocs>]
                
   The only required argument is `numTerms`: this is the size of the vocabulary.
    
6. Run one of the models. For example, for svd:
    
        spark-submit --class bda.lsa.svd.RunSVD \
                       target/scala-2.11/bda-project-lsa-assembly-1.0.jar  \
                       <k: default 100>
     
   The parameter `k` is the number of topics to infer, default to 100.
   
7. open a spark-shell and load the model:

        spark-shell --jars target/scala-2.11/bda-project-lsa-assembly-1.0.jar 
        val data = bda.lsa.getData(spark)
        val model = bda.lsa.svd.RunSVD.loadModel(spark)
        
8. Optionally, use the querier in the shell:

        val q = new bda.lsa.svd.SVDQueryEngine(model, data)
   
   
   
# Models available

## SVD

The class `bda.lsa.svd.RunSVD` makes it easy to compute an SVD model.

The results are saved in `{base.path}/svd`. More information about SVD can be found in the [svd package README](/derlin/bda-lsa-project/blob/master/src/main/scala/bda/lsa/svd/Readme.md).

After creating the model (see steps above), you can use the `bda.lsa.svd.SVDQueryEngine` to discover the results. From a spark-shell:

```
spark-shell --jars bda-project-lsa-assembly-1.0.jar
> val data = bda.lsa.getData(spark)
> val model = bda.lsa.svd.RunSVD.loadModel(spark)
> val q = new bda.lsa.svd.SVDQueryEngine(model, data)
```

See the [wiki](/derlin/bda-lsa-project/wiki) for our results and conclusion.
 
## LDA
 
 LDA models are available in two flavors: with _spark mllib_ and _spark ml_.
 
  We focused on the mllib implementation, mostly because the `org.apache.spark.mllib.clustering.DistributedLDAModel` offer more utility methods than it's ml counterpart. Our ml implementation will creates the model, but does not offer a useful query engine.
  
#### ml.LDA

 
To run the model:

    spark-submit --class bda.lsa.lda.ml.RunLDA \
          target/scala-2.11/bda-project-lsa-assembly-1.0.jar  \
          <k: default 100>  <maxIters: default 100> 
          
The model is then saved to `{base.path}/ml-lda`.

After creating the model (see steps above), you can use the `bda.lsa.lda.mllib.LDAQueryEngine` to discover the results. From a spark-shell:
 
 ```
 spark-shell --jars bda-project-lsa-assembly-1.0.jar
 > val data = bda.lsa.getData(spark)
 > val model = bda.lsa.lda.ml.RunLDA.loadModel(spark)
 ```
Note that there is no query engine available for this kind of model. 
 
#### mllib.LDA
 
 
To run the model:

    spark-submit --class bda.lsa.lda.mllib.RunLDA \
          target/scala-2.11/bda-project-lsa-assembly-1.0.jar  \
          <k: default 100>  <maxIters: default 100> <alpha: default -1>  <beta: default -1>
          
The model is then saved to `{base.path}/mllib-lda`.

After creating the model (see steps above), you can use the `bda.lsa.lda.mllib.LDAQueryEngine` to discover the results. From a spark-shell:
 
 ```
 spark-shell --jars bda-project-lsa-assembly-1.0.jar
 > val data = bda.lsa.getData(spark)
 > val model = bda.lsa.lda.mllib.RunLDA.loadModel(spark)
 > val q = new bda.lsa.lda.mllib.LDAQueryEngine(model, data)
 ```
 
 See the [wiki](/derlin/bda-lsa-project/wiki) for our results and conclusion.

