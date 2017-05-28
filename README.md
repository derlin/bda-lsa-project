# Building and Running

__Building the jar__

This project uses sbt. To create the jar, use:
    
    export JAVA_OPTS="-Xms256m -Xmx4g"
    sbt assembly

The jar is now available under `target/scala-2.11/bda-project-lsa-assembly-1.0.jar`.

__Project structure__

The project is made of multiple spark programs. Each program stores its output on disk, the actual location depending on the properties set in `config.properties`.
 
 A usual pipeline is:
 
 1. convert wikidump XML into plain text (`bda.lsa.preprocessing.XmlToParquetWriter`)
 2. create the vocabulary and the TF-IDF matrix (`bda.lsa.preprocessing.DocTermMatrixWriter`)
 3. create one of the models (`bda.lsa.svd.RunSVD`, `bda.lsa.lda.mllib.RunLDA`, `bda.lsa.lda.ml.RunLDA`)
 
 At this point, the model is persisted somewhere and you can load it inside a spark-shell to interact with it. The classes `bda.lsa.svd.SVDQueryEngine`, `bda.lsa.lda.mllib.LDAQueryEngine` and `bda.lsa.lda.ml.LDAQueryEngine` implement useful queries to analyse the models. 


__Example of pipeline__

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

        val q = new bda.lsa.svd.SVDQueryEngine(spark, data)
   
   
   
# Models available

TODO


# Running on the daplab

TODO

export environment variables:

    export SPARK_MAJOR_VERSION=2
    export HADOOP_CONF_DIR=/etc/hadoop/conf
    export LD_LIBRARY_PATH=/usr/hdp/current/hadoop-client/lib/native:$LD_LIBRARY_PATH
    
launch the shell on yarn:

    spark-shell --master yarn --deploy-mode client  --jars target/scala-2.11/bda-project-lsa-assembly-1.0.jar    
    spark-shell --master yarn --deploy-mode client  --jars target/scala-2.11/bda-project-lsa-assembly-1.0.jar  --driver-memory 2G --executor-memory 15G --executor-cores 8 
    
    spark-shell --master yarn-client --num-executors 10 --driver-memory 20G --executor-memory 10g --conf spark.rpc.message.maxSize=100 spark.shuffle.manager=SORT spark.yarn.executor.memoryOverhead=4096 --jars target/scala-2.11/bda-project-lsa-assembly-1.0.jar
    
    spark-shell --master yarn-client --num-executors 10 --driver-memory 20G --executor-memory 10g --conf spark.rpc.message.maxSize=100 spark.shuffle.manager=SORT spark.yarn.executor.memoryOverhead=4096 --jars target/scala-2.11/bda-project-lsa-assembly-1.0.jar
    
    
Local

    spark-submit --class bda.lsa.PrepareData --master "local[*]"  target/scala-2.11/bda-project-lsa-assembly-1.0.jar wikidump.xml wikidump.txt
    
    
    spark-submit --class bda.lsa.preprocessing.DocTermMatrixWriter  --master yarn-client --num-executors 3 --driver-memory 20G --executor-memory 10g target/scala-2.11/bda-project-lsa-assembly-1.0.jar 5000
    
    
sbt out of memory error: 

       export JAVA_OPTS="-Xms256m -Xmx4g"