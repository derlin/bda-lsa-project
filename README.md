# Running on the daplab

export environment variables:

    export SPARK_MAJOR_VERSION=2
    export HADOOP_CONF_DIR=/etc/hadoop/conf
    export LD_LIBRARY_PATH=/usr/hdp/current/hadoop-client/lib/native:$LD_LIBRARY_PATH
    
launch the shell on yarn:

    spark-shell --master yarn --deploy-mode client  --jars target/scala-2.11/bda-project-lsa-assembly-1.0.jar    
    spark-shell --master yarn --deploy-mode client  --jars target/scala-2.11/bda-project-lsa-assembly-1.0.jar  --driver-memory 2G --executor-memory 15G --executor-cores 8 
    
    spark-shell --master yarn-client --num-executors 10 --driver-memory 20G --executor-memory 10g --conf spark.rpc.message.maxSize=100 spark.shuffle.manager=SORT spark.yarn.executor.memoryOverhead=4096 --jars target/scala-2.11/bda-project-lsa-assembly-1.0.jar
    
    
    
Local

    spark-submit --class bda.lsa.PrepareData --master "local[*]"  target/scala-2.11/bda-project-lsa-assembly-1.0.jar wikidump.xml wikidump.txt
    
    
    spark-submit --class bda.lsa.PrepareData  --master yarn-client --num-executors 10 --driver-memory 20G --executor-memory 10g target/scala-2.11/bda-project-lsa-assembly-1.0.jar 