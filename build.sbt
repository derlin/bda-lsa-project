// useful commands:
//  * sbt clean => clean the target directories
//  * sbt assembly => create a jar file with manifest and dependencies (see `project/assembly.sbt` and https://github.com/sbt/sbt-assembly)
//
// example:
//  sbt clean assembly
//  spark-submit target/scala-2.11/bda-project-lsa-assembly-1.0.jar 

name := "bda-project-lsa"

version := "1.0"
scalaVersion := "2.11.8"

// set the main class of the jar
mainClass in Compile := Some("bda.lsa.preprocessing.XmlToParquetWriter")

val scalaMajorVersion = "2.11"
val sparkVersion = "2.1.1"

// spark dependencies: marked as provided.
// if you use IntellliJ:
// - open Project Settings > Modules
// - click "+" add the bottom > Add Jars or Directories
// - select "apache-spark/2.1.1/jar" folder in your machine
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided"
)

// dependencies for parsing Wikipedia dumps
libraryDependencies ++= Seq(
  "edu.umd" % "cloud9" % "1.5.0"
)

libraryDependencies ++= Seq(
  "com.databricks" %% "spark-csv" % "1.5.0"
)

// stanford corenlp (https://stanfordnlp.github.io/CoreNLP/) is used
// to clean the XML file into plainttext
libraryDependencies ++= Seq(
  "edu.stanford.nlp" % "stanford-corenlp" % "3.7.0",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.7.0" classifier "models",
  "info.bliki.wiki" % "bliki-core" % "3.0.19"
)

unmanagedJars in Compile += file("lib/spark-corenlp-0.2.0-s_2.11.jar")

//libraryDependencies += "databricks" % "spark-corenlp" % "0.2.0-s_2.11"
//resolvers ++= Seq(
//  // other resolvers here
//  "Maven Releases" at "https://dl.bintray.com/spark-packages/maven/"
//)
