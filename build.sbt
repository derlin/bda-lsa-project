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
mainClass in Compile := Some("bda.lsa.RunLSA")

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
  "edu.stanford.nlp" % "stanford-corenlp" % "3.4.1",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.4.1" classifier "models",
  "info.bliki.wiki" % "bliki-core" % "3.0.19"
)

// I don't think it is important after all...
//resolvers ++= Seq(
//  // other resolvers here
//  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
//)
