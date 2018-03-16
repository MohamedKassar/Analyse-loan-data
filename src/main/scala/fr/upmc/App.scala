package fr.upmc

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Hello world!
  *
  */
object App {
  def main(args: Array[String]) {
    var conf = new SparkConf()
    println("Hello World!")
    conf.setMaster("local").setAppName("Analyse Loan Data")

    val sc = new SparkContext(conf)
    val rdd = sc.textFile("loan0.csv")

    println(rdd.count)
  }

}
