package fr.upmc

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Hello world!
  *
  */
object App {
  def main(args: Array[String]) {

    var conf = new SparkConf()
    val sc = new SparkContext(conf)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    println("Hello World!")



    val rdd = sc.textFile("loan.csv")

    println(rdd.count)

    val dataf = sqlContext.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/home/osboxes/Desktop/DAR/Analyse-loan-data/loan.csv")
    display(dataf)
    conf.setMaster("local").setAppName("Analyse Loan Data")
  }
}
