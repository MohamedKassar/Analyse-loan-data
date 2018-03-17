package fr.upmc


import org.apache.spark._
import org.apache.spark.sql.{SQLContext, SparkSession}


/**
  * Hello world!
  *
  */
object App {
  def main(args: Array[String]) {

    val conf = new SparkConf()
    conf.setMaster("local").setAppName("Analyse Loan Data")

    //val sc = new SparkContext(conf)

    val sqlContext = SparkSession
      .builder()
      .appName("Analyse Loan Data").master("local[2]")
      .getOrCreate()

    println("Hello World!")


    val dataf = sqlContext.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("loan.csv")

    dataf.select("*").limit(100).show()
  }
}
