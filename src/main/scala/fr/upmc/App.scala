package fr.upmc


import org.apache.spark._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.bouncycastle.util.CollectionStore
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
/**
  * Hello world!
  * Hello world!
  *
  */
object App {


  def main(args: Array[String]) {

    val conf = new SparkConf()
    conf.setMaster("local").setAppName("Analyse Loan Data")

    val kk = new KMeans()


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



    val cleanDF = dataf.select("*").withColumnRenamed("earliest_cr_line","earliest_credit_line")
      .withColumnRenamed("inq_last_6mths","inquiries_last_6_months")
      .withColumnRenamed("int_rate","interest_rate")
      .withColumnRenamed("loan_amnt", "loan_amount")
      .withColumnRenamed("funded_amnt", "funded_amount")
      .withColumnRenamed("funded_amnt_inv", "investor_funds")
      .withColumnRenamed("int_rate", "interest_rate")
      .withColumnRenamed("annual_inc", "annual_income")
      .withColumnRenamed("delinq_2yrs", "delinquent_2_years")
      .withColumnRenamed("mths_since_last_delinq", "months_since_last_delinquent")
      .withColumnRenamed("open_acc", "open_account")
      .withColumnRenamed("pub_rec", "public_records")
      //.withColumnRenamed("open_acc", "open_account").limit(100).show()
      import sqlContext.implicits._
      val cachedData = cleanDF.select("loan_amount","annual_income").map(s => s).show()
    //Vectors.dense(s.map(_.toString.toDouble).toArray)

  }
}
