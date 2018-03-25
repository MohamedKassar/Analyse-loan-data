package fr.upmc


import java.io.FileWriter

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

    //val conf = new SparkConf()
    //conf.setMaster("local[2]").setAppName("Analyse Loan Data")


    //val sc = new SparkContext(conf)

    val sqlContext = SparkSession
      .builder()
      .appName("Analyse Loan Data").master("local[2]")
      .getOrCreate()

    sqlContext.sparkContext.setLogLevel("ERROR")

    println("Hello World!")


    val dataf = sqlContext.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("loan.csv")


    var cleanDF = dataf.select("*").withColumnRenamed("earliest_cr_line", "earliest_credit_line")
      .withColumnRenamed("inq_last_6mths", "inquiries_last_6_months")
      .withColumnRenamed("int_rate", "interest_rate")
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
    // useless : issue_d - pymnt_plan - url - desc - addr_state
    // recheck : earliest_cr_line - mths_since_last_delin - mths_since_last_record - initial_list_status
    println(cleanDF.count())
    val header = cleanDF.first
    cleanDF = cleanDF.filter(l => header != l)
    import sqlContext.implicits._
    val cachedData = cleanDF.select("dti", "delinquent_2_years", "inquiries_last_6_months",
      "zip_code", "home_ownership", "grade", "installment", "interest_rate", "term", "loan_amount", "funded_amount",
      "investor_funds", "sub_grade", "emp_title", "emp_length", "annual_income", "verification_status", "purpose", "title",
      "open_account", "public_records", "revol_bal", "revol_util", "total_acc") /*.where("dti < 500000")*/ .map(_.toString)
      .toDF().rdd.map(l => {
      val ss = l.toString().replaceAll("\\[", "").replaceAll("\\]", "").split(",");

      try {
        var i = -1

        def I = {
          i = i + 1;
          i
        }

        def convertS(s: String) = {
          if (s == null) "unknown".hashCode else s.trim.toLowerCase.hashCode
        }

        def convertD(s: String) = {
          s.toDouble
        }

        Vectors.dense(
          convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I).substring(0, 3)), convertS(ss(I)),
          convertS(ss(I)), convertS(ss(I)), convertS(ss(I)), convertS(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I)),
          convertS(ss(I)), convertS(ss(I)), convertS(ss(I)), convertD(ss(I)), convertS(ss(I)), convertS(ss(I)), convertS(ss(I)),
          convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I))
        )
      } catch {
        case _ => Vectors.dense(-1)
      }
    }).filter(l => l(0) != -1).cache()

    println(cachedData.count())
    var x = 1
    val fw = new FileWriter("test.txt", true)

    for (x <- 2 to 10) {
      println("---------------- k = " + x + "  -------------")
      val clusters = KMeans.train(cachedData, x, 20)
      clusters.clusterCenters.foreach(println)
      val WSSSE = clusters.computeCost(cachedData)
      println("---------------- WSSSE = " + WSSSE + "  -------------")
      try {
        fw.write("\nk = " + x + "\nWSSSE = " + WSSSE)
        fw.flush()
      }
    }
    fw.close()
    /*
    val clusters = KMeans.train(cachedData, 2, 20)
    clusters.clusterCenters.foreach(println)
    val WSSSE = clusters.computeCost(cachedData)
    */
    //println(clusters.predict(Vectors.dense(1.0, 4.0)))
    //Vectors.dense(s.map(_.toString.toDouble).toArray)


  }
}
