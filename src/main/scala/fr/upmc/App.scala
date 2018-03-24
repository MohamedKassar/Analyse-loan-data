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

    printf("5000.0".toDouble.toString)
    val header = cleanDF.first
    cleanDF = cleanDF.filter(l => header != l)
    import sqlContext.implicits._
    import sqlContext._
    val cachedData = cleanDF.select("loan_amount", "annual_income").where("annual_income < 500000").map(_.toString) /*.map(s=> {
        val ss = s.replaceAll("\\[","").replaceAll("\\]","").split(",");
        Array(ss(0).toDouble, ss(1).toDouble)
        })*/ .toDF().rdd.map(l => {

      val ss = l.toString().replaceAll("\\[", "").replaceAll("\\]", "").split(",");

      var a: Double = -1
      var b: Double = -1
      try {
        a = ss(0).toDouble
        b = ss(1).toDouble
      }catch {
        case _ => a = -1; b = -1
      }
      Vectors.dense(a/10000, b/10000)
    }).filter(l=>l(0) != -1 && l(1) != -1).cache()

    println(cachedData.count())

    val clusters = KMeans.train(cachedData, 6, 20)

    clusters.clusterCenters.foreach(println)
    //Vectors.dense(s.map(_.toString.toDouble).toArray)


  }
}
