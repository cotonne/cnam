package parsers

import config.Params
import java.time.{LocalDateTime, ZoneOffset}
import java.time.format.DateTimeFormatter._

import inlfuxdb.ToInflux
import models.ApacheLog
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import sttp.client._
import sttp.model.StatusCode

object ApacheLogParser {
  def main(args: Array[String]): Unit = {
    val config = new SparkConf().setMaster("local[4]").setAppName("My Application")
    val sc = new SparkContext(config)

    val apacheLog: RDD[String] = sc.textFile(Params.PATH_TO_APACHE_LOG)

    val apacheLogRegex = ("(.*?) \\- (.*?) (\\[.*?\\]) \"(.*?)\" ([0-9]+?|-) (\\d+|-) \"(.*?)\" \"(.*?)\" \"(.*?)\"").r

    val apacheLogMessages = apacheLog map {
      l => {
        l match {
          case apacheLogRegex(remoteHost,
          remoteUser,
          timeReceived,
          requestFirstLine,
          status,
          responseBytes,
          requestHeaderReferer,
          requestHeaderUserAgent,
          remoteLogName) => ApacheLog(remoteHost,
            "??",
            remoteUser,
            LocalDateTime.parse(timeReceived, ISO_DATE_TIME),
            requestFirstLine,
            "",
            "",
            "",
            status.toInt,
            responseBytes.toInt,
            requestHeaderReferer,
            requestHeaderUserAgent,
            "",
            "",
            "",
            "",
            remoteLogName)
        }
      }
    }

    val content: RDD[ApacheLog] = apacheLogMessages
      .map(EnrichWithGeoIP.apply)
      .map(EnrichUserAgent.apply)
      .map(DecomposeURL.apply)

    content.saveAsTextFile(Params.PATH_TO_EXPORT_LOG)

    implicit val backend: SttpBackend[Identity, Nothing, NothingT] = HttpURLConnectionBackend()

    content.foreach((f) => {
      val measurement = "apache_log"
      val key = ToInflux.format(f)
      val timestamp = f.timeReceived.toEpochSecond(ZoneOffset.UTC)
      val request = basicRequest
        .post(uri"${Params.INFLUX_DB}/api/v2/write?db=mydb")
        .body(s"$measurement,$key $timestamp")
      val response = request.send()
      if(response.code != StatusCode.NoContent) {
        println("Error while saving " + f)
      }
    })
  }
}
