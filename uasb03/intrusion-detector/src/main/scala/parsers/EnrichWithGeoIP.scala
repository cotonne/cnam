package parsers

import java.io.File
import java.net.InetAddress

import com.maxmind.geoip2.DatabaseReader
import models.ApacheLog

object EnrichWithGeoIP {
  private val database = new File(config.Params.PATH_TO_GEOLITE2)
  private val reader = new DatabaseReader.Builder(database).build();
  def apply(logApache: ApacheLog): ApacheLog = logApache.copy(isoCode = reader
      .country(InetAddress.getByName(logApache.remoteHost)).getCountry.getIsoCode)
}
