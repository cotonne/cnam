package inlfuxdb

import models.ApacheLog

import reflect.runtime.universe._
import reflect.runtime.currentMirror
import scala.collection.immutable

object ToInflux {
  def format(a: ApacheLog): String = {
    val r = currentMirror.reflect(a)
    val tmp = r.symbol.typeSignature.members.toStream
      .collect { case s: TermSymbol if !s.isMethod => r.reflectField(s) }
      .map(r => r.symbol.name.toString.trim + "=" + r.get.toString )
      .mkString(",")
    return ""
  }
}
