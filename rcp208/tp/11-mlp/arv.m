function [e] = arv(y, d)
  D = length(y)
  sum((y - d)^ 2)
  [e] = 