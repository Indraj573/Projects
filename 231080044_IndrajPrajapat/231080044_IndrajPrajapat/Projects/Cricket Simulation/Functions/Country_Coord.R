library(ggplot2)
west_indies <- c("Saint Lucia", "Anguilla", "Dominican Republic", "Guadeloupe", "Antigua", "Barbados", "Jamaica", "Trinidad", "Cayman Islands", "Bahamas", "Haiti", "Cuba", "Virgin Islands", "Martinique", "Saint Kitts", "Bermuda", "	
Saint Barthelemy")
r <- map_data("world")
colnames(r)[5] <- "teams"
for (i in 1:length(r$teams)) {
  if (r$teams[i] %in% west_indies) {
    r$teams[i] <- "West Indies"
  }
  if (r$teams[i] == "UK") {
    r$teams[i] <- "England"
  }
  if (r$teams[i] == "United Arab Emirates") {
    r$teams[i] <- "U.A.E"
  }
  if (r$teams[i] == "USA") {
    r$teams[i] <- "U.S.A."
  }
  if (r$teams[i] == "Papua New Guinea") {
    r$teams[i] <- "P.N.G."
  }
}
coord <- r
save(coord, file = "../Data Sets/Cricket_Coord.RData")
