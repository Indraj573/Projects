gdp <- read.csv("../gdp_per_capita.csv")
gdp
typeof(gdp$X1960)
for (i in 3:63) {
  colnames(gdp)[i] <- 1957 + i
}
i <- 1965
as.double((gdp %>% filter(Country.Name == "India"))[4] + (gdp %>% filter(Country.Name == "India"))[5])
for (i in 1:length(gdp$Country.Name)) {
  if (gdp$Country.Name[i] == "United Kingdom") {
    gdp$Country.Name[i] = "England"
  }
  if (gdp$Country.Name[i] == "United Arab Emirates") {
    gdp$Country.Name[i] = "U.A.E."
  }
  if (gdp$Code[i] == "HKG") {
    gdp$Country.Name[i] = "Hong Kong"
  }
  if (gdp$Country.Name[i] == "United States") {
    gdp$Country.Name[i] = "U.S.A."
  }
}
gdp
save(gdp, file = "../Data Sets/gdp_per_capita.csv")
