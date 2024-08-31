library(tidyverse)
library(ggplot2)
library(rvest)

load(file = "../Data Sets/Teams_Table.Rdata")
table2 <- NULL
for (i in 1971:2023) {
  print(i)
  html <- read_html(paste0("https://www.espncricinfo.com/records/year/team-match-results/", i, "-", i, "/one-day-internationals-2"))
  table1 <- html_table(html)[[1]]
  table1 <- table1 %>% filter(X1 != "Team 1")
  link <- html_elements(html, "a")
  link <- html_attr(link, "href")
  link <- unique(link)
  link <- link[substr(link, 1, 8) == "/series/"]
  table1 <- cbind(table1, paste0("https://www.espncricinfo.com", link))
  table1 <- cbind(table1, numeric(dim(table1)[1]))
  table1 <- cbind(table1, numeric(dim(table1)[1]))
  table2 <- rbind(table2, table1)
}

colnames(table2) <- c("Team 1", "Team 2", "Winner", "Margin", "Ground", "Match Date", "Scorecard", "Link", "Winner_Score", "Loser_Score")

table2 <- table2 %>% filter(Winner != "no result") %>% filter(Winner != "tied") %>% filter(Margin != "-")
table2 <- rbind(c("England", "New Zealand", "England", "0 runs", "Lord's", "2019-07-14", "Scorecard", "Link", 241, 241), table2)
d1 <- dim(table)[1]
d2 <- dim(table2)[1]
if (d1 == d2) {
  stop("No new data.")
}
table2 <- table2[c(d1 + 1: d2),]
table2 <- table2 %>% filter(!is.na(table2$`Team 1`))
links <- table2$Link
for (i in 1:dim(table2)[1]) {
  print(i)
  html <- read_html(links[i])
  tables <- html_table(html)
  table_1 <- tables[[1]]
  table_2 <- tables[[3]]
  name <- html_elements(html, "span")
  name <- html_text(name)
  name <- unique(name)
  c1 <- table_1$R[dim(table_1)[1] - 2]
  c2 <- table_2$R[dim(table_2)[1] - 2]
  if (!is.na(as.integer(substr(table_1$R[dim(table_1)[1] - 1], 1, 1)))) {
    c1 <- table_1$R[dim(table_1)[1] - 1]
  }
  if (!is.na(as.integer(substr(table_2$R[dim(table_2)[1] - 1], 1, 1)))) {
    c2 <- table_2$R[dim(table_2)[1] - 1]
  }
  total_1 <- NULL
  total_2 <- NULL
  if (substr(c1, nchar(c1) - 1, nchar(c1) - 1) == "/") {
    total_1 <- as.integer(substr(c1, 1, which(strsplit(c1, "")[[1]] == "/") - 1))
  } else {
    total_1 <- as.integer(c1)
  }
  if (substr(c2, nchar(c2) - 1, nchar(c2) - 1) == "/") {
    total_2 <- as.integer(substr(c2, 1, which(strsplit(c2, "")[[1]] == "/") - 1))
  } else {
    total_2 <- as.integer(c2)
  }
  
  winner <- table2$Winner[i]
  loser <- NULL
  if (table2$`Team 1`[i] == winner) {
    loser <- table2$`Team 2`[i]
  } else {
    loser <- table2$`Team 1`[i]
  }
  ind1 <- which(name == paste0(winner, " Innings"))
  ind2 <- which(name == paste0(loser, " Innings"))
  if (ind1 < ind2) {
    table2$Winner_Score[i] <- total_1
    table2$Loser_Score[i] <- total_2
  }
  else {
    table2$Loser_Score[i] <- total_1
    table2$Winner_Score[i] <- total_2
  }
}

t1 <- table2
Loser <- numeric(length(t1$`Team 1`))

for (i in 1:length(Loser)) {
  if (t1$Winner[i] == t1$`Team 1`[i]) {
    Loser[i] = t1$`Team 2`[i]
  }
  if (t1$Winner[i] == t1$`Team 2`[i]) {
    Loser[i] = t1$`Team 1`[i]
  }
}
table2 <- cbind(t1, Loser)
table2 <- table2 %>% select(Winner, Loser, Margin, Winner_Score, Loser_Score, Ground, `Match Date`)

mon <- month.abb

conv_date <- function(date) {
  dd <- strsplit(date, " ")[[1]]
  y <- dd[3]
  m <- which(mon == dd[1], arr.ind = TRUE)
  d <- dd[2]
  if (!(gregexpr(pattern ='-',d)[[1]][1])) {
    d <- substr(d, 1, nchar(d) - 1)
  }
  if (gregexpr(pattern ='-',d)[[1]][1]) {
    d <- substr(d, gregexpr(pattern ='-',d)[[1]][1] + 1, nchar(d) - 1)
  }
  date <- paste0(y, "/", m, "/", d)
  return(date)
}

for (i in 1:length(table2$`Match Date`)) {
  table2$`Match Date`[i] <- conv_date(table2$`Match Date`[i])
}
table2$`Match Date` <- as.Date(table2$`Match Date`)

Winning_Innings <- NULL
for (i in 1:length(table2$Margin)) {
  if ((substr(table2$Margin[i], nchar(table2$Margin[i]) - 6, nchar(table2$Margin[i]) - 1) == "wicket") | (substr(table2$Margin[i], nchar(table2$Margin[i]) - 5, nchar(table2$Margin[i])) == "wicket")) {
    Winning_Innings <- append(Winning_Innings, "Bowling")
  }
  else {
    Winning_Innings <- append(Winning_Innings, "Batting")
  }
}


table2 <- data.frame(table2, Winning_Innings)
colnames(table2)[7] <- "Match Date"
table <- rbind(table, table2)
save(table, team_filter, file = "../Data Sets/Teams_Table.Rdata")
