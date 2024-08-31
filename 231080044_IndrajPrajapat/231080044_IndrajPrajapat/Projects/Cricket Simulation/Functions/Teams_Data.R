library(tidyverse)
library(ggplot2)
library(rvest)

table <- NULL
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
  table <- rbind(table, table1)
}

colnames(table) <- c("Team 1", "Team 2", "Winner", "Margin", "Ground", "Match Date", "Scorecard", "Link", "Winner_Score", "Loser_Score")

table <- table %>% filter(Winner != "no result") %>% filter(Winner != "tied") %>% filter(Margin != "-")
links <- table$Link


for (i in 1:dim(table)[1]) {
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
  }
  else {
    total_1 <- as.integer(c1)
  }
  if (substr(c2, nchar(c2) - 1, nchar(c2) - 1) == "/") {
    total_2 <- as.integer(substr(c2, 1, which(strsplit(c2, "")[[1]] == "/") - 1))
  }
  else {
    total_2 <- as.integer(c2)
  }
  
  winner <- table$Winner[i]
  loser <- NULL
  if (table$`Team 1`[i] == winner) {
    loser <- table$`Team 2`[i]
  }
  else {
    loser <- table$`Team 1`[i]
  }
  ind1 <- which(name == paste0(winner, " Innings"))
  ind2 <- which(name == paste0(loser, " Innings"))
  if (ind1 < ind2) {
    table$Winner_Score[i] <- total_1
    table$Loser_Score[i] <- total_2
  }
  else {
    table$Loser_Score[i] <- total_1
    table$Winner_Score[i] <- total_2
  }
}
t1 <- table
Loser <- numeric(length(t1$`Team 1`))
for (i in 1:length(Loser)) {
  if (t1$Winner[i] == t1$`Team 1`[i]) {
    Loser[i] = t1$`Team 2`[i]
  }
  if (t1$Winner[i] == t1$`Team 2`[i]) {
    Loser[i] = t1$`Team 1`[i]
  }
}
table <- cbind(t1, Loser)
table <- table %>% select(Winner, Loser, Margin, Winner_Score, Loser_Score, Ground, `Match Date`)

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

for (i in 1:length(table$`Match Date`)) {
  table$`Match Date`[i] <- conv_date(table$`Match Date`[i])
}
d <- length(table$Winner)
v <- c("England", "New Zealand", "0 wickets", 241, 241, "Lord's", "2019-07-14", "Bowling")
v[7] <- as.Date(v[7])
table <- rbind(table, v)
table$`Match Date` <- as.Date(table$`Match Date`)
table$Winner_Score <- as.integer(table$Winner_Score)
table$Loser_Score <- as.integer(table$Loser_Score)
summary(table)
Winning_Innings <- NULL
for (i in 1:length(table$Margin)) {
  if ((substr(table$Margin[i], nchar(table$Margin[i]) - 6, nchar(table$Margin[i]) - 1) == "wicket") | (substr(table$Margin[i], nchar(table$Margin[i]) - 5, nchar(table$Margin[i])) == "wicket")) {
    Winning_Innings <- append(Winning_Innings, "Bowling")
  }
  else {
    Winning_Innings <- append(Winning_Innings, "Batting")
  }
}

colnames(table)[7] <- "Match Date"
table <- data.frame(table, Winning_Innings)
table[d + 1, 7] <- as.Date("2019-07-14")
team_filter <- function(team) {
  table <- table %>% select(Winner, Loser, Margin, Winning_Innings, Winner_Score, Loser_Score, Ground, `Match Date`)
  table1 <- table %>% filter(Winner == team | Loser == team)
  Team_Score <- NULL
  Opp_Score <- NULL
  Opponent <- NULL
  Team <- NULL
  for (i in 1:length(table1$Loser)) {
    if (table1$Winner[i] == team) {
      Team_Score = append(Team_Score, table1$Winner_Score[i])
      Opp_Score = append(Opp_Score, table1$Loser_Score[i])
      Team <- append(Team, team)
      Opponent = append(Opponent, table1$Loser[i])
    }
    if (table1$Loser[i] == team) {
      Team_Score = append(Team_Score, table1$Loser_Score[i])
      Opp_Score = append(Opp_Score, table1$Winner_Score[i])
      Team <- append(Team, team)
      Opponent = append(Opponent, table1$Winner[i])
    }
  }
  table1 <- cbind(table1, Team_Score, Opp_Score, Team, Opponent) %>% select(Team, Opponent, Winning_Innings, Team_Score, Opp_Score, Winner, Loser, Winner_Score, Loser_Score, Margin, Ground, 'Match Date')
  table1$Winner_Score <- as.integer(table1$Winner_Score)
  table1$Loser_Score <- as.integer(table1$Loser_Score)
  table1$Team_Score <- as.integer(table1$Team_Score)
  table1$Opp_Score <- as.integer(table1$Opp_Score)
  return(table1)
}

colnames(table)[7] <- "Match Date"
table <- table %>% arrange(table$`Match Date`)
save(table, team_filter, file = "../Data Sets/Teams_Table.Rdata")
