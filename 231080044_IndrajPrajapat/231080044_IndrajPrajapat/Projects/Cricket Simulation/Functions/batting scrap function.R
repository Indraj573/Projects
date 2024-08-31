# Loading all the libraries the i will use 
library(rvest)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(reshape2)
# Now i will form a function named as Scrap_Bating_data ,defing it 

Scrap_Bating_data <- function(name){
 # function will take input a name that will be a name of a cricket player 
  name1 <- gsub(pattern = " ", replacement = "+", name )         # here i will replace spaces with + so i can form a link 
  link <- paste0("http://www.cricmetric.com/playerstats.py?player=",name1,"&role=batsman&format=ODI&groupby=year")   # here i formed a link 

  data <- read_html(link) %>%            # here i scraped a batting data from the website Cricmetirc
    html_table()
  Data <- data[[1]]   # i will select first tibble that is of batting data that i targeted
  row_number <-dim(Data)[1]
  Data$Runs <-  gsub(pattern = ",",replacement = "",x = Data$Runs)   # removing commas from the digits
  Data$Balls <- gsub(pattern = ",",replacement = "",x = Data$Balls)
  Data$'4s' <- gsub(pattern = ",",replacement = "",x = Data$'4s')
  
  
  
  Data$Innings <- as.numeric(Data$Innings)       # taking all the charahtere value as a numeric value so that r can recoganise it 
  Data$Runs <- as.numeric(Data$Runs)
  Data$Balls <- as.numeric(Data$Balls)
  Data$Outs <- as.numeric(Data$Outs)
  Data$Avg <- as.numeric(Data$Avg)
  Data$SR <- as.numeric(Data$SR)
  Data$HS <- as.numeric(Data$HS)
  Data$'50' <- as.numeric(Data$'50')
  Data$'100' <- as.numeric(Data$'100')
  Data$'4s' <- as.numeric(Data$'4s')
  Data$'6s' <- as.numeric(Data$'6s')
  Data1 <- Data                 # making a copy of the data that is cleaned 
  Data <- Data[-c(row_number),]  # removing the last row that is of sum of values
  
  Main <- Data %>%            # now summarising data in useful way 
    select(Year ,Runs ,Balls, Innings , Outs) %>%
    summarise(Year = Year ,
              Rate = round({Runs/Balls}*6, 2),
              Runs_per_Inning =Runs/Innings  , 
              Balls_per_Inning = Balls/Innings,
              Not_Out_Rate = {Innings-Outs}/Innings)
  
  xy <- Main[,c(1,3,4)]        # making a subset of the data 
  xy <- melt(xy,"Year")             # now i will melt the data so it will be easy to plot in a effective way 
  text <- Main$Rate                 # now the below 4_5 lines are of ploting 
  text <- append(text,Main$Not_Out_Rate)
  text <- round(text,3)
  Data5 <- tibble(xy,text)
  ggp <- ggplot(Data5 , aes(x = Year , y = value , fill = variable )) +                    # here i defined the ggplot 
    geom_bar(stat = "identity", width = 0.5,position = position_dodge(width = 0.7)) + 
    geom_text(aes(label=text), vjust=1.5, colour="black", size=3) +
    labs(x = "Years" , y = "Number of Balls Or Runs ", title =paste0(name,"'s Performance") )
  
  
  xz  <- Data1[c(row_number),]    # now use the last row of the data set to make some usful  things that will help us to get some idea about the runs rates and probablities of the player to hit runs 
  Fours <- as.numeric(xz[1,11]/xz[1,4])
  Sixs <- as.numeric(xz[1,12]/xz[1,4])
  
  Fo <- as.numeric(xz[1,11])*4
  Six <- as.numeric(xz[1,12])*6
  Tot <- as.numeric(xz[1,3])
  ToB <- as.numeric(xz[1,4])
  pro <- {Tot-(Fo+Six)}/6
  
  Ones <- pro*3/ToB
  Twos <-  pro*2/ToB
  Threes <-  pro/ToB
  zeros <- 1-Ones - Twos - Threes - Fours - Sixs
  x <- NULL
  x[1] <- zeros                 # here i am saving the probablities and plot and cleaned data 
  x[2] <- Ones
  x[3] <- Twos
  x[4] <- Threes
  x[5] <- Fours
  x[6] <-Sixs
  output <- list(length(3))
  output[[1]] <- x
  output[[2]] <- Data
  output[[3]] <- ggp
  return(output)
  
}
