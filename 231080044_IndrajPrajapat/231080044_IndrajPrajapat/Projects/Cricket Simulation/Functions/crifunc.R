
library(dplyr)


Cricket_Match <- function(Team_Data_1 , Team_Data_2,Team_name_1,Team_name_2){
# here will make a function that will semulates a cricket match 
  
{  
  ball <- NULL          # defing ball as null to get values in it 
  
  for(i in 1:11){          # for i in 1:11 means there are 11 players in a team but will select 5 best bowlers 
    tryCatch({
      ball[i] <- Team_Data_2[[i]]$Bowling[[1]]
    }, error=function(e){})
  }
  ball[is.na(ball)] <- 0
  
  bowler_seq <- NULL
  for ( i in 1:5){
    bowler_seq[i] <- which(ball == max(ball), arr.ind = TRUE)[1]
    ball[bowler_seq[i]] <- 0
  }
  
  
  
  
  Baller_name <- NULL           # defining some variable that we will use in the function 
  Ball_runs <- NULL
  B_wicket <- NULL
  
  Order <- rep(bowler_seq,10)      # this sequence is make to define the order of bowling 
   
  
  batsman_name <- NULL
  Batsman_6 <- NULL
  Batsman_4 <- NULL
  batsman_run <- NULL
  batsman_outer <- NULL
  
  
  
  Runs <- 0
  Ind <- 1
  
  for(i in 1:50){             # i is the no of ovwers here that will be 50 and it will take one by one all the values 
    
    if(Ind == 11){
      break                 # this is for when 10 player got out and 11 player came to bets that the match will be over and so we beak the loop at 11
    }
    over <- NULL           # this will bw used to take data of oveers in it 
    
    
    Or <- Order[i]
    
    Wicket <- 0
    
    
    for(j in 1:6){       # here j is the no of ball in  a over 
      if(Ind == 11){         # this is also same when 11 player come to bet the loop will break and function will stop so that match will stop 
        break
      }
      batsman_run[6*(i-1) + j] <- 0        # i defined the values zero so that is this veriable in function not take any value then NA will not come zero will come if it takes value that it will be replaced
      Batsman_6[6*(i-1) + j] <- 0
      Batsman_4[6*(i-1) + j] <- 0
      Sam <- sample(c("Out","Not_Out"),1,replace = TRUE,prob = c(Team_Data_2[[Or]]$Bowling[[1]],1-Team_Data_2[[Or]]$Bowling[[1]]))    # here i take a sample of out or not on based of probablities that we calculated by data that was scraped 
      if(Sam == "Not_Out"){              # what if player will not out obeviously it will hit some runs or zero can be there
  #      print(Team_Data_1[[Ind]]$Batting[[1]])
        Run <- sample(c(0,1,2,3,4,6),1,replace = TRUE,prob = Team_Data_1[[Ind]]$Batting[[1]])    # this is the sample witch is based of players probablity to hit runs  according to previoud data 
          
        
         Runs <- Runs + Run
        over[j] <- Run                      ##### here i will store the values in variables
        batsman_name[6*(i-1) + j] <- Team_name_1[Ind]
        batsman_run[6*(i-1) + j] <- Run
        if(Run == 1 || Run == 3){            # we know when there are 1 or 3 run strike changes that i done here 
          
          
          nam <- Team_name_1[Ind]
          nam_ <- Team_name_1[Ind + 1]
          Team_name_1[Ind + 1] <- nam
          Team_name_1[Ind] <- nam_
          
          
          
          Runner <- Team_Data_1[[Ind]]$Batting              
          Facer <-Team_Data_1[[Ind +1]]$Batting
          Team_Data_1[[Ind+1]]$Batting <- Runner
          Team_Data_1[[Ind]]$Batting <- Facer
          
        }
        
        
        if (Run == 4){                    # if 4 runs i will store it in a variable 
          Batsman_4[6*(i-1)+j] <- 1
        }
        
        
        if (Run == 6){                       # this is for 6
          Batsman_6[6*(i-1)+j] <- 1
        } 
        
        
        
      } 
      
      else {
        
        batsman_name[6*(i-1) + j] <- Team_name_1[Ind]     # what if player got out           # here i store wicket and bowlers name
        batsman_outer[6*(i-1)+j] <- Team_name_2[Or]
        Ind <- Ind + 1
        Wicket <- Wicket + 1
        over[j] <- 0
      }
      if (j == 6 ){          # this is the last ball of the over strike will change so  i do it here 
        
        nam <- Team_name_1[Ind]
        nam_ <- Team_name_1[Ind + 1]
        Team_name_1[Ind + 1] <- nam
        Team_name_1[Ind] <- nam_
        
        
        B_wicket[i] <- Wicket
        Baller_name[i] <- Team_name_2[Or]
        Ball_runs[i] <- sum(over)
        
        Runner <- Team_Data_1[[Ind]]$Batting
        Facer <-Team_Data_1[[Ind +1]]$Batting
        Team_Data_1[[Ind+1]]$Batting <- Runner
        Team_Data_1[[Ind]]$Batting <- Facer
        
        # Over_Data[[i]] <- Over
      }
        
        
     }
      
      
      
      
    }
    
    baller_data <- tibble(Baller_name, Ball_runs,B_wicket )           # here i make tibble of all the data that a got from above loops 
    baller_data <- baller_data %>%                                    # here i arrange then in a useful way 
      group_by(Baller_name) %>%
      summarise(Runs = sum(Ball_runs),
                Wicket = sum(B_wicket)) %>%
      arrange(desc(Wicket))
  
  
    batsman_data <- tibble( batsman_name, Batsman_6 ,Batsman_4,batsman_run )
    batsman_data <- batsman_data %>%                                             # this is of batsman data 
      group_by(batsman_name) %>%
      summarise(six = sum(Batsman_6),
                four = sum(Batsman_4),
                Runs = sum(batsman_run))
  
    batsman_outer <- as.vector(na.omit(batsman_outer))
    len <- dim(batsman_data)[1]                                # some na are introduced so i remove them 
    
    if( length(batsman_outer) != len){
      for ( i in {length(batsman_outer) + 1}:len){
        batsman_outer[i] <- "Not Out"
      }
    }
    batsman_data$batsman_outer <- batsman_outer
    
    
    
    team_1_run <- sum(batsman_data$Runs)                  # savinf all useful thing in a list so i can take them in a return 
    team_2_wicket <- sum(baller_data$Wicket)
    
    output_1 <- list(length(2))
    output_1[[1]] <- batsman_data
    output_1[[2]] <- baller_data
  }
  #####################
  # Team 2 Bats                                    # here the same process for enning 2 so i will not explain it 
  ####################
  
  {  
    ball <- NULL
    for(i in 1:11){
      tryCatch({
        ball[i] <- Team_Data_1[[i]]$Bowling[[1]]
      }, error=function(e){})
    }
    ball[is.na(ball)] <- 0
    
    bowler_seq <- NULL
    for ( i in 1:5){
      bowler_seq[i] <- which(ball == max(ball), arr.ind = TRUE)[1]
      ball[bowler_seq[i]] <- 0
    }
    
    
    
    
    Baller_name <- NULL
    Ball_runs <- NULL
    B_wicket <- NULL
    
    Order <- rep(bowler_seq,10)
    
    
    batsman_name <- NULL
    Batsman_6 <- NULL
    Batsman_4 <- NULL
    batsman_run <- NULL
    batsman_outer <- NULL
    
    
    
    Runs <- 0
    Ind <- 1
    
    for(i in 1:50){
      
      if(Ind == 11){
        break
      }
      over <- NULL
      
      
      Or <- Order[i]
      
      Wicket <- 0
      
      
      for(j in 1:6){
        if(Ind == 11){
          break
        }
        batsman_run[6*(i-1) + j] <- 0 
        Batsman_6[6*(i-1) + j] <- 0
        Batsman_4[6*(i-1) + j] <- 0
        Sam <- sample(c("Out","Not_Out"),1,replace = TRUE,prob = c(Team_Data_1[[Or]]$Bowling[[1]],1-Team_Data_1[[Or]]$Bowling[[1]]))
        if(Sam == "Not_Out"){
          #      print(Team_Data_2[[Ind]]$Batting[[1]])
          Run <- sample(c(0,1,2,3,4,6),1,replace = TRUE,prob = Team_Data_2[[Ind]]$Batting[[1]])
          
          
          Runs <- Runs + Run
          over[j] <- Run
          batsman_name[6*(i-1) + j] <- Team_name_2[Ind]
          batsman_run[6*(i-1) + j] <- Run
          if(Run == 1 || Run == 3){
            
            
            nam <- Team_name_2[Ind]
            nam_ <- Team_name_2[Ind + 1]
            Team_name_2[Ind + 1] <- nam
            Team_name_2[Ind] <- nam_
            
            
            
            Runner <- Team_Data_2[[Ind]]$Batting
            Facer <-Team_Data_2[[Ind +1]]$Batting
            Team_Data_2[[Ind+1]]$Batting <- Runner
            Team_Data_2[[Ind]]$Batting <- Facer
            
          }
          
          
          if (Run == 4){
            Batsman_4[6*(i-1)+j] <- 1
          }
          
          
          if (Run == 6){
            Batsman_6[6*(i-1)+j] <- 1
          } 
          
          
          
        } 
        
        else {
          
          batsman_name[6*(i-1) + j] <- Team_name_2[Ind]
          batsman_outer[6*(i-1)+j] <- Team_name_1[Or]
          Ind <- Ind + 1
          Wicket <- Wicket + 1
          over[j] <- 0
        }
        if (j == 6 ){
          
          nam <- Team_name_2[Ind]
          nam_ <- Team_name_2[Ind + 1]
          Team_name_2[Ind + 1] <- nam
          Team_name_2[Ind] <- nam_
          
          
          B_wicket[i] <- Wicket
          Baller_name[i] <- Team_name_1[Or]
          Ball_runs[i] <- sum(over)
          
          Runner <- Team_Data_2[[Ind]]$Batting
          Facer <-Team_Data_2[[Ind +1]]$Batting
          Team_Data_2[[Ind+1]]$Batting <- Runner
          Team_Data_2[[Ind]]$Batting <- Facer
          
          # Over_Data[[i]] <- Over
        }
        
        
      }
      
      
      
      
    }
    
    baller_data <- tibble(Baller_name, Ball_runs,B_wicket )
    baller_data <- baller_data %>%
      group_by(Baller_name) %>%
      summarise(Runs = sum(Ball_runs),
                Wicket = sum(B_wicket)) %>%
      arrange(desc(Wicket))
    
    
    batsman_data <- tibble( batsman_name, Batsman_6 ,Batsman_4,batsman_run )
    batsman_data <- batsman_data %>%
      group_by(batsman_name) %>%
      summarise(six = sum(Batsman_6),
                four = sum(Batsman_4),
                Runs = sum(batsman_run))
    
    batsman_outer <- as.vector(na.omit(batsman_outer))
    len <- dim(batsman_data)[1]
    
    if( length(batsman_outer) != len){
      for ( i in {length(batsman_outer) + 1}:len){
        batsman_outer[i] <- "Not Out"
      }
    }
    batsman_data$batsman_outer <- batsman_outer
    
    
    team_2_run <- sum(batsman_data$Runs)
    team_1_wicket <- sum(baller_data$Wicket)
    
    output_2 <- list(length(2))
    output_2[[1]] <- batsman_data
    output_2[[2]] <- baller_data
  }
  
  output <- list(length(2))
  output[[1]] <- output_1
  output[[2]] <-output_2
  
  
  if (team_1_run == team_2_run){
    output[[3]] <- "Match Tie"
  }
  
  
  if ( team_1_run > team_2_run){
    by <- team_1_run - team_2_run
    output[[3]] <- paste( "Team 1 win by :" , by ,"runs")
  } else {
    by <- 11 - team_1_wicket
    output[[3]] <- paste( "Team 2 win by :" , by ,"wickets")
  }
    return(output)
}
  






###################################################################


Cricket_Match(Player_Data$India,Player_Data$Pakistan,Player_name$India,Player_name$Pakistan)

Team_Data_1 <- Player_Data$India
Team_Data_2 <- Player_Data$Pakistan
Team_name_1 <- Player_name$India
Team_name_2 <- Player_name$Pakistan







