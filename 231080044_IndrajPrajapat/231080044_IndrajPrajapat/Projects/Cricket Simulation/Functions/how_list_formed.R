Team <- read_html("https://www.thecricketer.com/Topics/mens-world-cup-2023/mens_cricket_world_cup_2023_squads_all_teams_player_lists.html") %>%
  html_elements("#wl_f813f7 h3") %>%
  html_text2()



Team_Player <- read_html("https://www.thecricketer.com/Topics/mens-world-cup-2023/mens_cricket_world_cup_2023_squads_all_teams_player_lists.html") %>%
  html_elements("#wl_f813f7 p") %>%
  html_text2()

Team_Player <- Team_Player[-c(1:3,14)]

Team_Player <- gsub("[(c)]","",Team_Player)

Team_Player[1] <- "Hashmatullah Shahidi , Rahmanullah Gurbaz, Ibrahim Zadran, Riaz Hassan, Rahmat Shah, Najibullah Zadran, Mohammad Nabi, Ikram Alikhil, Azmatullah Omarzai, Rashid Khan, Mujeeb ur Rahman, Noor Ahmad, Fazalhaq Farooqi, Abdul Rahman, Naveen-ul-Haq"
Team_Player[2] <- "Pat Cummins, Sean Abbott, Alex Carey, Cameron Green, Josh Hazlewood, Travis Head, Josh Inglis, Marnus Labuschagne, Mitchell Marsh, Glenn Maxwell, Steven Smith, Mitchell Starc, Marcus Stoinis, David Warner, Adam Zampa"
Team_Player[3] <- "Shakib Al Hasan , Liton Das , Najmul Hossain Shanto, Tanzid Hasan, Towhid Hridoy, Mahmudullah, Mushfiqur Rahim, Mehidy Hasan, Mahedi Hasan, Tanzim Hasan Sakib, Nasum Ahmed, Shoriful Islam, Hasan Mahmud, Taskin Ahmed, Mustafizur Rahman"
Team_Player[4] <- "Jos Buttler , Moeen Ali, Gus Atkinson, Jonny Bairstow, Sam Curran, Liam Livingstone, Dawid Malan, Adil Rashid, Joe Root, Jason Roy, Ben Stokes, Reece Topley, David Willey, Mark Wood, Chris Woakes"
Team_Player[6] <- "SA Edwards , Max O'Dowd, Bas de Leede, Vikram Singh, Teja Nidamanuru, Paul van Meekeren, Colin Ackermann, Roelof van der Merwe, Logan van Beek, Aryan Dutt, Ryan Klein, Wesley Barresi, Saqib Zulfiqar, Shariz Ahmad, Michael Rippon"


Afghanistan <- strsplit(x =Team_Player[1] , split = ",",fixed = TRUE )
Afghanistan[[1]][15] <- "Naveen-ul-Haq"
Australia <- strsplit(x =Team_Player[2] , split = ",",fixed = TRUE )
Bangladesh <- strsplit(x =Team_Player[3] , split = ",",fixed = TRUE )
England <- strsplit(x =Team_Player[4] , split = ",",fixed = TRUE )
India <- strsplit(x =Team_Player[5] , split = ",",fixed = TRUE )
Netherlands <- strsplit(x =Team_Player[6] , split = ",",fixed = TRUE )
New_Zealand <- strsplit(x =Team_Player[7] , split = ",",fixed = TRUE )[[1]][-c(8)]
Pakistan <- strsplit(x =Team_Player[8] , split = ",",fixed = TRUE )
Pakistan[[1]][14] <- "Shaheen Shah Afridi"
Pakistan[[1]][15] <- "Usama Mir"
South_Africa <- strsplit(x =Team_Player[9] , split = ",",fixed = TRUE )
Sri_Lanka <- strsplit(x =Team_Player[10] , split = ",",fixed = TRUE )

Sri_Lanka[[1]][5]









#################################################################################
# Assuming Team and Team_Player are defined as you provided

# Create an empty list to store the team players
Country <- vector("list", length = length(Team))

# Loop through each team
for(i in seq_along(Team)) {
  team_name <- Team[i]
  players <- strsplit(Team_Player[i], ", ")[[1]]
  
  # Assign the players to the list element with the team name as the list name
  Country[[team_name]] <- players
}



###############################################################################

country <- list()

for ( i in 1:length(Team)){
  team_name <- Team[i]
  for( j in 1:15){
    print(Country[[team_name]][j])
    country[[team_name]][[Country[[team_name]][j]]][["Batting"]] <- Scrap_Bating_data(Country[[team_name]][j])
    
  }
}
