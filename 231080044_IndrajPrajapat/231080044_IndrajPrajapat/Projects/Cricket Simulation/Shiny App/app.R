library(shiny)
library(ggplot2)
library(png)
library(magick)
library(shinyjs)
library(shinyalert)
library(dplyr)
library(shinythemes)
library(gt)
library(plotly)

load("../Data Sets/Cricket.RData")
load("../Data Sets/Teams_Table.RData")
load("../Data Sets/Cricket_Coord.RData")
happy <- read.csv("../Data Sets/Happiness_Index.csv")
gdp <- read.csv("../Data Sets/gdp_per_capita.csv")


year <- function(da) {
  return(as.integer(format(da, "%Y")))
}
t <- table
min(year(table$`Match Date`))
min(as.integer(year(table$`Match Date`)))
sum(is.na(table$`Match Date`))

ui <- fluidPage(theme = shinytheme("slate"),
                titlePanel("ODI"),
                tabsetPanel(
                  tabPanel("Player ODI Data",
                           sidebarLayout(
                             sidebarPanel(
                               selectInput("country", "Select Country:",
                                           choices = Country_name,
                                           selected = Country_name[5]),
                               uiOutput("player"),
                               uiOutput("ballorbat")
                             ),
                             mainPanel(
                               plotOutput("img"),
                               dataTableOutput("data"),
                               plotOutput("plot")
                             )
                           )
                  ),
                  tabPanel("Team ODI Data", 
                           tabsetPanel(
                             tabPanel("World View",
                                      sidebarLayout(
                                        sidebarPanel(
                                          sliderInput(
                                            inputId = "y_world",
                                            label = "Years : ",
                                            min = min(year(table$`Match Date`)),
                                            max = max(year(table$`Match Date`)),
                                            value = c(min(year(table$`Match Date`)), max(year(table$`Match Date`))),
                                            round = TRUE
                                          )
                                        ),
                                        mainPanel(
                                          plotOutput("world"),
                                          tableOutput("world_table")
                                        )
                                      )
                             ),
                             tabPanel("Team Overview",
                                      sidebarLayout(
                                        sidebarPanel(
                                          selectInput("team", "Select Team : ",
                                                      choices = unique(table$Winner)),
                                          uiOutput("years")
                                        ),
                                        mainPanel(
                                          verbatimTextOutput("team_summary"),
                                          plotOutput("plotsummary"),
                                          gt_output("h2h")
                                        )
                                      )
                             ),
                             tabPanel("Match Up",
                                      sidebarLayout(
                                        sidebarPanel(
                                          selectInput("team1", "Select Team 1 : ",
                                                      choices = unique(table$Winner),
                                                      selected = "Australia"),
                                          selectInput("team2", "Select Team 2 : ",
                                                      choices = unique(table$Winner),
                                                      selected = "England"),
                                          uiOutput("years_match")
                                        ),
                                        mainPanel(
                                          verbatimTextOutput("match_summary"),
                                          plotOutput("matchup"),
                                          tableOutput("matchuptab")
                                        )
                                      )
                             ),
                             tabPanel("Happiness Index",
                                      tabsetPanel(
                                        tabPanel("General",
                                                 sidebarLayout(
                                                   sidebarPanel(
                                                     checkboxGroupInput(inputId = "teams_happy_general", label = "Countries",
                                                                        choices = unique(table$Loser)[unique(table$Loser) %in% unique(happy$Country)],
                                                                        selected = c("Australia", "Pakistan", "New Zealand", "India", "Sri Lanka", "Zimbabwe", "Bangladesh", "South Africa", "Netherlands", "Ireland", "Afghanistan"))
                                                   ),
                                                   mainPanel(
                                                     plotOutput("bar_happy")
                                                   )
                                                 )
                                        ),
                                        tabPanel("Country",
                                                 sidebarLayout(
                                                   sidebarPanel(
                                                     selectInput("team_happy", "Select Team : ",
                                                                 choices = unique(table$Loser)[unique(table$Loser) %in% unique(happy$Country)],
                                                                 selected = "India")
                                                   ),
                                                   mainPanel(
                                                     plotOutput("happy_plot")
                                                   )
                                                 ))
                                      )
                             ),
                             tabPanel("GDP",
                                      tabsetPanel(
                                        tabPanel("General",
                                                 sidebarLayout(
                                                   sidebarPanel(
                                                     checkboxGroupInput(inputId = "teams_gdp_general", label = "Countries",
                                                                        choices = unique(table$Loser)[unique(table$Loser) %in% unique(gdp$Country.Name)],
                                                                        selected = c("Australia", "Pakistan", "New Zealand", "India", "Sri Lanka", "Zimbabwe", "Bangladesh", "South Africa", "Netherlands", "Kenya", "Ireland", "Afghanistan"))
                                                   ),
                                                   mainPanel(
                                                     plotOutput("bar_gdp")
                                                   )
                                                 )
                                        ),
                                        tabPanel("Country",
                                                 sidebarLayout(
                                                   sidebarPanel(
                                                     selectInput("team_gdp", "Select Team : ",
                                                                 choices = unique(table$Loser)[unique(table$Loser) %in% unique(gdp$Country.Name)],
                                                                 selected = "India")
                                                   ),
                                                   mainPanel(
                                                     plotOutput("gdp_plot")
                                                   )
                                                 ))
                                      )
                             )
                           )
                  ),
                  
                  tabPanel("Cricket Match", 
                           sidebarLayout(
                             sidebarPanel(
                               span(textOutput("hint")),
                               selectInput("team_select_1", "Select team 1", 
                                           choices = c(Country_name, "new_team (1)")),
                               conditionalPanel(
                                 condition = "input.team_select_1 == 'new_team (1)'", 
                                 selectInput("player_names", "Select Players (exect 11 players):", 
                                             choices = All_Player_Name, multiple = TRUE),
                                 span(textOutput("player_counter")),
                                 uiOutput("no_of_player")
                               ),
                               selectInput("team_select_2", "Select team 2", 
                                           choices = c(Country_name, "new_team (2)")),
                               conditionalPanel(
                                 condition = "input.team_select_2 == 'new_team (2)'", 
                                 selectInput("player_names_2", "Select Players (exect 11 players):", 
                                             choices = All_Player_Name, multiple = TRUE),
                                 span(textOutput("player_counter_2")),
                                 uiOutput("no_of_player_2")
                               ),
                               actionButton("start", "Start Match")
                             ),
                             mainPanel(
                               h1(textOutput("winorloss",container = span)),
                               h2(textOutput("team1",container = span)),
                               h3(textOutput("batting",container = span)),
                               dataTableOutput("team1_table"),
                               plotlyOutput("team_1_plot_bat"),
                               h3(textOutput("bowling",container = span)),
                               dataTableOutput("team1_table_"),
                               plotOutput("team_1_plot_bal"),
                               
                               h2(textOutput("team2",container = span)),
                               h3(textOutput("batting_",container = span)),
                               dataTableOutput("team2_table"),
                               plotlyOutput("team_2_plot_bat"),
                               h3(textOutput("bowling_",container = span)),
                               dataTableOutput("team2_table_"),
                               plotOutput("team_2_plot_bal"),
                               
                               
                             )
                           )
                           # UI components specific to Cricket Match tab
                           # Add your UI elements here
                  )
                )
)

server <- function(input, output) {
  team <- reactive({
    input$team
  })
  
  team_1 <- reactive({
    input$team1
  })
  
  team_2 <- reactive({
    input$team2
  })
  
  team_happy <- reactive({
    input$team_happy
  })
  
  team_gdp <- reactive({
    input$team_gdp
  })
  
  team_table <- reactive({
    team_filter(team())
  })
  
  teams_gdp_general <- reactive({
    input$teams_gdp_general
  })
  
  teams_happy_general <- reactive({
    input$teams_happy_general
  })
  
  team_1_table <- reactive({
    team_filter(team_1())
  })
  
  match_table <- reactive({
    team_1_table() %>% filter(Opponent == team_2())
  })
  
  team_happy_table <- reactive({
    team_filter(team_happy())
  })
  
  y_world_1 <- reactive({
    t <- input$y_world
    t[1]
  })
  
  y_world_2 <- reactive({
    t <- input$y_world
    t[2]
  })
  
  
  output$world_table <- renderTable({
    teams <- unique(table$Loser)[c(1:6, 8:19, 24:28)]
    W_L <- NULL
    t <- table %>% filter((year(`Match Date`) >= y_world_1()) & (year(`Match Date`) <= y_world_2()))
    for (i in 1:length(teams)) {
      win <- sum(t$Winner == teams[i])
      loss <- sum(t$Loser == teams[i])
      if (loss == 0) {
        W_L <- append(W_L, NA)
      }
      else {
        W_L <- append(W_L, win/loss)
      }
    }
    WL <- data.frame(teams, W_L)
    colnames(WL)[1] <- "Teams"
    colnames(WL)[2] <- "Win to Loss Ratio"
    WL %>% arrange(desc(W_L)) %>% filter(!is.na(W_L))
  })
  
  output$world <- renderPlot({
    teams <- unique(table$Loser)[c(1:6, 8:19, 24:28)]
    W_L <- NULL
    t <- table %>% filter((year(`Match Date`) >= y_world_1()) & (year(`Match Date`) <= y_world_2()))
    for (i in 1:length(teams)) {
      win <- sum(t$Winner == teams[i])
      loss <- sum(t$Loser == teams[i])
      if (loss == 0) {
        W_L <- append(W_L, NA)
      }
      else {
        W_L <- append(W_L, win/loss)
      }
    }
    WL <- data.frame(teams, W_L)
    data <- left_join(coord, WL, by = "teams")
    ggplot(data, aes(x = long, y = lat, group = group)) + geom_polygon(aes(fill = W_L), color = "black")
  })
  
  output$years <- renderUI({
    sliderInput(
      inputId = "y",
      label = "Years : ",
      min = min(year(team_table()$`Match Date`)),
      max = max(year(team_table()$`Match Date`)),
      value = c(min(year(team_table()$`Match Date`)), max(year(team_table()$`Match Date`))),
      round = TRUE
    )
  })
  
  output$years_match <- renderUI({
    sliderInput(
      inputId = "ym",
      label = "Years : ",
      min = min(year(match_table()$`Match Date`)),
      max = max(year(match_table()$`Match Date`)),
      value = c(min(year(match_table()$`Match Date`)), max(year(match_table()$`Match Date`))),
      round = TRUE
    )
  })
  
  output$years_happy <- renderUI({
    sliderInput(
      inputId = "yh",
      label = "Years : ",
      min = max(min(year(team_happy_table()$`Match Date`)), min((happy %>% filter(Country == team_happy()))$Year)),
      max = min(max(year(team_happy_table()$`Match Date`)), max((happy %>% filter(Country == team_happy()))$Year)),
      value = c(max(min(year(team_happy_table()$`Match Date`)), min((happy %>% filter(Country == team_happy()))$Year)), min(max(year(team_happy_table()$`Match Date`)), max((happy %>% filter(Country == team_happy()))$Year))),
      round = TRUE)
  })
  
  year_team_1 <- reactive({
    t <- input$y
    t[1]
  })
  year_team_2 <- reactive({
    t <- input$y
    t[2]
  })
  
  year_match_1 <- reactive({
    t <- input$ym
    t[1]
  })
  
  year_match_2 <- reactive({
    t <- input$ym
    t[2]
  })
  
  years_happiness <- reactive({
    input$yh
  })
  
  output$team_summary <- renderPrint({
    t <- team_table()
    t1 <- (t %>% filter((as.integer(year(t$'Match Date')) >= year_team_1()) & (as.integer(year(t$'Match Date')) <= year_team_2())))
    colnames(t1)[4] <- paste0(team(), " Score")
    colnames(t1)[5] <- "Opponent Score"
    colnames(t1)[8] <- "Winning Score"
    colnames(t1)[9] <- "Losing Score"
    t1 <- t1[c(4, 5, 8, 9, 12)]
    summary(t1)
  })
  
  output$plotsummary <- renderPlot({
    t <- team()
    tf <- team_filter(t)
    tab <- tf %>% filter((as.integer(year(tf$'Match Date')) <= year_team_2()) & (as.integer(year(tf$'Match Date')) >= year_team_1()))
    Date <- NULL
    Score <- NULL
    Win <- NULL
    
    for (i in 1:length(tab$Winner)) {
      if (tab$Winner[i] == t) {
        Date <- append(Date, tab$'Match Date'[i])
        Score <- append(Score, tab$Winner_Score[i])
        Win <- append(Win, "Won")
      }
      if (tab$Winner[i] != t) {
        Date <- append(Date, tab$'Match Date'[i])
        Score <- append(Score, tab$Loser_Score[i])
        Win <- append(Win, "Lost")
      }
    }
    tb <- data.frame(Date, Score, Win)
    ggplot(tb, aes(x = Date, y = Score, col = Win)) + geom_point()
  })
  
  output$matchup <- renderPlot({
    y1 <- year_match_1()
    y2 <- year_match_2()
    t1 <- team_1()
    t2 <- team_2()
    tf <- team_filter(t1)
    tab <- tf %>% filter((as.integer(year(tf$'Match Date')) <= y2) & (as.integer(year(tf$'Match Date')) >= y1)) %>% filter(Opponent == t2)
    Date <- NULL
    Score <- NULL
    Team <- NULL
    
    for (i in 1:length(tab$Winner)) {
      if (tab$Winner[i] == t1) {
        Date <- append(Date, tab$'Match Date'[i])
        Date <- append(Date, tab$'Match Date'[i])
        Score <- append(Score, tab$Winner_Score[i])
        Score <- append(Score, tab$Loser_Score[i])
        Team <- append(Team, t1)
        Team <- append(Team, t2)
      }
      if (tab$Winner[i] == t2) {
        Date <- append(Date, tab$'Match Date'[i])
        Date <- append(Date, tab$'Match Date'[i])
        Score <- append(Score, tab$Loser_Score[i])
        Score <- append(Score, tab$Winner_Score[i])
        Team <- append(Team, t1)
        Team <- append(Team, t2)
      }
    }
    t <- data.frame(Date, Score, Team)
    #  t
    ggplot(t, aes(x = Date, y = Score, col = Team)) + geom_point()
  })
  
  output$matchuptab <- renderTable({
    y1 <- year_match_1()
    y2 <- year_match_2()
    t1 <- team_1()
    t2 <- team_2()
    tf <- team_filter(t1)
    tab <- tf %>% filter((as.integer(year(tf$'Match Date')) <= y2) & (as.integer(year(tf$'Match Date')) >= y1)) %>% filter(Opponent == t2)
    Team_1_Batting <- length((tab %>% filter(Winner == t1) %>% filter(Winning_Innings == "Batting"))$Winner)
    Team_1_Bowling <- length((tab %>% filter(Winner == t1) %>% filter(Winning_Innings == "Bowling"))$Winner)
    Team_2_Batting <- length((tab %>% filter(Winner == t2) %>% filter(Winning_Innings == "Batting"))$Winner)
    Team_2_Bowling <- length((tab %>% filter(Winner == t2) %>% filter(Winning_Innings == "Bowling"))$Winner)
    Innings <- c("Batting", "Bowling")
    Team_1 <- c(Team_1_Batting, Team_1_Bowling)
    Team_2 <- c(Team_2_Batting, Team_2_Bowling)
    tb <- data.frame(Innings, Team_1, Team_2)
    colnames(tb)[2] <- paste0(t1)
    colnames(tb)[3] <- paste0(t2)
    tb
  })
  
  output$h2h <- render_gt({
    t2 <- team_table()
    t <- t2 %>% filter((as.integer(year(t2$'Match Date')) >= year_team_1()) & (as.integer(year(t2$'Match Date')) <= year_team_2()))
    Team <- unique(t$Opponent)
    other <- Team
    Won <- NULL
    Lost <- NULL
    Won_Bat <- NULL
    Won_Bowl <- NULL
    Lost_Bat <- NULL
    Lost_Bowl <- NULL
    for (i in 1:length(other)) {
      t1 <- t %>% filter(Opponent == other[i])
      Won <- append(Won, length((t1 %>% filter(Winner == team()))$Winner))
      Won_Bat <- append(Won_Bat, length((t1 %>% filter((Winner == team()) & (Winning_Innings == "Batting")))$Winner))
      Won_Bowl <- append(Won_Bowl, length((t1 %>% filter((Winner == team()) & (Winning_Innings == "Bowling")))$Winner))
      Lost_Bat <- append(Lost_Bat, length((t1 %>% filter((Winner == other[i]) & (Winning_Innings == "Bowling")))$Winner))
      Lost_Bowl <- append(Lost_Bowl, length((t1 %>% filter((Winner == other[i]) & (Winning_Innings == "Batting")))$Winner))
      Lost <- append(Lost, length((t1 %>% filter(Winner == other[i]))$Winner))
    }
    
    d <- data.frame(Team, Won_Bat, Lost_Bat, Won_Bowl, Lost_Bowl, Won/Lost)
    colnames(d)[6] <- "W_L Ratio"
    d$'W_L Ratio'[is.infinite(d$'W_L Ratio')] <- NA
    d$'W_L Ratio' <- as.double(d$'W_L Ratio')
    d <- d %>% arrange(desc(d$'W_L Ratio'))
    colnames(d)[6] <- "Win to Loss Ratio"
    d <- gt(d)
    tab <- tab_spanner(d, label = "Batting", columns = ends_with("Bat"))
    tab <- tab_spanner(tab, label = "Chasing", columns = ends_with("Bowl"))
    tab |> cols_label(starts_with("Won") ~ "Won", starts_with("Lost") ~ "Lost")
  })
  
  output$match_summary <- renderPrint({
    t <- match_table()
    t <- t %>% filter((year(`Match Date`) >= year_match_1()) & (year(`Match Date`) <= year_match_2()))
    colnames(t)[4] <- paste0(team_1(), " Score")
    colnames(t)[5] <- paste0(team_2(), " Score")
    colnames(t)[8] <- "Winning Score"
    colnames(t)[9] <- "Losing Score"
    t <- t[c(4, 5, 8, 9, 12)]
    summary(t)
  })
  
  output$happy_plot <- renderPlot({
    y1 <- happy %>% filter(Country == team_happy())
    y2 <- team_filter(team_happy())
    Year <- NULL
    for (i in unique(y1$Year)) {
      if (i %in% unique(year(y2$'Match Date'))) {
        Year <- append(Year, i)
      }
    }
    Win_To_Loss <- NULL
    Happiness <- NULL
    for (i in Year) {
      hap <- y1$Index[which(y1$Year == i, arr.ind = TRUE)]
      Happiness <- append(Happiness, as.double(hap))
      Win <- length((y2 %>% filter(Winner == team_happy()) %>% filter(year(`Match Date`) == i))$Winner)
      Loss <- length((y2 %>% filter(Loser == team_happy()) %>% filter(year(`Match Date`) == i))$Loser)
      if (Loss == 0) {
        Win_To_Loss <- append(Win_To_Loss, Win)
      }
      else {
        Win_To_Loss <- append(Win_To_Loss, Win/Loss)
      }
    }
    matplot(Year, cbind(Happiness, Win_To_Loss), col = c("red", "blue"), pch = c(13, 16), xlab = "Year", ylab = "Happiness Index + Win To Loss Ratio")
    legend("topright", legend = c("Happiness Index", "Win To Loss Ratio"), col = c("red", "blue"), pch = c(13, 16))
  })
  
  output$bar_happy <- renderPlot({
    #    Countries <- unique(table$Loser)[unique(table$Loser) %in% unique(gdp$Country.Name)]
    #    Countries <- c("Australia", "Pakistan", "New Zealand", "India", "Sri Lanka", "Zimbabwe", "Bangladesh", "South Africa", "Netherlands", "Kenya", "Ireland", "Afghanistan")
    Countries <- teams_happy_general()
    Correlation <- NULL
    for (c in Countries) {
      tab <- happy %>% filter(Country == c)
      Tf <- team_filter(c)
      y_team <- unique(year(Tf$'Match Date'))
      Year <- NULL
      for (i in unique(tab$Year)) {
        if (i %in% unique(year(Tf$'Match Date'))) {
          Year <- append(Year, i)
        }
      }
      Happiness <- NULL
      Win_To_Loss <- NULL
      for (i in Year) {
        hap <- tab$Index[which(tab$Year == i, arr.ind = TRUE)]
        Happiness <- append(Happiness, as.double(hap))
        Win <- length((Tf %>% filter(Winner == c) %>% filter(year(`Match Date`) == i))$Winner)
        Loss <- length((Tf %>% filter(Loser == c) %>% filter(year(`Match Date`) == i))$Loser)
        if (Loss == 0) {
          Win_To_Loss <- append(Win_To_Loss, Win)
        }
        else {
          Win_To_Loss <- append(Win_To_Loss, Win/Loss)
        }
      }
      Correlation <- append(Correlation, cor(Win_To_Loss, Happiness))
    }
    barplot(names = Countries, height = Correlation, xlab = "Countries", ylab = "Correlation")
  })
  
  output$gdp_plot <- renderPlot({
    t1 <- team_gdp()
    tab <- gdp %>% filter(Country.Name == t1)
    Tf <- team_filter(t1)
    y_team <- unique(year(Tf$'Match Date'))
    Years <- NULL
    for (i in 1960:2020) {
      if (!is.na(tab[i - 1957] == "NA") & (as.integer(i) %in% y_team)) {
        Years <- append(Years, as.integer(i))
      }
    }
    GDP <- NULL
    Win_To_Loss <- NULL
    for (i in Years) {
      GDP <- append(GDP, as.integer(tab[i - 1957]))
      Win <- length((Tf %>% filter(Winner == t1) %>% filter(year(`Match Date`) == i))$Winner)
      Loss <- length((Tf %>% filter(Loser == t1) %>% filter(year(`Match Date`) == i))$Loser)
      if (Loss == 0) {
        Win_To_Loss <- append(Win_To_Loss, Win)
      }
      else {
        Win_To_Loss <- append(Win_To_Loss, Win/Loss)
      }
    }
    # plot(Years, GDP, pch = 16, col = "red")
    # points(Years, GDP, pch = 12, col = "blue")
    # lines(Years, GDP, pch = 12, col = "blue")
    matplot(Years, cbind(GDP, 1000*Win_To_Loss), col = c("red", "blue"), pch = c(13, 16), xlab = "Years", ylab = "GDP + Win To Loss Ratio(x 1000)")
    legend("topright", legend = c("GDP per Capita", "Win To Loss Ratio(x 1000)"), col = c("red", "blue"), pch = c(13, 16))
  })
  
  output$bar_gdp <- renderPlot({
    #    Countries <- unique(table$Loser)[unique(table$Loser) %in% unique(gdp$Country.Name)]
    #    Countries <- c("Australia", "Pakistan", "New Zealand", "India", "Sri Lanka", "Zimbabwe", "Bangladesh", "South Africa", "Netherlands", "Kenya", "Ireland", "Afghanistan")
    Countries <- teams_gdp_general()
    Correlation <- NULL
    for (c in Countries) {
      tab <- gdp %>% filter(Country.Name == c)
      Tf <- team_filter(c)
      y_team <- unique(year(Tf$'Match Date'))
      Years <- NULL
      for (i in 1960:2020) {
        if (!is.na(tab[i - 1957] == "NA") & (as.integer(i) %in% y_team)) {
          Years <- append(Years, as.integer(i))
        }
      }
      GDP <- NULL
      Win_To_Loss <- NULL
      for (i in Years) {
        GDP <- append(GDP, as.integer(tab[i - 1957]))
        Win <- length((Tf %>% filter(Winner == c) %>% filter(year(`Match Date`) == i))$Winner)
        Loss <- length((Tf %>% filter(Loser == c) %>% filter(year(`Match Date`) == i))$Loser)
        if (Loss == 0) {
          Win_To_Loss <- append(Win_To_Loss, Win)
        }
        else {
          Win_To_Loss <- append(Win_To_Loss, Win/Loss)
        }
      }
      Correlation <- append(Correlation, cor(Win_To_Loss, GDP))
    }
    barplot(names = Countries, height = Correlation, xlab = "Countries", ylab = "Correlation")
  })
  
  output$player <- renderUI({
    selectInput("selectedplayer", "Select a Player:", choices = Player_name[[input$country]])
  }) 
  
  output$ballorbat <- renderUI({
    selectInput("selectb", "Batting or Bowlling :", choices = c("Batting" , "Bowling"))
  })
  
  output$img <- renderPlot({
    img <- image_read(image_links[[input$country]][[input$selectedplayer]])
    plot(img)
  })
  
  
  output$data <- renderDataTable({
    Player_Data[[input$country]][[input$selectedplayer]][[input$selectb]][[2]]
    
    
  })
  
  output$plot <- renderPlot({
    Player_Data[[input$country]][[input$selectedplayer]][[input$selectb]][[3]]
  })
  
  
  
  
  output$player_counter <- renderText({
    "Number of Player Selected"
  })
  
  output$no_of_player <- renderText( {
    length(input$player_names)
  })
  
  output$hint <- renderText({
    "Select 'new team' to create a team"
  })
  
  output$player_counter_2 <- renderText({
    "Number of Player Selected"
  })
  
  output$no_of_player_2 <- renderText( {
    length(input$player_names_2)
  })
  
  
  
  observeEvent(input$start, {
    if (input$team_select_1 == 'new_team (1)'){
      pyr_1 <- input$player_names
      if(length(pyr_1) != 11){
        showModal(modalDialog(
          title = "Player Selection Error",
          "Please select exactly 11 players for Team 1.",
          easyClose = TRUE
        ))
        return(NULL)
      }
      team_1 <- list()
      for (pyr in pyr_1){
        team_1[[pyr]] <- All_Player_Data[[pyr]]
      }
    } else {
      pyr_1 <- Player_name[[input$team_select_1]]
      team_1 <- Player_Data[[input$team_select_1]]
    }
    
    if (input$team_select_2 == 'new_team (2)'){
      pyr_2 <- input$player_names_2
      if(length(pyr_2) != 11){
        showModal(modalDialog(
          title = "Player Selection Error",
          "Please select exactly 11 players for Team 2.",
          easyClose = TRUE
        ))
        return(NULL)
      }
      team_2 <- list()
      for (pyr in pyr_2){
        team_2[[pyr]] <- All_Player_Data[[pyr]]
      }
    } else {
      pyr_2 <- Player_name[[input$team_select_2]]
      team_2 <- Player_Data[[input$team_select_2]]
    }
    
    if(input$team_select_2 == input$team_select_1){
      showModal(modalDialog(
        title = "Team Selection Error",
        "Please select different teams",
        easyClose = TRUE
      ))
      return(NULL)
    }
    data <- Cricket_Match(team_1,team_2,pyr_1,pyr_2)
    
    
    output$team_1_plot_bat <- renderPlotly({
      p <-  ggplot(data[[1]][[1]], aes(x = batsman_name, y = Runs, col = factor(four), size = six , shape = batsman_outer)) + 
        geom_point() + 
        labs(title ="Batsman Performance" ,x = "Batsman Name", y = "Total Runs", 
             shape = "Wicketeer", color = "No of Four", size = "No of Six") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1),legend.position = "none")
      ggplotly(p, tooltip = c("batsman_name", "Runs", "four", "six", "batsman_outer"))
    })
    
    output$team_2_plot_bat <- renderPlotly({
      p <- ggplot(data[[2]][[1]], aes(x = batsman_name, y = Runs, col = factor(four), size = six , shape = batsman_outer)) + 
        geom_point() + 
        labs(title ="Batsman Performance" ,x = "Batsman Name", y = "Total Runs", 
             shape = "Wicketeer", color = "No of Four", size = "No of Six") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1),
              legend.position = "none")  # This line removes the legend
      
      ggplotly(p, tooltip = c("batsman_name", "Runs", "four", "six", "batsman_outer"))
    })
    
    
    output$team_1_plot_bal <- renderPlot({
      ggplot(data[[1]][[2]], aes(x = "", y = Runs, fill = Baller_name)) +
        geom_bar(width = 1, stat = "identity", color = "black") +
        geom_text(aes(label = paste(Baller_name, "\n", "Wickets:", Wicket,"\n","Runs:" , Runs)), 
                  position = position_stack(vjust = 0.5), size = 3, vjust = 1, color = "white") +
        coord_polar("y") +
        theme_minimal() +  # Change the theme to minimal
        theme(
          legend.position = "none",
          axis.text.x = element_blank(),  # Remove x-axis text
          panel.grid = element_blank(),   # Remove grid lines
          plot.background = element_rect(fill = "lightblue"), 
          panel.background = element_rect(fill = "white"),    
          axis.line = element_line(color = "black"),          
          plot.title = element_text(hjust = 0.5, size = 16),   
          axis.title = element_text(size = 1),                
          axis.text.y = element_text(size = 10)                
        )
      
    })
    
    output$team_2_plot_bal <- renderPlot({
      ggplot(data[[2]][[2]], aes(x = "", y = Runs, fill = Baller_name)) +
        geom_bar(width = 1, stat = "identity", color = "black") +
        geom_text(aes(label = paste(Baller_name, "\n", "Wickets:", Wicket,"\n","Runs:" , Runs)), 
                  position = position_stack(vjust = 0.5), size = 3, vjust = 1, color = "white") +
        coord_polar("y") +
        theme_minimal() +  # Change the theme to minimal
        theme(
          legend.position = "none",
          axis.text.x = element_blank(),  # Remove x-axis text
          panel.grid = element_blank(),   # Remove grid lines
          plot.background = element_rect(fill = "lightblue"),  # Change plot background color
          panel.background = element_rect(fill = "white"),     # Change panel background color
          axis.line = element_line(color = "black"),           # Add a border around the plot
          plot.title = element_text(hjust = 0.5, size = 16),   # Center the title and increase font size
          axis.title = element_text(size = 1),                # Increase axis label font size
          axis.text.y = element_text(size = 10)                # Increase y-axis text font size
        )
      
    })
    
    
    output$team1 <- renderText({
      paste("First inning",":")
    })
    
    output$batting <- renderText({
      paste("Batting :",input$team_select_1)
    })
    output$team1_table <- renderDataTable({
      
      data[[1]][[1]]
    })
    
    output$bowling <- renderText({
      paste("Bowling :",input$team_select_2)
    })
    output$team1_table_ <- renderDataTable({
      
      data[[1]][[2]]
    })
    
    output$team2 <- renderText({
      paste("Second inning",":")
    })
    
    output$batting_ <- renderText({
      paste("Batting :",input$team_select_2)
    })
    output$team2_table <- renderDataTable({
      
      data[[2]][[1]]
    })
    
    output$bowling_ <- renderText({
      paste("Bowling :",input$team_select_1)
    })
    output$team2_table_ <- renderDataTable({
      
      data[[2]][[2]]
    })
    
    
    result <- substr(data[[3]],1,6)
    res <- substr(data[[3]],7,55)
    if(result == "Team 1"){
      output$winorloss <- renderText({
        paste(input$team_select_1,res)
      })
    } else {
      output$winorloss <- renderText({
        paste(input$team_select_2,res)
      })
    }
  })
  
  
}

shinyApp(ui = ui, server = server)
