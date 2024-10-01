# statcast_data_scrape_python
This allows you to scrape Statcast pitch level data from baseballsavant.mlb.com

Read in the 4 functions and put your date in quotation marks to the function Add_to_statcast2024.

An example will be: Add_to_statcast2024('2024-05-06', '2024-05-06')

This takes a feather file called statcast2024 and updates it with the pitches from whatever date we are choosing. Understand if you do a date twice, all pitches will be in the dataframe twice. 

I put an empty version of statcast2024.feather into this repository, so you can add whatever dates you would like. I personally have a running statcast2024 repository for the full 2024 season, but I wanted to make it as adjustable as you would like.

You can do date ranges instead of single days, but Baseball Savant does limit the maximum batch load that you can pull in a single request, so it can be easier to just go day by day.

This code is based off of R code in Bill Petti's BaseballR package. 

I have added in my own columns and converted the code to python, but this would not have been possible without the basis in his R code. 
