# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:45:44 2024

@author: pdepase
"""
from datetime import date
from datetime import datetime
import pandas as pd
import numpy as np
import os
import warnings
import requests
import re
import io
import joblib


## UPDATE TO WHATEVER YOU ARE KEEPING THE SUPPORTING FILES ##
os.chdir('C:/Users/phili/Documents/statcast_scrape_files_python')




def revamped_MLB_statcast_scrape(start_date, end_date, playerid, player_type):

    if start_date <= '2015-01-01':
        print("Some metrics such as Exit Velocity and Batted Ball Events have only been compiled since 2015.")

    if start_date < '2008-03-25':
        exit("The data are limited to the 2008 MLB season and after.")
        return(None)

    if start_date == date.today():
        print("The data are collected daily at 3 a.m. Some of today's games may not be included.")

    if start_date > end_date:
        exit("The start date is later than the end date.")
        return(None)

    playerid_var = "batters_lookup%5B%5D"

    if player_type == "pitcher":
        playerid_var = "pitchers_lookup%5B%5D"

    vars_df = pd.DataFrame({
        'var': [
            "all", "hfPT", "hfAB", "hfBBT", "hfPR", "hfZ", "stadium", "hfBBL", "hfNewZones",
            "hfGT", "hfC", "hfSea", "hfSit", "hfOuts", "opponent", "pitcher_throws",
            "batter_stands", "hfSA", "player_type", "hfInfield", "team", "position",
            "hfOutfield", "hfRO", "home_road", playerid_var, "game_date_gt",
            "game_date_lt", "hfFlag", "hfPull", "metric_1", "hfInn", "min_pitches",
            "min_results", "group_by", "sort_col", "player_event_sort",
            "h_launch_speed", "sort_order", "min_abs", "type"
        ],
        'value': [
            "true", "", "", "", "", "", "", "", "", "R%7CPO%7CS%7C", "",
            f"{datetime.strptime(start_date, '%Y-%m-%d').year}%7C", "", "", "", "", "",
            "", player_type, "", "", "", "", "", "", playerid if playerid is not None else "",
            start_date, end_date, "", "", "", "", "0", "0", "name", "pitches", "", "",
            "desc", "0", "details"
        ]})
    vars_df['pairs'] = vars_df.apply(
        lambda row: f"{row['var']}={row['value']}", axis=1)

    if playerid is None:
        vars_df = vars_df[~vars_df['var'].str.contains("lookup")]

    url_vars = "&".join(vars_df['pairs'])

    url = "".join(
        ["https://baseballsavant.mlb.com/statcast_search/csv?", url_vars])

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Send a GET request with the headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses

        # Suppress warnings while reading the CSV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Read the CSV data from the response content
            payload = pd.read_csv(io.StringIO(response.text), encoding='utf-8')

        # Display the first few rows to verify successful reading
        print(payload.head())

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        raise RuntimeError("No payload acquired") from e

    except Exception as e:
        print(e)
        raise RuntimeError("An unexpected error occurred") from e

    if len(payload) > 0:
        payload.columns = [
            "pitch_type", "game_date", "release_speed", "release_pos_x", "release_pos_z",
            "player_name", "batter", "pitcher", "events", "description", "spin_dir",
            "spin_rate_deprecated", "break_angle_deprecated", "break_length_deprecated",
            "zone", "des", "game_type", "stand", "p_throws", "home_team", "away_team",
            "type", "hit_location", "bb_type", "balls", "strikes", "game_year", "pfx_x",
            "pfx_z", "plate_x", "plate_z", "on_3b", "on_2b", "on_1b", "outs_when_up",
            "inning", "inning_topbot", "hc_x", "hc_y", "tfs_deprecated",
            "tfs_zulu_deprecated", "fielder_2", "umpire", "sv_id", "vx0", "vy0", "vz0",
            "ax", "ay", "az", "sz_top", "sz_bot", "hit_distance_sc", "launch_speed",
            "launch_angle", "effective_speed", "release_spin_rate", "release_extension",
                            "game_pk", "pitcher_1", "fielder_2_1", "fielder_3", "fielder_4", "fielder_5",
                            "fielder_6", "fielder_7", "fielder_8", "fielder_9", "release_pos_y",
                            "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
                            "woba_value", "woba_denom", "babip_value", "iso_value", "launch_speed_angle",
                            "at_bat_number", "pitch_number", "pitch_name", "home_score", "away_score",
                            "bat_score", "fld_score", "post_away_score", "post_home_score",
                            "post_bat_score", "post_fld_score", "if_fielding_alignment",
                            "of_fielding_alignment", "spin_axis", "delta_home_win_exp", "delta_run_exp",
                            "bat_speed", "swing_length"
        ]

        payload['game_date'] = pd.to_datetime(
            payload['game_date'], format='%Y-%m-%d')
        payload['des'] = payload['des'].astype(str)
        payload['inning'] = payload['inning'].astype(int)
        payload['at_bat_number'] = payload['at_bat_number'].astype(int)
        payload['pitch_number'] = payload['pitch_number'].astype(int)
        payload['game_pk'] = payload['game_pk'].astype(float)
        payload['hc_x'] = payload['hc_x'].astype(float)
        payload['hc_y'] = payload['hc_y'].astype(float)
        #payload['on_1b'] = payload['on_1b'].astype(int)
        #payload['on_2b'] = payload['on_2b'].astype(int)
        #payload['on_3b'] = payload['on_3b'].astype(int)
        payload['release_pos_x'] = payload['release_pos_x'].astype(float)
        payload['release_pos_y'] = payload['release_pos_y'].astype(float)
        payload['release_pos_z'] = payload['release_pos_z'].astype(float)
        payload['hit_distance_sc'] = payload['hit_distance_sc'].astype(float)
        payload['launch_speed'] = payload['launch_speed'].astype(float)
        payload['launch_angle'] = payload['launch_angle'].astype(float)
        payload['pfx_x'] = payload['pfx_x'].astype(float)
        payload['pfx_z'] = payload['pfx_z'].astype(float)
        payload['plate_x'] = payload['plate_x'].astype(float)
        payload['plate_z'] = payload['plate_z'].astype(float)
        payload['vx0'] = payload['vx0'].astype(float)
        payload['vy0'] = payload['vy0'].astype(float)
        payload['vz0'] = payload['vz0'].astype(float)
        payload['ax'] = payload['ax'].astype(float)
        payload['az'] = payload['az'].astype(float)
        payload['ay'] = payload['ay'].astype(float)
        payload['sz_bot'] = payload['sz_bot'].astype(float)
        payload['sz_top'] = payload['sz_top'].astype(float)
        payload['effective_speed'] = payload['effective_speed'].astype(float)
        payload['release_speed'] = payload['release_speed'].astype(float)
        payload['release_spin_rate'] = payload['release_spin_rate'].astype(float)
        payload['release_extension'] = payload['release_extension'].astype(float)
        payload['pitch_name'] = payload['pitch_name'].astype(str)
        payload['home_score'] = payload['home_score'].astype(int)
        payload['away_score'] = payload['away_score'].astype(int)
        payload['bat_score'] = payload['bat_score'].astype(int)
        payload['fld_score'] = payload['fld_score'].astype(int)
        payload['post_away_score'] = payload['post_away_score'].astype(int)
        payload['post_home_score'] = payload['post_home_score'].astype(int)
        payload['post_bat_score'] = payload['post_bat_score'].astype(int)
        payload['post_fld_score'] = payload['post_fld_score'].astype(int)
        payload['zone'] = payload['zone'].astype(int)
        payload['spin_axis'] = payload['spin_axis'].astype(float)
        payload['if_fielding_alignment'] = payload['if_fielding_alignment'].astype(str)
        payload['of_fielding_alignment'] = payload['of_fielding_alignment'].astype(str)

        cols_to_transform = ["batter", "pitcher", "fielder_2", "pitcher_1", "fielder_2_1",
                             "fielder_3", "fielder_4", "fielder_5", "fielder_6", "fielder_7",
                             "fielder_8", "fielder_9"]

    # Convert the specified columns to string, then to numeric, and replace NaN values
        payload[cols_to_transform] = (
            payload[cols_to_transform]
            .astype(str)                                    # Convert to string
            # Convert to numeric (NaNs for non-numeric)
            .apply(pd.to_numeric, errors='coerce')
            # Replace NaNs with 999999999
            .fillna(999999999)
        )

    else:
        print("No valid data")

    return payload









def Add_to_Names(start_date, end_date):
    NA_fixes = revamped_MLB_statcast_scrape(
        start_date=start_date, end_date=end_date, playerid=None, player_type='pitcher')
    NA_fixes['pitcher'] = NA_fixes['pitcher'].astype(str)

    masterExcel = pd.read_csv("masterExcel.csv",encoding='ISO-8859-1')
    Extra = pd.read_csv("Names_Add.csv",encoding='ISO-8859-1')

    ME_Add = masterExcel[['mlb_name','mlb_id']]

    Names_For_Add = pd.concat([ME_Add, Extra], axis=0, ignore_index=True)

    Names_For_Add['mlb_id'] = Names_For_Add['mlb_id'].astype(str)

    Names_For_Add = Names_For_Add.drop_duplicates()

    NA_fixes = NA_fixes.merge(
        Names_For_Add, how='left', left_on='pitcher', right_on='mlb_id')

    NA_fixes = NA_fixes[NA_fixes['mlb_name'].isna()]

    Names = NA_fixes.groupby(['player_name', 'pitcher']
                             ).size().reset_index().drop(columns=0)

    # Reformat the 'player_name' column by swapping first and last names
    Names['player_name'] = Names['player_name'].apply(
        lambda x: re.sub(r"(\w+),\s*(\w+)", r"\2 \1", x))

    Names = Names.rename(columns={
        "player_name": "mlb_name",
        "pitcher": "mlb_id"
        # "old_name": "new_name",  # Example of specific changes
    })
    Names_For_Add = pd.read_csv("Names_Add.csv",encoding='ISO-8859-1')

    Names_For_Add = pd.concat([Names_For_Add, Names],
                              axis=0, ignore_index=True)

    Names_For_Add.to_csv("Names_Add.csv", index=False)








def swing_decision_function(data_set):

    import pandas as pd
    import numpy as np
    import joblib

    
        
    
    
    data_set = data_set.dropna(subset=['plate_x'])
    data_set = data_set.dropna(subset=['plate_z'])
    data_set = data_set.dropna(subset=['pfx_x'])
    data_set = data_set.dropna(subset=['pfx_z'])
    data_set = data_set.dropna(subset=['balls'])
    data_set = data_set.dropna(subset=['strikes'])
    
    
    
    
    called_strike_gam= joblib.load('called_strike_gam.pkl')
    
    data_set['Called_Strike_Prob']= called_strike_gam.predict(data_set[['plate_x','plate_z']])

    
    swing_gam= joblib.load('swing_gam.pkl')
    
    data_set['Swing_Prob']= swing_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])


    single_gam= joblib.load('single_gam.pkl')
    
    data_set['Single_Prob']= single_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])
    
    
    
    double_gam= joblib.load('double_gam.pkl')
    
    data_set['Double_Prob']= double_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])
    
    
    triple_gam= joblib.load('triple_gam.pkl')
    
    data_set['Triple_Prob']= triple_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])
    
    
    
    homerun_gam= joblib.load('homerun_gam.pkl')
    
    data_set['HR_Prob']= homerun_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])
    
    
    
    out_gam= joblib.load('out_gam.pkl')
    
    data_set['Out_Prob']= out_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])
    
    
    
    
    
    conditions=[(data_set['on_1b'].isna() & data_set['on_2b'].isna() & data_set['on_3b'].isna()),
                (data_set['on_1b'].notna() & data_set['on_2b'].isna() & data_set['on_3b'].isna()),
                (data_set['on_1b'].isna() & data_set['on_2b'].notna() & data_set['on_3b'].isna()),
                (data_set['on_1b'].isna() & data_set['on_2b'].isna() & data_set['on_3b'].notna()),
                (data_set['on_1b'].notna() & data_set['on_2b'].notna() & data_set['on_3b'].isna()),
                (data_set['on_1b'].notna() & data_set['on_2b'].isna() & data_set['on_3b'].notna()),
                (data_set['on_1b'].isna() & data_set['on_2b'].notna() & data_set['on_3b'].notna()),
                (data_set['on_1b'].notna() & data_set['on_2b'].notna() & data_set['on_3b'].notna())
                ]


    choices= ['000','100', '010', '001','110','101', '011','111' ]
    
    
    data_set['Runners']=np.select(conditions, choices, default='ERROR')
    
    data_set['Runners'] = data_set['Runners'].astype(str)
    data_set['outs_when_up'] = data_set['outs_when_up'].astype(int)
    
    
    conditions2=[((data_set['Runners']=="000") & (data_set['outs_when_up']==0)),
                 ((data_set['Runners']=="000") & (data_set['outs_when_up']==1)),
                 ((data_set['Runners']=="000") & (data_set['outs_when_up']==2)),
                 ((data_set['Runners']=="100") & (data_set['outs_when_up']==0)),
                 ((data_set['Runners']=="100") & (data_set['outs_when_up']==1)),
                 ((data_set['Runners']=="100") & (data_set['outs_when_up']==2)),
                 ((data_set['Runners']=="010") & (data_set['outs_when_up']==0)),
                 ((data_set['Runners']=="010") & (data_set['outs_when_up']==1)),
                 ((data_set['Runners']=="010") & (data_set['outs_when_up']==2)),
                 ((data_set['Runners']=="001") & (data_set['outs_when_up']==0)),
                 ((data_set['Runners']=="001") & (data_set['outs_when_up']==1)),
                 ((data_set['Runners']=="001") & (data_set['outs_when_up']==2)),
                 ((data_set['Runners']=="110") & (data_set['outs_when_up']==0)),
                 ((data_set['Runners']=="110") & (data_set['outs_when_up']==1)),
                 ((data_set['Runners']=="110") & (data_set['outs_when_up']==2)),
                 ((data_set['Runners']=="101") & (data_set['outs_when_up']==0)),
                 ((data_set['Runners']=="101") & (data_set['outs_when_up']==1)),
                 ((data_set['Runners']=="101") & (data_set['outs_when_up']==2)),
                 ((data_set['Runners']=="011") & (data_set['outs_when_up']==0)),
                 ((data_set['Runners']=="011") & (data_set['outs_when_up']==1)),
                 ((data_set['Runners']=="011") & (data_set['outs_when_up']==2)),
                 ((data_set['Runners']=="111") & (data_set['outs_when_up']==0)),
                 ((data_set['Runners']=="111") & (data_set['outs_when_up']==1)),
                 ((data_set['Runners']=="111") & (data_set['outs_when_up']==2)),
                 ]


    choices2=[-0.25,-0.18,-0.11,-0.38,-0.33,-0.24,-0.44,-0.39,-0.34,-0.42,-0.62,-0.38,-0.55,-0.54,-0.46,-0.55,-0.69,-0.55,-0.61,-0.82,-0.61, -0.67,-0.86,-0.77]




    data_set['RV_Out']=np.select(conditions2, choices2, default='ERROR')


    data_set['RV_Out']=data_set['RV_Out'].astype(float)
    
    
    choices3=[0.41,0.28,0.13,0.6,0.43,0.22,0.38,0.27,0.12,0.37,0.24,0.17,0.75,0.63,0.31,0.51,0.39,0.22,0.26,0.2,0.16,1,1,1]


    data_set['RV_Walk']=np.select(conditions2, choices3, default='ERROR')
    
    
    
    data_set['RV_Walk']=data_set['RV_Walk'].astype(float)
    
    
    data_set['balls']=data_set['balls'].astype(int)
    data_set['strikes']=data_set['strikes'].astype(int)
    
    conditions4= [((data_set['balls']==0) & (data_set['strikes']==0)),
                  ((data_set['balls']==1) & (data_set['strikes']==0)),
                  ((data_set['balls']==2) & (data_set['strikes']==0)),
                  ((data_set['balls']==0) & (data_set['strikes']==1)),
                  ((data_set['balls']==1) & (data_set['strikes']==1)),
                  ((data_set['balls']==2) & (data_set['strikes']==1)),
                  ((data_set['balls']==0) & (data_set['strikes']==2)),
                  ((data_set['balls']==1) & (data_set['strikes']==2)),
                  ((data_set['balls']==2) & (data_set['strikes']==2)),
                  (data_set['balls']==3)
                  ]

    choices4=[0.036108717,0.069174515,0.146047634655311,0.0226912526927981,0.0486392876019061,0.118238221335809,0.0373043974898094,0.0500591067994056,0.0874434324885823,data_set['RV_Walk']]
    
    
    
    data_set['Count_Ball_RV']=np.select(conditions4, choices4, default='ERROR')
    
    
    data_set['Count_Ball_RV'] = pd.to_numeric(data_set['Count_Ball_RV'], errors='coerce')
    
    
    
    conditions5= [((data_set['balls']==0) & (data_set['strikes']==0)),
                  ((data_set['balls']==1) & (data_set['strikes']==0)),
                  ((data_set['balls']==2) & (data_set['strikes']==0)),
                  ((data_set['balls']==3) & (data_set['strikes']==0)),
                  ((data_set['balls']==0) & (data_set['strikes']==1)),
                  ((data_set['balls']==1) & (data_set['strikes']==1)),
                  ((data_set['balls']==2) & (data_set['strikes']==1)),
                  ((data_set['balls']==3) & (data_set['strikes']==1)),
                  (data_set['strikes']==2)
                  ]


    choices5=[-0.0500955001440635,-0.0635129642662081,-0.0840481921616983,-0.1118576054812,-0.0602540389132455,-0.0456408941162342,-0.0442210749187347,-0.0750158637659617,data_set['RV_Out']]


    data_set['Count_Strike_RV']=np.select(conditions5, choices5, default='ERROR')


    data_set['Count_Strike_RV'] = pd.to_numeric(data_set['Count_Strike_RV'], errors='coerce')
    
    data_set['xRV_Swing']= (
        0.883* data_set['Single_Prob']+
        1.238* data_set['Double_Prob']+
        1.558* data_set['Triple_Prob']+
        1.979* data_set['HR_Prob']+
        data_set['RV_Out'] * data_set['Out_Prob']
        )
    
    
    data_set['xRV_Take']= (
        data_set['Called_Strike_Prob']* data_set['Count_Strike_RV']+
        data_set['Count_Ball_RV']* (1-data_set['Called_Strike_Prob'])
        )
    
    
    data_set['Correct_Decision']=np.where(data_set['xRV_Swing']<data_set['xRV_Take'], "Take","Swing")



    data_set['IsCorrectDecision']=np.where(((data_set['Correct_Decision']=="Swing") & (data_set['IsSwing']==1))|((data_set['Correct_Decision']=="Take") & (data_set['IsSwing']==0)),1,0)
    
    
    
    data_set['RV_of_Result']= np.where(data_set['IsSwing']==1,(data_set['xRV_Swing']-data_set['xRV_Take'])*data_set['Swing_Prob'],
                                           np.where(data_set['IsSwing']==0,(data_set['xRV_Take']-data_set['xRV_Swing'])*data_set['Swing_Prob'],"Error"))


    
    data_set['RV_of_Result'] = pd.to_numeric(data_set['RV_of_Result'], errors='coerce')


    conditions6=[(data_set['RV_of_Result']>=0.01648372),
                 ((data_set['RV_of_Result']<0.01648372) & (data_set['RV_of_Result']>=0.007272974 )),
                 ((data_set['RV_of_Result']<0.007272974) & (data_set['RV_of_Result']>=0.001374903 )),
                 ((data_set['RV_of_Result']<0.001374903) & (data_set['RV_of_Result']>=-0.01711077  )),
                 (data_set['RV_of_Result']<-0.01711077 )
                 ]


    choices6=["A","B","C","D","F"]


    data_set['Swing_Decision_Grade']=np.select(conditions6, choices6, default='ERROR')


    data_set['Swing_Decision_Grade']=data_set['Swing_Decision_Grade'].astype(str)
    
    
    return data_set









def Add_to_statcast2024(start_date, end_date):
    Add_to_Names(start_date=start_date, end_date=end_date)

    LastNight = revamped_MLB_statcast_scrape(
        start_date=start_date, end_date=end_date, playerid=None, player_type='batter')

    LastNight['barrel'] = None

    LastNight['IsSingle'] = np.where(LastNight['events'] == "single", 1, np.where(LastNight['events'].isna(), 0, 0))
    LastNight['IsDouble'] = np.where(
        LastNight['events'] == "double", 1, np.where(LastNight['events'].isna(), 0, 0))
    LastNight['IsTriple'] = np.where(
        LastNight['events'] == "triple", 1, np.where(LastNight['events'].isna(), 0, 0))
    LastNight['IsHomerun'] = np.where(
        LastNight['events'] == "home_run", 1, np.where(LastNight['events'].isna(), 0, 0))
    LastNight['IsWalk'] = np.where(
        LastNight['events'] == "walk", 1, np.where(LastNight['events'].isna(), 0, 0))
    
    LastNight['IsOut'] = np.where(((LastNight['events'] == "double_play") |
                                   (LastNight['events'] == "field_out" )|
                                   (LastNight['events'] == "fielders_choice") |
                                   (LastNight['events'] == "fielders_choice_out") |
                                   (LastNight['events'] == "force_out") |
                                   (LastNight['events'] == "grounded_into_double_play" )|
                                   (LastNight['events'] == "strikeout" )|
                                   (LastNight['events'] == "strikeout_double_play") |
                                   (LastNight['events'] == "triple_play") ), 1, np.where(LastNight['events'].isna(), 0, 0))


    LastNight['IsInPlay'] = np.where(((LastNight['events'] == "double_play") |
                                        (LastNight['events'] == "field_out") |
                                        (LastNight['events'] == "fielders_choice") |
                                        (LastNight['events'] == "fielders_choice_out") |
                                        (LastNight['events'] == "force_out" )|
                                        (LastNight['events'] == "grounded_into_double_play" )|
                                        (LastNight['events'] == "single") |
                                        (LastNight['events'] == "double") |
                                        (LastNight['events'] == "triple" )|
                                        (LastNight['events'] == "home_run" )|
                                        (LastNight['events'] == "triple_play")|
                                        (LastNight['events'] == "field_error") ), 1, np.where(LastNight['events'].isna(), 0, 0))

    LastNight['IsStrikeout'] = np.where(((LastNight['events'] == "strikeout") |
                                        (LastNight['events'] == "strikeout_double_play") 
                                        ), 1, np.where(LastNight['events'].isna(), 0, 0))


    LastNight['IsSwMiss'] = np.where(((LastNight['description'] == "swinging_strike" )|
                                        (LastNight['description'] == "swinging_strike_blocked") 
                                        ), 1, np.where(LastNight['events'].isna(), 0, 0))


    LastNight['IsSwing'] = np.where(((LastNight['description'] == "swinging_strike") |
                                        (LastNight['description'] == "swinging_strike_blocked") |
                                        (LastNight['description'] == "foul") |
                                        (LastNight['description'] == "hit_into_play") |
                                        (LastNight['description'] == "hit_into_play_no_out") |
                                        (LastNight['description'] == "hit_into_play_score" )
                                        ), 1, 0)
    
    LastNight['BattingTeam']=np.where((LastNight['inning_topbot']=="Top") , LastNight['away_team'],LastNight['home_team'])


    LastNight['IsError']=np.where(LastNight['events']=="field_error",1, 
                                  np.where(LastNight['events'].isna(),0,0))


    LastNight['IsCalledStrike']= np.where(LastNight['description']=='called_strike',1,0)


    LastNight['IsBarrel']=np.where(LastNight['barrel'].isna(),None,"Error")

    def categorize(value):
        if value == "FF":
            return '1'
        elif value == "FT":
            return '2'
        elif value == "SI":
           return '3'
        elif value == "FC":
             return '4'
        elif value == "FS":
             return '5'
        elif value == "CH":
            return '6'
        elif value == "CS":
            return '7'
        elif value == "SL":
            return '8'
        elif value == "CU":
            return '9'
        elif value == "KC":
            return '10'
        else:
            return '0'

# Apply the function to the column
    LastNight['Pitch_type_code'] = LastNight['pitch_type'].apply(categorize)


    LastNight['Is95mphEV']= np.where(((LastNight['launch_speed']<95) | (LastNight['launch_speed'].isna())| (LastNight['IsInPlay']=="0")),0,1)

    LastNight['Is90mphEV']= np.where(((LastNight['launch_speed']<90) | (LastNight['launch_speed'].isna())| (LastNight['IsInPlay']=="0")),0,1)

    LastNight['IsRHP']= np.where(LastNight['p_throws']=="R",1,0)
    
    LastNight['IsFlyball']= np.where(LastNight['bb_type']=='fly_ball',1,0)
    
    LastNight['IsHBP']= np.where(LastNight['description']=='hit_by_pitch',1,0)

    LastNight['IsInZone']= np.where(((LastNight['plate_x']>(-0.94)) & (LastNight['plate_x']>(0.94)) & (LastNight['plate_z']>(1.6)) & (LastNight['plate_z']>(3.6))  ),1,0)

    LastNight['IsRHH'] = np.where(LastNight['stand']=="R",1,0)
    
    
    
    LastNight['IsSingle'] = LastNight['IsSingle'].fillna(0)
    LastNight['IsDouble'] = LastNight['IsDouble'].fillna(0)
    LastNight['IsTriple'] = LastNight['IsTriple'].fillna(0)
    LastNight['IsHomerun'] = LastNight['IsHomerun'].fillna(0)
    LastNight['IsWalk'] = LastNight['IsWalk'].fillna(0)
    LastNight['IsOut'] = LastNight['IsOut'].fillna(0)
    LastNight['IsInPlay'] = LastNight['IsOut'].fillna(0)
    LastNight['IsStrikeout'] = LastNight['IsStrikeout'].fillna(0)
    LastNight['IsError'] = LastNight['IsError'].fillna(0)
    LastNight['IsFlyball'] = LastNight['IsFlyball'].fillna(0)
    
    
   
    LastNight = swing_decision_function(LastNight)
    
    LastNight['VB']= 12* LastNight['pfx_z']
    
    LastNight['HB']=12* LastNight['pfx_x']
    
    
    
    
    single_EVLA_gam= joblib.load('single_EVLA_gam.pkl')
    
   
    def compute_xSingle_EVLA(row):
    # Check if launch_speed and launch_angle are numeric
        if pd.api.types.is_numeric_dtype(row['launch_speed']) and pd.api.types.is_numeric_dtype(row['launch_angle']):
            # If numeric, apply the prediction model
            return single_EVLA_gam.predict([[row['launch_speed'], row['launch_angle']]])[0]
        else:
            # If not numeric, return NaN
            return np.nan

# Apply the function to each row of the DataFrame
    LastNight['xSingle_EVLA'] = LastNight.apply(compute_xSingle_EVLA, axis=1)
    
    
    
    
    
    
    double_EVLA_gam= joblib.load('double_EVLA_gam.pkl')
    
    def compute_xDouble_EVLA(row):
    # Check if launch_speed and launch_angle are numeric
        if pd.api.types.is_numeric_dtype(row['launch_speed']) and pd.api.types.is_numeric_dtype(row['launch_angle']):
            # If numeric, apply the prediction model
            return double_EVLA_gam.predict([[row['launch_speed'], row['launch_angle']]])[0]
        else:
            # If not numeric, return NaN
            return np.nan

# Apply the function to each row of the DataFrame
    LastNight['xDouble_EVLA'] = LastNight.apply(compute_xDouble_EVLA, axis=1)


    triple_EVLA_gam= joblib.load('triple_EVLA_gam.pkl')
    
    def compute_xTriple_EVLA(row):
    # Check if launch_speed and launch_angle are numeric
        if pd.api.types.is_numeric_dtype(row['launch_speed']) and pd.api.types.is_numeric_dtype(row['launch_angle']):
            # If numeric, apply the prediction model
            return triple_EVLA_gam.predict([[row['launch_speed'], row['launch_angle']]])[0]
        else:
            # If not numeric, return NaN
            return np.nan

# Apply the function to each row of the DataFrame
    LastNight['xTriple_EVLA'] = LastNight.apply(compute_xTriple_EVLA, axis=1)
     

     


    HR_EVLA_gam= joblib.load('HR_EVLA_gam.pkl')
     
    def compute_xHR_EVLA(row):
    # Check if launch_speed and launch_angle are numeric
        if pd.api.types.is_numeric_dtype(row['launch_speed']) and pd.api.types.is_numeric_dtype(row['launch_angle']):
            # If numeric, apply the prediction model
            return HR_EVLA_gam.predict([[row['launch_speed'], row['launch_angle']]])[0]
        else:
            # If not numeric, return NaN
            return np.nan

# Apply the function to each row of the DataFrame
    LastNight['xHomeRun_EVLA'] = LastNight.apply(compute_xHR_EVLA, axis=1)
    
    
    
    
    LastNight['xSingle_EVLA']=LastNight['xSingle_EVLA'].astype(float)
    LastNight['xDouble_EVLA']=LastNight['xDouble_EVLA'].astype(float)
    LastNight['xTriple_EVLA']=LastNight['xTriple_EVLA'].astype(float)
    LastNight['xHomeRun_EVLA']=LastNight['xHomeRun_EVLA'].astype(float)
    
    LastNight['xSingle_EVLA'] = LastNight['xSingle_EVLA'].fillna(0)
    LastNight['xDouble_EVLA'] = LastNight['xDouble_EVLA'].fillna(0)
    LastNight['xTriple_EVLA'] = LastNight['xTriple_EVLA'].fillna(0)
    LastNight['xHomeRun_EVLA'] = LastNight['xHomeRun_EVLA'].fillna(0)
    
    
    LastNight['xSingle_EVLA'] = (LastNight['xSingle_EVLA']* LastNight['IsInPlay'])
    LastNight['xDouble_EVLA'] = (LastNight['xDouble_EVLA']* LastNight['IsInPlay'])
    LastNight['xTriple_EVLA'] = (LastNight['xTriple_EVLA']* LastNight['IsInPlay'])
    LastNight['xHomeRun_EVLA'] = (LastNight['xHomeRun_EVLA']* LastNight['IsInPlay'])
    
    
    
    
    masterExcel=pd.read_csv('masterExcel.csv',encoding='ISO-8859-1')
    
    Extra_For_Add=pd.read_csv('Names_Add.csv',encoding='ISO-8859-1')
    
    ME_For_Add= masterExcel[['mlb_id','mlb_name']]
    
    Names_For_Add = pd.concat([ME_For_Add, Extra_For_Add], ignore_index=True)
   
    Names_For_Add = Names_For_Add.drop_duplicates(subset=['mlb_id', 'mlb_name'], keep='first')
    
    Names_For_Add['mlb_id']=Names_For_Add['mlb_id'].astype(str)
    
    LastNight['pitcher']=LastNight['pitcher'].astype(str)
    
    
    Names_For_Add=Names_For_Add[['mlb_id','mlb_name']]
    
    LastNight = LastNight.merge(Names_For_Add, how='left', left_on='pitcher', right_on='mlb_id')
    
    sport_id = 1  # 1 represents baseball
    season = 2024

    url = f"https://statsapi.mlb.com/api/v1/sports/{sport_id}/players?season={season}"

    # Make a request to the MLB API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Convert the player data to a pandas DataFrame
        heights = pd.json_normalize(data['people'])
    else:
        print(f"Failed to fetch data: {response.status_code}")
    
    
    def convert_height_to_feet(height):
        try:
            # Check if height is a string and not empty
            if isinstance(height, str) and "'" in height:
                # Remove any extra spaces and quotes
                height = height.strip().replace('"', '')
                # Split the height by the foot marker "'"
                parts = height.split("'")
                
                # Convert feet and inches, ensuring they are numeric
                feet = int(parts[0].strip())
                inches = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 0
    
                # Calculate height in feet
                return feet + inches / 12
            else:
                # Return NaN if height is not in correct format
                return pd.NA
        except (ValueError, TypeError):
            # Return NaN for any conversion errors
            return pd.NA


    heights['height_ft'] = heights['height'].apply(convert_height_to_feet)
    
    
    
    
    
    
    
    heights['height_ft']=heights['height_ft'].astype(float)
    
    heights['id'] = heights['id'].astype(str)
    
    heights=heights[['id','height_ft']]
    
    LastNight = LastNight.merge(heights, how='left', left_on='pitcher', right_on='id')
    
    statcast2024=pd.read_feather('statcast2024.feather')
    
    Combined = pd.concat([statcast2024, LastNight], ignore_index=True)
    
    Combined['outs_when_up'] = pd.to_numeric(Combined['outs_when_up'], errors='coerce')
    
    Combined['Pitch_type_code'] = pd.to_numeric(Combined['Pitch_type_code'], errors='coerce')
    
    Combined.to_feather('statcast2024.feather')
    
    
    return Combined
   
    
   



 
##EXAMPLE OF HOW TO ADD A DAY TO A DATABASE CALLED statcast2024
statcast2024=Add_to_statcast2024('2024-05-06', '2024-05-06')


