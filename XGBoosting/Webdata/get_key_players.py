from Webdata.get_understat_data import get_player_data, get_team_data, get_match_data
import pandas as pd

def get_key_players(team, season, start_date=pd.to_datetime('01-11-2010'), end_date=pd.Timestamp.today(), n=3):
    '''Returns the three players with the highest xG in the last two months'''
    
    best_n_players = {str(i): 0 for i in range(1, n + 1)}
    best_n_players_defense = {str(i): 0 for i in range(1, n + 1)}
    players = get_team_data(team, season)['players']
    
    for p in players:
        player = p['player_name']
        player_id = p['id']
        player_matches = get_player_data(player_id)['matches']
        sum_xG = 0
        sum_xGBuildUp = 0
        for game in player_matches:
            sum_xG += float(game['xG']) 
            if 'D' in p['position']:
                sum_xGBuildUp += float(game['xGBuildup'])
            if pd.to_datetime(game['date']) < start_date or pd.to_datetime(game['date']) > end_date:
                break
            
        worst_key_player_key = min(best_n_players, key=best_n_players.get)
        if sum_xG > best_n_players[worst_key_player_key]:
            best_n_players[player]=sum_xG
            best_n_players.pop(worst_key_player_key)
            
        worst_key_player_key = min(best_n_players_defense, key=best_n_players_defense.get)
        if sum_xGBuildUp > best_n_players_defense[worst_key_player_key]:
            best_n_players_defense[player]=sum_xGBuildUp
            best_n_players_defense.pop(worst_key_player_key)
            
    return best_n_players, best_n_players_defense

def player_in_line_up(player_name, match_id, playground):
    line_up = list(get_match_data(match_id)['rosters'][playground].values())[:11]
    for player in line_up:
        if player['player'] == player_name:
            return True
    
    return False

def get_missing_xG(team, season, match_id, match_date, num_key_players, playground):
    
    missing_xG = 0
    missing_xG_buildup=0
    
    end_date = pd.to_datetime(match_date, dayfirst=True)
    start_date = end_date - pd.DateOffset(months=2)
    key_players, key_players_defense = get_key_players(team, season, start_date, end_date, n=num_key_players)
    for key_player in key_players.keys():
        if not player_in_line_up(key_player, match_id, playground):
            missing_xG += key_players[key_player]
    
    for key_player in key_players_defense:
        if not player_in_line_up(key_player, match_id, playground):
            missing_xG_buildup += key_players_defense[key_player]
        
    return missing_xG, missing_xG_buildup  
        
    