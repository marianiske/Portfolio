import random
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Webdata.win_probs import QuotaCalculator
import soccerdata as sd
from Webdata.get_key_players import get_missing_xG
import time
from functools import lru_cache
import signal

def _t():
    return time.perf_counter()

leagues = ['Bundesliga', 'EPL', 'Serie_A', 'La_Liga', 'Ligue_1']
years = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026']

teams_alias={'RB Leipzig': 'RasenBallsport_Leipzig', 'Ein Frankfurt':'Eintracht_Frankfurt', 'Heidenheim':'FC_Heidenheim',
             'Leverkusen':'Bayer_Leverkusen', 'Stuttgart':'VfB_Stuttgart', 'St Pauli':'St._Pauli', 'Dortmund': 'Borussia_Dortmund', 'Mainz':'Mainz_05',
             'FC Koln':'FC_Cologne', "M'gladbach": 'Borussia_M.Gladbach', 'Hamburg':'Hamburger_SV',
             'Newcastle':'Newcastle_United', 'Wolves':'Wolverhampton_Wanderers', 'Man City':'Manchester_City',
             "Nott'm Forest":'Nottingham_Forest', 'Man United':'Manchester_United',
             'Milan':'AC_Milan', 'Parma_Calcio_1913':'Parma', 'Vallecano':'Rayo_Vallecano',
             'Oviedo':'Real_Oviedo', 'Sociedad':'Real_Sociedad', 'Celta':'Celta_Vigo',
             'Ath Bilbao':'Athletic_Club', 'Espanol':'Espanyol', 'Ath Madrid':'Atletico_Madrid',
             'Betis':'Real_Betis', 'Paris SG':'Paris_Saint_Germain', 'West Brom':'West_Bromwich_Albion', 'Fortuna Dusseldorf':'Fortuna_Duesseldorf',
             'Spal': 'SPAL_2013', 'La Coruna': 'Deportivo_La_Coruna', 'Hannover':'Hannover_96', 'Schalke':'Schalke_04',
             'Hertha':'Hertha_Berlin', 'Nurnberg': 'Nuernberg', 'Bielefeld':'Arminia_Bielefeld', 'Greuther Furth':'Greuther_Fuerth',
             'QPR':'Queens_Park_Rangers', 'Sp Gijon':'Sporting_Gijon'}

def get_alias_team(team_name):
    try:
        team = teams_alias[team_name]
    except:
        team = team_name.replace('_', ' ')
    return team

class DataSet():
    
    def __init__(self):
        self.qc = QuotaCalculator()
        self.elo = sd.ClubElo()
    
    def fill_row(self, row, playground, xG, form, goals_last5, goals_conceeded_last5):
        row[f'xG {playground}'] = xG
        row[f'W {playground}'] = form['W']
        row[f'D {playground}'] = form['D']
        row[f'L {playground}'] = form['L']
        row[f'{playground} Goals last 5'] = goals_last5
        row[f'{playground} Goals conceeded last 5'] = goals_conceeded_last5
        return row
    
    @lru_cache(maxsize=None)
    def _team_history(self, team: str):
        # wird pro Team-String nur einmal geladen
        return self.elo.read_team_history(team)

    def get_elo_on_date(self, team: str, date):
        team_elo = self._team_history(team)
        # robust: falls Datum früher als erster Eintrag
        sub = team_elo[team_elo.index <= date]
        if len(sub) == 0:
            return team_elo.iloc[0]
        return sub.iloc[-1]
    
    def get_elo(self, team, team_alias, date):
        candidates = [
            team,
            team_alias,
            team.split("_", 1)[0],
            team.split("_", 1)[-1],
            team_alias.split(" ", 1)[0],
            team_alias.split(".")[-1],
            team_alias.split("'", 1)[-1],
            team.replace('o', 'oe'),
            team_alias.replace('o', 'oe').rsplit("_", 1)[-1].split(" ", 1)[-1],
            team.replace('u', 'ue'),
            team_alias.replace('u', 'ue'),
            team_alias.split(" ", 1)[-1],
        ]
        for candidate in candidates: 
            try:
                elo = self.get_elo_on_date(candidate, date)['elo']
                return elo
            except:
                continue
        print(f'No elo available for {candidates}.')
        print('Check more possible options')
        return 1550

    def get_ith_data_of_league_in_season(self, i, league, year, df):
        row_in_df={}
        game = df.loc[i]
        date = game['Date']
        date = pd.to_datetime(date, dayfirst=True)
        home_csv = game['HomeTeam']
        away_csv = game['AwayTeam']
        home = get_alias_team(home_csv)
        away = get_alias_team(away_csv)
        
        print(f'Match: {home} - {away}')
        
        t0 = _t()
        xG_home, xG_away, form_home, goals_last5_home, goals_conceeded_last5_home, form_away, goals_last5_away, goals_conceeded_last5_away, match_id = self.qc.get_xGoals_and_Form(home, away, league, season = year)
        print("get_xGoals_and_Form:", _t()-t0)
        
        row_in_df=self.fill_row(row_in_df, 'Home', xG_home, form_home, goals_last5_home, goals_conceeded_last5_home)
        row_in_df=self.fill_row(row_in_df, 'Away', xG_away, form_away, goals_last5_away, goals_conceeded_last5_away)
        
        p_home, p_away = self.qc.win_probability(xG_home, xG_away)
        p_draw = 1 - p_home - p_away
        
        row_in_df['Probability 1'] = p_home
        row_in_df['Probability X'] = p_draw
        row_in_df['Probability 2'] = p_away
        
        quota_double_chance = self.qc.double_chance_quota(p_home, p_away)
        
        row_in_df['Probability 1X'] = 1/float(quota_double_chance['1X'])
        row_in_df['Probability 12'] = 1/float(quota_double_chance['12'])
        row_in_df['Probability X2'] = 1/float(quota_double_chance['X2'])
        
        for go in range(1, 6): #+/- x.5
            prob_at_least_n_goals_val = self.qc.prob_at_least_n_goals(xG_home, xG_away, go)
            prob_less_than_n_goals = 1 - prob_at_least_n_goals_val
            row_in_df[f'Probability +{go-0.5}'] = prob_at_least_n_goals_val
            row_in_df[f'Probability -{go-0.5}'] = prob_less_than_n_goals
        
        t0 = _t()
        elo_home = self.get_elo(home, home_csv, date)
        elo_away = self.get_elo(away, away_csv, date)
        print("get_elo total:", _t()-t0)
        
        row_in_df['Elo Home']=elo_home
        row_in_df['Elo Away']=elo_away
        
        p_home_book = 1/game['PSH']
        p_draw_book = 1/game['PSD']
        p_away_book = 1/game['PSA']
        
        p_sum = p_home_book + p_draw_book + p_away_book
        
        p_home_normed = p_home_book / p_sum
        p_draw_normed = p_draw_book / p_sum
        p_away_normed = p_away_book / p_sum   
        
        row_in_df['P Home normed'] = p_home_normed
        row_in_df['P Draw normed'] = p_draw_normed
        row_in_df['P Away normed'] = p_away_normed
        
        try:
            row_in_df['Avg Quota H'] = game['AvgH']
            row_in_df['Avg Quota D'] = game['AvgD']
            row_in_df['Avg Quota A'] = game['AvgA']
        except:
            row_in_df['Avg Quota H'] = game['PSH']
            row_in_df['Avg Quota D'] = game['PSD']
            row_in_df['Avg Quota A'] = game['PSA']
        
        try:
            row_in_df['Avg Quota >2.5'] = game['P>2.5']
            row_in_df['Avg Quota <2.5'] = game['P<2.5']
        except:
            row_in_df['Avg Quota >2.5'] = game['BbAv>2.5']
            row_in_df['Avg Quota <2.5'] = game['BbAv<2.5']
        
        t0 = _t()
        missing_xG_home, missing_xG_build_up_home = get_missing_xG(home, year, match_id, date, num_key_players = 5, playground = 'h')
        missing_xG_away, missing_xG_build_up_away = get_missing_xG(away, year, match_id, date, num_key_players = 5, playground = 'a')
        print("get_missing_xG total:", _t()-t0)
        
        row_in_df['Missing xGH']=missing_xG_home
        row_in_df['Missing xGA']=missing_xG_away
        
        row_in_df['Missing Defense xGH']=missing_xG_build_up_home
        row_in_df['Missing Defense xGA']=missing_xG_build_up_away
        
        row_in_label={}
        if game['FTR'] == 'H':
            row_in_label['H'] = 1
            row_in_label['D'] = 0
            row_in_label['A'] = 0
        elif game['FTR'] == 'D':
            row_in_label['H'] = 0
            row_in_label['D'] = 1
            row_in_label['A'] = 0
        else:
            row_in_label['H'] = 0
            row_in_label['D'] = 0
            row_in_label['A'] = 1
        
        if game['FTR'] == 'H' or game['FTR'] == 'D':
            row_in_label['1X'] = 1
        else:
            row_in_label['1X'] = 0
            
        if game['FTR'] == 'H' or game['FTR'] == 'A':
            row_in_label['12'] = 1
        else:
            row_in_label['12'] = 0
        
        if game['FTR'] == 'D' or game['FTR'] == 'A':
            row_in_label['X2'] = 1
        else:
            row_in_label['X2'] = 0
            
        goals_in_match = int(game['FTHG']) + int(game['FTAG'])
        
        if goals_in_match > 1.5:
            row_in_label['+1.5']=1
            if goals_in_match > 2.5:
                row_in_label['+2.5']=1
                if goals_in_match > 3.5:
                    row_in_label['+3.5']=1
                    if goals_in_match > 4.5:
                        row_in_label['+4.5']=1
                    else:
                        row_in_label['+4.5']=0
                else:
                    row_in_label['+3.5']=0
                    row_in_label['+4.5']=0
            else:
                row_in_label['+2.5']=0
                row_in_label['+3.5']=0
                row_in_label['+4.5']=0
        else:
            row_in_label['+1.5']=0
            row_in_label['+2.5']=0
            row_in_label['+3.5']=0
            row_in_label['+4.5']=0
            

        return row_in_df, row_in_label
        
    def get_all_data_points(self):
        data_list = []
        labels = []
        for year in years:
            try:
                for league in leagues:
                    df = pd.read_csv('DataSet/'+league+year+'.csv')
                    year = int(year) - 1
                    for i in range(len(df)):
                        print(f'Dates collected: {i}')
                        row_in_df, row_in_label = self.get_ith_data_of_league_in_season(i, league, year, df)
                        data_list.append(row_in_df)
                        labels.append(row_in_label)
            except:
                print(f'An Error occured at match = {i}, in league {league} in season {year}/{year+1}')
                available_data = pd.DataFrame(data_list) 
                available_labels = pd.DataFrame(labels)
                available_data.to_json("dataset.json")
                available_labels.to_json("labels.json")
                break
                        
            available_data = pd.DataFrame(data_list) 
            available_labels = pd.DataFrame(labels)
            return available_data, available_labels
    
    def get_random_data(self, data_points):
        data_list = []
        labels = []
        while len(data_list) < data_points:
            league=random.choice(leagues)
            year=random.choice(years)
            
            df = pd.read_csv('DataSet/'+league+year+'.csv')
            idx = random.sample(range(len(df)), 10)
            year = int(year) - 1
            for i in idx:
                print(f'Dates collected: {len(data_list)}')
                row_in_df, row_in_label = self.get_ith_data_of_league_in_season(i, league, year, df)
                data_list.append(row_in_df)
                labels.append(row_in_label)
        available_data = pd.DataFrame(data_list) 
        available_labels = pd.DataFrame(labels)
        return available_data, available_labels
    
class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException()
    
def main():

    signal.signal(signal.SIGALRM, _timeout_handler)
        
    ds = DataSet()
    
    data_rows = []
    label_rows = []
    i=0
    season_year=0
    for league in leagues:
        for year_str in years:
            try:
                df = pd.read_csv(f"../CSVs/{league}{year_str}.csv")
                season_year = int(year_str) - 1
    
                for i in range(len(df)):
                    print(f"League: {league}, Year: {year_str}, Dates collected: {i}")
                    signal.alarm(20 * 60)  # set alarm for 20 Min
                    try:
                        row_in_df, row_in_label = ds.get_ith_data_of_league_in_season(i, league, season_year, df)
                        data_rows.append(row_in_df)
                        label_rows.append(row_in_label)
                    except Exception as e:
                        print(f"Timeout at i={i} -> skipped: {e}")
                        continue
                    finally:
                        signal.alarm(0)  
            
            except Exception as e:
                print(f"An Error occured at match={i}, league={league}, season={season_year}/{season_year+1}")
                print(e)
                pd.DataFrame(data_rows).to_json(f"dataset_{league}_{year_str}.json")
                pd.DataFrame(label_rows).to_json(f"labels_{league}_{year_str}.json")
                raise 
            finally:
                if data_rows:
                    pd.DataFrame(data_rows).to_json(
                        f"dataset_{league}_{year_str}.json"
                    )
                if label_rows:
                    pd.DataFrame(label_rows).to_json(
                        f"labels_{league}_{year_str}.json"
                    )
            pd.DataFrame(data_rows).to_json(f"dataset_{league}_{year_str}.json")
            pd.DataFrame(label_rows).to_json(f"labels_{league}_{year_str}.json")
    
    df_data = pd.DataFrame(data_rows)
    df_label = pd.DataFrame(label_rows)

if __name__ == '__main__':
    main() 


    