from Webdata.get_understat_data import get_league_data, get_team_data
import numpy as np
import math
from collections import Counter

class QuotaCalculator:

    last_n_games = 10
    league_games_per_matchday = 10
    current_matchday = 0
    
    def get_current_matchday(self, league : str, season : str):
        self.current_matchday = []
        matches = get_league_data(league, season)
        for idx, m in enumerate(matches):
            if not m['isResult']:
                self.current_matchday.append(matches[idx])
                if len(self.current_matchday) >= self.league_games_per_matchday:
                    break
        return self.current_matchday
    
    def get_stats_list(self, team : str, season="2026", possible_opponent = '-', playground = 'h'):
        playground_opposite = 'a' if playground == 'h' else 'h'
        team_match_data = get_team_data(team, season)['dates']
        n=0
        match_id=0
        xGoals=[]
        xGoals_home = []
        xGoals_away = []
        xGoals_against=[]
        xGoals_against_home=[]
        xGoals_against_away=[]
        results = []
        goals_scored = []
        goals_conceeded = []
        while n < len(team_match_data) and team_match_data[n]['isResult']:
            if (team_match_data[n][playground]['title']==team.replace('_', ' ') and team_match_data[n][playground_opposite]['title']==possible_opponent.replace('_', ' ')):
                match_id = team_match_data[n]['id']
                break
            
            side = team_match_data[n]['side']
            side_op = 'a' if side == 'h' else 'h'
            start_len = len(xGoals)
            try:
                xGoals.append(float(team_match_data[n]['xG'][side]))
                xGoals_against.append(float(team_match_data[n]['xG'][side_op]))
                goals_scored.append(int(team_match_data[n]['goals'][side]))
                goals_conceeded.append(int(team_match_data[n]['goals'][side_op]))
            except:
                del xGoals[start_len:]
                del xGoals_against[start_len:]
                del goals_scored[start_len:]
                del goals_conceeded[start_len:]
                print('Error with understat!')
                n+=1
                continue
            
            if side == 'a':
                xGoals_away.append(float(team_match_data[n]['xG'][side]))
                xGoals_against_away.append(float(team_match_data[n]['xG'][side_op]))
            else:
                xGoals_home.append(float(team_match_data[n]['xG'][side]))
                xGoals_against_home.append(float(team_match_data[n]['xG'][side_op]))
                    
            results.append(team_match_data[n]['result'].upper())
            n+=1
        if len(results) < 3:
            try:
                (xGoals_prev,
                    xGoals_home_prev,
                    xGoals_away_prev,
                    xGoals_against_prev,
                    xGoals_against_home_prev,
                    xGoals_against_away_prev,
                    results_prev,
                    goals_scored_prev,
                    goals_conceeded_prev, _) = self.get_stats_list(team, int(season)-1, possible_opponent = '-', playground = 'h')
                
                xGoals_prev = xGoals_prev[-(3-len(results)):]
                xGoals = xGoals + xGoals_prev
                
                xGoals_home_prev = xGoals_home_prev[-(3-len(results)):]
                xGoals_home = xGoals_home + xGoals_home_prev
                
                xGoals_away_prev = xGoals_away_prev[-(3-len(results)):]
                xGoals_away = xGoals_away + xGoals_away_prev
                
                xGoals_against_prev = xGoals_against_prev[-(3-len(results)):]
                xGoals_against = xGoals_against + xGoals_against_prev
                
                xGoals_against_home_prev = xGoals_against_home_prev[-(3-len(results)):]
                xGoals_against_home = xGoals_against_home + xGoals_against_home_prev
                
                xGoals_against_away_prev = xGoals_against_away_prev[-(3-len(results)):]
                xGoals_against_away = xGoals_against_away + xGoals_against_away_prev
                
                results_prev = results_prev[-(3-len(results)):]
                results = results + results_prev
                
                goals_scored_prev = goals_scored_prev[-(3-len(results)):]
                goals_scored = goals_scored + goals_scored_prev
                
                goals_conceeded_prev = goals_conceeded_prev[-(3-len(results)):]
                goals_conceeded = goals_conceeded + goals_conceeded_prev
            except:
                print('No previous data available!')
            
        return (
            xGoals,
                xGoals_home,
                xGoals_away,
                xGoals_against,
                xGoals_against_home,
                xGoals_against_away,
                results,
                goals_scored,
                goals_conceeded,
                match_id
        )
    
    def get_form_last_n_games(results, goals_scored, goals_conceeded, n):
        results = results[-n:]
        form = Counter(results)
        
        goals_last_n = sum(goals_scored[-n:])
        goals_conceeded_last_n = sum(goals_conceeded[-n:])
        
        return form, goals_last_n, goals_conceeded_last_n
    
    def get_xGoals_last_n(self, xGoals, xGoals_against, n):
        last_n_xGoals = xGoals[-n:]
        last_n_xGoals_against = xGoals_against[-n:]
        return last_n_xGoals, last_n_xGoals_against
    
    def calc_home_advantage_last_n(self, xGoals_home, xGoals_away, xGoals_against_home, xGoals_against_away, n_last_games):
        home_advantage_attack = np.mean(xGoals_home) / np.mean(xGoals_away)
        home_advantage_defense = np.mean(xGoals_against_away) / np.mean(xGoals_against_home)
        home_advantage_shrunk = 1+(2/n_last_games)*(home_advantage_attack * home_advantage_defense-1) #shrink home advantage
        home_advantage_clipped = float(np.clip(home_advantage_shrunk, 0.5, 1.7)) #clip for unrealistic values
        return home_advantage_clipped

    def xGoals_team(self, team : str, n_last_games : int, season="2026", possible_opponent = '-', playground = 'h'):
        xGoals,xGoals_home, xGoals_away,xGoals_against,xGoals_against_home,xGoals_against_away,results,goals_scored,goals_conceeded, match_id = self.get_stats_list(team, season, possible_opponent, playground)
            
        home_advantage_clipped = self.calc_home_advantage_last_n(xGoals_home, xGoals_away, xGoals_against_home, xGoals_against_away, n_last_games)
        
        last_n_xGoals, last_n_xGoals_against = self.get_xGoals_last_n(xGoals, xGoals_against, n_last_games)
        
        results = results[-5:]
        form = Counter(results)
        
        goals_last5 = sum(goals_scored[-5:])
        goals_conceeded_last5 = sum(goals_conceeded[-5:])
        
        return (
            np.mean(last_n_xGoals),
            np.mean(last_n_xGoals_against), 
            home_advantage_clipped,
            form, 
            goals_last5,
            goals_conceeded_last5,
            match_id
        )
    
    def poisson(self, lamb, k):
        return lamb**k*np.exp(-lamb)/math.factorial(k)
    
    def league_avg_xG(self, league: str, season: str):
    
        matches = get_league_data(league, season)
    
        total_xG = 0.0
        num_matches = 0
    
        for idx, m in enumerate(matches):
            if not m['isResult']:
                if self.current_matchday == None: 
                    self.current_matchday = matches[idx : idx+self.league_games_per_matchday]
                break
            xG_home = float(m['xG']['h'])
            xG_away = float(m['xG']['a'])
    
            total_xG += xG_home + xG_away
            num_matches += 1
            
        if num_matches>0:
            avg_xG_per_match = total_xG / num_matches
            avg_xG_per_team = total_xG / (2 * num_matches)
        else:
            avg_xG_per_match, avg_xG_per_team = (0,0)
        
        return avg_xG_per_match, avg_xG_per_team
    
    def get_xGoals_and_Form(self, team_home : str, team_away : str, league : str, season):
        _, xGoals_team_league = self.league_avg_xG(league, season)
        
        av_xG_home, av_xGA_home, home_advantage, form_home, goals_last5_home, goals_conceeded_last5_home, match_id =self.xGoals_team(team_home, self.last_n_games, season, possible_opponent=team_away, playground ='h')
        av_xG_away, av_xGA_away, _, form_away, goals_last5_away, goals_conceeded_last5_away, _ =self.xGoals_team(team_away, self.last_n_games, season, possible_opponent=team_home, playground='a')
        
        xG_home = av_xG_home*av_xGA_away/xGoals_team_league*home_advantage
        xG_away = av_xG_away*av_xGA_home/xGoals_team_league* 1.0 #set disadvantage to 1 since we already have home advantage
        
        return xG_home, xG_away, form_home, goals_last5_home, goals_conceeded_last5_home, form_away, goals_last5_away, goals_conceeded_last5_away, match_id
    
    def prob_at_least_n_goals(self, xG_home, xG_away, n):
        lamb_total = xG_home + xG_away
        p_leq_n_minus_1 = 0.0
        for k in range(0, n):
            p_leq_n_minus_1 += self.poisson(lamb_total, k)
        return 1.0 - p_leq_n_minus_1
        
    
    def win_probability(self, xG_home : str, xG_away : str):
        
        sum_prob_home = 0
        sum_prob_away = 0
        for i in range(0, 7): #usually until infinity but values already neglectably small
            sum_poisson_home = 0
            sum_poisson_away = 0
            for j in range(0,i):
                sum_poisson_home += self.poisson(xG_home, j)
                sum_poisson_away += self.poisson(xG_away, j)
            sum_prob_home += self.poisson(xG_home, i)*sum_poisson_away
            sum_prob_away += self.poisson(xG_away, i)*sum_poisson_home
        
        return sum_prob_home, sum_prob_away
    
    def double_chance_quota(self, win_prob_home, win_prob_away):
        draw_prob = 1-win_prob_home-win_prob_away
        return {'12': f'{1/(win_prob_home+win_prob_away)}', '1X': f'{1/(draw_prob+win_prob_home)}', 'X2': f'{1/(win_prob_away+draw_prob)}'}
    
    def get_predictions_for_teams(self, home_team, away_team, league, season):
        xG_home, xG_away, _, _, _, _, _, _, _ = self.get_xGoals_and_Form(home_team, away_team, league, season)
        win_prob_home, win_prob_away = self.win_probability(xG_home, xG_away)
        quota = {f'{home_team} - {away_team}': {'winner':{'1': f'{1/win_prob_home}', 'X': f'{1/(1-win_prob_home-win_prob_away)}', '2': f'{1/win_prob_away}'}}}
        quota[f'{home_team} - {away_team}']['double chance'] = self.double_chance_quota(win_prob_home, win_prob_away)
        
        for n in range(1, 6):
            prob_at_least_n_goals_val = self.prob_at_least_n_goals(xG_home, xG_away, n)
            quota_above_n_goals = {'+' : f'{1/prob_at_least_n_goals_val}', '-': f'{1/(1-prob_at_least_n_goals_val)}'}
            quota[f'{home_team} - {away_team}'][f'+/- {n-0.5}'] = quota_above_n_goals
        return quota
    
    def current_matchday_predictions(self, league: str, season : str):
        matchday = self.get_current_matchday(league, season)
        quotas = []
        for m in matchday:
            home = m['h']['title']
            away = m['a']['title']
            quota = self.get_predictions_for_teams(home, away, league, season)
            quotas.append(quota)
        return quotas
    

def main():
    league = 'Bundesliga'
    qc = QuotaCalculator()
    if league == 'Bundesliga' or league == 'Ligue_1':
        qc.league_games_per_matchday = 9
    current_quotas = qc.current_matchday_predictions(league, '2025')
    for q in current_quotas:
        print(f'{q}\n')
    
if __name__ == '__main__':
    main() 
