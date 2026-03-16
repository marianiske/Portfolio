import requests
import json

def get_league_data(league: str, season: str):
    url = f"https://understat.com/getLeagueData/{league}/{season}"

    headers = {
        # Einfach einen typischen Browser-User-Agent verwenden
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        # Referer wie im Browser – aus deinem Screenshot ableitbar
        "Referer": f"https://understat.com/league/{league}/{season}",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    text = r.text

    # Robust parsen (manchmal ist es bereits JSON, manchmal escaped)
    try:
        return r.json()['dates']
    except ValueError:
        try:
            return json.loads(text)
        except ValueError:
            text_unescaped = text.encode("utf-8").decode("unicode_escape")
            return json.loads(text_unescaped)
        
def get_team_data(team: str, season: str):
    # team so kodieren, wie es der Browser im Request macht ("Bayern Munich" -> "Bayern%20Munich")
    team_encoded = team.replace(' ', '_') #quote(team)

    url = f"https://understat.com/getTeamData/{team_encoded}/{season}"
    
    # Für den Referer wird der Slug mit Unterstrich verwendet
    team_slug = team.replace(" ", "_")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://understat.com/team/{team_slug}/{season}",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    text = r.text

    try:
        return r.json()
    except ValueError:
        try:
            return json.loads(text)
        except ValueError:
            text_unescaped = text.encode("utf-8").decode("unicode_escape")
            return json.loads(text_unescaped)
        
def get_match_data(match_id: str):

    url = f"https://understat.com/getMatchData/{match_id}"
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://understat.com/match/{match_id}",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    text = r.text

    try:
        return r.json()
    except ValueError:
        try:
            return json.loads(text)
        except ValueError:
            text_unescaped = text.encode("utf-8").decode("unicode_escape")
            return json.loads(text_unescaped)
        
def get_player_data(player_id: str):
    url = f"https://understat.com/getPlayerData/{player_id}"
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://understat.com/player/{player_id}",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    text = r.text

    try:
        return r.json()
    except ValueError:
        try:
            return json.loads(text)
        except ValueError:
            text_unescaped = text.encode("utf-8").decode("unicode_escape")
            return json.loads(text_unescaped)
    

if __name__ == "__main__":
    data = get_team_data('Freiburg', '2016')