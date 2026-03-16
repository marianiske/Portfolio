import re
import requests
from bs4 import BeautifulSoup

LEAGUES = {
    "EPL":  "https://www.rotowire.com/soccer/lineups.php",
    "LIGA": "https://www.rotowire.com/soccer/lineups.php?league=LIGA",
    "SERI": "https://www.rotowire.com/soccer/lineups.php?league=SERI",
    "BUND": "https://www.rotowire.com/soccer/lineups.php?league=BUND",
    "FRAN": "https://www.rotowire.com/soccer/lineups.php?league=FRAN",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}
POS = {"GK","DL","DC","DR","ML","MR","MC","DMC","FWL","FWR","FW","AMC","AML","AMR"}

def extract_players(block_text: str):
    lines = [l.strip() for l in block_text.splitlines() if l.strip()]
    players = []
    for i, l in enumerate(lines[:-1]):
        if l in POS:
            name = lines[i+1]
            name = re.sub(r"\s+(QUES|OUT|SUS|GTD)$", "", name).strip()
            players.append(name)
    return players

def is_probable_team_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    bad = {
        "Alerts", "Predicted Lineup", "Injuries", "Lineups",
        "Display Preference", "Show Display Settings",
        "Sportsbook Odds", "Composite", "DraftKings", "FanDuel", "BetMGM", "PointsBet",
        "Yes", "No", "Standard", "Compact", "Simple",
    }
    if s in bad:
        return False
    # Uhrzeiten / Datumszeilen grob raus
    if re.match(r"^\d{1,2}:\d{2}\s*(AM|PM)\b", s):
        return False
    if re.match(r"^(January|February|March|April|May|June|July|August|September|October|November|December)\b", s):
        return False
    # Teamkürzel (NEW, CHE, BVB, FCB etc.) raus
    if re.match(r"^[A-Z0-9]{2,4}$", s):
        return False
    # UI-Kram
    if re.search(r"(sign up|log in|subscribe|newsletter|cookie|privacy|terms)", s, re.I):
        return False
    return True

def parse_rotowire(url: str):
    html = requests.get(url, headers=HEADERS, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    # Kandidaten: Container mit genau zwei "Predicted Lineup" (= typischerweise ein Spiel)
    candidates = []
    for node in soup.find_all(["div", "section", "article"]):
        t = node.get_text("\n", strip=True)
        if t.count("Predicted Lineup") == 2 and len(t) > 400:
            candidates.append(node)

    results = []
    seen = set()

    for node in candidates:
        txt = node.get_text("\n", strip=True)

        # 1) Teamnamen nur aus dem Teil VOR dem ersten "Predicted Lineup"
        preamble = txt.split("Predicted Lineup", 1)[0]
        pre_lines = [l.strip() for l in preamble.splitlines() if l.strip()]
        team_lines = [l for l in pre_lines if is_probable_team_line(l)]

        if len(team_lines) < 2:
            continue

        # In der Praxis sind die beiden Teamnamen sehr weit hinten im Preamble
        home_team, away_team = team_lines[-2], team_lines[-1]

        # 2) Spieler aus den beiden Predicted-Blöcken
        parts = txt.split("Predicted Lineup")
        if len(parts) < 3:
            continue
        home_block = "Predicted Lineup" + parts[1]
        away_block = "Predicted Lineup" + parts[2]

        home_xi = extract_players(home_block)
        away_xi = extract_players(away_block)

        # Plausibilität: mind. 10 Namen
        if len(home_xi) < 10 or len(away_xi) < 10:
            continue

        key = (home_team, away_team, tuple(home_xi), tuple(away_xi))
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "home_team": home_team,
            "away_team": away_team,
            "home_predicted_xi": home_xi,
            "away_predicted_xi": away_xi,
        })

    return results


if __name__ == "__main__":
    all_matches = {code: parse_rotowire(url) for code, url in LEAGUES.items()}
    print({k: len(v) for k, v in all_matches.items()})

