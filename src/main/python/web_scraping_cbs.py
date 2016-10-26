from bs4 import BeautifulSoup
import configobj
from crate import client
from dateutil import parser
from datetime import date, timedelta
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import logging
import urllib2
from teams import ncaa_fb_teams, nfl_teams, nba_teams
from xml.etree import ElementTree as etree
from pytz import timezone
import re
import time
import traceback

logging.basicConfig(filename="/Users/smacmullin/sports/webscores.log",format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def process_page(url,gamedate):


    try:
        page = urllib2.urlopen(url,gamedate)
        soup = BeautifulSoup(page)
    except:
        logging.error("Could not open page. %s"%traceback.format_exc())


    teams = [team for team in nba_teams]

    try:
        all_tables = soup.find_all('table')

        results = []

        for table in all_tables:

            game_result = []

            for row in table.findAll('tr'):

                team_cells = row.findAll('td', {'class': 'teamName'})
                score_cells = row.findAll('td', {'class': 'finalScore'})

                for cell in team_cells:

                    guesses = process.extract(cell.text, teams, limit=4)
                    matches = []
                    for guess in guesses:
                        ratio = fuzz.ratio(guess, cell.text)
                        matches.append({'team': guess[0],
                                        'fuzz': guess[1],
                                        'ratio': ratio})

                    max_match = max(matches, key=lambda x: x['ratio'])
                    team_longname = max_match['team']
                    team = nba_teams[team_longname]

                    # print max_match
                    # print team_longname
                    # print team
                    # raw_input()

                for cell in score_cells:
                    score = float(cell.text.split('\n')[0])

                    game_result.append({"Team": team, "Score": score})

            results.append(game_result)

        results = [game for game in results if len(game) > 0]
        rows = []
        for game in results:

            home_team = game[1]['Team']
            away_team = game[0]['Team']
            home_score = game[1]['Score']
            away_score = game[0]['Score']
            game_id = gamedate + '_' + away_team + '_' + home_team
            rows.append([game_id, away_score, home_score])

        return rows

    except:
        logging.error(traceback.format_exc())
        return []

def insert_results_to_crate(games, schema):

    config = configobj.ConfigObj("/Users/smacmullin/sports/crate.ini")
    crate_host = config["crate"]["host_url"]
    connection = client.connect(crate_host)
    print connection.client._active_servers
    cursor = connection.cursor()

    for game in games:
        try:
            query = '''SELECT "GameId" from %s.results where "GameId"='%s' '''%(schema,game[0])
            cursor.execute(query)
            if len([row[0] for row in cursor])==0:
                query = '''INSERT INTO %s.results ("GameId","AwayScore","HomeScore") VALUES (?,?,?)'''%schema
                cursor.execute(query, game)
            else:
                pass
        except:
            logging.error(str(traceback.format_exc()))

if __name__=='__main__':

    yesterday = date.today() - timedelta(1)
    gamedate = yesterday.strftime('%Y%m%d')
    url = 'http://www.cbssports.com/nba/scoreboard/%s'%gamedate
    games = process_page(url, gamedate)
    insert_results_to_crate(games, 'nba')