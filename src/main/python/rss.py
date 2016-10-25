import configobj
from crate import client
from dateutil import parser
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

logging.basicConfig(filename="/Users/smacmullin/sports/rss.log",format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def get_line_feed(url):

    file = urllib2.urlopen(url)
    data = file.read()
    root = etree.fromstring(data)
    item = root.findall('channel/item')
    feed = []
    for entry in item:
        desc = entry.findtext('description')
        feed.append([desc])
    return feed


def parse_line_feed(feed, sport):

    games = []
    lines = []

    for f in feed:

        try:

            parsed_string = (re.sub(r'<.+?>', '', f[0]))
            atoms = parsed_string.split('\n')

            # line 0 = gametime
            # line 1 = away spread
            # line 2 = away moneyline
            # line 3 = home spread
            # line 4 = home moneyline
            # line 5 = over
            # line 6 = under

            gametime = atoms[0]
            dt = parser.parse(gametime)
            localtz = timezone('US/Eastern')
            dt_aware = localtz.localize(dt)
            unixtime = time.mktime(dt_aware.timetuple())

            teams = [team for team in sport]

            guesses = process.extract(atoms[1], teams, limit=4)
            matches = []
            for guess in guesses:
                ratio = fuzz.ratio(guess, atoms[1])
                matches.append({'team': guess[0],
                                'fuzz': guess[1],
                                'ratio': ratio})

            max_match = max(matches, key=lambda x: x['ratio'])
            away_team_longname = max_match['team']
            away_team = sport[away_team_longname]

            # print atoms[1]
            # print guesses
            # print matches
            # print away_team_longname
            # print away_team
            # raw_input()

            guesses = process.extract(atoms[3], teams, limit=4)
            matches = []
            for guess in guesses:
                ratio = fuzz.ratio(guess, atoms[3])
                matches.append({'team': guess[0],
                                'fuzz': guess[1],
                                'ratio': ratio})

            max_match = max(matches, key=lambda x: x['ratio'])
            home_team_longname  = max_match['team']
            home_team = sport[home_team_longname]

            # print atoms[3]
            # print guesses
            # print matches
            # print home_team_longname
            # print home_team
            # raw_input()


            #todo: add warning if match confidence it too low

            away_payout = re.sub('[(){}<>]', '', atoms[1].split(' ')[-1])
            if away_payout == 'even': away_payout = 100
            away_payout = float(away_payout)
            away_spread = float(atoms[1].split(' ')[-2])

            home_payout = re.sub('[(){}<>]', '', atoms[3].split(' ')[-1])
            if home_payout == 'even': home_payout = 100
            home_payout = float(home_payout)
            home_spread = float(atoms[3].split(' ')[-2])

            away_ml = atoms[2].split(":")[-1]
            if (away_ml == 'even' or away_ml == 'even.'): away_ml = 100
            away_ml = float(away_ml)

            home_ml = atoms[4].split(":")[-1]
            if (home_ml == 'even' or home_ml == 'even.'): home_ml = 100
            home_ml = float(home_ml)

            over_payout = re.sub('[(){}<>]', '', atoms[5].split(' ')[-1])
            if (over_payout == 'even' or over_payout == 'even.'): over_payout = 100
            over_payout = float(over_payout)
            over_points = float(atoms[5].split(' ')[-2])

            under_payout = re.sub('[(){}<>]', '', atoms[6].split(' ')[-1])
            if (under_payout == 'even' or under_payout == 'even.'): under_payout = 100
            under_payout = float(under_payout)
            under_points = float(atoms[6].split(' ')[-2])

            game_id = str(dt_aware.year) + str(dt_aware.month) + str(dt_aware.day) + '_' + away_team + '_' + home_team

            games.append([game_id, str(dt_aware), unixtime, home_team, away_team])
            lines.append([game_id, time.time(), over_points, under_points, over_payout, under_payout,
                          home_spread, home_payout, away_spread, away_payout, home_ml, away_ml])

        except:

            logging.error(str(traceback.format_exc()))



    return games, lines


def insert_lines_to_crate(games, lines, schema):

    config = configobj.ConfigObj("/Users/smacmullin/sports/crate.ini")
    crate_host = config["crate"]["host_url"]


    connection = client.connect(crate_host)
    print connection.client._active_servers
    cursor = connection.cursor()

    for game in games:

        try:

            query = '''SELECT "GameId" from %s.games where "GameId"='%s' '''%(schema,game[0])
            cursor.execute(query)

            if len([row[0] for row in cursor])==0:

                query = '''INSERT INTO %s.games ("GameId","GameTime","UnixGameTime",
                                "HomeTeam", "AwayTeam") VALUES (?,?,?,?,?)'''%schema

                cursor.execute(query, game)

            else:
                pass

        except:
            logging.error(str(traceback.format_exc()))

    try:

        query = '''INSERT INTO %s.lines ("GameId", "QueryTime",
                        "OverPoints","UnderPoints","OverPayout","UnderPayout",
                        "HomeSpread","HomePayout","AwaySpread","AwayPayout",
                        "HomeMoneyline","AwayMoneyline") VALUES (?,?,?,?,?,?,?,?,?,?,?,?)'''%schema

        cursor.executemany(query, lines)

    except:

        logging.error(str(traceback.format_exc()))



if __name__=='__main__':

    while True:

        url = 'https://www.sportsbook.ag/rss/nba-basketball'
        feed = get_line_feed(url)
        if len(feed) > 0:
            games, lines = parse_line_feed(feed, nba_teams)
            insert_lines_to_crate(games, lines, "nba")
            logging.info("Inserted %s game lines to nba" % len(lines))
        else:
            logging.info("No new nba games to upload")
            pass

        url = 'https://www.sportsbook.ag/rss/nfl-football'
        feed = get_line_feed(url)
        if len(feed) > 0:
            games, lines = parse_line_feed(feed, nfl_teams)
            insert_lines_to_crate(games, lines, "nfl")
            logging.info("Inserted %s game lines to nfl" % len(lines))
        else:
            logging.info("No new nfl games to upload")
            pass


        # #try:
        # url = 'https://www.sportsbook.ag/rss/ncaa-football'
        # feed = get_line_feed(url)
        # if len(feed) > 0:
        #     games, lines = parse_line_feed(feed, ncaa_fb_teams)
        #     #insert_lines_to_crate(games, lines, "ncaa_fb")
        #     logging.info("Inserted %s game lines to ncaa fb" % len(games, lines))
        # else:
        #     logging.info("No new ncaa fb games to upload")
        #     pass
        #
        # # except:
        # #     logging.error("Error from ncaa fb lines: " % traceback.format_exc())
        # #     pass

        time.sleep(3600)

