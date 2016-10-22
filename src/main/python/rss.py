import configobj
from crate import client
from dateutil import parser
import logging
logging.basicConfig(filename="/Users/smacmullin/sports/rss.log",format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
import urllib2
from teams import ncaa_fb_teams, nfl_teams
from xml.etree import ElementTree as etree
from pytz import timezone
import re
import time
import traceback

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

    rows = []

    for f in feed:

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

        for team in sport:
            if team in atoms[1]: away_team = sport[team]
            if team in atoms[3]: home_team = sport[team]

        away_payout = re.sub('[(){}<>]', '', atoms[1].split(' ')[-1])
        if away_payout == 'even': away_payout = 100
        away_payout = float(away_payout)
        away_spread = float(atoms[1].split(' ')[-2])

        home_payout = re.sub('[(){}<>]', '', atoms[3].split(' ')[-1])
        if home_payout == 'even': home_payout = 100
        home_payout = float(home_payout)
        home_spread = float(atoms[3].split(' ')[-2])

        away_ml = atoms[2].split(":")[-1]
        if away_ml == 'even': away_ml = 100
        away_ml = float(away_ml)

        home_ml = atoms[4].split(":")[-1]
        if home_ml == 'even': home_ml = 100
        home_ml = float(home_ml)

        over_payout = re.sub('[(){}<>]', '', atoms[5].split(' ')[-1])
        if over_payout == 'even': over_payout = 100
        over_payout = float(over_payout)
        over_points = float(atoms[5].split(' ')[-2])

        under_payout = re.sub('[(){}<>]', '', atoms[6].split(' ')[-1])
        if under_payout == 'even': under_payout = 100
        under_payout = float(under_payout)
        under_points = float(atoms[6].split(' ')[-2])

        game_id = str(dt_aware.year) + str(dt_aware.month) + str(dt_aware.day) + '_' + away_team + '_' + home_team

        rows.append([game_id, str(dt_aware), unixtime, time.time(), home_team, away_team, over_points,
               under_points, over_payout, under_payout, home_spread, home_payout, away_spread,
               away_payout,home_ml, away_ml])
    return rows


def insert_lines_to_crate(rows, schema):

    config = configobj.ConfigObj("/Users/smacmullin/sports/crate.ini")
    crate_host = config["crate"]["host_url"]


    connection = client.connect(crate_host)
    print connection.client._active_servers
    cursor = connection.cursor()

    query = '''INSERT INTO %s.lines ("GameId","GameTime","UnixGameTime","QueryTime",
                    "HomeTeam", "AwayTeam","OverPoints","UnderPoints","OverPayout","UnderPayout",
                    "HomeSpread","HomePayout","AwaySpread","AwayPayout",
                    "HomeMoneyline","AwayMoneyline") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''%schema

    cursor.executemany(query,rows)


if __name__=='__main__':

    while True:
        try:
            url = 'https://www.sportsbook.ag/rss/nfl-football'
            feed = get_line_feed(url)
            rows = parse_line_feed(feed, nfl_teams)
            insert_lines_to_crate(rows, "nfl")
            logging.info("Inserted %s game lines to nfl"%len(rows))
        except:
            logging.error("Error from nfl lines: " % traceback.format_exc())

        try:
            url = 'https://www.sportsbook.ag/rss/ncaa-football'
            feed = get_line_feed(url)
            rows = parse_line_feed(feed, ncaa_fb_teams)
            insert_lines_to_crate(rows, "ncaa_fb")
            logging.info("Inserted %s game lines to ncaa fb" % len(rows))
        except:
            logging.error("Error from ncaa fb lines: " % traceback.format_exc())

        time.sleep(3600)

