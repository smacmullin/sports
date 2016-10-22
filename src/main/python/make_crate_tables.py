import configobj
from crate import client

config = configobj.ConfigObj("../../../crate.ini")

crate_host = config["crate"]["host_url"]
base_schema = config["crate"]["base_schema"]
shards = int(config["crate"]["shards"])
replicas = int(config["crate"]["replicas"])

connection = client.connect(crate_host)
print connection.client._active_servers
cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS nfl.lines")
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS nfl.lines (
    "GameId" string,
    "GameTime" string,
    "UnixGameTime" double,
    "QueryTime" double,
    "HomeTeam" string,
    "AwayTeam" string,
    "OverPoints" float,
    "UnderPoints" float,
    "OverPayout" float,
    "UnderPayout" float,
    "HomeSpread" float,
    "HomePayout" float,
    "AwaySpread" float,
    "AwayPayout" float,
    "HomeMoneyline" float,
    "AwayMoneyline" float
)
CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
""" % (shards, replicas))

cursor.execute("DROP TABLE IF EXISTS ncaa_fb.lines")
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS ncaa_fb.lines (
    "GameId" string,
    "GameTime" string,
    "UnixGameTime" double,
    "QueryTime" double,
    "HomeTeam" string,
    "AwayTeam" string,
    "OverPoints" float,
    "UnderPoints" float,
    "OverPayout" float,
    "UnderPayout" float,
    "HomeSpread" float,
    "HomePayout" float,
    "AwaySpread" float,
    "AwayPayout" float,
    "HomeMoneyline" float,
    "AwayMoneyline" float
)
CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
""" % (shards, replicas))