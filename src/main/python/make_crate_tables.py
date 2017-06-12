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

# cursor.execute("DROP TABLE IF EXISTS nfl.lines")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS nfl.lines (
#     "GameId" string,
#     "QueryTime" double,
#     "OverPoints" float,
#     "UnderPoints" float,
#     "OverPayout" float,
#     "UnderPayout" float,
#     "HomeSpread" float,
#     "HomePayout" float,
#     "AwaySpread" float,
#     "AwayPayout" float,
#     "HomeMoneyline" float,
#     "AwayMoneyline" float
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))
#
# cursor.execute("DROP TABLE IF EXISTS nfl.games")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS nfl.games (
#     "GameId" string,
#     "GameTime" string,
#     "GameDate" string,
#     "UnixGameTime" double,
#     "HomeTeam" string,
#     "AwayTeam" string
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))
#
# cursor.execute("DROP TABLE IF EXISTS nfl.results")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS nfl.results (
#     "GameId" string,
#     "HomeScore" float,
#     "AwayScore" float
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))


# cursor.execute("DROP TABLE IF EXISTS nba.lines")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS nba.lines (
#     "GameId" string,
#     "QueryTime" double,
#     "OverUnder" float,
#     "HomeSpread" float
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))
#
# cursor.execute("DROP TABLE IF EXISTS nba.games")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS nba.games (
#     "GameId" string,
#     "GameDate" int,
#     "HomeTeam" string,
#     "AwayTeam" string
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))
#
# cursor.execute("DROP TABLE IF EXISTS nba.results")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS nba.results (
#     "GameId" string,
#     "HomeScore" float,
#     "AwayScore" float
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))

# cursor.execute("DROP TABLE IF EXISTS nba.predictions")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS nba.predictions (
#     "GameId" string,
#     "Spread" float,
#     "SpreadStdev" float,
#     "OU" float,
#     "OUStdev" float,
#     "N" integer
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))


# cursor.execute("DROP TABLE IF EXISTS mlb.lines")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS mlb.lines (
#     "GameId" string,
#     "OverUnder" float,
#     "HomeLine" float
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))

# cursor.execute("DROP TABLE IF EXISTS mlb.games")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS mlb.games (
#     "GameId" string,
#     "GameDate" int,
#     "HomeTeam" string,
#     "AwayTeam" string,
#     "HomeStarter" string,
#     "AwayStarter" string
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))

# cursor.execute("DROP TABLE IF EXISTS mlb.results")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS mlb.results (
#     "GameId" string,
#     "HomeScore" float,
#     "AwayScore" float
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))

# cursor.execute("DROP TABLE IF EXISTS mlb.pitchers")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS mlb.pitchers (
#     "Season" int,
#     "Name" string,
#     "UniqueName" string,
#     "Team" string,
#     "WLPct" float,
#     "ERA" float,
#     "Wins" int,
#     "Losses" int,
#     "Games" int,
#     "GamesStarted" int,
#     "GamesFinished" int,
#     "InningsPitched" float
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))

# cursor.execute("DROP TABLE IF EXISTS mlb.pitching")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS mlb.pitching (
#     "Season" int,
#     "Team" string,
#     "RAPG" float,
#     "ERA" float
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))

# cursor.execute("DROP TABLE IF EXISTS mlb.batting")
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS mlb.batting (
#     "Season" int,
#     "Team" string,
#     "RPG" float,
#     "OBP" float
# )
# CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
# """ % (shards, replicas))


cursor.execute("DROP TABLE IF EXISTS mlb.model")
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS mlb.model (
    "GameId" string,
    "HomeRPG" float,
    "AwayRPG" float,
    "HomeOBP" float,
    "AwayOBP" float,
    "HomeERA" float,
    "AwayERA" float,
    "HomeStarterERA" float,
    "AwayStarterERA" float,
    "Result" int
)
CLUSTERED INTO %d SHARDS WITH (number_of_replicas=%d)
""" % (shards, replicas))