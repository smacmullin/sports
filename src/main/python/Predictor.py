import configobj
import cPickle
from crate import client
import logging
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import traceback

logging.basicConfig(filename="/Users/smacmullin/sports/modelvalidation.log",format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

from teams import nba_teams
team_keys = nba_teams


def moneyline_from_implied_odds(p):
    if p < 0.5:
        return int(-1. * (100. * (p - 1.0)) / p)
    else:
        return int((100. * p) / (p - 1.0))


class Predictor(object):

    def __init__(self, *args, **kwargs):

        if args:
            raise ValueError("Only keyword arguments are allowed in constructor")
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                raise ValueError("%s has no attribute %s" % (type(self).__name__, key))

    def _open_db_connection(self, inifile="/Users/smacmullin/sports/crate.ini"):

        config = configobj.ConfigObj(inifile)
        crate_host = config["crate"]["host_url"]

        connection = client.connect(crate_host)
        print connection.client._active_servers
        return connection

    def get_test_dataset(self, startdate=None, enddate=None):

        connection = self._open_db_connection()

        sql = '''
        SELECT
        nba.games."GameId" as "GameId",
        nba.games."GameDate" as "GameDate",
        nba.games."HomeTeam" as "HomeTeam",
        nba.games."AwayTeam" as "AwayTeam",
        nba.results."AwayScore" as "AwayScore",
        nba.results."HomeScore" as "HomeScore"
        FROM nba.games, nba.results
        WHERE nba.games."GameId" = nba.results."GameId"
        AND nba.games."GameDate" > %s AND nba.games."GameDate" < %s
        ORDER BY nba.games."GameDate"
        LIMIT 1000000
        '''%(startdate,enddate)

        df = pd.read_sql(sql, connection)

        teams = df.HomeTeam.unique()
        teams = pd.DataFrame(teams, columns=['Teams'])
        teams['i'] = teams.index

        df = pd.merge(df, teams, left_on='HomeTeam', right_on='Teams', how='left')
        df = df.rename(columns={'i': 'i_home'})
        df = pd.merge(df, teams, left_on='AwayTeam', right_on='Teams', how='left')
        df = df.rename(columns={'i': 'i_away'})

        #print df.head()

        observed_home_score = df['HomeScore'].values
        observed_away_score = df['AwayScore'].values

        home_team = df['i_home'].values
        away_team = df['i_away'].values

        num_teams = len(df.i_home.drop_duplicates())

        return {'teams':teams,
                'num_teams':num_teams,
                'home_team':home_team,
                'away_team':away_team,
                'observed_home_score':observed_home_score,
                'observed_away_score':observed_away_score}

    def model(self, test_dataset):

        num_teams = test_dataset['num_teams']
        home_team = test_dataset['home_team']
        away_team = test_dataset['away_team']
        observed_home_score = test_dataset['observed_home_score']
        observed_away_score = test_dataset['observed_away_score']

        with pm.Model() as model:
            # global model parameters
            baseline_home = pm.Normal('baseline_home', 0., 0.0005)
            baseline_away = pm.Normal('baseline_away', 0., 0.0005)
            tau = pm.Gamma('tau', 1., 2.)  # tau for a normal distribution is 1/sigma**2

            # team-specific model parameters
            team_skills = pm.Normal("team_skills",
                                    mu=0.0,
                                    tau=tau,
                                    shape=num_teams)

            team_skill = pm.Deterministic('team_skill', team_skills - tt.mean(team_skills))

            home_theta = np.exp(baseline_home + team_skill[home_team])# - team_skill[away_team])
            away_theta = np.exp(baseline_away + team_skill[away_team])# - team_skill[home_team])

            # likelihood of observed data
            home_points = pm.Poisson('home_points', mu=home_theta, observed=observed_home_score)
            away_points = pm.Poisson('away_points', mu=away_theta, observed=observed_away_score)

        with model:
            start = pm.find_MAP()
            step = pm.NUTS(state=start)
            trace = pm.sample(5000, step, start=start)

        return trace

    def save_model(self, trace, file='/Users/smacmullin/Desktop/modeltrace.pkl'):

        with open(file,"wb") as fp:
            cPickle.dump(trace ,fp,cPickle.HIGHEST_PROTOCOL)

    def load_model(self, file='/Users/smacmullin/Desktop/modeltrace.pkl'):

        model = pm.Model()
        with open(file,"rb") as fp:
            with model:
                trace = cPickle.load(fp)

        return trace

    def predict_game(self, trace, teams, away_team = None, home_team = None):

        baseline_home = trace['baseline_home']
        baseline_away = trace['baseline_away']
        team_skills_likelihood = trace['team_skill']
        team_skill = {}
        for val in teams.values:
            team_skill[val[0]] = [j[val[1]] for j in team_skills_likelihood]

        # theta
        home_theta = np.exp(baseline_home + team_skill[home_team] - team_skill[away_team])
        away_theta = np.exp(baseline_away + team_skill[away_team] - team_skill[home_team])

        home_scores = np.random.poisson(home_theta)
        away_scores = np.random.poisson(away_theta)

        predicted_home_score = np.average(home_scores)
        stdev_home_score = np.std(home_scores)
        predicted_away_score = np.average(away_scores)
        stdev_away_score = np.std(away_scores)

        # predict the score
        print "Predicted Away Score (%s): %s +/- %s" % (away_team, predicted_away_score, stdev_away_score)
        print "Predicted Home Score (%s): %s +/- %s" % (home_team, predicted_home_score, stdev_home_score)

        # predict the spread
        predicted_spread = np.average([aws - hs for hs, aws in zip(home_scores, away_scores)])
        std_spread = np.std([aws - hs for hs, aws in zip(home_scores, away_scores)])
        print "Predicted Home Spread: %s +/- %s" % (predicted_spread, std_spread)
        # predict the o/u
        predicted_ou = np.average([aws + hs for hs, aws in zip(home_scores, away_scores)])
        std_ou = np.std([aws + hs for hs, aws in zip(home_scores, away_scores)])
        print "Predicted Over Under: %s +/- %s" % (predicted_ou, std_ou)

    def validate_model(self, trace, teams, startdate=None, enddate=None):

        connection = self._open_db_connection()

        baseline_home = trace['baseline_home']
        baseline_away = trace['baseline_away']
        team_skills_likelihood = trace['team_skill']
        team_skill = {}
        for val in teams.values:
            team_skill[val[0]] = [j[val[1]] for j in team_skills_likelihood]

        # query for a test set of games
        sql = '''
        SELECT
        nba.games."GameId" as "GameId",
        nba.games."GameDate" as "GameDate",
        nba.games."HomeTeam" as "HomeTeam",
        nba.games."AwayTeam" as "AwayTeam",
        nba.results."AwayScore" as "AwayScore",
        nba.results."HomeScore" as "HomeScore",
        nba.lines."HomeSpread" as "HomeSpread",
        nba.lines."OverUnder" as "OverUnder"
        FROM nba.games, nba.lines, nba.results
        WHERE nba.games."GameId" = nba.results."GameId"
        AND nba.games."GameId" = nba.lines."GameId"
        AND nba.games."GameDate" > %s AND nba.games."GameDate" < %s
        ORDER BY nba.games."GameDate"
        LIMIT 1000000
        '''%(startdate, enddate)

        df = pd.read_sql(sql, connection)
        #print df.head()
        test_records = df.to_dict('records')

        ou_counter = 0.0
        spread_counter = 0.0

        results = []

        for game in test_records:

            away_team = game["AwayTeam"]
            home_team = game["HomeTeam"]

            # theta
            home_theta = np.exp(baseline_home + team_skill[home_team])# - team_skill[away_team])
            away_theta = np.exp(baseline_away + team_skill[away_team])# - team_skill[home_team])

            home_scores = np.random.poisson(home_theta)
            away_scores = np.random.poisson(away_theta)

            predicted_home_score = np.median(home_scores)
            predicted_away_score = np.median(away_scores)

            predicted_spread = np.average([aws - hs for hs, aws in zip(home_scores, away_scores)])
            predicted_ou = predicted_away_score + predicted_home_score

            # over/under validation

            if predicted_ou > game['OverUnder']:
                ou_bet = 1
            else:
                ou_bet = 0

            if (game['HomeScore'] + game['AwayScore']) > game['OverUnder']:
                ou_outcome = 1
            else:
                ou_outcome = 0

            if ou_outcome == ou_bet:
                ou_counter += 1.0
                ou_bet_outcome = "Win"
            else:
                ou_bet_outcome = "Lose"

            # spread bet validation

            if predicted_spread < game["HomeSpread"]:
                home_bet = 1
            else:
                home_bet = 0

            if (game['AwayScore'] - game['HomeScore']) < game["HomeSpread"]:
                home_outcome = 1
            else:
                home_outcome = 0

            if home_bet == home_outcome:
                spread_counter += 1.0
                spread_bet_outcome = "Win"
            else:
                spread_bet_outcome = "Lose"

            result = {"HomeTeam": home_team,
                      "AwayTeam": away_team,
                      "PredictedHomeScore": int(predicted_home_score),
                      "PredictedAwayScore": int(predicted_away_score),
                      "ActualHomeScore": game["HomeScore"],
                      "ActualAwayScore": game["AwayScore"],
                      "PredictedOverUnder": predicted_ou,
                      "OfferedOverUnder": game["OverUnder"],
                      "OverUnderOutcome": ou_bet_outcome,
                      "ActualTotalScore": game["HomeScore"] + game["AwayScore"],
                      "PredictedSpread": predicted_spread,
                      "OfferedSpread": game["HomeSpread"],
                      "SpreadBetOutcome": spread_bet_outcome}

            results.append(result)

        results_df = pd.DataFrame(results)

        print "O/U win percentage: %s" %(ou_counter / len(test_records))
        print "Spread Bet Win Percentage: %s"%(spread_counter / len(test_records))

        print results_df
        return results_df

if __name__=='__main__':

    predictor = Predictor()
    test_dataset = predictor.get_test_dataset(startdate=20151026, enddate=20160110)

    trace = predictor.model(test_dataset)
    predictor.save_model(trace,file='/Users/smacmullin/sports/test/modeltest.pkl')

    #trace = predictor.load_model(file='/Users/smacmullin/sports/test/modeltest.pkl')

    #predictor.predict_game(trace, test_dataset['teams'], away_team='LAL', home_team='NYK')
    results = predictor.validate_model(trace, test_dataset['teams'], startdate=20160111, enddate=20160120)

