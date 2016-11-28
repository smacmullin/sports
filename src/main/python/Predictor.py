import configobj
import cPickle
from crate import client
import datetime as dt
import logging
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import traceback
from teams import nba_teams

logging.basicConfig(filename="/Users/smacmullin/sports/modelvalidation.log",format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def index_teams(team_keys):

    team_index = {}
    for i, key in enumerate(team_keys):
        team_index[team_keys[key]] = i

    return team_index

team_index = index_teams(nba_teams)


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

    def get_test_dataset(self, date=None, ngames=None):

        connection = self._open_db_connection()

        frames = []

        for team in team_index.keys():

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
            AND (nba.games."HomeTeam" = '%s' or nba.games."AwayTeam" = '%s')
            AND nba.games."GameDate" < %i
            ORDER BY nba.games."GameDate" DESC
            LIMIT %i
            ''' % (team, team, date, ngames)

            df = pd.read_sql(sql, connection)

            frames.append(df)

        df = pd.concat(frames)

        observed_home_score = df['HomeScore'].values
        observed_away_score = df['AwayScore'].values

        home_team = [team_index[i] for i in df['HomeTeam'].values]
        away_team = [team_index[i] for i in df['AwayTeam'].values]

        num_teams = len(team_index)

        return {'teams':team_index.keys(),
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
            baseline_home = pm.Normal('baseline_home', 0., tau=0.01)
            tau_offense = pm.Gamma('tau_offense', .1, .1)  # tau for a normal distribution is 1/sigma**2
            tau_defense = pm.Gamma('tau_defense', .1, .1)
            intercept = pm.Normal('intercept',  4.4, tau=0.1)

            # team-specific model parameters
            offense_skills = pm.Normal("offense_skills",
                                    mu=0.0,
                                    tau=tau_offense,
                                    shape=num_teams)

            defense_skills = pm.Normal("defense_skills",
                                    mu=0.0,
                                    tau=tau_defense,
                                    shape=num_teams)

            offense_skill = pm.Deterministic('offense_skill', offense_skills - tt.mean(offense_skills))
            defense_skill = pm.Deterministic('defense_skill', defense_skills - tt.mean(defense_skills))

            home_theta = np.exp(intercept + baseline_home + offense_skill[home_team] - defense_skill[away_team])
            away_theta = np.exp(intercept + offense_skill[away_team] - defense_skill[home_team])
            # likelihood of observed data
            home_points = pm.Poisson('home_points', mu=home_theta, observed=observed_home_score)
            away_points = pm.Poisson('away_points', mu=away_theta, observed=observed_away_score)

        with model:
            start = pm.find_MAP()
            step = pm.NUTS(state=start)
            trace = pm.sample(1000, step, start=start)
            #pm.traceplot(trace)
            #plt.show()

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

        baseline_home = np.asarray(trace['baseline_home'])
        offense_skills_likelihood = np.asarray(trace['offense_skill'])
        defense_skills_likelihood = np.asarray(trace['defense_skill'])
        intercept = np.asarray(trace['intercept'])

        offense_skill = {}
        defense_skill ={}

        for team in teams:
            ix = team_index[team]
            offense_skill[team] = np.asarray([val[ix] for val in offense_skills_likelihood])
            defense_skill[team] = np.asarray([val[ix] for val in defense_skills_likelihood])

        home_theta = np.exp(intercept + baseline_home + offense_skill[home_team] - defense_skill[away_team])
        away_theta = np.exp(intercept + offense_skill[away_team] - defense_skill[home_team])

        home_scores = np.random.poisson(home_theta)
        away_scores = np.random.poisson(away_theta)

        home_spreads = away_scores - home_scores
        home_wins = 0.
        for s in home_spreads:
            if s < 0:
                home_wins+=1.0
        p_home = moneyline_from_implied_odds(home_wins/len(home_spreads))


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
        print "Predicted Home Money: %s" %(p_home)


def generate_test_times():

    start = dt.date(2015, 10, 25)
    testing_delta = dt.timedelta(days=7)
    training_delta = dt.timedelta(days=7)

    testing_times = []
    training_times = []

    for i in range(20):
        t = start + training_delta
        t2 = t + testing_delta

        training_times.append((int(start.strftime('%Y') + start.strftime('%m') + start.strftime('%d')),
                               int(t.strftime('%Y') + t.strftime('%m') + t.strftime('%d'))))
        testing_times.append((int(t.strftime('%Y') + t.strftime('%m') + t.strftime('%d')),
                              int(t2.strftime('%Y') + t2.strftime('%m') + t2.strftime('%d'))))
        start = start + testing_delta

    return training_times, testing_times

def generate_test_times2():

    start = dt.date(2016, 02, 20)

    test_times = [int(start.strftime('%Y') + start.strftime('%m') + start.strftime('%d'))]

    delta = dt.timedelta(days=1)

    for d in range(175):

        t = start + delta

        test_times.append(int(t.strftime('%Y') + t.strftime('%m') + t.strftime('%d')))

        start = t

    return test_times

if __name__ == '__main__':

    predictor = Predictor()
    test_dataset = predictor.get_test_dataset(date=20161125, ngames=8)
    #trace = predictor.model(test_dataset)
    #predictor.save_model(trace, file='/Users/smacmullin/sports/test/nba_model_20161125_8.pkl')
    trace = predictor.load_model(file='/Users/smacmullin/sports/test/nba_model_20161125_8.pkl')
    predictor.predict_game(trace,test_dataset['teams'], away_team='ATL', home_team='LAL')

