def moneyline_from_implied_odds(p):
    if p < 0.5:
        return int(-1. * (100. * (p - 1.0)) / p)
    else:
        return int((100. * p) / (p - 1.0))

def implied_odds_from_moneyline(ml):

    if ml < 0.:
        return ( -1.0* ml)  / (( -1.0*ml )  + 100.)
    else:
        return 100. / ( ml + 100. )

def profit_from_odds(stake, ml):
    
    if ml > 0:
        return stake * (ml/100.)
    else:
        return stake / (-1.*ml/100.)
