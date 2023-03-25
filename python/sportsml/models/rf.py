from sklearn.ensemble import RandomForestRegressor


def train_rf(X_train, y_train, **kwargs):
    rf = RandomForestRegressor(**kwargs)
    rf.fit(X_train, y_train)
    return rf
