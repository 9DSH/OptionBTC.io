import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.utils import resample
import warnings



def option_probabilities_with_greeks(df, current_price, instrument_filter=None):
    df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])
    df['Option Type'] = LabelEncoder().fit_transform(df['Option Type'])
    df['Time to Expiration'] = (df['Expiration Date'] - pd.to_datetime('today')).dt.days

    delta_threshold = 0.5
    df['Exercise'] = (df['Delta'] > delta_threshold).astype(int)

    features = ['Option Type', 'Strike Price', 'Bid IV', 'Ask IV', 'Gamma', 'Theta', 'Vega', 'Time to Expiration']
    target = 'Exercise'
    
    X = df[features]
    y = df[target]
    
    df_majority = df[df.Exercise == 0]
    df_minority = df[df.Exercise == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    X_balanced = df_balanced[features]
    y_balanced = df_balanced[target]
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Convert to DataFrame to maintain feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
    X_valid_scaled = pd.DataFrame(X_valid_scaled, columns=features)

    xgb_search = BayesSearchCV(
        xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, verbosity=0),
        {'n_estimators': (50, 200), 'learning_rate': (0.01, 0.1), 'max_depth': (3, 10)},
        n_iter=20,
        cv=3,
        scoring='accuracy'
    )
    xgb_search.fit(X_train_scaled, y_train)
    best_xgb_model = xgb_search.best_estimator_
    
    lgb_model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', verbose=-1)
    lgb_search = BayesSearchCV(
        lgb_model,
        {'num_leaves': (31, 100), 'learning_rate': (0.01, 0.1), 'n_estimators': (50, 200)},
        n_iter=20,
        cv=3,
        scoring='accuracy'
    )
    
    lgb_search.fit(X_train_scaled, y_train, feature_name=features)
    best_lgb_model = lgb_search.best_estimator_
    
    voting_model = VotingClassifier(estimators=[('xgb', best_xgb_model), ('lgbm', best_lgb_model)], voting='soft')
    voting_model.fit(X_train_scaled, y_train)
    
    y_proba = voting_model.predict_proba(X_valid_scaled)[:, 1]
    
    results_df = df.iloc[y_valid.index].copy()
    results_df['Exercise_Probability (%)'] = y_proba * 100

    # Modify the Exercise Probability
    results_df['Exercise_Probability (%)'] = results_df['Exercise_Probability (%)'].apply(lambda x: int(round((x - int(x)) * 1000))/10)

    lower_bound = current_price - 50000
    upper_bound = current_price + 50000
    filtered_results = results_df[(results_df['Strike Price'] >= lower_bound) & 
                                  (results_df['Strike Price'] <= upper_bound)]
    
    if instrument_filter is not None:
        filtered_results = filtered_results[filtered_results['Instrument'] == instrument_filter]

    sorted_results = filtered_results.sort_values(by='Exercise_Probability (%)', ascending=False)
    
    return sorted_results[['Instrument', 'Exercise_Probability (%)']]

# Example usage
df = pd.read_csv("options_data.csv")
current_price = 107000  # Example current price
result = option_probabilities_with_greeks(df, current_price, instrument_filter='BTC-23MAY25-107000-C')
print(result)