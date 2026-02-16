import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler


def build_mlp_model(random_state=123):
    return MLPRegressor(
        hidden_layer_sizes=(5, 5, 5),
        activation='relu',
        solver='adam',
        learning_rate_init=0.003,
        alpha=0.01,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=100,
        random_state=random_state
    )


def load_data(train_path="CW1_train.csv", test_path="CW1_test.csv"):
    trn = pd.read_csv(train_path)
    tst = pd.read_csv(test_path)
    return trn, tst


def encode_categoricals(trn, tst, categorical_cols=None):
    if categorical_cols is None:
        categorical_cols = ["cut", "color", "clarity"]

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        trn[col] = le.fit_transform(trn[col])
        label_encoders[col] = le

    for col in categorical_cols:
        tst[col] = label_encoders[col].transform(tst[col])

    return trn, tst


def drop_low_corr_features(trn, tst, threshold=0.05, target_col="outcome"):
    correlations = trn.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    low_corr_features = correlations[correlations < threshold].index.tolist()

    X_full = trn.drop(columns=[target_col])
    y = trn[target_col]
    X = X_full.drop(columns=low_corr_features)
    tst = tst.reindex(columns=X.columns)
    return X, y, tst


def split_and_scale(X, y, tst, test_size=0.2, random_state=123):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    tst_scaled = scaler.transform(tst)

    return X_train_scaled, X_val_scaled, y_train, y_val, tst_scaled, scaler


def train_mlp(X_train_scaled, y_train, random_state=123):
    model = build_mlp_model(random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model


def predict_test(model, tst_scaled):
    return model.predict(tst_scaled)


def main():
    np.random.seed(123)

    trn, tst = load_data()
    trn, tst = encode_categoricals(trn, tst)
    X, y, tst = drop_low_corr_features(trn, tst)
    X_train_scaled, X_val_scaled, y_train, y_val, tst_scaled, _ = split_and_scale(X, y, tst)

    model = train_mlp(X_train_scaled, y_train)
    yhat_train = model.predict(X_train_scaled)
    yhat_val = model.predict(X_val_scaled)
    r2_train = r2_score(y_train, yhat_train)
    r2_val = r2_score(y_val, yhat_val)
    generalization_gap = r2_train - r2_val

    print(f"Train R2: {r2_train:.4f}")
    print(f"Val R2: {r2_val:.4f}")
    print(f"Generalization gap (Train - Val): {generalization_gap:.4f}")

    yhat_test = predict_test(model, tst_scaled)

    out = pd.DataFrame({"yhat": yhat_test})
    out.to_csv("CW1_submission_K23067889.csv", index=False)


if __name__ == "__main__":
    main()
    
    # At test time, we will use the true outcomes
    # tst = pd.read_csv('CW1_test_with_true_outcome.csv') # You do not have access to this

    # # This is the R^2 function
    # def r2_fn(yhat):
    #     eps = y_tst - yhat
    #     rss = np.sum(eps ** 2)
    #     tss = np.sum((y_tst - y_tst.mean()) ** 2)
    #     r2 = 1 - (rss / tss)
    #     return r2

    # # How does the linear model do?
    # print(r2_fn(yhat_lm))
