import os
import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense

from lime.lime_tabular import LimeTabularExplainer

class CNNModel:
    def __init__(self,
                 data_file="stock_data.xlsx", #Assuming we store the yfinance data in an excel file
                 window_size=60,
                 feature_cols=None,
                 model_path="cnn_model.h5",
                 scaler_path="scaler.pkl"):
        """
        data_file: Excel workbook with one sheet per ticker
        window_size: number of past days per example
        feature_cols: list of columns to use as features; if None, inferred
        """
        self.data_file = data_file
        self.window = window_size
        self.model_path = model_path
        self.scaler_path = scaler_path

        # scaler
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.scaler = StandardScaler()

        # model
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            self.model = None

        self.explainer = None
        self.feature_cols = feature_cols

    def build_model(self, n_features):
        m = Sequential([
            Conv1D(32, 3, activation="relu", input_shape=(self.window, n_features)),
            Conv1D(64, 3, activation="relu"),
            Dropout(0.5),
            Flatten(),
            Dense(100, activation="relu"),
            Dense(3, activation="softmax")
        ])
        m.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
        self.model = m

    def train(self,
              symbol_list,
              epochs=10,
              batch_size=32,
              threshold=0.0):
        """
        Train on sheets in data_file; generates labels by
        comparing next-day 'close' vs today:
          > +threshold → BUY
          < -threshold → SELL
          otherwise → HOLD
        """
        LABELS = ["BUY", "SELL", "HOLD"]
        X_wins, y = [], []

        for sym in symbol_list:
            df = pd.read_excel(self.data_file, sheet_name=sym)
            # derive label column from next-day close
            df["future_ret"] = df["close"].shift(-1) - df["close"]
            df = df.iloc[:-1]  # drop last row w/o future
            df["label"] = np.where(
                df["future_ret"] > threshold, "BUY",
                np.where(df["future_ret"] < -threshold, "SELL", "HOLD")
            )

            # infer features if needed
            if self.feature_cols is None:
                self.feature_cols = [c for c in df.columns
                                     if c not in ("date", "future_ret", "label")]

            arr = df[self.feature_cols].values
            for i in range(len(arr) - self.window):
                X_wins.append(arr[i:i+self.window])
                y.append(LABELS.index(df["label"].iloc[i + self.window]))

        # stack & one-hot
        X = np.stack(X_wins)
        y_cat = to_categorical(y, num_classes=3)

        # scale
        n_samps, w, n_feat = X.shape
        X_flat = X.reshape(n_samps*w, n_feat)
        self.scaler.fit(X_flat)
        X_sc = self.scaler.transform(X_flat).reshape(n_samps, w, n_feat)
        joblib.dump(self.scaler, self.scaler_path)

        # build if needed
        if self.model is None:
            self.build_model(n_feat)

        # fit
        self.model.fit(
            X_sc, y_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1
        )
        self.model.save(self.model_path)

        # LIME explainer
        flat_train = X_sc.reshape(n_samps, w*n_feat)
        feat_names = [f"{feat}_t{t}"
                      for t in range(w)
                      for feat in self.feature_cols]
        self.explainer = LimeTabularExplainer(
            flat_train,
            feature_names=feat_names,
            class_names=LABELS,
            mode="classification"
        )

    def _prepare_input(self, symbol):
        df = pd.read_excel(self.data_file, sheet_name=symbol)
        arr = df[self.feature_cols].values[-self.window:]
        return self.scaler.transform(arr).reshape(1, self.window, len(self.feature_cols))

    def predict(self, symbol):
        X_in = self._prepare_input(symbol)
        probas = self.model.predict(X_in)[0]
        idx = np.argmax(probas)
        action = self.explainer.class_names[idx]
        flat = X_in.reshape(1, -1)
        return action, probas, flat

    def explain(self, flat_input, num_features=5):
        if self.explainer is None:
            raise RuntimeError("Explainer not ready—train first.")
        return self.explainer.explain_instance(
            flat_input.flatten().tolist(),
            classifier_fn=lambda x: self.model.predict(
                x.reshape(-1, self.window, len(self.feature_cols))),
            num_features=num_features
        )

# === Main App ===
class StockApp:
    def __init__(self, model):
        self.model = model

    def execute_action(self, action, symbol):
        print(f"Executing {action} on {symbol}")

    def log_denial(self, uid, symbol, pred, user_act):
        entry = {
            "user_id": uid,
            "symbol": symbol,
            "predicted": pred,
            "user_action": user_act,
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open("denials.log", "a") as f:
            f.write(str(entry) + "\n")

    def decision_flow(self, uid, symbol):
        action, probas, flat = self.model.predict(symbol)
        print(f"\nModel → {symbol}: {action} "
              f"(Buy: {probas[0]:.2f}, Sell: {probas[1]:.2f}, Hold: {probas[2]:.2f})")

        # show LIME
        exp = self.model.explain(flat)
        print("\nLIME feature weights:")
        for feat, w in exp.as_list():
            print(f"  {feat}: {w:.4f}")

        choice = input("\nACCEPT or DENY? ").strip().lower()
        if choice == "accept":
            self.execute_action(action, symbol)
        elif choice == "deny":
            override = input("Your action (BUY/SELL/HOLD): ").strip().upper()
            if override in ("BUY", "SELL", "HOLD"):
                self.log_denial(uid, symbol, action, override)
                self.execute_action(override, symbol)
            else:
                print("Invalid override—aborted.")
        else:
            print("Invalid input—no action.")

if __name__ == "__main__":
    model = CNNModel(
        data_file="stock_data.xlsx",
        window_size=60,
        feature_cols=None,
        model_path="cnn_model.h5",
        scaler_path="scaler.pkl"
    )

    # train check
    need = False
    if model.model is None or model.explainer is None:
        print("No trained model found.")
        need = True
    else:
        if input("Retrain? (y/n): ").strip().lower() == "y":
            need = True

    if need:
        syms = input("Tickers (comma-separated, e.g. AAPL,TSLA): ")
        lst = [s.strip().upper() for s in syms.split(",") if s.strip()]
        model.train(symbol_list=lst)
        print("Training done.")

    app = StockApp(model)
    print("\n--- Stock App Ready (type 'exit' to quit) ---")

    while True:
        uid = input("\nUser ID: ").strip()
        if uid.lower()=="exit": break
        sym = input("Ticker: ").strip().upper()
        if sym.lower()=="exit": break
        app.decision_flow(uid, sym)

