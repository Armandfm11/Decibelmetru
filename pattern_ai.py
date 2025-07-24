        # Librarii

# Ziua si ora colectarii esantionului
import datetime

# Array-uri
import numpy as np

# Serializare si deserializare
import pickle

# Path model antrenat
import os

# Antrenare
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


class PatternAI:

    # Initializare
    def __init__(self, save_path="patternai_state.pkl"):
        self.history = []
        self.scaler = StandardScaler()
        self.model = SGDRegressor(max_iter=1, tol=None, penalty='l2', alpha=1e-3)
        self.initialized = False
        self.save_path = save_path
        self._load_state()

    # Adaugare valori noi
    def add_observation(self, value: float):
        now = datetime.datetime.now()
        # Se tine cont doar de ziua saptamanii, nu de ziua lunii
        week_day = now.weekday()
        hour = now.hour + now.minute/60
        self.history.append((week_day, hour, value))

        # Antrenare model
        # Daca nu sunt suficiente date (ex.: inceputul antrenarii)
        if not self.initialized and len(self.history) >= 100:
            X = np.array([[d, h] for d, h, _ in self.history])
            y = np.array([v for _, _, v in self.history])
            Xs = self.scaler.fit_transform(X)
            self.model.partial_fit(Xs, y)
            self.initialized = True
        # Odată ce modelul a fost initializat (invatare online)    
        elif self.initialized:
            X_new = np.array([[week_day, hour]])
            Xs_new = self.scaler.transform(X_new)
            self.model.partial_fit(Xs_new, [value])
        # Se salveaza progresul
        self._save_state()

    # Predictie valoare zgomot in GUI, folosind modelul antrenat
    def predict_current_pattern(self):
        if not self.initialized:
            return None
        now = datetime.datetime.now()
        week_day = now.weekday()
        hour = now.hour + now.minute/60
        Xs = self.scaler.transform([[week_day, hour]])
        return float(self.model.predict(Xs)[0])

    # Salvare model curent
    def _save_state(self):
        with open(self.save_path, "wb") as f:
            pickle.dump({
                "history": self.history,
                "scaler": self.scaler,
                "model": self.model,
                "initialized": self.initialized
            }, f)
        

    # Incarcare model antrenat anterior
    def _load_state(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "rb") as f:
                    state = pickle.load(f)
                    self.history = state.get("history", [])
                    self.scaler = state.get("scaler", StandardScaler())
                    self.model = state.get("model", SGDRegressor(max_iter=1, tol=None, penalty='l2', alpha=1e-3))
                    self.initialized = state.get("initialized", False)
                print(f"[PatternAI] State loaded from {self.save_path}")
            # Daca modelul nu poate fi incarcat/gasit, se incepe de la zero
            except Exception:
                self.history = []
                self.scaler = StandardScaler()
                self.model = SGDRegressor(max_iter=1, tol=None, penalty='l2', alpha=1e-3)
                self.initialized = False
                print(f"[PatternAI] Failed to load state from {self.save_path}, starting fresh.")
