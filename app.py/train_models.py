import os
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# إنشاء مجلد models إذا لم يكن موجوداً
os.makedirs("models", exist_ok=True)

TRAINING_FILE = "data/Training.csv"

df = pd.read_csv(TRAINING_FILE)

df.replace({
    'prognosis':{
        'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,
        'Drug Reaction':4,'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,
        'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
        'Migraine':11,'Cervical spondylosis':12,
        'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,
        'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
        'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,
        'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
        'Common Cold':26,'Pneumonia':27,
        'Dimorphic hemmorhoids(piles)':28,
        'Heart attack':29,'Varicose veins':30,
        'Hypothyroidism':31,'Hyperthyroidism':32,
        'Hypoglycemia':33,'Osteoarthristis':34,
        'Arthritis':35,
        '(vertigo) Paroymsal  Positional Vertigo':36,
        'Acne':37,'Urinary tract infection':38,
        'Psoriasis':39,'Impetigo':40
    }
}, inplace=True)

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

print("Training Decision Tree...")
dt = DecisionTreeClassifier()
dt.fit(X, y)
joblib.dump(dt, "models/decision_tree.pkl")

print("Training Random Forest...")
rf = RandomForestClassifier()
rf.fit(X, y)
joblib.dump(rf, "models/random_forest.pkl")

print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X, y)
joblib.dump(nb, "models/naive_bayes.pkl")

print("\nAll models saved successfully.")