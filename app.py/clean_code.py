# ==========================================
# Disease Prediction System
# Developed by Ahmed Al Salman
# Version: 2.0 Professional Edition
# ==========================================

import warnings
from pathlib import Path
from datetime import datetime
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import joblib
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph,Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from tkinter import *
from tkinter import messagebox
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import Image

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)

TRAINING_FILE = DATA_DIR / "Training.csv"
TESTING_FILE = DATA_DIR / "Testing.csv"

RESULT_FILE = RESULT_DIR / "Prediction_History.csv"

# ==========================================
# Symptoms
# ==========================================

SYMPTOMS = []
# ==========================================
# Diseases
# ==========================================
DISEASES = []
    
l1 = SYMPTOMS
disease = DISEASES

# ==========================================
# Disease Encoding
# ==========================================

DISEASE_MAPPING = {
    'Fungal infection': 0,
    'Allergy': 1,
    'GERD': 2,
    'Chronic cholestasis': 3,
    'Drug Reaction': 4,
    'Peptic ulcer diseae': 5,
    'AIDS': 6,
    'Diabetes': 7,
    'Gastroenteritis': 8,
    'Bronchial Asthma': 9,
    'Hypertension': 10,
    'Migraine': 11,
    'Cervical spondylosis': 12,
    'Paralysis (brain hemorrhage)': 13,
    'Jaundice': 14,
    'Malaria': 15,
    'Chicken pox': 16,
    'Dengue': 17,
    'Typhoid': 18,
    'hepatitis A': 19,
    'Hepatitis B': 20,
    'Hepatitis C': 21,
    'Hepatitis D': 22,
    'Hepatitis E': 23,
    'Alcoholic hepatitis': 24,
    'Tuberculosis': 25,
    'Common Cold': 26,
    'Pneumonia': 27,
    'Dimorphic hemmorhoids(piles)': 28,
    'Heart attack': 29,
    'Varicose veins': 30,
    'Hypothyroidism': 31,
    'Hyperthyroidism': 32,
    'Hypoglycemia': 33,
    'Osteoarthristis': 34,
    'Arthritis': 35,
    '(vertigo) Paroymsal  Positional Vertigo': 36,
    'Acne': 37,
    'Urinary tract infection': 38,
    'Psoriasis': 39,
    'Impetigo': 40
}


# ==========================================
# Read Dataset
# ==========================================

df = pd.read_csv(TRAINING_FILE)
tr = pd.read_csv(TESTING_FILE)


# استخراج الأعراض تلقائياً
SYMPTOMS = df.columns.drop("prognosis").tolist()

l1 = SYMPTOMS


# استخراج الأمراض تلقائياً
DISEASES = df["prognosis"].unique().tolist()

disease = DISEASES


print("Symptoms count:", len(l1))
print("First symptoms:", l1[:10])

print("Diseases count:", len(disease))


df.replace(
    {"prognosis": DISEASE_MAPPING},
    inplace=True
)

tr.replace(
    {"prognosis": DISEASE_MAPPING},
    inplace=True
)


X = df[l1]

y = np.ravel(
    df["prognosis"]
)


X_test = tr[l1]

y_test = np.ravel(
    tr["prognosis"]
)


dt_model = joblib.load("models/decision_tree.pkl")

rf_model = joblib.load("models/random_forest.pkl")

nb_model = joblib.load("models/naive_bayes.pkl")
# ------------------------------------------------------------------------------------------------------

def get_selected_symptoms():
    return [
        Symptom1.get(),
        Symptom2.get(),
        Symptom3.get(),
        Symptom4.get(),
        Symptom5.get()
    ]

def get_selected_symptoms_text():

    symptoms = []

    for symptom in get_selected_symptoms():

        if symptom != "Select Symptom":

            symptoms.append(symptom.replace("_", " ").title())

    return symptoms

def create_input_vector(symptoms):

    vector = [0] * len(l1)

    for symptom in symptoms:
        if symptom in l1:
            vector[l1.index(symptom)] = 1

    return [vector]


def save_prediction(model_name, prediction, confidence):

    record = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Patient": Name.get(),
        "Model": model_name,
        "Prediction": prediction,
        "Confidence": confidence
    }

    df_result = pd.DataFrame([record])

    if RESULT_FILE.exists():
        df_result.to_csv(
            RESULT_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        df_result.to_csv(
            RESULT_FILE,
            index=False
        )



def get_confidence(model, inputtest):

    try:
        probabilities = model.predict_proba(inputtest)

        confidence = max(probabilities[0]) * 100

        top_indexes = np.argsort(probabilities[0])[-3:][::-1]

        top_predictions = []

        for index in top_indexes:
            disease_name = disease[index]
            score = probabilities[0][index] * 100

            top_predictions.append(
                f"{disease_name} : {score:.2f}%"
            )

        return confidence, top_predictions

    except:

        return 0, []

def save_prediction(model_name, prediction, confidence):

    record = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Patient": Name.get(),
        "Model": model_name,
        "Prediction": prediction,
        "Confidence": confidence
    }

    df_result = pd.DataFrame([record])

    if RESULT_FILE.exists():

        df_result.to_csv(
            RESULT_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        df_result.to_csv(
            RESULT_FILE,
            index=False
        )
def export_pdf():

    filename = RESULT_DIR / (
        f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    doc = SimpleDocTemplate(str(filename))

    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    title_style.alignment = TA_CENTER

    elements = []

    # ==========================
    # Logo
    # ==========================

    logo_path = BASE_DIR / "assets" / "medical_logo.png"

    if logo_path.exists():

        logo = Image(
            str(logo_path),
            width=2.5 * cm,
            height=2.5 * cm
        )

        elements.append(logo)

        elements.append(Spacer(1, 15))

    # ==========================
    # Title
    # ==========================

    elements.append(
        Paragraph(
            "<font color='#0B5ED7'><b>Disease Prediction System</b></font>",
            title_style
        )
    )

    elements.append(
        Paragraph(
            "<b>Machine Learning Medical Report</b>",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            "Developed by Ahmed Al Salman",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))

    # ==========================
    # Patient Info
    # ==========================

    elements.append(
        Paragraph(
            f"<b>Patient Name:</b> {Name.get()}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))

    # ==========================
    # Symptoms
    # ==========================

    elements.append(
        Paragraph(
            "<b>Selected Symptoms</b>",
            styles["Heading3"]
        )
    )

    for symptom in get_selected_symptoms():

        if symptom != "Select Symptom":

            elements.append(
                Paragraph(
                    "• " + symptom.replace("_", " ").title(),
                    styles["Normal"]
                )
            )

    elements.append(Spacer(1, 20))

    # ==========================
    # Prediction Table
    # ==========================

    data = [

        ["Model", "Prediction"],

        ["Decision Tree",
         t1.get("1.0", END).strip()],

        ["Random Forest",
         t2.get("1.0", END).strip()],

        ["Naive Bayes",
         t3.get("1.0", END).strip()]

    ]

    table = Table(data, colWidths=[6 * cm, 9 * cm])

    table.setStyle(TableStyle([

        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2C3E50")),

        ("TEXTCOLOR", (0,0), (-1,0), colors.white),

        ("GRID", (0,0), (-1,-1), 1, colors.black),

        ("BACKGROUND", (0,1), (-1,-1), colors.beige),

        ("ALIGN", (0,0), (-1,-1), "CENTER"),

        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),

        ("BOTTOMPADDING", (0,0), (-1,0), 10),

    ]))

    elements.append(table)

    elements.append(Spacer(1, 20))

    # ==========================
    # Confidence
    # ==========================

    elements.append(
        Paragraph(
            "<b>Confidence</b>",
            styles["Heading3"]
        )
    )

    elements.append(
        Paragraph(
            confidence_text.get("1.0", END).strip(),
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))

    # ==========================
    # Top 3 Predictions
    # ==========================

    elements.append(
        Paragraph(
            "<b>Top 3 Predictions</b>",
            styles["Heading3"]
        )
    )

    for line in top3_text.get("1.0", END).splitlines():

        if line.strip():

            elements.append(
                Paragraph(
                    line,
                    styles["Normal"]
                )
            )

    elements.append(Spacer(1, 25))

    # ==========================
    # Footer
    # ==========================

    elements.append(
        Paragraph(
            "<i>Generated automatically by Disease Prediction System v3.0</i>",
            styles["Italic"]
        )
    )

    doc.build(elements)

    messagebox.showinfo(
        "Success",
        f"PDF Report Saved Successfully!\n\n{filename}"
    )
          
def get_confidence(model, inputtest):

    try:
        probabilities = model.predict_proba(inputtest)

        confidence = max(probabilities[0]) * 100

        top_indexes = np.argsort(probabilities[0])[-3:][::-1]

        top_predictions = []

        for index in top_indexes:

            top_predictions.append(
                f"{disease[index]} : {probabilities[0][index]*100:.2f}%"
            )

        return confidence, top_predictions

    except Exception:

        return 0, []
    
def DecisionTree():

    clf3 = dt_model

    psymptoms = get_selected_symptoms()

    inputtest = create_input_vector(psymptoms)


    prediction = clf3.predict(inputtest)

    result = disease[prediction[0]]


    confidence, top3 = get_confidence(
        clf3,
        inputtest
    )


    t1.delete("1.0", END)

    t1.insert(
        END,
        result
    )


    confidence_text.delete(
        "1.0",
        END
    )

    confidence_text.insert(
        END,
        f"{confidence:.2f}%"
    )


    top3_text.delete(
        "1.0",
        END
    )

    for item in top3:
        top3_text.insert(
            END,
            item+"\n"
        )


    save_prediction(
        "Decision Tree",
        result,
        confidence
    )




def randomforest():

    clf4 = rf_model

    psymptoms = get_selected_symptoms()

    inputtest = create_input_vector(psymptoms)


    prediction = clf4.predict(inputtest)

    result = disease[prediction[0]]


    confidence, top3 = get_confidence(
        clf4,
        inputtest
    )


    t2.delete("1.0", END)

    t2.insert(
        END,
        result
    )


    confidence_text.delete(
        "1.0",
        END
    )

    confidence_text.insert(
        END,
        f"{confidence:.2f}%"
    )


    top3_text.delete(
        "1.0",
        END
    )


    for item in top3:

        top3_text.insert(
            END,
            item+"\n"
        )


    save_prediction(
        "Random Forest",
        result,
        confidence
    )





def NaiveBayes():

    gnb = nb_model

    psymptoms = get_selected_symptoms()

    inputtest = create_input_vector(psymptoms)


    prediction = gnb.predict(inputtest)

    result = disease[prediction[0]]


    confidence, top3 = get_confidence(
        gnb,
        inputtest
    )


    t3.delete("1.0", END)

    t3.insert(
        END,
        result
    )


    confidence_text.delete(
        "1.0",
        END
    )

    confidence_text.insert(
        END,
        f"{confidence:.2f}%"
    )


    top3_text.delete(
        "1.0",
        END
    )


    for item in top3:

        top3_text.insert(
            END,
            item+"\n"
        )

    save_prediction(
      "Naive Bayes",
    result,
    confidence

    )

# gui_stuff------------------------------------------------------------------------------------

# ==========================================================
# Main Window
# ==========================================================
def reset_fields():
    Name.set("")
    Symptom1.set("Select Symptom")
    Symptom2.set("Select Symptom")
    Symptom3.set("Select Symptom")
    Symptom4.set("Select Symptom")
    Symptom5.set("Select Symptom")

    t1.delete("1.0", END)
    t2.delete("1.0", END)
    t3.delete("1.0", END)

    confidence_text.delete("1.0", END)
    top3_text.delete("1.0", END)

    global l2
    l2 = [0] * len(l1)
root = ttk.Window(themename="darkly")
root.title("🩺 Disease Prediction System v3.0")
root.geometry("1050x650")
root.resizable(False, False)
# ==========================
# Variables
# ==========================

Name = StringVar()

Symptom1 = StringVar(value="Select Symptom")
Symptom2 = StringVar(value="Select Symptom")
Symptom3 = StringVar(value="Select Symptom")
Symptom4 = StringVar(value="Select Symptom")
Symptom5 = StringVar(value="Select Symptom")

# ==========================
# Heading
# ==========================

title = Label(
    root,
    text="🩺 Disease Prediction System",
    font=("Segoe UI", 24, "bold"),
    bg="#2C3E50",
    fg="white"
)

title.grid(row=0, column=0, columnspan=4, pady=(20,5))

subtitle = Label(
    root,
    text="Developed by Ahmed Al Salman",
    font=("Segoe UI", 12),
    bg="#2C3E50",
    fg="#BDC3C7"
)
subtitle.grid(row=1, column=0, columnspan=4, pady=(0,20))

# ==========================
# Labels
# ==========================

Label(root,text="Patient Name",font=("Segoe UI",11,"bold"),
      bg="#2C3E50",fg="white").grid(row=2,column=0,padx=15,pady=10,sticky=W)

Label(root,text="Symptom 1",font=("Segoe UI",10),
      bg="#2C3E50",fg="white").grid(row=3,column=0,padx=15,pady=8,sticky=W)

Label(root,text="Symptom 2",font=("Segoe UI",10),
      bg="#2C3E50",fg="white").grid(row=4,column=0,padx=15,pady=8,sticky=W)

Label(root,text="Symptom 3",font=("Segoe UI",10),
      bg="#2C3E50",fg="white").grid(row=5,column=0,padx=15,pady=8,sticky=W)

Label(root,text="Symptom 4",font=("Segoe UI",10),
      bg="#2C3E50",fg="white").grid(row=6,column=0,padx=15,pady=8,sticky=W)

Label(root,text="Symptom 5",font=("Segoe UI",10),
      bg="#2C3E50",fg="white").grid(row=7,column=0,padx=15,pady=8,sticky=W)

# ==========================
# Inputs
# ==========================

OPTIONS = sorted(l1)

NameEn = Entry(root,textvariable=Name,width=35,font=("Segoe UI",10))
NameEn.grid(row=2,column=1,pady=10)

S1En = ttk.OptionMenu(
    root,
    Symptom1,
    "Select Symptom",
    *OPTIONS
)

S2En = ttk.OptionMenu(
    root,
    Symptom2,
    "Select Symptom",
    *OPTIONS
)

S3En = ttk.OptionMenu(
    root,
    Symptom3,
    "Select Symptom",
    *OPTIONS
)

S4En = ttk.OptionMenu(
    root,
    Symptom4,
    "Select Symptom",
    *OPTIONS
)

S5En = ttk.OptionMenu(
    root,
    Symptom5,
    "Select Symptom",
    *OPTIONS
)


# ==========================
# Display Symptom Menus
# ==========================

S1En.grid(row=3,column=1,sticky="ew")
S2En.grid(row=4,column=1,sticky="ew")
S3En.grid(row=5,column=1,sticky="ew")
S4En.grid(row=6,column=1,sticky="ew")
S5En.grid(row=7,column=1,sticky="ew")


# ==========================
# Buttons
# ==========================

ttk.Button(
    root,
    text="Decision Tree",
    command=DecisionTree,
    bootstyle="primary",
    width=18
).grid(row=3,column=2,padx=20)

ttk.Button(
    root,
    text="Random Forest",
    command=randomforest,
    bootstyle="success",
    width=18
).grid(row=4,column=2,padx=20)

ttk.Button(
    root,
    text="Naive Bayes",
    command=NaiveBayes,
    bootstyle="secondary",
    width=18
).grid(row=5,column=2,padx=20)

ttk.Button(
    root,
    text="Export PDF Report",
    command=export_pdf,
    bootstyle="danger",
    width=22
).grid(row=6,column=2,padx=20,pady=10)
# ==========================
# Result Labels
# ==========================

ttk.Label(
    root,
    text="🔵 Decision Tree",
    bootstyle="primary",
    font=("Segoe UI",10,"bold")
).grid(row=9,column=0,pady=10,sticky=W)

ttk.Label(
    root,
    text="🟢 Random Forest",
    bootstyle="success",
    font=("Segoe UI",10,"bold")
).grid(row=10,column=0,pady=10,sticky=W)

ttk.Label(
    root,
    text="🟣 Naive Bayes",
    bootstyle="secondary",
    font=("Segoe UI",10,"bold")
).grid(row=11,column=0,pady=10,sticky=W)

# ==========================
# Result Boxes
# ==========================

t1 = Text(
    root,
    height=1,
    width=35,
    font=("Segoe UI", 10, "bold"),
    bg="#D6EAF8",
    fg="#154360",
    relief="flat",
    bd=2
)
t1.grid(row=9, column=1, pady=5)


t2 = Text(
    root,
    height=1,
    width=35,
    font=("Segoe UI", 10, "bold"),
    bg="#D5F5E3",
    fg="#145A32",
    relief="flat",
    bd=2
)
t2.grid(row=10, column=1, pady=5)


t3 = Text(
    root,
    height=1,
    width=35,
    font=("Segoe UI", 10, "bold"),
    bg="#EBDEF0",
    fg="#512E5F",
    relief="flat",
    bd=2
)
t3.grid(row=11, column=1, pady=5)
# ==========================
# Confidence
# ==========================

ttk.Label(
    root,
    text="⭐ Confidence (%)",
    bootstyle="warning",
    font=("Segoe UI",10,"bold")
).grid(row=12,column=0,pady=10,sticky=W)


confidence_text = Text(root,height=1,width=35,font=("Segoe UI",10))
confidence_text.grid(row=12,column=1,pady=5)

# ==========================
# Top 3 Diseases
# ==========================

ttk.Label(
    root,
    text="📋 Top 3 Predictions",
    bootstyle="info",
    font=("Segoe UI",10,"bold")
).grid(row=13,column=0,pady=10,sticky=W)

top3_text = Text(root,height=5,width=35,font=("Segoe UI",10))
top3_text.grid(row=13,column=1,pady=5)

root.mainloop()