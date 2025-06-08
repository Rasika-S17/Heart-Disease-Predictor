import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("heart_model.pkl")

# Feature names and their input types
features = {
    "age": "entry",
    "sex": "combo",        # 0 = female, 1 = male
    "cp": "combo",         # chest pain type (0-3)
    "trestbps": "entry",   # resting blood pressure
    "chol": "entry",       # serum cholesterol
    "fbs": "combo",        # fasting blood sugar > 120 mg/dl (0 = false, 1 = true)
    "restecg": "combo",    # resting electrocardiographic results (0-2)
    "thalach": "entry",    # max heart rate achieved
    "exang": "combo",      # exercise induced angina (0 = no, 1 = yes)
    "oldpeak": "entry",    # ST depression induced by exercise
    "slope": "combo",      # slope of the peak exercise ST segment (0-2)
    "ca": "combo",         # number of major vessels (0-3) colored by fluoroscopy
    "thal": "combo"        # 1 = normal; 2 = fixed defect; 3 = reversible defect
}

# Possible dropdown values for categorical features
dropdown_options = {
    "sex": ["0", "1"],
    "cp": ["0", "1", "2", "3"],
    "fbs": ["0", "1"],
    "restecg": ["0", "1", "2"],
    "exang": ["0", "1"],
    "slope": ["0", "1", "2"],
    "ca": ["0", "1", "2", "3"],
    "thal": ["1", "2", "3"]
}

# Create main window
root = tk.Tk()
root.title("Heart Disease Risk Predictor")
root.geometry("400x650")

entries = {}

# Create labels and inputs
row = 0
for feature, ftype in features.items():
    label = tk.Label(root, text=feature.upper(), anchor='w')
    label.grid(row=row, column=0, padx=10, pady=5, sticky="w")

    if ftype == "entry":
        entry = tk.Entry(root)
        entry.grid(row=row, column=1, padx=10, pady=5)
        entries[feature] = entry
    else:  # combo box
        combo = ttk.Combobox(root, values=dropdown_options[feature], state="readonly")
        combo.current(0)
        combo.grid(row=row, column=1, padx=10, pady=5)
        entries[feature] = combo

    row += 1

def predict():
    try:
        # Collect input values as floats or ints as appropriate
        input_data = []
        for f, ftype in features.items():
            val = entries[f].get()
            if val == "":
                raise ValueError(f"{f} cannot be empty.")
            # Convert to float for entry fields, int for combo fields
            if ftype == "entry":
                val = float(val)
            else:
                val = int(val)
            input_data.append(val)

        df = pd.DataFrame([input_data], columns=features.keys())
        result = model.predict(df)

        if result[0] == 1:
            messagebox.showwarning("Result", "⚠️ High Risk of Heart Disease")
        else:
            messagebox.showinfo("Result", "✅ No Risk Detected")

    except ValueError as e:
        messagebox.showerror("Invalid input", str(e))

# Predict button
predict_btn = tk.Button(root, text="Predict", command=predict, bg="green", fg="white")
predict_btn.grid(row=row, column=0, columnspan=2, pady=20)

root.mainloop()
