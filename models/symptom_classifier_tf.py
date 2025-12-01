# models/symptom_classifier_tf.py
import pandas as pd
import difflib

def predict_department_tf(user_input: str, csv_path="data/symptoms_to_department.csv"):
    """
    Predict the most appropriate medical department based on user symptoms.
    
    Args:
        user_input (str): User's symptom description
        csv_path (str): Path to symptoms CSV file
    
    Returns:
        str: Recommended department name
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Using fallback logic.")
        # Fallback symptom mapping if CSV not found
        return fallback_symptom_mapping(user_input)
    
    text = user_input.lower()
    
    # Check for direct substring matches first
    for _, row in df.iterrows():
        if row["symptom"].lower() in text:
            return row["department"]
    
    # Fuzzy match if no direct match
    symptoms = df["symptom"].str.lower().tolist()
    closest = difflib.get_close_matches(text, symptoms, n=1, cutoff=0.4)
    if closest:
        matched_symptom = closest[0]
        dept = df[df["symptom"].str.lower() == matched_symptom]["department"].iloc[0]
        return dept
    
    # If no match found
    return "General Medicine"

def fallback_symptom_mapping(user_input: str):
    """
    Fallback function for symptom to department mapping when CSV is not available.
    
    Args:
        user_input (str): User's symptom description
    
    Returns:
        str: Recommended department name
    """
    text = user_input.lower()
    
    # Define symptom keywords for each department
    symptom_map = {
        "Cardiology": [
            "chest pain", "heart", "cardiac", "palpitation", "angina", 
            "hypertension", "blood pressure", "arrhythmia", "shortness of breath"
        ],
        "Neurology": [
            "headache", "migraine", "seizure", "stroke", "paralysis", 
            "numbness", "dizziness", "vertigo", "tremor", "memory loss"
        ],
        "Orthopedics": [
            "bone", "fracture", "joint pain", "arthritis", "back pain", 
            "sprain", "knee pain", "shoulder pain", "hip pain", "muscle pain"
        ],
        "Dermatology": [
            "skin", "rash", "acne", "eczema", "psoriasis", "itching", 
            "allergy", "hives", "pimples", "skin infection"
        ],
        "Gastroenterology": [
            "stomach", "abdominal pain", "diarrhea", "constipation", 
            "vomiting", "nausea", "indigestion", "gastric", "liver", "intestine"
        ],
        "ENT": [
            "ear", "nose", "throat", "hearing", "tonsil", "sinus", 
            "voice", "vertigo", "ear pain", "sore throat"
        ],
        "Ophthalmology": [
            "eye", "vision", "cataract", "glaucoma", "blurred vision", 
            "eye pain", "red eye", "conjunctivitis"
        ],
        "Pulmonology": [
            "lung", "breathing", "cough", "asthma", "bronchitis", 
            "pneumonia", "chest congestion", "respiratory"
        ],
        "Pediatrics": [
            "child", "infant", "baby", "vaccination", "growth", 
            "childhood illness", "pediatric"
        ],
        "Gynecology": [
            "pregnancy", "menstrual", "period", "ovarian", "uterus", 
            "pcos", "gynec", "women's health"
        ],
        "Urology": [
            "kidney", "bladder", "urinary", "prostate", "urine", 
            "uti", "kidney stone"
        ],
        "Psychiatry": [
            "depression", "anxiety", "mental", "stress", "insomnia", 
            "mood", "psychological", "panic attack"
        ],
        "Endocrinology": [
            "diabetes", "thyroid", "hormone", "sugar", "insulin", 
            "metabolic", "gland"
        ],
        "Oncology": [
            "cancer", "tumor", "chemotherapy", "radiation", "malignant", 
            "oncology"
        ]
    }
    
    # Check each department's keywords
    for department, keywords in symptom_map.items():
        for keyword in keywords:
            if keyword in text:
                return department
    
    # If nothing matches, return General Medicine
    return "General Medicine"