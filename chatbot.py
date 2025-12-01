# app.py

import streamlit as st
import pandas as pd
import os
import uuid
import time
import re
import dateparser
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils.pdf_generator import generate_pdf

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="MedBot AI", page_icon="💬", layout="centered")

# ---------------- LOAD DATA ----------------
hospitals = pd.read_csv("data/hospitals.csv")
doctors = pd.read_csv("data/doctors.csv")
reviews = pd.read_csv("data/reviews.csv")

# ---------------- LOAD DEEP LEARNING MODEL ----------------
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

@st.cache_resource
def load_symptom_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=14   # 14 medical departments
    )
    return tokenizer, model

tokenizer, model = load_symptom_model()

DEPARTMENTS = [
    "Cardiology", "Neurology", "Orthopedics", "Dermatology",
    "Gastroenterology", "ENT", "Ophthalmology", "Pulmonology",
    "Pediatrics", "Gynecology", "Urology", "Psychiatry",
    "Endocrinology", "General Medicine"
]

def predict_department(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    return DEPARTMENTS[pred]

# ---------------- SESSION STATE ----------------
if "user" not in st.session_state:
    st.session_state.user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "city" not in st.session_state:
    st.session_state.city = None
if "pending_booking" not in st.session_state:
    st.session_state.pending_booking = {}
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

# ---------------- HEADER ----------------
st.title("💬 MedBot AI – Deep Learning Symptom Analyzer")

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.user:
    st.subheader("🔐 Login / Sign Up")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login"):
            if email == "demo@demo.com" and password == "demo123":
                st.session_state.user = {
                    "name": "Demo User",
                    "email": email,
                    "city": "Kochi",
                    "age": 25,
                    "gender": "Male",
                }
                st.success("Logged in successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with tab2:
        name = st.text_input("Name")
        age = st.number_input("Age", 0, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        email2 = st.text_input("Email")
        city = st.text_input("City")
        pw = st.text_input("Password", type="password")

        if st.button("Create Account"):
            if name and email2 and city:
                st.session_state.user = {
                    "name": name,
                    "email": email2,
                    "city": city,
                    "age": age,
                    "gender": gender
                }
                st.success(f"Welcome, {name}!")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Fill all fields.")

# ---------------- MAIN APP ----------------
else:
    user = st.session_state.user

    # SIDEBAR
    st.sidebar.write(f"👋 Logged in as **{user['name']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    # CHAT INTERFACE
    st.subheader("💬 Chat with MedBot AI")

    for chat in st.session_state.chat_history:
        st.chat_message(chat["sender"]).markdown(chat["text"])

    user_message = st.chat_input("Type your message...")
    if user_message:
        msg = user_message.lower()
        st.session_state.chat_history.append({"sender": "user", "text": user_message})
        st.chat_message("user").markdown(user_message)

        time.sleep(0.4)
        response = ""
        city = user.get("city", "Kochi")

        # Detect if city is mentioned
        for c in hospitals["city"].unique():
            if c.lower() in msg:
                city = c
                st.session_state.city = c
                break

        # ---------------- INTENT DETECTION ----------------
        if any(w in msg for w in ["pain", "fever", "rash", "ache", "vomit", "dizzy", "cough"]):
            intent = "symptom"
        elif "hospital" in msg:
            intent = "hospital"
        elif "doctor" in msg:
            intent = "doctor"
        elif "book" in msg or "appointment" in msg:
            intent = "booking"
        else:
            intent = st.session_state.last_intent or "unknown"

        # ---------------- SYMPTOM ANALYSIS USING DEEP LEARNING ----------------
        if intent == "symptom":
            dept = predict_department(user_message)
            response = f"🩺 Based on your symptoms, you should visit the **{dept}** department."

            # List hospitals for that department
            results = hospitals[
                (hospitals["city"].str.lower() == city.lower()) &
                (hospitals["department"].str.lower() == dept.lower())
            ]

            if not results.empty:
                response += f"\n🏥 Top hospitals for **{dept}** in **{city}**:\n"
                for _, row in results.sort_values("rating", ascending=False).head(3).iterrows():
                    response += f"- **{row['hospital_name']}** ({row['rating']}⭐)\n"
            else:
                response += f"\nNo hospitals found for {dept} in {city}."

            st.session_state.pending_booking = {"department": dept, "city": city}
            st.session_state.last_intent = "symptom"

        # ---------------- HOSPITAL LIST ----------------
        elif intent == "hospital":
            city_hosp = hospitals[hospitals["city"].str.lower() == city.lower()]
            if not city_hosp.empty:
                response = f"🏥 Hospitals in **{city}**:\n"
                for _, row in city_hosp.head(5).iterrows():
                    response += f"- **{row['hospital_name']}** (⭐{row['rating']})\n"
            else:
                response = f"No hospitals found in {city}."
            st.session_state.last_intent = "hospital"

        # ---------------- DOCTOR LIST ----------------
        elif intent == "doctor":
            dept = st.session_state.pending_booking.get("department")
            if not dept:
                response = "Please describe your symptoms first."
            else:
                doctor_list = doctors[
                    (doctors["department"].str.lower() == dept.lower()) &
                    (doctors["hospital_id"].isin(
                        hospitals[hospitals["city"].str.lower() == city.lower()]["hospital_id"]
                    ))
                ]

                if not doctor_list.empty:
                    response = f"👩‍⚕️ Doctors in **{dept}** at **{city}**:\n"
                    for _, doc in doctor_list.head(5).iterrows():
                        hosp_name = hospitals[hospitals["hospital_id"] == doc["hospital_id"]]["hospital_name"].iloc[0]
                        response += (
                            f"- **{doc['doctor_name']}** ({hosp_name})\n"
                            f"  🕒 {doc['start_time']} - {doc['end_time']} | ₹{doc['fee']}\n"
                        )
                else:
                    response = f"No doctors found for {dept} in {city}."
            st.session_state.last_intent = "doctor"

        # ---------------- BOOKING ----------------
        elif intent == "booking" and st.session_state.pending_booking:
            dept = st.session_state.pending_booking["department"]
            parsed_date = dateparser.parse(msg)
            date_str = parsed_date.strftime("%Y-%m-%d") if parsed_date else datetime.now().strftime("%Y-%m-%d")

            time_match = re.search(r"(\d{1,2}):(\d{2})", msg)
            time_str = time_match.group(0) if time_match else "10:00"

            hosp = hospitals[
                (hospitals["city"].str.lower() == city.lower()) &
                (hospitals["department"].str.lower() == dept.lower())
            ]

            if not hosp.empty:
                top = hosp.sort_values("rating", ascending=False).iloc[0]
                doctor = doctors[
                    (doctors["hospital_id"] == top.hospital_id) &
                    (doctors["department"].str.lower() == dept.lower())
                ].iloc[0]

                booking_id = "A" + uuid.uuid4().hex[:6].upper()

                booking = {
                    "Booking ID": booking_id,
                    "Patient": user["name"],
                    "Age": user["age"],
                    "Gender": user["gender"],
                    "Hospital": top.hospital_name,
                    "Department": dept,
                    "Doctor": doctor["doctor_name"],
                    "Date": date_str,
                    "Time": time_str
                }

                os.makedirs("receipts", exist_ok=True)
                pdf_path = f"receipts/{booking_id}.pdf"
                generate_pdf(booking, pdf_path)

                response = (
                    f"✅ Appointment booked with **{doctor['doctor_name']}** at **{top.hospital_name}**.\n"
                    f"🗓️ {date_str} | ⏰ {time_str}\n"
                    "Download your receipt below."
                )

                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download Receipt", f, file_name=f"{booking_id}.pdf")

            st.session_state.last_intent = "booking"

        # ---------------- DEFAULT ----------------
        else:
            response = (
                "I'm MedBot 🤖.\n"
                "Tell me your **symptoms** and I'll predict the right department using AI.\n"
                "Example: *I have chest pain and I'm sweating*"
            )
            st.session_state.last_intent = "unknown"

        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append({"sender": "assistant", "text": response})
