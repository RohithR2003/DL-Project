# app.py

import streamlit as st
import pandas as pd
import os
import uuid
import time
import re
import dateparser
from datetime import datetime
from models.symptom_classifier_tf import predict_department_tf
from utils.pdf_generator import generate_pdf

# ---------------- APP CONFIGURATION ----------------
st.set_page_config(page_title="MedBot AI", page_icon="💬", layout="centered")

# ---------------- LOAD DATA ----------------
hospitals = pd.read_csv("data/hospitals.csv")
doctors = pd.read_csv("data/doctors.csv")
reviews = pd.read_csv("data/reviews.csv")

# ---------------- SESSION STATES ----------------
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
st.title("💬 MedBot AI – Smart Medical Assistant & Appointment Booking")

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.user:
    st.subheader("🔐 Login / Sign Up")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login", key="login_btn"):
            if email == "demo@demo.com" and password == "demo123":
                st.session_state.user = {
                    "name": "Demo User",
                    "email": email,
                    "city": "Kochi",
                    "age": 25,
                    "gender": "Male",
                }
                st.success("Login successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials. Try demo@demo.com / demo123")

    with tab2:
        name = st.text_input("Name", key="signup_name")
        age = st.number_input("Age", min_value=0, max_value=120, key="signup_age")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="signup_gender")
        email2 = st.text_input("Email", key="signup_email")
        city = st.text_input("City", key="signup_city")
        pw = st.text_input("Password", type="password", key="signup_pw")
        if st.button("Create Account", key="signup_btn"):
            if name and email2 and city:
                st.session_state.user = {
                    "name": name,
                    "email": email2,
                    "city": city,
                    "age": age,
                    "gender": gender,
                }
                st.success(f"Welcome, {name}! You’re now signed in.")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Please fill in all fields.")
else:
    user = st.session_state.user
    st.sidebar.write(f"👋 Logged in as **{user['name']} ({user['gender']}, {user['age']} yrs)**")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.chat_history = []
        st.session_state.city = None
        st.session_state.pending_booking = {}
        st.session_state.last_intent = None
        st.rerun()

    # ---------------- CHATBOT INTERFACE ----------------
    st.subheader("💬 Chat with MedBot AI")

    for chat in st.session_state.chat_history:
        if chat["sender"] == "user":
            st.chat_message("user").markdown(chat["text"])
        else:
            st.chat_message("assistant").markdown(chat["text"])

    user_message = st.chat_input("Type your message...")
    if user_message:
        msg = user_message.lower()
        st.session_state.chat_history.append({"sender": "user", "text": user_message})
        st.chat_message("user").markdown(user_message)
        time.sleep(0.4)

        response = ""
        city = st.session_state.city or user.get("city", "Chennai")

        # --- Detect city mentioned ---
        for c in hospitals["city"].unique():
            if c.lower() in msg:
                st.session_state.city = c
                city = c
                break

        # --- Intent detection ---
        if any(word in msg for word in ["hospital", "hospitals", "nearby", "list hospitals"]):
            intent = "hospital_list"
        elif any(word in msg for word in ["doctor", "doctors", "available", "specialist"]):
            intent = "doctor_list"
        elif any(word in msg for word in ["pain", "fever", "vomit", "rash", "headache", "ache", "itch", "cough", "breath", "skin"]):
            intent = "symptom_analysis"
        elif any(word in msg for word in ["book", "appointment", "yes", "confirm", "on", "pm", "am", "date"]):
            intent = "booking"
        else:
            # if user continues a previous intent
            intent = st.session_state.last_intent or "unknown"

        # ------------------ Hospital list ------------------
        if intent == "hospital_list":
            city_hosp = hospitals[hospitals["city"].str.lower() == city.lower()]
            if not city_hosp.empty:
                response = f"🏥 Hospitals in **{city}**:\n"
                for _, row in city_hosp.head(5).iterrows():
                    response += f"- **{row['hospital_name']}** ({row['department']}, ⭐{row['rating']})\n"
                response += "\nWould you like to book an appointment?"
            else:
                response = f"Sorry, no hospitals found in {city}."
            st.session_state.last_intent = "hospital_list"

        # ------------------ Doctor list ------------------
        elif intent == "doctor_list":
            dept = None
            # Case 1: user typed department directly
            for d in hospitals["department"].unique():
                if d.lower() in msg:
                    dept = d
                    break
            # Case 2: fallback to pending department from symptom
            if not dept and "department" in st.session_state.pending_booking:
                dept = st.session_state.pending_booking["department"]

            if dept:
                doctor_list = doctors[
                    (doctors["department"].str.lower() == dept.lower())
                    & (doctors["hospital_id"].isin(
                        hospitals[hospitals["city"].str.lower() == city.lower()]["hospital_id"]
                    ))
                ]
                if not doctor_list.empty:
                    response = f"👩‍⚕️ Available doctors in **{dept}** at **{city}**:\n"
                    for _, doc in doctor_list.head(5).iterrows():
                        hosp_name = hospitals[hospitals["hospital_id"] == doc["hospital_id"]]["hospital_name"].iloc[0]
                        response += (
                            f"- **{doc['doctor_name']}** ({hosp_name})\n"
                            f"  🕒 {doc['start_time']} - {doc['end_time']} | 💰 ₹{doc['fee']}\n"
                        )
                    response += "\nWould you like to book an appointment?"
                    st.session_state.pending_booking = {"department": dept, "city": city}
                else:
                    response = f"Sorry, no doctors found for **{dept}** in **{city}**."
            else:
                response = "Please mention which department you'd like to know doctors for."
            st.session_state.last_intent = "doctor_list"

        # ------------------ Symptom analysis ------------------
        elif intent == "symptom_analysis":
            dept = predict_department_tf(user_message)
            response = f"🩺 Based on your symptoms, you should visit the **{dept}** department."

            results = hospitals[
                (hospitals["city"].str.lower() == city.lower())
                & (hospitals["department"].str.lower() == dept.lower())
            ]
            if not results.empty:
                response += f"\n🏥 Hospitals for **{dept}** in **{city}**:\n"
                for _, row in results.sort_values(by="rating", ascending=False).head(3).iterrows():
                    response += f"- **{row['hospital_name']}** ({row['rating']}⭐)\n"
                response += "\nWould you like to book an appointment?"
                st.session_state.pending_booking = {"department": dept, "city": city}
            else:
                response += f"\nSorry, no hospitals found for {dept} in {city}."
            st.session_state.last_intent = "symptom_analysis"

        # ------------------ Booking ------------------
        elif intent == "booking" and st.session_state.pending_booking:
            city = st.session_state.pending_booking["city"]
            dept = st.session_state.pending_booking["department"]

            parsed_date = dateparser.parse(msg, settings={"PREFER_DATES_FROM": "future"})
            time_match = re.search(r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?", msg)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                am_pm = time_match.group(3)
                if am_pm and am_pm.lower() == "pm" and hour != 12:
                    hour += 12
                time_str = f"{hour:02d}:{minute:02d}"
            else:
                time_str = "10:00"

            if parsed_date:
                date_str = parsed_date.strftime("%Y-%m-%d")
                st.session_state.pending_booking["date"] = date_str
            else:
                date_str = st.session_state.pending_booking.get("date", datetime.now().strftime("%Y-%m-%d"))

            hosp = hospitals[
                (hospitals["city"].str.lower() == city.lower())
                & (hospitals["department"].str.lower() == dept.lower())
            ]

            if not hosp.empty:
                top_hosp = hosp.sort_values(by="rating", ascending=False).iloc[0]
                doctor = doctors[
                    (doctors["hospital_id"] == top_hosp.hospital_id)
                    & (doctors["department"].str.lower() == dept.lower())
                ].iloc[0]

                start_t = datetime.strptime(doctor["start_time"], "%H:%M").time()
                end_t = datetime.strptime(doctor["end_time"], "%H:%M").time()
                booking_t = datetime.strptime(time_str, "%H:%M").time()

                if not (start_t <= booking_t <= end_t):
                    response = (
                        f"{doctor['doctor_name']} is available between "
                        f"{start_t.strftime('%I:%M %p')} and {end_t.strftime('%I:%M %p')}.\n"
                        f"Please choose a time within this window."
                    )
                
                else:
                    booking_id = "A" + uuid.uuid4().hex[:6].upper()
                    booking = {
                        "Booking ID": booking_id,
                        "Patient": user["name"],
                        "Age": user["age"],
                        "Gender": user["gender"],
                        "Hospital": top_hosp.hospital_name,
                        "Department": dept,
                        "Doctor": doctor["doctor_name"],
                        "Date": date_str,
                        "Time": time_str
                    }

                    os.makedirs("receipts", exist_ok=True)
                    pdf_path = f"receipts/{booking_id}.pdf"
                    generate_pdf(booking, pdf_path)

                    response = (
                        f"✅ Appointment booked with **{doctor['doctor_name']}** at **{top_hosp.hospital_name}**.\n"
                        f"🗓️ Date: **{date_str}** | ⏰ Time: **{time_str} hrs**\n\n"
                        "📄 You can download your receipt below."
                    )

                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "📄 Download Appointment Receipt",
                            f,
                            file_name=f"{booking_id}.pdf",
                            key=f"receipt_{booking_id}",
                        )
            else:
                response = f"Sorry, no hospitals found for {dept} in {city}."
            st.session_state.last_intent = "booking"

        # ------------------ Default ------------------
        else:
            response = (
                "I'm MedBot 🤖 — I can help you with:\n"
                "- Listing hospitals 🏥\n"
                "- Finding doctors 👩‍⚕️\n"
                "- Detecting departments from symptoms 🩺\n"
                "- Booking appointments 📅\n\n"
                "Please tell me your **city** or your **symptoms** to begin."
            )
            st.session_state.last_intent = "unknown"

        st.session_state.chat_history.append({"sender": "assistant", "text": response})
        st.chat_message("assistant").markdown(response)
