# app.py

import streamlit as st
import pandas as pd
import os
import uuid
import time
import re
import dateparser
from datetime import datetime
from pathlib import Path

# ML imports (wrap in try/except so app still runs if not installed)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# local project imports
# You must have these implemented in your project:
# - models.symptom_classifier_tf.predict_department_tf : returns department string for symptom text
# - utils.pdf_generator.generate_pdf : creates a PDF from booking dict and writes to given path
try:
    from models.symptom_classifier_tf import predict_department_tf
except Exception:
    # fallback simple rule-based department mapping if user hasn't implemented predict_department_tf
    def predict_department_tf(text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["chest", "palpit", "breath"]):
            return "Cardiology"
        if any(w in t for w in ["headache", "migraine", "seizure", "numbness"]):
            return "Neurology"
        if any(w in t for w in ["skin", "rash", "acne", "itch"]):
            return "Dermatology"
        if any(w in t for w in ["stomach", "nausea", "diarrhea", "vomit", "acid"]):
            return "Gastroenterology"
        if any(w in t for w in ["cough", "asthma", "breath"]):
            return "Pulmonology"
        return "General Medicine"

try:
    from utils.pdf_generator import generate_pdf
except Exception:
    # simple placeholder generator if not available. Creates a tiny text-based PDF-like file.
    def generate_pdf(booking: dict, path: str):
        # create a small text report and save as .pdf extension (not a real PDF)
        with open(path, "w", encoding="utf-8") as f:
            f.write("Booking Receipt\n\n")
            for k, v in booking.items():
                f.write(f"{k}: {v}\n")
        return

# ----------------- Initialize Models -----------------
sentiment_model = None
chat_tokenizer = None
chat_model = None

if TRANSFORMERS_AVAILABLE:
    try:
        # sentiment pipeline (lightweight)
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    except Exception as e:
        sentiment_model = None

    try:
        # small conversational model - DialoGPT-medium (or fallback to a different small LM)
        chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    except Exception:
        chat_tokenizer = None
        chat_model = None

# helper: sentiment wrapper
def get_review_sentiment(text: str):
    """
    Returns (label, score) or ('NEUTRAL', 0.0) if model missing.
    label is e.g. 'POSITIVE' / 'NEGATIVE'
    """
    if sentiment_model is None:
        return "NEUTRAL", 0.0
    try:
        res = sentiment_model(text, truncation=True)
        if isinstance(res, list) and len(res) > 0:
            return res[0]["label"], float(res[0].get("score", 0.0))
    except Exception:
        pass
    return "NEUTRAL", 0.0

# helper: medbot reply generator using sentiment + LLM (with fallback)
def medbot_reply(user_input: str) -> str:
    """
    Compose a medical-aware prompt using sentiment and optionally generate a reply with the chat_model.
    If chat_model missing, return a templated reply with department suggestion.
    """
    sentiment, score = get_review_sentiment(user_input)
    # tone prefix
    if sentiment == "NEGATIVE":
        prefix = "I'm sorry to hear that. Please know I'm here to help. "
    elif sentiment == "POSITIVE":
        prefix = "That's good to hear. Here is some helpful information. "
    else:
        prefix = "Let's look into that. "

    # Ask the symptom classifier for a department suggestion
    try:
        dept = predict_department_tf(user_input)
    except Exception:
        dept = None

    base_instructions = (
        f"{prefix}You are an empathetic and careful medical assistant. "
        "Do NOT provide a definitive diagnosis. Offer likely departments, first-aid suggestions, "
        "and advise seeing a clinician when appropriate. Keep the reply concise (2-4 sentences). "
    )

    if dept:
        base_instructions += f"Based on the symptoms, suggest visiting the {dept} department when relevant. "

    prompt = f"{base_instructions}User: \"{user_input}\"\nAssistant:"

    # If generation model available, use it
    if chat_model is not None and chat_tokenizer is not None:
        try:
            inputs = chat_tokenizer.encode(prompt + chat_tokenizer.eos_token, return_tensors="pt", truncation=True)
            outputs = chat_model.generate(
                inputs,
                max_length=inputs.shape[-1] + 150,
                pad_token_id=chat_tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                eos_token_id=chat_tokenizer.eos_token_id,
            )
            # decode only the newly generated tokens
            generated = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # try to extract assistant part after "Assistant:" marker
            if "Assistant:" in generated:
                assistant_part = generated.split("Assistant:", 1)[1].strip()
                return assistant_part
            return generated
        except Exception:
            # fallback below
            pass

    # fallback templated reply if model not available
    reply = prefix
    if dept:
        reply += f"It may be appropriate to consult {dept}. "
    reply += "Please rest, stay hydrated, and if symptoms are severe or worsen (high fever, breathing difficulty, severe pain), seek immediate medical attention or visit the nearest hospital."
    return reply

# ----------------- Streamlit App -----------------
st.set_page_config(page_title="MedBot AI Healthcare", page_icon="🏥", layout="wide")

# -- Theme toggle (persist in session)
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
theme_choice = st.sidebar.radio("Theme", ("dark", "light"), index=0 if st.session_state.theme == "dark" else 1)
st.session_state.theme = theme_choice

# minimal CSS for chat bubble contrast depending on theme
if st.session_state.theme == "dark":
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background-color: #0b0f13; color: #e6eef6; }
        [data-testid="stChatMessageContainer"] { background: #0b0f13 !important; }
        [data-testid="stChatMessage-user"] { background: #0b3d17 !important; color: #e8f5e9 !important; border-right: 4px solid #66bb6a !important; }
        [data-testid="stChatMessage"]:not([data-testid="stChatMessage-user"]) { background: #052f66 !important; color: #e3f2fd !important; border-left: 4px solid #64b5f6 !important; }
        .stDownloadButton > button { background: #2e7d32 !important; color: white !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background-color: #f7fbfc; color: #0a0a0a; }
        [data-testid="stChatMessageContainer"] { background: #ffffff !important; }
        [data-testid="stChatMessage-user"] { background: #e8f5e9 !important; color: #1b5e20 !important; border-right: 4px solid #2e7d32 !important; }
        [data-testid="stChatMessage"]:not([data-testid="stChatMessage-user"]) { background: #e3f2fd !important; color: #0d47a1 !important; border-left: 4px solid #1565c0 !important; }
        .stDownloadButton > button { background: #1565c0 !important; color: white !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -- Load data (CSV)
DATA_DIR = Path("data")
hosp_csv = DATA_DIR / "hospitals.csv"
doctors_csv = DATA_DIR / "doctors.csv"
reviews_csv = DATA_DIR / "reviews.csv"

if not hosp_csv.exists() or not doctors_csv.exists():
    st.error("Required data files not found in `data/` (hospitals.csv and doctors.csv). Please add them and reload.")
    st.stop()

hospitals = pd.read_csv(hosp_csv)
doctors = pd.read_csv(doctors_csv)

# Normalize column names for doctors and hospitals to safe snake_case
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

hospitals = normalize_columns(hospitals)
doctors = normalize_columns(doctors)

# Ensure required columns exist
required_hosp_cols = {"hospital_id", "hospital_name", "city", "department", "rating"}
required_doc_cols = {"doctor_id", "doctor_name", "hospital_id", "department", "fee"}
if not required_hosp_cols.issubset(set(hospitals.columns)):
    st.error(f"Missing required hospital columns. Found: {hospitals.columns.tolist()}")
    st.stop()
if not required_doc_cols.issubset(set(doctors.columns)):
    st.error(f"Missing required doctor columns. Found: {doctors.columns.tolist()}")
    st.stop()

# Some doctor datasets may have start_time/end_time columns; ensure they exist (create default if missing)
if "start_time" not in doctors.columns or "end_time" not in doctors.columns:
    # create default availability window 09:00 - 17:00
    doctors["start_time"] = doctors.get("start_time", "09:00")
    doctors["end_time"] = doctors.get("end_time", "17:00")

# Optional reviews (for sentiment/display)
if reviews_csv.exists():
    reviews = pd.read_csv(reviews_csv)
    reviews = normalize_columns(reviews)
else:
    reviews = pd.DataFrame(columns=["hospital_id", "review_text"])

# ---------------- Session variables ----------------
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

# ---------------- UI: Header / Login ----------------
st.title("💬 MedBot AI — Healthcare Assistant")

if st.session_state.user is None:
    st.subheader("🔐 Login (demo available)")
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", value="demo@demo.com")
        password = st.text_input("Password", type="password", value="demo123")
        submitted = st.form_submit_button("Login")
        if submitted:
            if email == "demo@demo.com" and password == "demo123":
                st.session_state.user = {
                    "name": "Demo User",
                    "email": email,
                    "city": "Kochi",
                    "age": 25,
                    "gender": "Male"
                }
                st.success("Logged in as Demo User.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Invalid credentials for demo. Use demo@demo.com / demo123")
    st.caption("Or add your own signup flow and user storage.")
    st.stop()

# Logged-in UI
user = st.session_state.user
with st.sidebar:
    st.markdown(f"**👤 Logged in as:** {user['name']}")
    st.markdown(f"**🏙 City:** {st.session_state.city or user.get('city','')}")

    # Theme toggle already shown; add a button to clear context
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.pending_booking = {}
        st.session_state.last_intent = None
        st.rerun()

    if st.button("Logout"):
        st.session_state.user = None
        st.session_state.chat_history = []
        st.session_state.pending_booking = {}
        st.session_state.last_intent = None
        st.rerun()

# ---------------- Display Chat History ----------------
for entry in st.session_state.chat_history:
    st.chat_message(entry["sender"]).markdown(entry["text"])

# Input
user_input = st.chat_input("Type your message here... (e.g., 'I have fever', 'Find hospitals in Kochi')")
if user_input:
    msg = user_input.strip()
    st.session_state.chat_history.append({"sender": "user", "text": msg})
    st.chat_message("user").markdown(msg)

    # Basic normalization
    msg_l = msg.lower()

    # Detect city mention
    detected_city = None
    for c in hospitals["city"].unique():
        if c.lower() in msg_l:
            detected_city = c
            st.session_state.city = c
            break

    # priority intent detection (greeting > symptom > explicit booking > doctor list > hospital list > fallback)
    response_text = ""
    intent = None

    # Greeting
    if any(w in msg_l for w in ["hi", "hello", "hey", "good morning", "good evening"]):
        intent = "greeting"
        response_text = (f"👋 Hello {user['name']}! I'm MedBot — I can help find hospitals, list doctors, "
                         "analyze symptoms, and assist with booking appointments. Tell me your city or symptoms to begin.")

    # Symptom detection
    elif any(w in msg_l for w in ["fever", "pain", "headache", "vomit", "cough", "rash", "nausea", "dizziness", "migraine"]):
        intent = "symptom"
        dept = predict_department_tf(msg)
        st.session_state.pending_booking = {"department": dept, "city": st.session_state.city or user.get("city")}
        # create medical-aware reply using medbot_reply
        generated = medbot_reply(msg)
        # include hospitals list
        city_for_search = st.session_state.pending_booking["city"]
        matches = hospitals[
            (hospitals["city"].str.lower() == (city_for_search or "").lower())
            & (hospitals["department"].str.lower() == dept.lower())
        ]
        response_text = f"🩺 {generated}\n\n"
        if not matches.empty:
            response_text += f"🏥 Hospitals for **{dept}** in **{city_for_search}**:\n"
            for _, r in matches.sort_values(by="rating", ascending=False).head(5).iterrows():
                response_text += f"- **{r['hospital_name']}** ({r['rating']}⭐)\n"
            response_text += "\nWould you like to book an appointment at one of these?"
        else:
            response_text += f"\nI couldn't find hospitals for {dept} in {city_for_search}."

    # Explicit booking
    elif any(w in msg_l for w in ["book", "appointment", "schedule", "reserve"]):
        intent = "booking"
        booking_info = st.session_state.pending_booking.copy() if st.session_state.pending_booking else {}
        # allow booking by specifying hospital name directly: "book in Lakeshore Hospital"
        selected_hospital = None
        for hn in hospitals["hospital_name"].unique():
            if hn.lower() in msg_l:
                selected_hospital = hospitals[hospitals["hospital_name"].str.lower() == hn.lower()].iloc[0]
                booking_info["city"] = selected_hospital["city"]
                break

        # parse date & time from message
        parsed_date = dateparser.parse(msg, settings={"PREFER_DATES_FROM": "future"})
        time_match = re.search(r"(\d{1,2})(?::|\.?)(\d{2})?\s*(am|pm)?", msg, re.IGNORECASE)
        parsed_time = None
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            ampm = time_match.group(3)
            if ampm:
                if ampm.lower() == "pm" and hour != 12:
                    hour += 12
                if ampm.lower() == "am" and hour == 12:
                    hour = 0
            parsed_time = f"{hour:02d}:{minute:02d}"

        # if user specified a department explicitly in the booking message, prefer it
        for d in hospitals["department"].unique():
            if d.lower() in msg_l:
                booking_info["department"] = d
                break

        # fallback values
        department = booking_info.get("department")
        city_search = booking_info.get("city") or st.session_state.city or user.get("city")

        if not department or not city_search:
            response_text = "I need the department and city to book. Please tell me what department (e.g., Cardiology) and your city."
        else:
            # find hospital list
            hosp_matches = hospitals[
                (hospitals["city"].str.lower() == city_search.lower())
                & (hospitals["department"].str.lower() == department.lower())
            ]
            if hosp_matches.empty:
                response_text = f"Sorry, I couldn't find any {department} hospitals in {city_search}."
            else:
                chosen = selected_hospital if selected_hospital is not None else hosp_matches.sort_values(by="rating", ascending=False).iloc[0]
                # choose a doctor available at chosen hospital & department
                available_docs = doctors[
                    (doctors["hospital_id"] == chosen["hospital_id"])
                    & (doctors["department"].str.lower() == department.lower())
                ]
                if available_docs.empty:
                    response_text = f"Sorry — no doctors listed for {department} at {chosen['hospital_name']}."
                else:
                    doc = available_docs.iloc[0]
                    # decide date/time for booking; must use parsed_date if provided else ask user
                    if parsed_date is None or parsed_time is None:
                        # ask for the missing info
                        missing = []
                        if parsed_date is None:
                            missing.append("date")
                        if parsed_time is None:
                            missing.append("time")
                        response_text = f"Please provide the {' and '.join(missing)} for the appointment (e.g., 'Dec 15 at 2 PM')."
                        # remember partial booking context
                        st.session_state.pending_booking.update({"hospital_id": chosen["hospital_id"], "department": department, "city": city_search})
                    else:
                        date_str = parsed_date.strftime("%Y-%m-%d")
                        time_str = parsed_time
                        # verify doctor availability window if start_time & end_time exist
                        try:
                            start_t = datetime.strptime(str(doc["start_time"]), "%H:%M").time()
                            end_t = datetime.strptime(str(doc["end_time"]), "%H:%M").time()
                            requested_t = datetime.strptime(time_str, "%H:%M").time()
                            if not (start_t <= requested_t <= end_t):
                                response_text = (
                                    f"⚠️ Dr. {doc['doctor_name']} is available between {start_t.strftime('%I:%M %p')} "
                                    f"and {end_t.strftime('%I:%M %p')}. Please pick a time in that range."
                                )
                            else:
                                booking_id = "A" + uuid.uuid4().hex[:6].upper()
                                booking = {
                                    "Booking ID": booking_id,
                                    "Patient": user["name"],
                                    "Age": user.get("age", "N/A"),
                                    "Gender": user.get("gender", "N/A"),
                                    "Hospital": chosen["hospital_name"],
                                    "Department": department,
                                    "Doctor": doc["doctor_name"],
                                    "Date": date_str,
                                    "Time": time_str,
                                }
                                os.makedirs("receipts", exist_ok=True)
                                pdf_path = os.path.join("receipts", f"{booking_id}.pdf")
                                generate_pdf(booking, pdf_path)
                                response_text = (
                                    f"✅ Appointment booked with **{doc['doctor_name']}** at **{chosen['hospital_name']}**.\n"
                                    f"📅 {date_str}  ⏰ {time_str}\n\nYou can download your receipt below."
                                )
                                try:
                                    with open(pdf_path, "rb") as f:
                                        st.download_button("📄 Download Receipt", f, file_name=f"{booking_id}.pdf")
                                except Exception:
                                    # fallback: show path
                                    response_text += f"\n(Receipt saved to {pdf_path})"
                                # clear pending booking
                                st.session_state.pending_booking = {}
                        except Exception:
                            # if time parsing/availability check fails, still create booking
                            booking_id = "A" + uuid.uuid4().hex[:6].upper()
                            booking = {
                                "Booking ID": booking_id,
                                "Patient": user["name"],
                                "Hospital": chosen["hospital_name"],
                                "Department": department,
                                "Doctor": doc["doctor_name"],
                                "Date": parsed_date.strftime("%Y-%m-%d") if parsed_date else str(datetime.now().date()),
                                "Time": parsed_time or "10:00",
                            }
                            os.makedirs("receipts", exist_ok=True)
                            pdf_path = os.path.join("receipts", f"{booking_id}.pdf")
                            generate_pdf(booking, pdf_path)
                            response_text = (
                                f"✅ Appointment booked with **{doc['doctor_name']}** at **{chosen['hospital_name']}**.\n"
                                f"📅 {booking['Date']}  ⏰ {booking['Time']}\n\nYou can download your receipt below."
                            )
                            try:
                                with open(pdf_path, "rb") as f:
                                    st.download_button("📄 Download Receipt", f, file_name=f"{booking_id}.pdf")
                            except Exception:
                                response_text += f"\n(Receipt saved to {pdf_path})"

    # Doctor listing
    elif any(w in msg_l for w in ["who is", "which doctors", "doctors in", "available doctors", "list doctors"]):
        intent = "doctor_list"
        dept = None
        for d in hospitals["department"].unique():
            if d.lower() in msg_l:
                dept = d
                break
        if dept is None and st.session_state.pending_booking:
            dept = st.session_state.pending_booking.get("department")
        if dept:
            city_search = st.session_state.city or user.get("city")
            docs = doctors[
                (doctors["department"].str.lower() == dept.lower())
                & (doctors["hospital_id"].isin(hospitals[hospitals["city"].str.lower() == (city_search or "").lower()]["hospital_id"]))
            ]
            if not docs.empty:
                response_text = f"👩‍⚕️ Doctors for **{dept}** in **{city_search}**:\n"
                for _, d in docs.head(10).iterrows():
                    hosp_name = hospitals[hospitals["hospital_id"] == d["hospital_id"]]["hospital_name"].iloc[0]
                    response_text += f"- **{d['doctor_name']}** ({hosp_name}) • {d.get('start_time','09:00')}-{d.get('end_time','17:00')} • ₹{d.get('fee','N/A')}\n"
                response_text += "\nWould you like to book an appointment with one of them?"
                st.session_state.pending_booking = {"department": dept, "city": city_search}
            else:
                response_text = f"No doctors found for {dept} in {city_search}."
        else:
            response_text = "Which department do you want doctors for? (e.g., Cardiology, Neurology)"

    # Hospital listing
    elif any(w in msg_l for w in ["hospital", "hospitals", "nearby"]):
        intent = "hospital_list"
        city_search = st.session_state.city or user.get("city")
        if detected_city:
            city_search = detected_city
        if city_search:
            matches = hospitals[hospitals["city"].str.lower() == city_search.lower()]
            if not matches.empty:
                response_text = f"🏥 Hospitals available in **{city_search}**:\n"
                for _, r in matches.head(10).iterrows():
                    response_text += f"- **{r['hospital_name']}** ({r['department']}, ⭐{r['rating']})\n"
                response_text += "\nWould you like to book an appointment at one of them?"
            else:
                response_text = f"No hospitals found in {city_search}."
        else:
            response_text = "Please tell me your city to find nearby hospitals (e.g., 'I'm in Kochi')."

    # Fallback / general conversation -> use medbot_reply (LLM or templated)
    else:
        intent = "general"
        response_text = medbot_reply(msg)

    # append assistant response and render
    st.session_state.chat_history.append({"sender": "assistant", "text": response_text})
    st.chat_message("assistant").markdown(response_text)
