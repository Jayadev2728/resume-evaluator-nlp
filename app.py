import streamlit as st
import PyPDF2
import re, string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# -------------------- Setup --------------------

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

st.set_page_config(
    page_title="Automated Resume Evaluator",
    layout="wide"
)

# -------------------- Styling --------------------

st.markdown("""
<style>
.big-number {font-size:28px;font-weight:700;}
.card {padding:16px;border-radius:12px;background:#0f1720;color:white;}
</style>
""", unsafe_allow_html=True)

# -------------------- Functions --------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "\n".join(text)
    except:
        return ""


def extract_contact_info(text):
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    phones = re.findall(r'(?:\+?\d{1,3}[\s-]?)?(?:\d{10})', text)

    email = emails[0] if emails else ""
    phone = phones[0] if phones else ""

    return email, phone


def skill_match_list(resume_text, required_skills):

    resume_words = set(resume_text.split())

    matched = [s for s in required_skills if s in resume_words]
    missing = [s for s in required_skills if s not in resume_words]

    score = 0

    if required_skills:
        score = round(len(matched) / len(required_skills) * 100, 2)

    return matched, missing, score


def ats_feedback(text):

    words = len(text.split())

    if words < 200:
        return "Resume is too short"
    elif words > 2000:
        return "Resume is too long"
    else:
        return "Resume length looks good"


# -------------------- Sidebar --------------------

st.sidebar.header("Settings")

required_skills_input = st.sidebar.text_area(
    "Required Skills (comma separated)",
    value="python, machine learning, sql"
)

threshold = st.sidebar.slider(
    "Shortlist Threshold %",
    0, 100, 40
)

# -------------------- UI --------------------

st.title("Automated Resume Evaluator")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

job_description = st.text_area("Paste Job Description")


# -------------------- Processing --------------------

if st.button("Process Resumes"):

    if not uploaded_files:
        st.warning("Upload resumes first")
        st.stop()

    if not job_description:
        st.warning("Enter job description")
        st.stop()

    records = []

    raw_texts = []

    for f in uploaded_files:

        raw = extract_text_from_pdf(f)

        clean = clean_text(raw)

        email, phone = extract_contact_info(raw)

        records.append({
            "filename": f.name,
            "raw_text": raw,
            "clean_text": clean,
            "email": email,
            "phone": phone
        })

        raw_texts.append(clean)

    clean_jd = clean_text(job_description)

    corpus = raw_texts + [clean_jd]

    vectorizer = TfidfVectorizer(max_features=5000)

    tfidf = vectorizer.fit_transform(corpus)

    jd_vec = tfidf[-1]

    scores = cosine_similarity(tfidf[:-1], jd_vec).flatten()

    required_skills = [s.strip().lower() for s in required_skills_input.split(",")]

    rows = []

    for i, r in enumerate(records):

        matched, missing, skill_score = skill_match_list(
            r["clean_text"],
            required_skills
        )

        similarity = round(scores[i] * 100, 2)

        composite = round((similarity * 0.7) + (skill_score * 0.3), 2)

        shortlisted = composite >= threshold

        rows.append({
            "Name": r["filename"],
            "Email": r["email"],
            "Phone": r["phone"],
            "Similarity %": similarity,
            "Skill Match %": skill_score,
            "Composite Score %": composite,
            "Shortlisted": shortlisted,
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing),
            "ATS Feedback": ats_feedback(r["raw_text"])
        })

    df = pd.DataFrame(rows)

    df_sorted = df.sort_values(
        by="Composite Score %",
        ascending=False
    )

# -------------------- Metrics --------------------

    total = len(df_sorted)

    shortlisted_count = df_sorted["Shortlisted"].sum()

    rate = round(shortlisted_count / total * 100, 2)

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Candidates", total)
    c2.metric("Shortlisted", shortlisted_count)
    c3.metric("Shortlist Rate", f"{rate}%")

# -------------------- Best Candidate --------------------

    best = df_sorted.iloc[0]

    st.success(
        f"Best Candidate: {best['Name']} | Score: {best['Composite Score %']}%"
    )

# -------------------- Top Candidates --------------------

    st.subheader("Top 5 Candidates")

    st.dataframe(
        df_sorted.head(5)[[
            "Name",
            "Email",
            "Composite Score %",
            "Similarity %",
            "Skill Match %",
            "Shortlisted"
        ]]
    )

# -------------------- Score Chart --------------------

    fig_bar = px.bar(
        df_sorted,
        x="Name",
        y="Composite Score %",
        title="Candidate Ranking"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

# -------------------- Score Distribution --------------------

    st.subheader("Score Distribution")

    fig_hist = px.histogram(
        df_sorted,
        x="Composite Score %",
        nbins=10
    )

    st.plotly_chart(fig_hist, use_container_width=True)

# -------------------- Candidate Search --------------------

    st.subheader("Search Candidate")

    search = st.text_input("Enter candidate filename")

    if search:
        filtered = df_sorted[
            df_sorted["Name"].str.contains(search, case=False)
        ]
        st.dataframe(filtered)

# -------------------- Shortlisted Table --------------------

    st.subheader("Shortlisted Candidates")

    st.dataframe(
        df_sorted[df_sorted["Shortlisted"] == True]
    )

# -------------------- Full Results --------------------

    st.subheader("All Candidates")

    st.dataframe(df_sorted)

# -------------------- CSV Export --------------------

    csv = df_sorted.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Results CSV",
        csv,
        "resume_results.csv",
        "text/csv"
    )