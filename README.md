# Automated Resume Evaluator using NLP

This project is a Streamlit-based web application that evaluates resumes using Natural Language Processing (NLP). It analyzes resumes based on skills, keywords, and job description relevance to help identify the best candidates.

## 🚀 Features

* Upload multiple resume PDFs
* Extracts text from resumes automatically
* Matches resumes with job description using NLP
* Skill matching based on required skills
* Provides similarity score and ranking
* Displays shortlisted candidates
* Generates ATS feedback (resume length check)
* Download results as CSV file

## 🛠️ Tech Stack

* Python
* Streamlit
* NLTK
* Scikit-learn
* Pandas
* Plotly
* PyPDF2

## 📂 Project Structure

resume-evaluator-nlp/
│
├── app.py
├── requirements.txt
├── README.md

## ▶️ How to Run

1. Clone the repository:

git clone https://github.com/Jayadev2728/resume-evaluator-nlp.git
cd resume-evaluator-nlp

2. Install dependencies:

pip install -r requirements.txt

3. Run the application:

streamlit run app.py

## 📌 How to Use

1. Upload one or more resume PDFs
2. Paste the job description
3. Enter required skills (comma-separated)
4. Click "Process Resumes"
5. View results and download CSV

## 📊 Output

The system provides:

* Similarity Score (%)
* Skill Match (%)
* Composite Score (%)
* Shortlisted Candidates
* Candidate Ranking
* ATS Feedback

## 📎 Future Improvements

* Advanced skill extraction using NLP
* Resume formatting analysis
* Integration with job portals
* UI enhancements

## 👤 Author

Jayadev H N
Aspiring Web Developer
