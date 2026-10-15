# Student Performance Prediction Using Concept Learning

An academic analytics and machine learning web application built for college engineering/data science curricula. The project uses Tom Mitchell's **Find-S Concept Learning algorithm** to classify students into performance cohorts, evaluate their academic parameters against general-to-specific hypothesis constraints, and prescribe tailored pedagogical interventions.

---

## 🏛️ Project Overview

- **Project Title:** Student Performance Prediction Using Concept Learning
- **Target Concept:** $c(x) = 1$ (`Good Performer`) vs $c(x) = 0$ (`Needs Academic Support`)
- **Core ML Algorithm:** Find-S (Find-Specific Concept Learning)
- **Primary Tech Stack:**
  - **Frontend:** Semantic HTML5, Vanilla CSS3 (Custom Academic Theme), Vanilla JavaScript (ES6+), Chart.js
  - **Backend:** Python Flask 3.x
  - **Machine Learning Engine:** Pure Python concept learning with attribute discretization, step-by-step hypothesis evolution, and empirical confusion matrix evaluation

---

## 🎨 Design & Aesthetics

- **Color Palette:** Premium Navy-Blue (`#0a192f`, `#0d233a`), Crisp White (`#ffffff`), Sky/Ice Blue (`#f0f7ff`, `#e0f2fe`), and subtle Emerald/Amber indicators.
- **Layout:** Fully responsive across mobile smartphones, tablets, and desktop workstations.
- **Components:** Interactive sidebar, live KPI cards, Chart.js donut distribution, transparent hypothesis vector inspection chips, searchable/paginated student records repository, and pedagogical algorithm training trace.

---

## 📂 Project Directory Structure

```text
STUDENT PERFORMANCE PREDICTION USING CONCEPT LEARNING/
├── app.py                      # Flask backend application & REST API routes
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & run guide
│
├── model/                      # Decoupled Machine Learning package
│   ├── __init__.py             # Model package exports
│   ├── concept_learning.py     # Find-S algorithm, discretization & evaluation logic
│   └── dataset.py              # Sample student dataset generator & loader
│
├── data/
│   └── students_dataset.csv    # Labeled student academic records (120 samples)
│
├── static/
│   ├── css/
│   │   └── style.css           # Premium academic analytics stylesheet
│   └── js/
│       └── app.js              # Frontend controller, API calls & Chart.js logic
│
└── templates/
    └── index.html              # Modern, accessible dashboard UI
```

---

## 🧮 Concept Learning & The Find-S Algorithm

### 1. Attribute Formulation & Discretization

Continuous academic metrics are mapped into discrete conceptual attributes:

| Feature Parameter | Scale | Discretization Benchmark | Conceptual States |
| :--- | :--- | :--- | :--- |
| **Attendance** | 0–100% | $\ge 75.0\%$ | `Regular` vs `Irregular` |
| **Internal Assessment** | 0–50 | $\ge 30.0$ | `Satisfactory` vs `Unsatisfactory` |
| **Assignment Completion** | 0–100% | $\ge 70.0\%$ | `Completed` vs `Incomplete` |
| **Daily Study Hours** | 0–24 hrs | $\ge 2.5\text{ hrs}$ | `Sufficient` vs `Insufficient` |
| **Previous Exam Marks** | 0–100% | $\ge 60.0\%$ | `Passed` vs `BelowAverage` |

### 2. Find-S Generalization Process

1. Initialize hypothesis $h$ to the most specific null vector:
   $$h_0 = \langle \emptyset, \emptyset, \emptyset, \emptyset, \emptyset \rangle$$
2. For each positive training instance $x$ where $c(x) = \text{Good Performer}$:
   - For each attribute constraint $a_i$ in $h$:
     - If $a_i = \emptyset$, update $a_i = x_i$
     - If $a_i \ne x_i$ and $a_i \ne '?'$, generalize to $a_i = '?'$
3. Negative training instances ($label = \text{Needs Academic Support}$) are ignored.
4. Final learned hypothesis:
   $$h = \langle \text{High}, \text{High}, ?, \text{High}, \text{High} \rangle$$

---

## 🔌 REST API Specification: `POST /predict`

The web application connects the interactive frontend to the Flask backend via `POST /predict`.

### Request Payload (JSON)
```json
{
  "student_id": "STU-2026",
  "attendance": 88.0,
  "internal_marks": 42.0,
  "assignment_completion": 92.0,
  "study_hours": 4.0,
  "previous_exam_marks": 80.0
}
```

### Response Payload (JSON)
```json
{
  "status": "success",
  "student_id": "STU-2026",
  "predicted_category": "Good Performer",
  "input_features_used": {
    "numeric": {
      "attendance": 88.0,
      "internal_marks": 42.0,
      "assignment_completion": 92.0,
      "study_hours": 4.0,
      "previous_exam_marks": 80.0
    },
    "categorical": {
      "Attendance": "High",
      "Internal Assessment": "High",
      "Study Hours": "High",
      "Assignment Completion": "High",
      "Previous Examination Marks": "High"
    }
  },
  "learned_hypothesis": ["High", "High", "?", "High", "High"],
  "formatted_hypothesis": "⟨ High, High, ?, High, High ⟩",
  "explanation": "The student's features satisfy all constraints in the learned Find-S hypothesis. Classified as 'Good Performer'.",
  "academic_suggestion": "Maintain consistent study routines, explore advanced elective topics, and consider peer mentoring or undergraduate research participation.",
  "matches_hypothesis": true,
  "disclaimer": "This prediction is an educational estimate derived from the symbolic Find-S Concept Learning algorithm on synthetic demonstration data."
}
```

---

## 🚀 How to Run the Project

### Prerequisites
Make sure **Python 3.10+** is installed on your computer.

### Step 1: Open Terminal in the Project Directory
Navigate to the project root directory:
```bash
cd "c:\Users\lsail\OneDrive\Desktop\STUDENT PERFORMANCE PREDICTION USING CONCEPT LEARNING"
```

### Step 2: Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

### Step 3: Run the Automated Unit Tests
Verify that all ML components, threshold transformations, and API endpoints pass:
```bash
python -m unittest test_concept_learning.py
```

### Step 4: Start the Flask Application
```bash
python app.py
```

### Step 5: Access the Website
Open your web browser (Chrome, Edge, Firefox, Safari) and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🔍 Features Walkthrough

1. **Dashboard:**
   - Displays project title and academic context.
   - Shows live counts: Total Students (120), Good Performers, and Needs Academic Support.
   - Live Doughnut chart for cohort distribution.
   - Current learned hypothesis vector: `⟨ High, High, ?, High, High ⟩`.

2. **Student Prediction Form:**
   - Input fields: Student ID (optional), Attendance %, Internal Marks (/50), Assignment Completion %, Study Hours/day, and Previous Exam Marks %.
   - Preset buttons to instantly populate **High Achiever**, **Steady Performer**, or **Struggling Student** profiles for laboratory demonstrations.
   - Real-time client and backend validation.
   - Asynchronous AJAX `fetch` submission displaying loading spinner without refreshing the page.

3. **Prediction Results:**
   - Predicted category badge (`Good Performer` vs `Needs Academic Support`).
   - Transparent feature breakdown table indicating raw input, categorical conversion, hypothesis constraint, and satisfied/violated status.
   - Actionable pedagogical academic improvement suggestions.
   - Clear educational estimate disclaimer.

4. **Student Records:**
   - Searchable and filterable table of sample student records with pagination.
   - **Load in Predictor** button on each row to test any existing record with one click.

5. **About the Model:**
   - Pedagogical explanations of Concept Learning and Find-S.
   - Interactive step-by-step training trace table showing hypothesis updates for each training example.
   - Scientific limitations of Find-S (negative instance exclusion, noise sensitivity).
