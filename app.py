"""
app.py
======
Flask Application for Student Performance Prediction Using Concept Learning.
Connects the web frontend to the Find-S machine learning algorithm in concept_learning.py.
"""

from flask import Flask, render_template, request, jsonify
from concept_learning import (
    FindSConceptLearner,
    convert_numeric_to_categorical,
    get_sample_training_data,
    get_find_s_limitations,
    DEFAULT_THRESHOLDS,
    POSITIVE_LABEL,
    NEGATIVE_LABEL,
    FEATURE_NAMES
)
from model.dataset import load_dataset

app = Flask(__name__)

# Initialize and train the Find-S Concept Learning model on the demonstration dataset
find_s_model = FindSConceptLearner()
sample_training_data = get_sample_training_data()
find_s_model.fit(sample_training_data)

# Preload student records repository for table browsing
student_records = load_dataset()


@app.route("/")
def index():
    """Renders the main academic analytics dashboard."""
    return render_template("index.html")


@app.route("/api/summary", methods=["GET"])
def get_summary():
    """
    Returns summary metrics, hypothesis state, and distribution statistics.
    NOTE: Data is synthetic and used strictly for academic demonstration.
    """
    total = len(student_records)
    good_count = sum(1 for r in student_records if r["label"] == POSITIVE_LABEL)
    support_count = sum(1 for r in student_records if r["label"] == NEGATIVE_LABEL)

    good_pct = round((good_count / total) * 100, 1) if total > 0 else 0
    support_pct = round((support_count / total) * 100, 1) if total > 0 else 0

    return jsonify({
        "status": "success",
        "project": {
            "title": "Student Performance Prediction Using Concept Learning",
            "subtitle": "College Machine Learning Project Dashboard",
            "description": (
                "An educational machine learning analytics dashboard demonstrating Tom Mitchell's "
                "Find-S Concept Learning algorithm to identify performance categories and provide "
                "targeted academic improvement suggestions."
            )
        },
        "metrics": {
            "total_students": total,
            "good_performers": good_count,
            "good_performers_pct": good_pct,
            "support_needed": support_count,
            "support_needed_pct": support_pct,
            "training_samples": len(sample_training_data),
            "feature_count": len(find_s_model.feature_names)
        },
        "distribution": {
            "labels": [POSITIVE_LABEL, NEGATIVE_LABEL],
            "counts": [good_count, support_count],
            "percentages": [good_pct, support_pct]
        },
        "learned_hypothesis": {
            "vector": find_s_model.hypothesis,
            "formatted": f"⟨ {', '.join(find_s_model.hypothesis)} ⟩",
            "features": find_s_model.feature_names
        },
        "thresholds": DEFAULT_THRESHOLDS,
        "notice": "All data is synthetic and provided for educational demonstration only."
    })


def process_prediction(data: dict):
    """
    Core validation and prediction pipeline:
    1. Validates numeric inputs against allowable bounds.
    2. Converts numeric inputs to categorical features ('High'/'Low').
    3. Runs the Find-S prediction algorithm.
    4. Constructs the JSON response payload.
    """
    errors = {}

    # 1. Student ID (optional)
    student_id = str(data.get("student_id", "")).strip()
    if not student_id:
        student_id = "STU-PROSPECTIVE"

    # 2. Attendance (0 to 100%)
    try:
        attendance = float(data.get("attendance", ""))
        if not (0.0 <= attendance <= 100.0):
            errors["attendance"] = "Attendance must be between 0% and 100%."
    except (TypeError, ValueError):
        errors["attendance"] = "Please enter a valid numeric value for attendance (0–100)."

    # 3. Internal Assessment Marks (0 to 50)
    try:
        internal_marks = float(data.get("internal_marks", ""))
        if not (0.0 <= internal_marks <= 50.0):
            errors["internal_marks"] = "Internal assessment marks must be between 0 and 50."
    except (TypeError, ValueError):
        errors["internal_marks"] = "Please enter a valid numeric value for internal marks (0–50)."

    # 4. Assignment Completion (0 to 100%)
    try:
        assignment_completion = float(data.get("assignment_completion", ""))
        if not (0.0 <= assignment_completion <= 100.0):
            errors["assignment_completion"] = "Assignment completion percentage must be between 0% and 100%."
    except (TypeError, ValueError):
        errors["assignment_completion"] = "Please enter a valid numeric value for assignment completion (0–100)."

    # 5. Study Hours Per Day (0 to 24 hrs)
    try:
        study_hours = float(data.get("study_hours", ""))
        if not (0.0 <= study_hours <= 24.0):
            errors["study_hours"] = "Study hours must be between 0 and 24 hours per day."
    except (TypeError, ValueError):
        errors["study_hours"] = "Please enter a valid numeric value for study hours (0–24)."

    # 6. Previous Examination Marks (0 to 100%)
    previous_exam_marks = None
    prev_raw = data.get("previous_exam_marks")
    if prev_raw is not None and str(prev_raw).strip() != "":
        try:
            previous_exam_marks = float(prev_raw)
            if not (0.0 <= previous_exam_marks <= 100.0):
                errors["previous_exam_marks"] = "Previous exam marks must be between 0% and 100%."
        except (TypeError, ValueError):
            errors["previous_exam_marks"] = "Please enter a valid numeric value for previous exam marks (0–100)."

    # Return validation errors if any field is invalid
    if errors:
        return jsonify({
            "status": "validation_error",
            "message": "Input validation failed. Please check the provided values.",
            "errors": errors
        }), 422

    try:
        # Step 2: Convert numeric inputs into categorical features ('High' or 'Low')
        categorical_features = convert_numeric_to_categorical(
            attendance=attendance,
            internal_assessment=internal_marks,
            study_hours=study_hours,
            assignment_completion=assignment_completion,
            previous_exam_marks=previous_exam_marks,
            thresholds=DEFAULT_THRESHOLDS
        )

        # Step 3: Run the Find-S prediction model
        prediction_result = find_s_model.predict(categorical_features)

        # Step 4: Assemble comprehensive academic suggestions list
        suggestions = []
        if categorical_features.get("Attendance") == "Low":
            suggestions.append({
                "category": "Class Attendance & Engagement",
                "severity": "critical" if attendance < 65 else "moderate",
                "current": f"{attendance}% (Benchmark: ≥ {DEFAULT_THRESHOLDS['attendance']}%)",
                "advice": (
                    f"Current attendance is {attendance}%, categorized as 'Low'. Regular lecture attendance "
                    f"is vital to cross the {DEFAULT_THRESHOLDS['attendance']}% benchmark and satisfy hypothesis constraints."
                )
            })

        if categorical_features.get("Internal Assessment") == "Low":
            suggestions.append({
                "category": "Internal Assessment Marks",
                "severity": "critical" if internal_marks < 25 else "moderate",
                "current": f"{internal_marks} / 50 (Benchmark: ≥ {DEFAULT_THRESHOLDS['internal_assessment']}/50)",
                "advice": (
                    f"Internal assessment score is {internal_marks}/50, categorized as 'Low'. Participate in faculty "
                    f"consultation hours and review mid-term problem sets to achieve at least {DEFAULT_THRESHOLDS['internal_assessment']}/50."
                )
            })

        if categorical_features.get("Assignment Completion") == "Low":
            suggestions.append({
                "category": "Assignments & Coursework",
                "severity": "critical" if assignment_completion < 50 else "moderate",
                "current": f"{assignment_completion}% (Benchmark: ≥ {DEFAULT_THRESHOLDS['assignment_completion']}%)",
                "advice": (
                    f"Assignment completion rate of {assignment_completion}% is categorized as 'Low'. "
                    f"Submit coursework on time to satisfy the {DEFAULT_THRESHOLDS['assignment_completion']}% benchmark."
                )
            })

        if categorical_features.get("Study Hours") == "Low":
            suggestions.append({
                "category": "Daily Independent Study",
                "severity": "moderate",
                "current": f"{study_hours} hrs/day (Benchmark: ≥ {DEFAULT_THRESHOLDS['study_hours']} hrs/day)",
                "advice": (
                    f"Daily study time of {study_hours} hours is categorized as 'Low'. Target at least "
                    f"{DEFAULT_THRESHOLDS['study_hours']} hours of focused daily study using Pomodoro blocks."
                )
            })

        if categorical_features.get("Previous Examination Marks") == "Low":
            suggestions.append({
                "category": "Previous Examination Preparation",
                "severity": "critical" if (previous_exam_marks or 0) < 50 else "moderate",
                "current": f"{previous_exam_marks}% (Benchmark: ≥ {DEFAULT_THRESHOLDS['previous_exam_marks']}%)",
                "advice": (
                    f"Previous examination mark of {previous_exam_marks}% is categorized as 'Low'. "
                    f"Review prerequisite core concepts and practice previous exam question papers."
                )
            })

        if not suggestions:
            suggestions.append({
                "category": "Academic Excellence & Enrichment",
                "severity": "positive",
                "current": "All measured indicators categorized as 'High'",
                "advice": (
                    "The student satisfies all hypothesis criteria. Maintain consistent study schedules, "
                    "explore advanced elective topics, and consider peer tutoring."
                )
            })

        # Assemble input features used dictionary
        numeric_inputs = {
            "attendance": attendance,
            "internal_marks": internal_marks,
            "assignment_completion": assignment_completion,
            "study_hours": study_hours
        }
        if previous_exam_marks is not None:
            numeric_inputs["previous_exam_marks"] = previous_exam_marks

        input_features_used = {
            "numeric": numeric_inputs,
            "categorical": categorical_features
        }

        # Step 5: Return complete JSON response with all requested fields
        # - Predicted category
        # - Input features used
        # - Learned hypothesis
        # - Simple explanation
        # - Appropriate academic suggestion
        return jsonify({
            "status": "success",
            "student_id": student_id,
            "predicted_category": prediction_result["predicted_category"],
            "prediction": prediction_result["prediction"],
            "input_features_used": input_features_used,
            "raw_inputs": numeric_inputs,
            "categorical_features": categorical_features,
            "learned_hypothesis": prediction_result["learned_hypothesis"],
            "formatted_hypothesis": prediction_result["formatted_hypothesis"],
            "explanation": prediction_result["explanation"],
            "academic_suggestion": prediction_result["academic_suggestion"],
            "suggestions": suggestions,
            "feature_breakdown": prediction_result["feature_breakdown"],
            "matches_hypothesis": prediction_result["matches_hypothesis"],
            "disclaimer": (
                "This prediction is an educational estimate derived from the symbolic Find-S Concept Learning "
                "algorithm on synthetic demonstration data. It does not represent a guaranteed or "
                "scientifically proven academic outcome."
            )
        }), 200

    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": f"An unexpected error occurred during prediction: {str(exc)}"
        }), 500


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    """
    POST API Endpoint for Student Performance Prediction.
    Accepts JSON payload with student indicators and returns the Find-S prediction.
    Accessible via both /predict and /api/predict.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "status": "error",
            "message": "Invalid request payload. Expected JSON data."
        }), 400

    return process_prediction(data)


@app.route("/api/students", methods=["GET"])
def get_students():
    """Returns student records for table browsing, search, and filtering."""
    query = request.args.get("query", "").strip().lower()
    label_filter = request.args.get("label", "").strip()
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 10))

    filtered = student_records

    if label_filter and label_filter.lower() != "all":
        filtered = [r for r in filtered if r["label"].lower() == label_filter.lower()]

    if query:
        filtered = [
            r for r in filtered
            if query in r["student_id"].lower() or query in r["label"].lower()
        ]

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size

    return jsonify({
        "status": "success",
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 1,
        "records": filtered[start:end]
    })


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """
    Returns pedagogical explanations of Concept Learning, the Find-S algorithm,
    training step trace, and theoretical limitations.
    """
    return jsonify({
        "status": "success",
        "concept_learning_theory": {
            "title": "Concept Learning in Machine Learning",
            "summary": (
                "Concept learning is the task of inferring a boolean-valued function from training "
                "examples of its input and output attributes. The algorithm searches through the "
                "hypothesis space H to identify a hypothesis consistent with training instances."
            ),
            "target_concept": f"c(x) = '{POSITIVE_LABEL}' vs c(x) = '{NEGATIVE_LABEL}'",
            "hypothesis_representation": (
                "Hypothesis h is represented as a conjunction of attribute constraints: "
                f"⟨ {', '.join(find_s_model.feature_names)} ⟩."
            )
        },
        "find_s_algorithm": {
            "name": "Find-S (Find-Specific Algorithm)",
            "procedure": [
                f"1. Initialize hypothesis h to the most specific hypothesis: h = ⟨ {', '.join(['Ø']*len(find_s_model.feature_names))} ⟩.",
                "2. For each positive training instance x:",
                "   a. For each attribute constraint a_i in h:",
                "      i. If a_i is satisfied by x, do nothing.",
                "      ii. Else replace a_i with the next more general constraint (if a_i = Ø, set a_i = x_i; if a_i != x_i, set a_i = '?').",
                "3. Negative training instances are completely ignored.",
                "4. Output the final learned hypothesis h."
            ],
            "learned_hypothesis": find_s_model.hypothesis,
            "formatted_hypothesis": f"⟨ {', '.join(find_s_model.hypothesis)} ⟩"
        },
        "training_trace": find_s_model.training_steps,
        "thresholds": DEFAULT_THRESHOLDS,
        "limitations": get_find_s_limitations(),
        "synthetic_notice": (
            "Training data is synthetic and designed strictly for academic demonstration "
            "of the Find-S concept learning algorithm."
        )
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
