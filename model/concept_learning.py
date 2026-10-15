"""
Concept Learning Engine using the Find-S Algorithm.
Implements attribute discretization, hypothesis generalization,
step-by-step training trace, prediction, feature influence explanation,
and empirical evaluation metrics.
"""

from typing import Dict, List, Tuple, Any
from .dataset import load_dataset


FEATURE_KEYS = [
    "attendance",
    "internal_marks",
    "assignment_completion",
    "study_hours",
    "previous_exam_marks"
]

FEATURE_LABELS = {
    "attendance": "Attendance Percentage",
    "internal_marks": "Internal Assessment Marks",
    "assignment_completion": "Assignment Completion",
    "study_hours": "Daily Study Hours",
    "previous_exam_marks": "Previous Exam Marks"
}

# Thresholds for discretizing continuous metrics into conceptual states
THRESHOLDS = {
    "attendance": {
        "benchmark": 75.0,
        "high_label": "Regular",
        "low_label": "Irregular",
        "unit": "%",
        "max": 100.0
    },
    "internal_marks": {
        "benchmark": 30.0,
        "high_label": "Satisfactory",
        "low_label": "Unsatisfactory",
        "unit": " marks (out of 50)",
        "max": 50.0
    },
    "assignment_completion": {
        "benchmark": 70.0,
        "high_label": "Completed",
        "low_label": "Incomplete",
        "unit": "%",
        "max": 100.0
    },
    "study_hours": {
        "benchmark": 2.5,
        "high_label": "Sufficient",
        "low_label": "Insufficient",
        "unit": " hrs/day",
        "max": 24.0
    },
    "previous_exam_marks": {
        "benchmark": 60.0,
        "high_label": "Passed",
        "low_label": "BelowAverage",
        "unit": "%",
        "max": 100.0
    }
}

POSITIVE_CLASS = "Good Performer"
NEGATIVE_CLASS = "Needs Academic Support"


def discretize_instance(record: Dict[str, Any]) -> List[str]:
    """
    Map raw numerical student attributes into discrete conceptual categories.
    Returns: [attendance_cat, internal_cat, assignment_cat, study_hours_cat, prev_exam_cat]
    """
    concept_vector = []
    
    # 1. Attendance (>= 75.0% -> Regular, else Irregular)
    att = float(record["attendance"])
    concept_vector.append("Regular" if att >= THRESHOLDS["attendance"]["benchmark"] else "Irregular")

    # 2. Internal Assessment Marks (>= 30.0 / 50 -> Satisfactory, else Unsatisfactory)
    im = float(record["internal_marks"])
    concept_vector.append("Satisfactory" if im >= THRESHOLDS["internal_marks"]["benchmark"] else "Unsatisfactory")

    # 3. Assignment Completion (>= 70.0% -> Completed, else Incomplete)
    ac = float(record["assignment_completion"])
    concept_vector.append("Completed" if ac >= THRESHOLDS["assignment_completion"]["benchmark"] else "Incomplete")

    # 4. Study Hours (>= 2.5 hrs/day -> Sufficient, else Insufficient)
    sh = float(record["study_hours"])
    concept_vector.append("Sufficient" if sh >= THRESHOLDS["study_hours"]["benchmark"] else "Insufficient")

    # 5. Previous Exam Marks (>= 60.0% -> Passed, else BelowAverage)
    pm = float(record["previous_exam_marks"])
    concept_vector.append("Passed" if pm >= THRESHOLDS["previous_exam_marks"]["benchmark"] else "BelowAverage")

    return concept_vector


class FindSConceptLearner:
    """
    Find-S (Find-Specific) Concept Learning Algorithm.
    Operates on attribute-value representations to find the most specific
    hypothesis consistent with all positive training examples.
    """

    def __init__(self):
        self.hypothesis: List[str] = ["Ø"] * len(FEATURE_KEYS)
        self.training_trace: List[Dict[str, Any]] = []
        self.is_trained: bool = False
        self.evaluation_metrics: Dict[str, Any] = {}
        self.total_records_count: int = 0
        self.train_count: int = 0
        self.test_count: int = 0

    def fit(self, training_data: List[Dict[str, Any]]) -> "FindSConceptLearner":
        """
        Train the Find-S model on labeled data.
        Find-S only updates hypothesis from positive instances (POSITIVE_CLASS).
        """
        self.hypothesis = ["Ø"] * len(FEATURE_KEYS)
        self.training_trace = []
        
        step_number = 1
        for record in training_data:
            discrete_vector = discretize_instance(record)
            label = record["label"]
            is_positive = (label == POSITIVE_CLASS)
            
            prev_hypothesis = list(self.hypothesis)
            action = "Ignored (Negative example - Find-S disregards negative instances)"

            if is_positive:
                updated = False
                for i in range(len(self.hypothesis)):
                    if self.hypothesis[i] == "Ø":
                        self.hypothesis[i] = discrete_vector[i]
                        updated = True
                    elif self.hypothesis[i] != discrete_vector[i] and self.hypothesis[i] != "?":
                        self.hypothesis[i] = "?"
                        updated = True

                action = "Generalization applied" if updated else "Consistent with current hypothesis"

            # Log trace for educational transparency
            if len(self.training_trace) < 25:  # Keep informative sample steps for UI display
                self.training_trace.append({
                    "step": step_number,
                    "student_id": record.get("student_id", f"STU-{step_number}"),
                    "features_raw": {k: record[k] for k in FEATURE_KEYS},
                    "features_discrete": discrete_vector,
                    "label": label,
                    "is_positive": is_positive,
                    "previous_hypothesis": prev_hypothesis,
                    "current_hypothesis": list(self.hypothesis),
                    "action_description": action
                })
            step_number += 1

        self.is_trained = True
        return self

    def predict_instance(self, discrete_vector: List[str]) -> Tuple[str, bool, List[Dict[str, Any]]]:
        """
        Predict performance category given a discrete attribute vector.
        Returns:
          - category: "Good Performer" or "Needs Academic Support"
          - matches_all: bool
          - feature_analysis: list of analysis dicts for each attribute
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        feature_analysis = []
        matches_all = True

        for i, key in enumerate(FEATURE_KEYS):
            h_val = self.hypothesis[i]
            x_val = discrete_vector[i]
            matched = (h_val == "?" or h_val == x_val)

            if not matched:
                matches_all = False

            feature_analysis.append({
                "feature_key": key,
                "feature_name": FEATURE_LABELS[key],
                "student_value": x_val,
                "hypothesis_constraint": h_val,
                "satisfied": matched
            })

        prediction = POSITIVE_CLASS if matches_all else NEGATIVE_CLASS
        return prediction, matches_all, feature_analysis

    def predict_raw(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-level prediction from raw continuous attributes.
        Returns complete diagnostic bundle with educational guidance.
        """
        discrete_vector = discretize_instance(record)
        category, matches_all, feature_analysis = self.predict_instance(discrete_vector)

        # Generate intelligent recommendations based on attribute deficits
        suggestions = self.generate_improvement_suggestions(record, feature_analysis)

        return {
            "prediction": category,
            "status": "success",
            "matches_hypothesis": matches_all,
            "learned_hypothesis": self.hypothesis,
            "feature_analysis": feature_analysis,
            "discrete_vector": discrete_vector,
            "suggestions": suggestions,
            "confidence_note": (
                "Based on the empirical concept hypothesis learned via Find-S. "
                "All constraints must be satisfied to guarantee classification as a Good Performer."
            ),
            "disclaimer": (
                "This prediction is an educational estimate derived from the symbolic Find-S Concept Learning "
                "algorithm on academic indicators. It is not a guaranteed academic outcome."
            )
        }

    def generate_improvement_suggestions(self, record: Dict[str, Any], feature_analysis: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Generate actionable academic improvement suggestions tailored
        specifically to the features that fell short or need reinforcement.
        """
        suggestions = []

        for item in feature_analysis:
            key = item["feature_key"]
            raw_val = float(record[key])
            bench = THRESHOLDS[key]["benchmark"]
            satisfied = item["satisfied"]

            if not satisfied or raw_val < bench:
                if key == "attendance":
                    suggestions.append({
                        "category": "Attendance & Classroom Engagement",
                        "severity": "critical" if raw_val < 65 else "moderate",
                        "current": f"{raw_val}% (Benchmark: {bench}%)",
                        "advice": (
                            f"Current attendance is {raw_val}%. Attend upcoming lectures and tutorials consistently "
                            f"to cross the {bench}% threshold. Active lecture participation directly reinforces continuous learning."
                        )
                    })
                elif key == "internal_marks":
                    suggestions.append({
                        "category": "Internal Assessments & Mid-terms",
                        "severity": "critical" if raw_val < 25 else "moderate",
                        "current": f"{raw_val} / 50 marks (Benchmark: {bench}/50)",
                        "advice": (
                            f"Internal score of {raw_val} is below the {bench}-mark target. Schedule doubt-clearing "
                            f"sessions with course instructors and review mid-semester question papers to bridge conceptual gaps."
                        )
                    })
                elif key == "assignment_completion":
                    suggestions.append({
                        "category": "Assignments & Coursework",
                        "severity": "critical" if raw_val < 50 else "moderate",
                        "current": f"{raw_val}% (Benchmark: {bench}%)",
                        "advice": (
                            f"Assignment submission rate is {raw_val}%. Establish a weekly study calendar to complete "
                            f"and submit problem sets on time, securing full continuous evaluation credit."
                        )
                    })
                elif key == "study_hours":
                    suggestions.append({
                        "category": "Independent Study Routines",
                        "severity": "moderate",
                        "current": f"{raw_val} hrs/day (Benchmark: {bench} hrs/day)",
                        "advice": (
                            f"Dedicated study time is {raw_val} hours per day. Target at least {bench} to 3.5 hours of daily "
                            f"focused self-study using 25-minute Pomodoro sessions to master difficult subject modules."
                        )
                    })
                elif key == "previous_exam_marks":
                    suggestions.append({
                        "category": "Semester Exam Preparation",
                        "severity": "critical" if raw_val < 50 else "moderate",
                        "current": f"{raw_val}% (Benchmark: {bench}%)",
                        "advice": (
                            f"Previous exam score stands at {raw_val}%. Strengthen core prerequisite concepts and "
                            f"undertake timed mock examinations to build test endurance and analytical speed."
                        )
                    })

        # If all satisfied, provide enrichment advice
        if not suggestions:
            suggestions.append({
                "category": "Excellence & Academic Enrichment",
                "severity": "positive",
                "current": "All academic benchmarks currently met",
                "advice": (
                    "Student demonstrates consistent academic habits across all measured parameters. "
                    "Maintain current study schedules, explore advanced elective topics, and consider peer tutoring or research projects."
                )
            })

        return suggestions

    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the learned hypothesis on unseen test data.
        Calculates Confusion Matrix, Accuracy, Precision, Recall, and F1-Score.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained prior to evaluation.")

        tp = 0  # True Positives
        fp = 0  # False Positives
        tn = 0  # True Negatives
        fn = 0  # False Negatives

        for row in test_data:
            discrete_vec = discretize_instance(row)
            pred, _, _ = self.predict_instance(discrete_vec)
            actual = row["label"]

            if actual == POSITIVE_CLASS:
                if pred == POSITIVE_CLASS:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == POSITIVE_CLASS:
                    fp += 1
                else:
                    tn += 1

        total = len(test_data)
        accuracy = round((tp + tn) / total, 4) if total > 0 else 0.0
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1_score = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

        self.evaluation_metrics = {
            "total_test_samples": total,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1_score * 100, 2),
            "confusion_matrix": [
                [tn, fp],
                [fn, tp]
            ],
            "evaluation_note": (
                f"Evaluated on a strictly held-out test split of {total} student records. "
                f"Obtained an empirical accuracy of {round(accuracy * 100, 1)}%."
            )
        }
        return self.evaluation_metrics


# Global singleton instance for the application
_model_singleton = None

def get_trained_model() -> Tuple[FindSConceptLearner, List[Dict[str, Any]]]:
    """
    Load data, split train/test, train the Find-S model, and evaluate it.
    Returns: (learner, all_records)
    """
    global _model_singleton
    if _model_singleton is not None:
        all_records = load_dataset()
        return _model_singleton, all_records

    all_records = load_dataset()
    
    # 75% train / 25% test split
    split_index = int(len(all_records) * 0.75)
    train_data = all_records[:split_index]
    test_data = all_records[split_index:]

    learner = FindSConceptLearner()
    learner.total_records_count = len(all_records)
    learner.train_count = len(train_data)
    learner.test_count = len(test_data)
    learner.fit(train_data)
    learner.evaluate(test_data)

    _model_singleton = learner
    return _model_singleton, all_records
