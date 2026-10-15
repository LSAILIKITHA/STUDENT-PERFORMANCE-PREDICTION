"""
test_concept_learning.py
========================
Automated test suite verifying:
1. Numeric-to-categorical feature conversion & configurable thresholds.
2. The Find-S algorithm learning logic on positive instances.
3. Find-S ignoring negative instances.
4. Edge case handling when a dataset contains zero positive examples.
5. Student prediction matching logic.
6. Input validation and error handling.
"""

import unittest
from concept_learning import (
    convert_numeric_to_categorical,
    FindSConceptLearner,
    get_sample_training_data,
    DEFAULT_THRESHOLDS,
    POSITIVE_LABEL,
    NEGATIVE_LABEL
)
from app import app


class TestNumericToCategoricalConversion(unittest.TestCase):
    """Test numeric-to-categorical conversion functionality and thresholds."""

    def test_all_high_values(self):
        """Values exceeding thresholds should all be mapped to 'High'."""
        cat = convert_numeric_to_categorical(
            attendance=85.0,
            internal_assessment=40.0,
            study_hours=4.5,
            assignment_completion=90.0
        )
        self.assertEqual(cat["Attendance"], "High")
        self.assertEqual(cat["Internal Assessment"], "High")
        self.assertEqual(cat["Study Hours"], "High")
        self.assertEqual(cat["Assignment Completion"], "High")

    def test_all_low_values(self):
        """Values below thresholds should all be mapped to 'Low'."""
        cat = convert_numeric_to_categorical(
            attendance=60.0,
            internal_assessment=20.0,
            study_hours=1.5,
            assignment_completion=50.0
        )
        self.assertEqual(cat["Attendance"], "Low")
        self.assertEqual(cat["Internal Assessment"], "Low")
        self.assertEqual(cat["Study Hours"], "Low")
        self.assertEqual(cat["Assignment Completion"], "Low")

    def test_exact_boundary_thresholds(self):
        """Values exactly equal to thresholds should be categorized as 'High'."""
        cat = convert_numeric_to_categorical(
            attendance=DEFAULT_THRESHOLDS["attendance"],
            internal_assessment=DEFAULT_THRESHOLDS["internal_assessment"],
            study_hours=DEFAULT_THRESHOLDS["study_hours"],
            assignment_completion=DEFAULT_THRESHOLDS["assignment_completion"]
        )
        self.assertEqual(cat["Attendance"], "High")
        self.assertEqual(cat["Internal Assessment"], "High")
        self.assertEqual(cat["Study Hours"], "High")
        self.assertEqual(cat["Assignment Completion"], "High")

    def test_custom_configurable_thresholds(self):
        """Verify that passing custom thresholds alters the conversion boundaries."""
        custom_thresholds = {
            "attendance": 90.0,
            "internal_assessment": 45.0,
            "study_hours": 6.0,
            "assignment_completion": 95.0
        }
        # 80.0 is High under default, but Low under 90.0 custom
        cat = convert_numeric_to_categorical(
            attendance=80.0,
            internal_assessment=35.0,
            study_hours=4.0,
            assignment_completion=80.0,
            thresholds=custom_thresholds
        )
        self.assertEqual(cat["Attendance"], "Low")
        self.assertEqual(cat["Internal Assessment"], "Low")
        self.assertEqual(cat["Study Hours"], "Low")
        self.assertEqual(cat["Assignment Completion"], "Low")


class TestFindSAlgorithm(unittest.TestCase):
    """Test Find-S concept learning algorithm mechanics."""

    def test_find_s_hypothesis_evolution(self):
        """Find-S should generalize from null hypothesis using positive instances and ignore negatives."""
        training_data = [
            # Positive 1: initializes null hypothesis
            {
                "Attendance": "High",
                "Internal Assessment": "High",
                "Study Hours": "High",
                "Assignment Completion": "High",
                "label": POSITIVE_LABEL
            },
            # Negative: should be completely ignored
            {
                "Attendance": "Low",
                "Internal Assessment": "Low",
                "Study Hours": "Low",
                "Assignment Completion": "Low",
                "label": NEGATIVE_LABEL
            },
            # Positive 2: Study Hours differs ('Low') -> generalizes Study Hours constraint to '?'
            {
                "Attendance": "High",
                "Internal Assessment": "High",
                "Study Hours": "Low",
                "Assignment Completion": "High",
                "label": POSITIVE_LABEL
            }
        ]

        learner = FindSConceptLearner()
        learner.fit(training_data)

        # Expected hypothesis: ['High', 'High', '?', 'High']
        self.assertEqual(learner.hypothesis, ["High", "High", "?", "High"])

    def test_find_s_ignoring_negative_examples(self):
        """Find-S must ignore negative examples even if they contradict positive examples."""
        training_data = [
            {
                "Attendance": "High",
                "Internal Assessment": "High",
                "Study Hours": "High",
                "Assignment Completion": "High",
                "label": POSITIVE_LABEL
            },
            {
                "Attendance": "High",
                "Internal Assessment": "High",
                "Study Hours": "High",
                "Assignment Completion": "High",
                "label": NEGATIVE_LABEL  # Contradictory negative example
            }
        ]

        learner = FindSConceptLearner()
        learner.fit(training_data)

        # Hypothesis must remain unchanged because negative example was ignored
        self.assertEqual(learner.hypothesis, ["High", "High", "High", "High"])

    def test_find_s_zero_positive_examples_edge_case(self):
        """When dataset contains no positive examples, hypothesis remains at null state < Ø, Ø, Ø, Ø >."""
        only_negative_data = [
            {
                "Attendance": "Low",
                "Internal Assessment": "Low",
                "Study Hours": "Low",
                "Assignment Completion": "Low",
                "label": NEGATIVE_LABEL
            },
            {
                "Attendance": "High",
                "Internal Assessment": "Low",
                "Study Hours": "Low",
                "Assignment Completion": "Low",
                "label": NEGATIVE_LABEL
            }
        ]

        learner = FindSConceptLearner()
        learner.fit(only_negative_data)

        self.assertFalse(learner.has_positive_examples)
        self.assertEqual(learner.hypothesis, ["Ø", "Ø", "Ø", "Ø"])

        # Prediction should safely classify as NEGATIVE_LABEL without crashing
        test_student = {
            "Attendance": "High",
            "Internal Assessment": "High",
            "Study Hours": "High",
            "Assignment Completion": "High"
        }
        res = learner.predict(test_student)
        self.assertEqual(res["prediction"], NEGATIVE_LABEL)
        self.assertIn("warning", res)

    def test_predict_before_fit_raises_error(self):
        """Calling predict() before fit() must raise a RuntimeError."""
        learner = FindSConceptLearner()
        with self.assertRaises(RuntimeError):
            learner.predict({"Attendance": "High"})

    def test_prediction_classification_logic(self):
        """Verify matching instances predict Positive and non-matching predict Negative."""
        sample_data = get_sample_training_data()
        learner = FindSConceptLearner()
        learner.fit(sample_data)

        # Learned hypothesis is: ['High', 'High', '?', 'High']

        # Matching student (Study Hours can be High or Low due to '?')
        matching_student = {
            "Attendance": "High",
            "Internal Assessment": "High",
            "Study Hours": "Low",
            "Assignment Completion": "High"
        }
        res_good = learner.predict(matching_student)
        self.assertEqual(res_good["prediction"], POSITIVE_LABEL)
        self.assertTrue(res_good["matches_hypothesis"])

        # Violating student (Attendance is Low)
        violating_student = {
            "Attendance": "Low",
            "Internal Assessment": "High",
            "Study Hours": "High",
            "Assignment Completion": "High"
        }
        res_bad = learner.predict(violating_student)
        self.assertEqual(res_bad["prediction"], NEGATIVE_LABEL)
        self.assertFalse(res_bad["matches_hypothesis"])


class TestFlaskBackendValidationAndAPI(unittest.TestCase):
    """Test Flask backend API routes, input validation, and model integration."""

    def setUp(self):
        self.client = app.test_client()

    def test_predict_endpoint_success(self):
        """POST /predict returns 200 with all 5 required response fields."""
        payload = {
            "student_id": "TEST-2026",
            "attendance": 88.0,
            "internal_marks": 42.0,
            "assignment_completion": 92.0,
            "study_hours": 4.0,
            "previous_exam_marks": 80.0
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "success")

        # 1. Predicted category
        self.assertIn("predicted_category", data)
        self.assertEqual(data["predicted_category"], POSITIVE_LABEL)

        # 2. Input features used
        self.assertIn("input_features_used", data)
        self.assertIn("numeric", data["input_features_used"])
        self.assertIn("categorical", data["input_features_used"])
        self.assertEqual(data["input_features_used"]["categorical"]["Attendance"], "High")

        # 3. Learned hypothesis
        self.assertIn("learned_hypothesis", data)
        self.assertIn("formatted_hypothesis", data)

        # 4. Simple explanation of the classification
        self.assertIn("explanation", data)
        self.assertTrue(len(data["explanation"]) > 0)

        # 5. Appropriate academic suggestion
        self.assertIn("academic_suggestion", data)
        self.assertTrue(len(data["academic_suggestion"]) > 0)

    def test_predict_endpoint_support_needed_response(self):
        """POST /predict for struggling student yields Needs Academic Support and specific suggestion."""
        payload = {
            "student_id": "TEST-STRUGGLE",
            "attendance": 55.0,  # Low
            "internal_marks": 20.0,  # Low
            "assignment_completion": 45.0,  # Low
            "study_hours": 1.5,  # Low
            "previous_exam_marks": 40.0  # Low
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["predicted_category"], NEGATIVE_LABEL)
        self.assertIn("violated", data["explanation"].lower())
        self.assertIn("attendance", data["academic_suggestion"].lower())

    def test_invalid_out_of_bound_inputs(self):
        """POST /api/predict with negative numbers or numbers exceeding range returns 422 validation error."""
        invalid_payload = {
            "attendance": 150.0,  # Invalid: > 100%
            "internal_marks": -5.0,  # Invalid: < 0
            "assignment_completion": 80.0,
            "study_hours": 28.0,  # Invalid: > 24 hrs
            "previous_exam_marks": 50.0
        }
        response = self.client.post("/api/predict", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

        data = response.get_json()
        self.assertEqual(data["status"], "validation_error")
        self.assertIn("attendance", data["errors"])
        self.assertIn("internal_marks", data["errors"])
        self.assertIn("study_hours", data["errors"])

    def test_non_numeric_inputs(self):
        """POST /api/predict with non-numeric strings returns 422 validation error."""
        invalid_payload = {
            "attendance": "NotANumber",
            "internal_marks": "Invalid",
            "assignment_completion": 80.0,
            "study_hours": 2.0,
            "previous_exam_marks": 70.0
        }
        response = self.client.post("/api/predict", json=invalid_payload)
        self.assertEqual(response.status_code, 422)
        data = response.get_json()
        self.assertIn("attendance", data["errors"])


if __name__ == "__main__":
    unittest.main()
