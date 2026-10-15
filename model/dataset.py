"""
Dataset generator and manager for Student Performance Prediction.
Provides realistic sample student records and handles CSV persistence.
"""

import os
import csv
import random

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "students_dataset.csv")

def generate_student_records(count=120, seed=42):
    """
    Generate realistic student academic records.
    Features:
      - student_id: string
      - attendance: percentage (0-100)
      - internal_marks: marks (0-50)
      - assignment_completion: percentage (0-100)
      - study_hours: daily hours (0-12)
      - previous_exam_marks: percentage (0-100)
      - label: 'Good Performer' or 'Needs Academic Support'
    """
    random.seed(seed)
    records = []

    for i in range(1, count + 1):
        student_id = f"STU-{1000 + i}"
        
        # Decide latent academic archetype to create realistic correlated distribution
        archetype = random.choices(
            ["high_achiever", "steady_performer", "struggling", "at_risk"],
            weights=[0.35, 0.25, 0.25, 0.15],
            k=1
        )[0]
        
        if archetype == "high_achiever":
            attendance = round(random.uniform(80.0, 98.0), 1)
            internal_marks = round(random.uniform(36.0, 49.0), 1)
            assignment_completion = round(random.uniform(85.0, 100.0), 1)
            study_hours = round(random.uniform(3.5, 7.0), 1)
            previous_exam_marks = round(random.uniform(75.0, 96.0), 1)
            label = "Good Performer"

        elif archetype == "steady_performer":
            attendance = round(random.uniform(75.0, 88.0), 1)
            internal_marks = round(random.uniform(30.0, 42.0), 1)
            assignment_completion = round(random.uniform(72.0, 90.0), 1)
            study_hours = round(random.uniform(2.0, 5.0), 1)
            previous_exam_marks = round(random.uniform(65.0, 82.0), 1)
            label = "Good Performer"

        elif archetype == "struggling":
            attendance = round(random.uniform(55.0, 74.0), 1)
            internal_marks = round(random.uniform(20.0, 31.0), 1)
            assignment_completion = round(random.uniform(45.0, 68.0), 1)
            study_hours = round(random.uniform(1.0, 2.8), 1)
            previous_exam_marks = round(random.uniform(45.0, 62.0), 1)
            label = "Needs Academic Support"

        else: # at_risk
            attendance = round(random.uniform(35.0, 60.0), 1)
            internal_marks = round(random.uniform(10.0, 24.0), 1)
            assignment_completion = round(random.uniform(20.0, 50.0), 1)
            study_hours = round(random.uniform(0.5, 1.8), 1)
            previous_exam_marks = round(random.uniform(25.0, 48.0), 1)
            label = "Needs Academic Support"

        records.append({
            "student_id": student_id,
            "attendance": attendance,
            "internal_marks": internal_marks,
            "assignment_completion": assignment_completion,
            "study_hours": study_hours,
            "previous_exam_marks": previous_exam_marks,
            "label": label
        })

    return records


def ensure_dataset():
    """Ensure that the dataset CSV exists, creating it if necessary."""
    if not os.path.exists(DATASET_PATH):
        records = generate_student_records(count=120)
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        with open(DATASET_PATH, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "student_id",
                "attendance",
                "internal_marks",
                "assignment_completion",
                "study_hours",
                "previous_exam_marks",
                "label"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    return DATASET_PATH


def load_dataset():
    """Load dataset records from CSV."""
    path = ensure_dataset()
    records = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "student_id": row["student_id"],
                "attendance": float(row["attendance"]),
                "internal_marks": float(row["internal_marks"]),
                "assignment_completion": float(row["assignment_completion"]),
                "study_hours": float(row["study_hours"]),
                "previous_exam_marks": float(row["previous_exam_marks"]),
                "label": row["label"]
            })
    return records


if __name__ == "__main__":
    records = load_dataset()
    print(f"Loaded {len(records)} student records.")
    good_count = sum(1 for r in records if r['label'] == 'Good Performer')
    print(f"Good Performers: {good_count}, Needs Support: {len(records) - good_count}")
