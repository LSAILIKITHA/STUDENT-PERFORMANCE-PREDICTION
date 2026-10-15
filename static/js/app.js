/**
 * Student Performance Prediction Using Concept Learning
 * Application JavaScript Controller
 */

// Global State
const state = {
    currentPage: "dashboardPage",
    summaryData: null,
    modelInfoData: null,
    recordsPage: 1,
    recordsPageSize: 10,
    recordsSearch: "",
    recordsFilter: "all",
    chartInstance: null
};

// Preset sample profiles for quick testing/demo
const PRESETS = {
    high: {
        student_id: "STU-DEMO-HIGH",
        attendance: 92.5,
        internal_marks: 44.0,
        assignment_completion: 95.0,
        study_hours: 5.5,
        previous_exam_marks: 88.0
    },
    steady: {
        student_id: "STU-DEMO-STEADY",
        attendance: 82.0,
        internal_marks: 34.0,
        assignment_completion: 78.0,
        study_hours: 3.0,
        previous_exam_marks: 72.0
    },
    struggling: {
        student_id: "STU-DEMO-RISK",
        attendance: 58.0,
        internal_marks: 22.0,
        assignment_completion: 48.0,
        study_hours: 1.5,
        previous_exam_marks: 46.0
    }
};

// DOM Ready initialization
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initSidebarToggle();
    initPredictionForm();
    initRecordsControls();
    
    // Fetch initial data
    loadDashboardSummary();
    loadStudentRecords();
    loadModelInfo();
});

/* ==========================================================================
   NAVIGATION CONTROLS
   ========================================================================== */
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const target = item.dataset.target;
            if (target) {
                navigateTo(target);
            }
        });
    });
}

function navigateTo(targetPageId) {
    // Hide all sections
    document.querySelectorAll(".page-section").forEach(sec => {
        sec.classList.remove("active");
    });

    // Show target section
    const targetEl = document.getElementById(targetPageId);
    if (targetEl) {
        targetEl.classList.add("active");
        state.currentPage = targetPageId;
    }

    // Update sidebar nav active state
    document.querySelectorAll(".nav-item").forEach(item => {
        if (item.dataset.target === targetPageId) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });

    // Close mobile sidebar if open
    const sidebar = document.getElementById("appSidebar");
    if (sidebar && sidebar.classList.contains("open")) {
        sidebar.classList.remove("open");
    }

    // Window scroll to top
    window.scrollTo({ top: 0, behavior: "smooth" });
}

function initSidebarToggle() {
    const toggleBtn = document.getElementById("sidebarToggle");
    const sidebar = document.getElementById("appSidebar");
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener("click", () => {
            sidebar.classList.toggle("open");
        });
    }
}

/* ==========================================================================
   1. DASHBOARD & SUMMARY
   ========================================================================== */
async function loadDashboardSummary() {
    try {
        const response = await fetch("/api/summary");
        if (!response.ok) throw new Error("Failed to load dashboard summary");
        
        const data = await response.json();
        state.summaryData = data;

        // Stat cards
        document.getElementById("statTotalStudents").textContent = data.metrics.total_students;
        document.getElementById("statSplitText").textContent = 
            `Partition: ${data.metrics.training_samples} train / ${data.metrics.test_samples} test`;
        
        document.getElementById("statGoodPerformers").textContent = data.metrics.good_performers;
        document.getElementById("statGoodPct").textContent = `${data.metrics.good_performers_pct}% of student body`;

        document.getElementById("statSupportNeeded").textContent = data.metrics.support_needed;
        document.getElementById("statSupportPct").textContent = `${data.metrics.support_needed_pct}% of student body`;

        document.getElementById("statAccuracy").textContent = `${data.evaluation.accuracy}%`;
        document.getElementById("statEvalNote").textContent = 
            `Precision: ${data.evaluation.precision}% | Recall: ${data.evaluation.recall}%`;

        // Sidebar and card hypothesis display
        const hypFormatted = data.learned_hypothesis.formatted;
        document.getElementById("sidebarHypothesis").textContent = hypFormatted;
        document.getElementById("hypothesisDisplay").innerHTML = `<code>${hypFormatted}</code>`;

        renderHypothesisChips(data.learned_hypothesis);
        renderDistributionChart(data.distribution);

    } catch (err) {
        console.error("Dashboard summary error:", err);
        showToast("Error loading dashboard metrics: " + err.message);
    }
}

function renderHypothesisChips(hypothesisData) {
    const container = document.getElementById("hypothesisChips");
    if (!container) return;

    container.innerHTML = "";
    hypothesisData.features.forEach((featureName, idx) => {
        const val = hypothesisData.vector[idx];
        const chip = document.createElement("div");
        chip.className = "hypothesis-chip";
        chip.innerHTML = `
            <span class="hc-name">${featureName}:</span>
            <span class="hc-val">${val}</span>
        `;
        container.appendChild(chip);
    });
}

function renderDistributionChart(distribution) {
    const canvas = document.getElementById("categoryChart");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (state.chartInstance) {
        state.chartInstance.destroy();
    }

    if (window.Chart) {
        state.chartInstance = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: distribution.labels,
                datasets: [{
                    data: distribution.counts,
                    backgroundColor: [
                        "#059669", // Success Emerald for Good Performer
                        "#d97706"  // Amber for Needs Support
                    ],
                    borderWidth: 2,
                    borderColor: "#ffffff"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const count = context.raw;
                                const pct = distribution.percentages[context.dataIndex];
                                return ` ${context.label}: ${count} (${pct}%)`;
                            }
                        }
                    }
                },
                cutout: "68%"
            }
        });
    }

    // Legend
    const legendEl = document.getElementById("chartLegend");
    if (legendEl) {
        legendEl.innerHTML = `
            <div class="legend-item">
                <span class="legend-color-box" style="background:#059669"></span>
                <span>Good Performers (${distribution.counts[0]})</span>
            </div>
            <div class="legend-item">
                <span class="legend-color-box" style="background:#d97706"></span>
                <span>Needs Support (${distribution.counts[1]})</span>
            </div>
        `;
    }
}

/* ==========================================================================
   2. STUDENT PREDICTION
   ========================================================================== */
function fillSample(type) {
    const preset = PRESETS[type];
    if (!preset) return;

    document.getElementById("studentId").value = preset.student_id;
    document.getElementById("attendance").value = preset.attendance;
    document.getElementById("internalMarks").value = preset.internal_marks;
    document.getElementById("assignmentCompletion").value = preset.assignment_completion;
    document.getElementById("studyHours").value = preset.study_hours;
    document.getElementById("previousExamMarks").value = preset.previous_exam_marks;

    clearFieldErrors();
    showToast(`Loaded preset profile: ${type.toUpperCase()}`);
}

function clearFieldErrors() {
    document.querySelectorAll(".field-error").forEach(el => {
        el.textContent = "";
        el.classList.remove("show");
    });
    document.querySelectorAll(".form-control").forEach(input => {
        input.classList.remove("is-invalid");
    });
}

function initPredictionForm() {
    const form = document.getElementById("predictionForm");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        clearFieldErrors();

        const student_id = document.getElementById("studentId").value.trim();
        const attendance = parseFloat(document.getElementById("attendance").value);
        const internal_marks = parseFloat(document.getElementById("internalMarks").value);
        const assignment_completion = parseFloat(document.getElementById("assignmentCompletion").value);
        const study_hours = parseFloat(document.getElementById("studyHours").value);
        const previous_exam_marks = parseFloat(document.getElementById("previousExamMarks").value);

        // Local client validation
        let hasError = false;
        if (isNaN(attendance) || attendance < 0 || attendance > 100) {
            setFieldError("attendance", "Attendance must be between 0% and 100%.");
            hasError = true;
        }
        if (isNaN(internal_marks) || internal_marks < 0 || internal_marks > 50) {
            setFieldError("internal_marks", "Internal marks must be between 0 and 50.");
            hasError = true;
        }
        if (isNaN(assignment_completion) || assignment_completion < 0 || assignment_completion > 100) {
            setFieldError("assignment_completion", "Assignment completion must be between 0% and 100%.");
            hasError = true;
        }
        if (isNaN(study_hours) || study_hours < 0 || study_hours > 24) {
            setFieldError("study_hours", "Study hours must be between 0 and 24 hours per day.");
            hasError = true;
        }
        if (isNaN(previous_exam_marks) || previous_exam_marks < 0 || previous_exam_marks > 100) {
            setFieldError("previous_exam_marks", "Previous exam marks must be between 0% and 100%.");
            hasError = true;
        }

        if (hasError) return;

        // UI Spinner
        const submitBtn = document.getElementById("predictBtn");
        const spinner = document.getElementById("predictSpinner");
        submitBtn.disabled = true;
        spinner.style.display = "inline-block";

        try {
            const payload = {
                student_id: student_id || "STU-PROSPECTIVE",
                attendance,
                internal_marks,
                assignment_completion,
                study_hours,
                previous_exam_marks
            };

            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok) {
                if (result.status === "validation_error" && result.errors) {
                    for (const [key, msg] of Object.entries(result.errors)) {
                        setFieldError(key, msg);
                    }
                } else {
                    showToast(result.message || "Prediction error occurred.");
                }
                return;
            }

            // Successfully evaluated
            displayPredictionResults(result);
            navigateTo("resultsPage");

            // Display badge indicator on Results navigation tab
            const badge = document.getElementById("resultsBadge");
            if (badge) badge.style.display = "inline-block";

        } catch (err) {
            console.error("Prediction request failed:", err);
            showToast("Server communication error. Please try again.");
        } finally {
            submitBtn.disabled = false;
            spinner.style.display = "none";
        }
    });
}

function setFieldError(fieldKey, message) {
    const errorEl = document.getElementById(`err_${fieldKey}`);
    const inputEl = document.getElementById(fieldKey.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase()));
    if (errorEl) {
        errorEl.textContent = message;
        errorEl.classList.add("show");
    }
    if (inputEl) {
        inputEl.classList.add("is-invalid");
    }
}

/* ==========================================================================
   3. PREDICTION RESULTS DISPLAY
   ========================================================================== */
function displayPredictionResults(data) {
    document.getElementById("noResultsState").style.display = "none";
    const resultsContent = document.getElementById("resultsContent");
    resultsContent.style.display = "flex";

    const isGood = (data.prediction === "Good Performer" || data.predicted_category === "Good Performer");
    const banner = document.getElementById("resultBanner");
    banner.className = `card result-banner-card ${isGood ? "status-good" : "status-support"}`;

    document.getElementById("resStudentId").textContent = data.student_id;
    document.getElementById("resCategory").textContent = data.predicted_category || data.prediction;
    document.getElementById("resBadgeText").textContent = isGood ? "Hypothesis Match (Positive)" : "Hypothesis Constraint Violated";

    // Display learned hypothesis and simple explanation
    const hypEl = document.getElementById("resHypothesis");
    if (hypEl && data.formatted_hypothesis) {
        hypEl.textContent = data.formatted_hypothesis;
    }

    const summaryEl = document.getElementById("resSummary");
    if (summaryEl) {
        summaryEl.textContent = data.explanation || (isGood
            ? `All student features satisfy the learned Find-S concept hypothesis ${data.formatted_hypothesis}.`
            : `One or more student features violated the learned Find-S concept hypothesis ${data.formatted_hypothesis}.`);
    }

    // Display primary academic suggestion
    const suggestionEl = document.getElementById("resAcademicSuggestion");
    if (suggestionEl) {
        suggestionEl.textContent = data.academic_suggestion || "No specific suggestion required.";
    }

    // Render Feature Analysis Table with raw input, categorical conversion, and hypothesis constraint
    const tableBody = document.getElementById("featureAnalysisTable");
    tableBody.innerHTML = "";

    const rawInputs = (data.input_features_used && data.input_features_used.numeric) ? data.input_features_used.numeric : (data.raw_inputs || {});
    const categorical = (data.input_features_used && data.input_features_used.categorical) ? data.input_features_used.categorical : (data.categorical_features || {});
    const breakdown = data.feature_breakdown || [];

    const featureKeysMap = {
        "Attendance": "attendance",
        "Internal Assessment": "internal_marks",
        "Assignment Completion": "assignment_completion",
        "Study Hours": "study_hours",
        "Previous Examination Marks": "previous_exam_marks"
    };

    breakdown.forEach(item => {
        const featName = item.feature;
        const rawKey = featureKeysMap[featName];
        const rawVal = rawInputs[rawKey] !== undefined ? rawInputs[rawKey] : "N/A";
        const unit = getUnitForFeature(rawKey);
        const catVal = categorical[featName] || item.student_value;

        const row = document.createElement("tr");
        row.innerHTML = `
            <td><strong>${featName}</strong></td>
            <td><code>${rawVal}${unit}</code></td>
            <td><span class="badge-concept" style="font-weight:700; color:#1e40af; background:#eff6ff; padding:2px 8px; border-radius:4px;">${catVal}</span></td>
            <td><code style="font-weight:700; color:#0f172a;">${item.hypothesis_constraint}</code></td>
            <td>
                <span class="status-tag ${item.satisfied ? "tag-satisfied" : "tag-violated"}">
                    ${item.satisfied ? "✓ Satisfied" : "✗ Violated"}
                </span>
                <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">${item.reason}</div>
            </td>
        `;
        tableBody.appendChild(row);
    });

    // Render Suggestions
    const suggestionsContainer = document.getElementById("suggestionsContainer");
    suggestionsContainer.innerHTML = "";

    (data.suggestions || []).forEach(sg => {
        const card = document.createElement("div");
        card.className = `suggestion-card ${sg.severity}`;
        card.innerHTML = `
            <div>
                <div class="sg-header">
                    <span class="sg-cat">${sg.category}</span>
                    <span class="sg-current">${sg.current}</span>
                </div>
                <p class="sg-advice">${sg.advice}</p>
            </div>
        `;
        suggestionsContainer.appendChild(card);
    });

    // Disclaimer
    document.getElementById("resDisclaimer").textContent = data.disclaimer;
}

function getUnitForFeature(key) {
    switch (key) {
        case "attendance": return "%";
        case "internal_marks": return " / 50";
        case "assignment_completion": return "%";
        case "study_hours": return " hrs/day";
        case "previous_exam_marks": return "%";
        default: return "";
    }
}

/* ==========================================================================
   4. STUDENT RECORDS
   ========================================================================== */
function initRecordsControls() {
    const searchInput = document.getElementById("recordsSearch");
    const filterSelect = document.getElementById("recordsFilter");
    const prevBtn = document.getElementById("prevPageBtn");
    const nextBtn = document.getElementById("nextPageBtn");

    let debounceTimer;
    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                state.recordsSearch = e.target.value.trim();
                state.recordsPage = 1;
                loadStudentRecords();
            }, 300);
        });
    }

    if (filterSelect) {
        filterSelect.addEventListener("change", (e) => {
            state.recordsFilter = e.target.value;
            state.recordsPage = 1;
            loadStudentRecords();
        });
    }

    if (prevBtn) {
        prevBtn.addEventListener("click", () => {
            if (state.recordsPage > 1) {
                state.recordsPage--;
                loadStudentRecords();
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener("click", () => {
            state.recordsPage++;
            loadStudentRecords();
        });
    }
}

async function loadStudentRecords() {
    try {
        const url = `/api/students?page=${state.recordsPage}&page_size=${state.recordsPageSize}&query=${encodeURIComponent(state.recordsSearch)}&label=${encodeURIComponent(state.recordsFilter)}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error("Failed to load student records");

        const data = await res.json();
        renderRecordsTable(data.records);

        // Update pagination
        const total = data.total;
        const totalPages = data.total_pages || 1;
        const start = total === 0 ? 0 : (data.page - 1) * data.page_size + 1;
        const end = Math.min(data.page * data.page_size, total);

        document.getElementById("paginationInfo").textContent = `Showing ${start} to ${end} of ${total} records`;
        document.getElementById("pageIndicator").textContent = `Page ${data.page} of ${totalPages}`;

        document.getElementById("prevPageBtn").disabled = (data.page <= 1);
        document.getElementById("nextPageBtn").disabled = (data.page >= totalPages);

    } catch (err) {
        console.error("Load records error:", err);
    }
}

function renderRecordsTable(records) {
    const tbody = document.getElementById("studentRecordsTable");
    if (!tbody) return;

    tbody.innerHTML = "";
    if (records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 2rem; color: #94a3b8;">No matching student records found.</td></tr>`;
        return;
    }

    records.forEach(r => {
        const isGood = (r.label === "Good Performer");
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong><code>${r.student_id}</code></strong></td>
            <td>${r.attendance}%</td>
            <td>${r.internal_marks} / 50</td>
            <td>${r.assignment_completion}%</td>
            <td>${r.study_hours} hrs</td>
            <td>${r.previous_exam_marks}%</td>
            <td>
                <span class="${isGood ? "badge-label-good" : "badge-label-support"}">
                    ${r.label}
                </span>
            </td>
            <td>
                <button class="btn btn-outline btn-sm" onclick='loadRecordIntoForm(${JSON.stringify(r)})'>
                    Load in Predictor
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function loadRecordIntoForm(record) {
    document.getElementById("studentId").value = record.student_id;
    document.getElementById("attendance").value = record.attendance;
    document.getElementById("internalMarks").value = record.internal_marks;
    document.getElementById("assignmentCompletion").value = record.assignment_completion;
    document.getElementById("studyHours").value = record.study_hours;
    document.getElementById("previousExamMarks").value = record.previous_exam_marks;

    clearFieldErrors();
    navigateTo("predictPage");
    showToast(`Loaded ${record.student_id} into prediction form.`);
}

/* ==========================================================================
   5. ABOUT THE MODEL & TRACE
   ========================================================================== */
async function loadModelInfo() {
    try {
        const res = await fetch("/api/model-info");
        if (!res.ok) throw new Error("Failed to load model details");

        const data = await res.json();
        state.modelInfoData = data;

        // Render Training Trace Table
        if (data.training_trace) {
            renderTrainingTrace(data.training_trace);
        }

        // Render Limitations
        if (data.limitations) {
            renderLimitations(data.limitations);
        }

    } catch (err) {
        console.error("Model info load error:", err);
    }
}

function renderTrainingTrace(trace) {
    const tbody = document.getElementById("trainingTraceTable");
    if (!tbody || !trace) return;

    tbody.innerHTML = "";
    trace.forEach(step => {
        const isPos = step.is_positive;
        const featureVals = step.features ? Object.values(step.features).join(", ") : "";
        const hypVals = step.hypothesis_after ? step.hypothesis_after.join(", ") : "";
        
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><code>#${step.step}</code></td>
            <td><strong>${step.student_id}</strong></td>
            <td><small>⟨ ${featureVals} ⟩</small></td>
            <td>
                <span class="${isPos ? "badge-label-good" : "badge-label-support"}">
                    ${step.label}
                </span>
            </td>
            <td><small>${step.action}</small></td>
            <td><code style="color: #2563eb; font-weight:700;">⟨ ${hypVals} ⟩</code></td>
        `;
        tbody.appendChild(tr);
    });
}

function renderLimitations(limitations) {
    const container = document.getElementById("limitationsGrid");
    if (!container || !limitations) return;

    container.innerHTML = "";
    limitations.forEach(lim => {
        const item = document.createElement("div");
        item.className = "limitation-item";
        item.innerHTML = `
            <h4 class="lim-title">${lim.title}</h4>
            <p class="lim-desc">${lim.description}</p>
        `;
        container.appendChild(item);
    });
}

/* ==========================================================================
   TOAST NOTIFICATION UTILITY
   ========================================================================== */
function showToast(message) {
    const toast = document.getElementById("toast");
    if (!toast) return;

    toast.textContent = message;
    toast.style.display = "block";

    setTimeout(() => {
        toast.style.display = "none";
    }, 3200);
}
