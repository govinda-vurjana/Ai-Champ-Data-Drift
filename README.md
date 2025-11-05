# Data Drift Detection RL Task - Comprehensive README

## 📋 Table of Contents
1. [Problem Statement](#problem-statement)
2. [Objective](#objective)
3. [Background](#background)
4. [Architecture](#architecture)
5. [Model Information](#model-information)
6. [Test Suite Overview](#test-suite-overview)
7. [Test Cases Breakdown](#test-cases-breakdown)
8. [Why These Tests Matter](#why-these-tests-matter)
9. [Running the Evaluation](#running-the-evaluation)
10. [Understanding Results](#understanding-results)

---

## 🎯 Problem Statement

### The Challenge
Models deployed to production naturally degrade as real-world data distributions change over time. This phenomenon is called **data drift**.

**The Problem:**
- ✗ Models that performed well during development become outdated
- ✗ User behaviors, market conditions, and data characteristics evolve
- ✗ Performance metrics silently decrease without explicit monitoring
- ✗ ML engineers face exhausting cycles of:
  - Model evaluation
  - Retraining
  - Redeployment
  - Monitoring
  - Repeat...

### Types of Data Drift

**1. Covariate Drift**
- Input distribution changes
- Model's output quality remains stable
- Example: User income distribution shifts, but model accuracy stays same
- Action: Usually needs retraining on new input distribution

**2. Concept Drift**
- Output quality degrades
- Input distribution remains stable
- Example: User preferences change, model accuracy drops
- Action: Requires model retraining with new logic

**3. Both Drifts**
- Both input AND output change
- Most dangerous scenario
- Action: Complete model review and retraining

---

## 🎯 Objective

This evaluation tests whether an AI model (Claude) can implement **production-grade data drift detection and response functions**.

### Specific Goals

1. **Implement Drift Detection**
   - Accurately detect covariate drift (>20% input shift + stable quality)
   - Accurately detect concept drift (>10% quality drop + stable input)
   - Classify drift type correctly

2. **Calculate Business Impact**
   - Compute affected predictions
   - Calculate error counts
   - Estimate financial impact ($$$)
   - Handle edge cases (fractional days, extreme values)

3. **Recommend Actions**
   - Map drift severity to recommended action
   - Use thresholds: MONITOR → INVESTIGATE → RETRAIN → ESCALATE
   - Handle unknown scenarios gracefully

4. **Achieve Target Accuracy**
   - Pass rate between 10-40%
   - Indicates model competency without over-fitting to test cases
   - Prevents gaming or memorization

---

<img width="1994" height="684" alt="image" src="https://github.com/user-attachments/assets/e6b8d97b-8c41-4dd5-ad3a-9ca60acdb81f" />




## 📚 Background

### Why Data Drift Matters in Production

**Real-World Scenarios:**
- E-commerce: Product popularity shifts seasonally
- Finance: Market volatility changes transaction patterns
- Healthcare: Disease prevalence evolves
- Ad Tech: User interests shift over time

**Cost of Not Detecting Drift:**
```
Undetected Drift
    ↓
Silently Declining Accuracy
    ↓
Poor Predictions
    ↓
Business Loss (customers, revenue, trust)
    ↓
Crisis Mode Retraining
    ↓
Expensive Recovery
```

### The Continuous Monitoring Loop

```
Deploy Model
    ↓
Monitor for Drift
    ↓
Drift Detected? 
    ├─ YES → Analyze Impact → Retrain → Redeploy → Monitor
    └─ NO → Continue Monitoring
```

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│         Data Drift Detection System                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Production Model                                       │
│  ├─ Input Data (X)                                      │
│  ├─ Predictions (ŷ)                                    │
│  └─ Quality Metrics (Accuracy, F1, AUC)                │
│          ↓                                              │
│  Drift Detector                                         │
│  ├─ detect_covariate_drift()    [Input → Stable]      │
│  ├─ detect_concept_drift()      [Quality → Drop]      │
│  └─ classify_drift()            [Type classification]  │
│          ↓                                              │
│  Impact Calculator                                      │
│  ├─ calculate_drift_impact()    [$ impact]            │
│  └─ determine_response_action() [Action mapping]      │
│          ↓                                              │
│  Response                                               │
│  ├─ MONITOR: Continue watching                         │
│  ├─ INVESTIGATE: Review model & data                   │
│  ├─ RETRAIN: Update with new data                      │
│  └─ ESCALATE: Critical - immediate action              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🤖 Model Information

### Model Used: Claude Sonnet 4.5

**Deployment:** Google Vertex AI
```
API: Claude via Vertex AI
Region: Global
Project: august-beaker-470006-s8
Model: claude-sonnet-4-5
Max Tokens: 3,000
Temperature: Default (0.7)
```

### Why Claude Sonnet 4.5?
- Fast inference (suitable for repeated evaluations)
- Strong reasoning (can handle complex logic)
- Tool use capable (can test code with python_expression)
- Cost-effective for multiple runs (10 runs per evaluation)

### Evaluation Method

**Agent Loop:**
```
1. Claude reads task prompt
2. Claude writes Python functions
3. Claude tests with python_expression tool
4. Claude iterates until satisfied
5. Claude submits with submit_answer tool
6. System grades submitted code
7. Repeat 10 times (concurrent runs)
```

**Scoring: Binary (5-point scale)**
- 1 point per function if ALL its tests pass
- 0 points if ANY test fails
- Total: 0-5 points per run

---

## 📊 Test Suite Overview

### Test Statistics

```
Total Functions:    5
Total Test Cases:   31
Easy Tests:         7
Medium Tests:      10
Hard Tests:         8
Extreme Tests:      6

Expected Pass Rate: 10-40%
Time per Run:       ~2-3 minutes
Total Time (10x):   ~20-30 minutes
```

### Test Distribution

| Function | Tests | Difficulty | Pass Requirement |
|----------|-------|-----------|------------------|
| detect_covariate_drift | 8 | Hard | ≥6/8 |
| detect_concept_drift | 5 | Hard | ≥2/5 |
| classify_drift | 6 | Easy | 6/6 (all) |
| calculate_drift_impact | 4 | Medium-Hard | ≥2/4 |
| determine_response_action | 8 | Medium | ≥5/8 |

---

## 🧪 Test Cases Breakdown

### Function 1: detect_covariate_drift (8 tests)

**Rule:** Detect if `input_shift > 20% AND |quality_change| ≤ 5%`

| # | Shift | Quality | Expected | Type | Why Useful |
|---|-------|---------|----------|------|-----------|
| 1 | +40% | -1% | ✓ DETECT | Sanity | Basic functionality |
| 2 | +15% | -6% | ✗ NO | Trap | Quality drop veto |
| 3 | +19% | -2% | ✗ NO | Boundary | Just below 20% threshold |
| 4 | +21% | +1% | ✓ DETECT | Boundary | Just above 20% threshold |
| 5 | -25% | -1% | ✓ DETECT | Negative | Magnitude matters |
| 6 | +20% | -7% | ✗ NO | Veto | Quality drop always rejects |
| 7 | +5% | +1% | ✗ NO | Easy | No drift |
| 8 | +30% | -4.5% | ✓ DETECT | Extreme | Borderline tolerance |

**Key Insights:**
- ✓ Tests exact 20% threshold (±1%)
- ✓ Tests quality drop veto logic
- ✓ Tests negative shifts (magnitude)
- ✓ Tests floating-point precision

---

### Function 2: detect_concept_drift (5 tests)

**Rule:** Detect if `input_stable AND quality_drop > 10%`

| # | Quality | Input | Expected | Type | Why Useful |
|---|---------|-------|----------|------|-----------|
| 1 | -16% | Same | ✓ DETECT | Sanity | Clear degradation |
| 2 | -9% | Same | ✗ NO | Boundary | Below 10% threshold |
| 3 | -11% | +20% | ✗ NO | Veto | Input changed = covariate |
| 4 | +2% | Same | ✗ NO | Improve | No degradation |
| 5 | -11% | +5% | ✓ DETECT | Edge | At input tolerance |

**Key Insights:**
- ✓ Tests exact 10% quality threshold
- ✓ Tests input stability requirement
- ✓ Tests that improvement rejects detection
- ✓ Tests tolerance boundaries

---

### Function 3: classify_drift (6 tests)

**Rule:** Simple if-logic mapping

| # | Input Shifted | Quality Dropped | Expected |
|---|---------------|-----------------|----------|
| 1 | True | False | 'covariate' |
| 2 | False | True | 'concept' |
| 3 | True | True | 'both' |
| 4 | False | False | 'none' |
| 5 | True | False | 'covariate' (repeat) |
| 6 | True | True | 'both' (repeat) |

**Key Insights:**
- ✓ All tests should pass easily
- ✓ Tests consistency (tests 5-6 repeat earlier)
- ✓ Used as baseline to ensure basic competency

---

### Function 4: calculate_drift_impact (4 tests)

**Formula:**
```
predictions_affected = daily_predictions × days
errors = predictions_affected × error_rate
financial_impact = errors × cost
```

| # | Daily | Days | Rate | Cost | Expected Errors | Tolerance |
|---|-------|------|------|------|-----------------|-----------|
| 1 | 10k | 5 | 0.02 | $50 | 1,000 | ±1% |
| 2 | 10k | 7 | 0.0001 | $50 | 7 | ±5% |
| 3 | 10k | 2.5 | 0.01 | $100 | 250 | ±5% |
| 4 | 50k | 1 | 0.05 | $200 | 2,500 | ±2% |

**Key Insights:**
- ✓ Test 1: Standard case with tight tolerance
- ✓ Test 2: Extreme precision (error rate 0.0001)
- ✓ Test 3: Fractional days (floating-point)
- ✓ Test 4: Large scale ($500k+ impact)

---

### Function 5: determine_response_action (8 tests)

**Thresholds:**
```
0.0 - 0.3  → MONITOR
0.3 - 0.5  → INVESTIGATE
0.5 - 0.9  → RETRAIN
> 0.9      → ESCALATE
```

| # | Drift Type | Severity | Expected | Why |
|---|-----------|----------|----------|-----|
| 1 | covariate | 0.29 | MONITOR | Low severity |
| 2 | concept | 0.35 | INVESTIGATE | Mid-range |
| 3 | both | 0.51 | RETRAIN | High severity |
| 4 | concept | 0.91 | ESCALATE | Critical |
| 5 | covariate | 0.5 | INVESTIGATE | Exact boundary |
| 6 | both | 0.75 | RETRAIN | Normal high |
| 7 | both | 0.95 | ESCALATE | Very critical |
| 8 | unknown | 0.45 | INVESTIGATE | Unknown type |

**Key Insights:**
- ✓ Tests exact threshold boundaries (0.3, 0.5, 0.9)
- ✓ Tests unknown drift type handling
- ✓ Tests severity mapping accuracy

---

## 💡 Why These Tests Matter

### 1. Boundary Testing (Most Important)

**Why Critical:** Off-by-one errors are common
```python
# Wrong: if shift >= 0.20
# Right: if shift > 0.20

# 20% exactly:
# Wrong approach: DETECT ✗
# Right approach: NO (at boundary) ✓
```

**Test Examples:**
- Covariate: 19% vs 20% vs 21% shifts
- Concept: 9% vs 10% vs 11% quality drops
- Action: Severity 0.29 vs 0.30 vs 0.31

### 2. Veto Logic (Real-World Requirement)

**Why Matters:** Real systems have rejections
```
Covariate Detection:
- Quality drop > 5% → ALWAYS reject (it's concept drift, not covariate)

Concept Detection:
- Input changed > 5% → ALWAYS reject (it's covariate, not concept)
- Quality improved → ALWAYS reject (degradation requirement)
```

### 3. Precision Testing (Edge Cases)

**Why Matters:** Financial impact calculations need accuracy
```
Error rate 0.0001 × 70,000 predictions = 7 errors EXACTLY
Not 6.9 or 7.1 - EXACTLY 7
```

### 4. Scale Variation (Production Readiness)

**Why Matters:** Models must work at all scales
```
From: 1 prediction, $1 cost
To: 50,000 predictions/day, $500,000 impact
```

### 5. Consistency (Reliability)

**Why Matters:** Production systems must be deterministic
```
classify_drift(True, False) → ALWAYS returns 'covariate'
Never random, never depends on call order
```

---

## 🚀 Running the Evaluation

### Quick Start

```bash
# Run with progress tracking (concurrent)
python main_WITH_PROGRESS.py

# Expected output:
# [████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 8/10
# Score: 4/5 | Avg: 3.2 | Max: 5/5 | ETA: 120s
```

### Understanding Progress

```
Progress Bar:   [████░░░░░░] Shows overall completion
Current Score:  5/5 (Run N passed all 5 functions)
Average Score:  3.5 (Average across all runs so far)
Max Score:      5/5 (Best score achieved)
ETA:            120s (Estimated time remaining)
```

### Configuration Options

```python
# main_WITH_PROGRESS.py
asyncio.run(main(
    num_runs=10,      # Number of runs (default: 10)
    concurrent=True   # Parallel execution (default: True)
))
```

---

## 📈 Understanding Results

### Final Report Example

```
FINAL REPORT
======================================================================
API Mode: Vertex AI
Model: Claude Sonnet 4.5 (Vertex AI)
Total Runs: 10
Fully Passed Runs (5/5): 2/10
Pass Rate: 20.0%
Target Range (10-40%): ✓ YES
Score Distribution: [5, 4, 4, 3, 3, 2, 2, 1, 1, 0]
======================================================================
```

### Interpretation

| Pass Rate | Status | Meaning |
|-----------|--------|---------|
| 0-10% | Below Target | Tests too hard; adjust tolerances |
| 10-40% | ✓ Optimal | Model has good understanding |
| 40-70% | Above Target | Tests too easy; add complexity |
| 70%+ | Way Too Easy | Tests insufficient |

### Score Distribution Analysis

```
Score: [5, 4, 4, 3, 3, 2, 2, 1, 1, 0]
        ↓
Strong passes (5/5):    2 runs   - 20% perfect
Good passes (3-4/5):    4 runs   - 40% mostly correct
Weak passes (1-2/5):    3 runs   - 30% partial success
Failures (0/5):         1 run    - 10% complete failure

Average: 2.5/5 = 50% of test suite passing
Interpretation: Model understands core concepts but struggles
               with boundaries and edge cases
```

---

## 📋 File Reference

### main_WITH_PROGRESS.py
Updated main file with real-time progress tracking
- ✅ Progress bar during concurrent runs
- ✅ Real-time statistics (avg, max score)
- ✅ ETA countdown
- ✅ Final summary statistics

### Key Changes from Original
```python
# NEW: ProgressTracker class
progress = ProgressTracker(num_runs)
progress.update(run_id, score)  # Called per run
progress.final_stats()          # Called at end

# Shows:
# ✓ Real-time progress bar
# ✓ Current/average/max scores
# ✓ Time elapsed and ETA
```

---

## 🔍 Troubleshooting

### Issue: Tests Taking Too Long
```
Solution: Reduce num_runs or set concurrent=False for debugging
python -c "asyncio.run(main(num_runs=3, concurrent=True))"
```

### Issue: All Runs Failing (0% Pass Rate)
```
Likely Cause: Claude returning wrong data types
Check: Are functions returning dicts with correct keys?
       detect_covariate_drift() → {'detected': bool}
       classify_drift() → {'type': str}
       calculate_drift_impact() → {'predictions_affected': int, 'errors': int, 'financial_impact': float}
```

### Issue: Some Functions Always Pass, Others Always Fail
```
Example: classify_drift always passes, covariate always fails
Likely Cause: Easy vs hard difficulty difference
Solution: Adjust thresholds in grader for specific function
```

---

## 📊 Expected Performance

### Baseline (Generic Implementation)
- Pass Rate: ~30% (3/5 functions)
- Strong: classify_drift (easy logic)
- Weak: covariate/concept drift (boundary logic)

### Strong Implementation
- Pass Rate: ~50% (2-3 functions per run)
- All functions partially working
- Struggles with boundary precision

### Expert Implementation
- Pass Rate: ~70%+ (4-5 functions per run)
- All boundaries correct
- Handles all edge cases
- Consistent performance

---

## 🎓 Learning Outcomes

After this evaluation, you should understand:

1. ✅ **Data Drift Types**
   - Covariate drift (input change)
   - Concept drift (quality change)
   - Combined drift

2. ✅ **Threshold Precision**
   - Why exact boundaries matter
   - Impact of off-by-one errors
   - Tolerance handling

3. ✅ **Business Impact Calculation**
   - Financial impact metrics
   - Scale considerations
   - Precision requirements

4. ✅ **Production ML**
   - Continuous monitoring needs
   - Automated retraining triggers
   - Alert escalation strategies

---

**Last Updated:** November 6, 2025
**Task:** Data Drift Detection RL Evaluation
**Status:** Production Ready ✅
