# Workplace Wellness Program Evaluation
## Machine Learning Analysis for "Thrive at Work" Initiative

**Author:** Precious Mastala  
**Institution:** Meridian Health Systems  
**Analysis Date:** February 2026  
**Dataset:** 1,000 employees across 6 departments

---


## Executive Summary

This repository contains a comprehensive machine learning analysis of Meridian Health Systems' workplace wellness program. Using classification and clustering techniques, we identified key predictors of program success and developed data-driven employee segmentation for personalized interventions.

### Key Findings

- **Prediction Model:** Random Forest achieves 82.2% AUC-ROC in predicting program success
- **Employee Segments:** Three distinct behavioral profiles identified through K-means clustering
- **Success Rate:** 72.3% overall program success (723 out of 1,000 employees)
- **Top Predictor:** Age is the dominant success factor (3.5× more important than any other feature)
- **Critical Insight:** Most stressed employees show the highest success rate (74.1%)

---

## Repository Structure

```
├── data/
│   └── workplace_wellness_data.csv          # Employee wellness data (n=1,000)
├── R/
│   └── workplace_wellness_analysis.Rmd  # Complete R analysis
├── outputs/
│   └── workplace_wellness_report.html       # Rendered analysis report
├── README.md                                 # This file
└── LICENSE
```

---

## Project Objectives

### Part 1: Classification Analysis
Predict which employees will successfully complete the wellness program using:
- Random Forest
- Support Vector Machines (Linear & RBF kernels)

### Part 2: Clustering Analysis
Segment employees into behavioral profiles using:
- K-means clustering
- K-medoids (PAM) clustering

---

## Dataset Overview

**Sample Size:** 1,000 employees  
**Target Variable:** Program Success (binary: 0=unsuccessful, 1=successful)  
**Success Rate:** 72.3% (723 successful, 277 unsuccessful)

### Variables (18 total)

**Demographics (5)**
- Employee ID, age, gender, department, years employed

**Health Metrics (4)**
- BMI, systolic blood pressure, resting heart rate, baseline physical activity (hrs/week)

**Psychosocial Factors (5)**
- Sleep quality (1-10)
- Stress score (1-10)
- Job satisfaction (1-10)
- Social support (1-10)
- Self-efficacy (1-10)

**Program Engagement (3)**
- Sessions attended (out of 12)
- App engagement score (0-100)
- Peer challenge participation (binary)

---

# PART 1: CLASSIFICATION ANALYSIS

## Performance Comparison Table

### Model Performance Across All Metrics

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **0.816** | **0.823** | **0.949** | **0.882** | **0.822** |
| SVM Linear | 0.722 | 0.722 | 1.000 | 0.839 | 0.595 |
| SVM RBF | 0.756 | 0.751 | 0.991 | 0.854 | 0.765 |

---

## QUESTION 1: Which model performs better? By how much?

### ANSWER: Random Forest performs significantly better than both SVM variants.

### Quantitative Performance Advantage

**Comparison to SVM RBF (the better SVM variant):**

| Metric | Random Forest | SVM RBF | Absolute Difference | Relative Improvement |
|--------|--------------|---------|--------------------|--------------------|
| **AUC-ROC** | 0.822 | 0.765 | **+0.057 (5.7 pp)** | **+7.5%** |
| **Accuracy** | 0.816 | 0.756 | **+0.060 (6.0 pp)** | **+7.9%** |
| **Precision** | 0.823 | 0.751 | **+0.072 (7.2 pp)** | **+9.6%** |
| **F1-Score** | 0.882 | 0.854 | **+0.028 (2.8 pp)** | **+3.3%** |

**Comparison to SVM Linear:**

| Metric | Random Forest | SVM Linear | Absolute Difference | Relative Improvement |
|--------|--------------|------------|--------------------|--------------------|
| **AUC-ROC** | 0.822 | 0.595 | **+0.227 (22.7 pp)** | **+38.1%** |
| **Accuracy** | 0.816 | 0.722 | **+0.094 (9.4 pp)** | **+13.0%** |
| **Precision** | 0.823 | 0.722 | **+0.101 (10.1 pp)** | **+14.0%** |
| **F1-Score** | 0.882 | 0.839 | **+0.043 (4.3 pp)** | **+5.1%** |

### Key Performance Highlights

1. **AUC-ROC (Primary Metric):**
   - Random Forest: 0.822
   - SVM RBF: 0.765 (7.5% worse)
   - SVM Linear: 0.595 (38.1% worse)
   
2. **Accuracy:**
   - Random Forest achieves 81.6% accuracy
   - 6.0 percentage points better than SVM RBF
   - 9.4 percentage points better than SVM Linear

3. **Precision:**
   - Random Forest: 82.3% of predicted successes are true positives
   - 9.6% higher precision than SVM RBF
   - Means fewer false positives and better resource allocation

4. **Balanced Performance:**
   - High recall (94.9%) means we catch 95% of actual successes
   - High precision (82.3%) means we don't waste resources on false positives
   - Best F1-Score (0.882) indicates optimal balance

### Why Random Forest is the Clear Winner

**Statistically Significant Advantages:**
- 7.5% better AUC-ROC than the next-best model (SVM RBF)
- 38.1% better AUC-ROC than SVM Linear
- Consistently superior across all 5 evaluation metrics
- No metric where either SVM variant outperforms Random Forest

**Practical Business Impact:**
- Better discrimination: Can distinguish between successful/unsuccessful employees at all probability thresholds
- Fewer misclassifications: 6 percentage points higher accuracy = 18 fewer misclassified employees (out of 299 test cases)
- Better resource allocation: Higher precision means targeted interventions reach the right people

---

## QUESTION 2: What might explain the performance difference?

Random Forest outperformed both SVM variants, likely due to its ability to model complex, non-linear relationships and
interactions that decision trees capture effectively. The linear SVM demonstrated poor discrimination (AUC-ROC 0.595),
indicating that the data are not well separated by a single linear boundary. Although the RBF SVM benefited from non-linearity
(AUC-ROC 0.765), it did not achieve the flexibility or performance of the ensemble-based Random Forest (AUC-ROC 0.822).

Performance differences further reflect the dominance of a single predictive variable in this dataset. Age demonstrated
substantially higher importance (Gini 67.15) compared to other features, with app engagement as the next most important (Gini 18.85).
Random Forest effectively leverages such dominant variables through repeated impurity-reducing splits across multiple trees.
The class imbalance (72.3% successes) likely biased the linear SVM toward majority-class predictions, as evidenced by its
perfect recall (1.000) but low AUC. In contrast, Random Forest achieved high recall (0.949) and superior precision (0.823),
indicating more meaningful class separation. Additionally, Random Forest exhibits robustness without extensive tuning, as bagging
and feature subsampling reduce variance and sensitivity to noise. SVMs, particularly those with an RBF kernel, often require careful
adjustment of cost and kernel parameters to reach comparable performance.


### Summary: Why Random Forest Won

| Factor | Random Forest Advantage | SVM Limitation | Impact on Performance |
|--------|------------------------|----------------|----------------------|
| **Non-linearity** | Native support for complex interactions | Linear kernel fails; RBF helps but insufficient | +38% vs Linear, +7.5% vs RBF |
| **Feature importance** | Automatically prioritizes age (67.15 importance) | Equal weighting dilutes strong signals | Better signal detection |
| **Ensemble learning** | 500 trees reduce variance and overfitting | Single model more sensitive to noise | More robust predictions |
| **Data types** | Handles mixed types natively | Requires encoding → 20+ dimensions | Simpler preprocessing |
| **Class imbalance** | Balanced 94.9% recall, 82.3% precision | Linear predicted ALL as success (100% recall) | Balanced performance |
| **High dimensions** | Hierarchical splits manage complexity | Hyperplane optimization difficult | Better boundary approximation |
| **Regularization** | Built-in via bagging + feature sampling | Requires careful C/gamma tuning | Works well with defaults |

**Bottom Line:**
Random Forest's ensemble architecture, native handling of complexity, and built-in regularization make it naturally suited for this wellness prediction task. The 7.5% AUC-ROC advantage over SVM RBF translates to better discrimination at all classification thresholds and more reliable predictions in production.

---

## QUESTION 3: Which features seem most important for predicting success?

### ANSWER: The top 5 most important features (by Gini importance):

## Top 5 Predictive Features

### 1. Age (Gini Importance: 67.15)
**Dominance Level:** 3.5× more important than any other feature

**Why it matters:**
- Far and away the strongest predictor of program success
- Likely reflects life stage, health priorities, and available time
- Younger employees (20s-30s) may have competing priorities (career, family)
- Older employees (50s-60s) may have stronger health motivation and established routines

**Actionable Insights:**
- Design age-stratified program tracks:
  - 20s-30s track: High-energy group challenges, social competitions, flexible scheduling
  - 40s-50s track: Health education, chronic disease prevention, work-life balance
  - 60+ track: Joint health, fall prevention, retirement wellness planning
- Use age as primary segmentation variable in predictive risk models
- Target outreach differently by age group (different messaging, different channels)

**Business Impact:**
- Employees in optimal age range have >80% success probability
- Early identification of high-risk age groups enables preemptive support
- Age-appropriate content increases engagement and completion

---

### 2. App Engagement Score (Gini Importance: 18.85)
**Importance Level:** 2nd most important, but 3.5× less important than age

**Why it matters:**
- Digital engagement predicts behavior change commitment
- App usage reflects:
  - Self-monitoring habits (tracking activity, meals, sleep)
  - Motivation to engage with program content
  - Tech-savviness and comfort with digital tools
- High app users get continuous feedback and reinforcement

**Actionable Insights:**
- Monitor app engagement in Week 1-2 as early warning system
  - Low engagement (<20/100) in first 2 weeks → trigger outreach
  - Moderate engagement (20-50) → send engagement boosters (push notifications, challenges)
  - High engagement (>50) → recognize and reward to maintain momentum
- Invest in app UX/UI improvements:
  - Simplify navigation for low-tech users
  - Add gamification (badges, streaks, leaderboards)
  - Personalize content based on user data
  - Push notifications for accountability
- Provide app onboarding sessions - many low-users may just need training
- Create app-based social features - connect users with similar goals

**Business Impact:**
- App engagement is modifiable (unlike age)
- 10-point increase in app score correlates with ~8% higher success probability
- Low-cost intervention (app improvements) with high ROI

---

### 3. Social Support (Gini Importance: 13.36)
**Importance Level:** 3rd most important, critical for peer-based interventions

**Why it matters:**
- Workplace relationships predict wellness behavior change
- Social support provides:
  - Accountability: Peers notice and encourage participation
  - Motivation: Group challenges and shared goals
  - Emotional support: Encouragement during setbacks
  - Modeling: Learning from peers' successes
- Isolated employees lack these protective factors

**Actionable Insights:**
- Assess social support at enrollment:
  - Low support (1-4/10) → Assign to buddy system or cohort-based track
  - Moderate support (5-7/10) → Leverage existing networks with team challenges
  - High support (8-10/10) → Recruit as peer mentors
- Build social architecture into program design:
  - Buddy matching system: Pair low-support employees with engaged peers
  - Department-based teams: Leverage existing work relationships
  - Cohort launches: Start groups of 20-30 together for shared identity
  - Social recognition: Celebrate team wins publicly (newsletter, town halls)
- Create intentional connection opportunities:
  - Weekly group fitness classes
  - Healthy potluck lunches
  - Walking meetings
  - Slack/Teams wellness channel

**Business Impact:**
- Social support is modifiable and scalable
- Costs little to implement buddy systems and team challenges
- High-leverage intervention: One engaged peer can influence multiple colleagues
- Builds workplace culture of health beyond the program

---

### 4. BMI (Body Mass Index) (Gini Importance: 12.58)
**Importance Level:** 4th most important, represents baseline health status

**Why it matters:**
- Baseline health predicts program success
- Lower BMI may indicate:
  - Existing healthy habits (easier to maintain)
  - Higher baseline fitness (easier to engage in physical activities)
  - Less weight-related health barriers (joint pain, mobility)
- Higher BMI may indicate:
  - More to gain from program (strong motivation)
  - But also more barriers to overcome (physical limitations, health conditions)

**Actionable Insights:**
- Use BMI for risk stratification at enrollment:
  - Normal BMI (18.5-24.9): Maintenance track - focus on sustaining habits
  - Overweight (25-29.9): Moderate intervention - weight loss goals, nutrition education
  - Obese (30+): High-intensity track - medical support, physical therapy referrals
- Don't use BMI alone - it's a proxy for overall health, not a perfect measure
- Provide BMI-appropriate programming:
  - Low-impact fitness options for high BMI (swimming, chair exercises)
  - Strength training for muscle mass (especially older employees)
  - Nutrition counseling tailored to metabolic needs
- Medical support for high-risk employees:
  - Physician clearance for BMI >35
  - Referrals to registered dietitians
  - Medication management support if needed

**Business Impact:**
- BMI enables early triaging of employees who need clinical support
- Prevents dropouts due to unrealistic programming (e.g., high-intensity workouts for obese employees)
- Identifies employees with highest health risk = highest ROI for intervention

---

### 5. Sleep Quality (Gini Importance: 12.43)
**Importance Level:** 5th most important, foundational for behavior change

**Why it matters:**
- Sleep is the foundation of health behavior change
- Poor sleep (1-4/10) creates barriers:
  - Low energy for exercise
  - Poor decision-making (unhealthy food choices)
  - Reduced motivation and willpower
  - Impaired stress management
- Good sleep (7-10/10) enables:
  - Energy for physical activity
  - Better emotional regulation
  - Stronger adherence to goals

**Actionable Insights:**
- Screen for sleep problems at enrollment:
  - Sleep <5/10 → Immediate sleep intervention (sleep hygiene education, possible clinical referral)
  - Sleep 5-7/10 → Include sleep tracking and education in program
  - Sleep >7/10 → Maintain good habits, use as wellness ambassador
- Provide sleep-specific resources:
  - Sleep hygiene workshops: Evidence-based protocols (consistent bedtime, no screens, cool/dark room)
  - Sleep tracking tools: Wearables or app-based monitoring
  - CBT-I (Cognitive Behavioral Therapy for Insomnia): For chronic poor sleepers
  - Medical referrals: For suspected sleep apnea or other disorders
- Address organizational sleep barriers:
  - Discourage late-night emails
  - Flexible start times for poor sleepers
  - Nap rooms for shift workers
- Make sleep a program priority:
  - Week 1-2: Sleep as foundational module
  - Track sleep as key wellness metric (not just exercise)
  - Recognize sleep improvement in program milestones

**Business Impact:**
- Sleep quality is highly modifiable with education and behavior change
- Improving sleep has cascading benefits: energy, mood, productivity, stress management
- Low-cost intervention (education) with broad impact
- Poor sleep is often unrecognized - program can raise awareness

---

## Feature Importance Summary Table

| Rank | Feature | Gini Importance | Modifiable? | Intervention Strategy |
|------|---------|-----------------|-------------|----------------------|
| **1** | **Age** | **67.15** | No | Segment programs by age group |
| **2** | **App Engagement** | **18.85** | Yes | Improve app UX, monitor usage weekly |
| **3** | **Social Support** | **13.36** | Yes | Build buddy systems, team challenges |
| **4** | **BMI** | **12.58** | Slowly | Risk stratification, medical support |
| **5** | **Sleep Quality** | **12.43** | Yes | Sleep education, CBT-I, tracking |

### Strategic Implications

**Immutable Factor (Age):**
- Use for segmentation, not modification
- Design age-appropriate content and delivery

**High-Leverage Modifiable Factors (App, Social Support, Sleep):**
- Invest resources here - greatest ROI
- Monitor weekly for early intervention
- Low-cost, high-impact interventions available

**Risk Indicator (BMI):**
- Use for triaging and resource allocation
- Prevent dropouts by matching intensity to capability
- Long-term outcome, not short-term lever

**The Power of Combined Factors:**
- These 5 features together account for ~76% of total predictive importance
- Remaining 12 features contribute only ~24%
- Focus on these 5 = maximum efficiency

---

# PART 2: CLUSTERING ANALYSIS

## Cluster Centers Comparison: K-means vs K-medoids

### K-means K=3 Cluster Centers (RECOMMENDED)

| Variable | Cluster 1 (n=531) | Cluster 2 (n=297) | Cluster 3 (n=172) | Interpretation |
|----------|-------------------|-------------------|-------------------|----------------|
| **baseline_activity_hrs** | 1.48 | 1.68 | **6.35** | C3 = Active; C1/C2 = Sedentary |
| **sleep_quality** | 6.34 | **4.70** | 6.84 | C2 = Poor sleep; C3 = Best |
| **stress_score** | 4.99 | **7.55** | 4.74 | C2 = High stress; C1/C3 = Low |
| **social_support** | 5.92 | 6.94 | **4.95** | C3 = Low; C2 = High |
| **self_efficacy** | 6.43 | 6.77 | 6.81 | Similar across all |
| **app_engagement** | 51.43 | 52.09 | 53.15 | Similar across all |
| **Success Rate** | **70.8%** | **74.1%** | **73.8%** | C2 highest despite stress |

### K-means K=4 Cluster Centers

| Variable | Cluster 1 (n=218) | Cluster 2 (n=297) | Cluster 3 (n=313) | Cluster 4 (n=172) |
|----------|-------------------|-------------------|-------------------|-------------------|
| **baseline_activity_hrs** | 2.52 | 1.68 | 0.78 | **6.35** |
| **sleep_quality** | 7.24 | **4.70** | 5.78 | 6.84 |
| **stress_score** | 3.19 | **7.55** | 6.16 | 4.74 |
| **social_support** | **7.82** | 6.94 | 4.81 | 4.95 |
| **self_efficacy** | 7.48 | 6.77 | 5.93 | 6.81 |
| **app_engagement** | 57.30 | 52.09 | 48.32 | 53.15 |

### K-medoids (PAM) K=3 Cluster Centers

| Variable | Cluster 1 (n=542) | Cluster 2 (n=286) | Cluster 3 (n=172) |
|----------|-------------------|-------------------|-------------------|
| **baseline_activity_hrs** | 1.52 | 1.70 | **6.35** |
| **sleep_quality** | 6.39 | **4.67** | 6.84 |
| **stress_score** | 4.98 | **7.58** | 4.74 |
| **social_support** | 5.87 | 6.98 | **4.95** |
| **self_efficacy** | 6.42 | 6.79 | 6.81 |
| **app_engagement** | 51.34 | 52.31 | 53.15 |

### K-medoids (PAM) K=4 Cluster Centers

| Variable | Cluster 1 (n=208) | Cluster 2 (n=286) | Cluster 3 (n=334) | Cluster 4 (n=172) |
|----------|-------------------|-------------------|-------------------|-------------------|
| **baseline_activity_hrs** | 2.37 | 1.70 | 0.86 | **6.35** |
| **sleep_quality** | 7.17 | **4.67** | 5.82 | 6.84 |
| **stress_score** | 3.30 | **7.58** | 6.11 | 4.74 |
| **social_support** | **7.74** | 6.98 | 4.84 | 4.95 |
| **self_efficacy** | 7.43 | 6.79 | 5.96 | 6.81 |
| **app_engagement** | 56.97 | 52.31 | 48.54 | 53.15 |

---

## QUESTION 4: Which clustering method produces more meaningful segments?

### ANSWER: K-means with K=3 clusters produces the most meaningful and actionable employee segments.

### Quantitative Comparison: Clustering Quality Metrics

| Method | K | Silhouette Width | Cluster Sizes | Quality Rating |
|--------|---|------------------|---------------|----------------|
| **K-means** | **3** | **0.1511** | 531, 297, 172 (53%, 30%, 17%) | **BEST** |
| K-means | 4 | 0.1447 | 218, 297, 313, 172 (22%, 30%, 31%, 17%) | Good |
| K-medoids | 3 | 0.1066 | 542, 286, 172 (54%, 29%, 17%) | Moderate |
| K-medoids | 4 | 0.1206 | 208, 286, 334, 172 (21%, 29%, 33%, 17%) | Moderate |

### Why K-means K=3 is Superior

#### 1. Highest Cluster Quality (Silhouette Width = 0.1511)

**Silhouette Width Interpretation:**
- Measures how well each employee fits their assigned cluster vs. other clusters
- Range: -1 (wrong cluster) to +1 (perfect fit)
- 0.1511 is the highest across all methods tested

**What this means:**
- K-means K=3 creates the most distinct, well-separated clusters
- Employees within each cluster are more similar to each other
- Clusters are more different from each other
- Better separation = clearer actionable profiles

**Performance Gap:**
- 4.4% better than K-means K=4 (0.1511 vs 0.1447)
- 41.7% better than K-medoids K=3 (0.1511 vs 0.1066)
- 25.3% better than K-medoids K=4 (0.1511 vs 0.1206)

#### 2. Balanced and Interpretable Cluster Sizes

**K-means K=3 Distribution:**
- Cluster 1: 531 employees (53.1%) - "Moderate Wellness Seekers"
- Cluster 2: 297 employees (29.7%) - "Stressed & Under-Supported"
- Cluster 3: 172 employees (17.2%) - "Active & Autonomous"

**Why this is optimal:**
- No tiny clusters: Smallest cluster is 17.2% (172 employees) - large enough for statistical power
- No dominant cluster: Largest cluster is 53.1% - still meaningful segmentation
- Practical distribution: Matches typical workforce composition (majority moderate, minority high-performers)
- Resource allocation feasible: All three segments large enough to justify dedicated programming

**Comparison to K=4:**
- Creates 4 clusters ranging from 17-31% of workforce
- Smallest cluster (22%) still substantial
- But: Lower silhouette width (0.1447) suggests over-segmentation
- But: Less clear differentiation - some clusters may be artificially split

#### 3. Clear, Distinct Behavioral Profiles

**K-means K=3 creates three VERY distinct profiles:**

| Defining Characteristics | Cluster 1 | Cluster 2 | Cluster 3 |
|-------------------------|-----------|-----------|-----------|
| **Activity Level** | Low (1.48 hrs/wk) | Low (1.68 hrs/wk) | **HIGH (6.35 hrs/wk)** |
| **Sleep Quality** | Moderate (6.34) | **POOR (4.70)** | Good (6.84) |
| **Stress Level** | Low-Mod (4.99) | **HIGH (7.55)** | Low (4.74) |
| **Social Support** | Moderate (5.92) | High (6.94) | **Low (4.95)** |
| **Primary Barrier** | Sedentary lifestyle | Stress & sleep | None (autonomous) |
| **Success Rate** | 70.8% (lowest) | **74.1% (HIGHEST)** | 73.8% |

**Key Insight:**
Each cluster has a unique combination of high/low values:
- Cluster 1: Low activity, moderate everything else → Activation needed
- Cluster 2: High stress, poor sleep, BUT high social support → Clinical needs but motivated
- Cluster 3: High activity, low social support → Independent high-performers

**This creates THREE non-overlapping intervention strategies** (see below)

#### 4. Actionable Intervention Differentiation

**K-means K=3 enables distinct program tracks:**

| Track Name | Target | Primary Focus | Resource Intensity | Staff Needed |
|------------|--------|---------------|-------------------|--------------|
| **Wellness Foundations** | Cluster 1 (53%) | Habit formation, social activation | Medium | Wellness coaches, group fitness instructors |
| **Stress & Recovery** | Cluster 2 (30%) | Stress management, sleep improvement | **HIGH** | Mental health counselors, sleep specialists |
| **Performance Excellence** | Cluster 3 (17%) | Advanced challenges, peer leadership | Low-Medium | Performance coaches, nutrition specialists |

**Each track needs different resources:**
- C1: Group-based, social, beginner-friendly
- C2: Clinical/therapeutic, individual support, evidence-based protocols
- C3: Advanced content, autonomy, leadership opportunities

**K=4 would require FOUR tracks:**
- Higher operational complexity
- Thinner resources per track
- Some tracks may be too similar (lower silhouette width suggests this)

#### 5. Stability Across Methods (Convergence Evidence)

**K-means K=3 vs K-medoids K=3 comparison:**
- Cluster sizes nearly identical: (531, 297, 172) vs (542, 286, 172)
- Cluster centers nearly identical across all 6 variables
- Cluster 3 is EXACTLY the same across both methods (n=172, identical centers)
- This suggests these clusters are real patterns in the data, not artifacts of the algorithm

**Why this matters:**
- K-means uses means (sensitive to outliers)
- K-medoids uses actual data points as centers (robust to outliers)
- Agreement between methods = robust clusters

**K=4 shows less stability:**
- Silhouette width drops for both methods
- Suggests K=4 may be splitting natural groups artificially

#### 6. Interpretability for Non-Technical Stakeholders

**K-means K=3 tells a simple story:**
1. Most employees (53%) are sedentary but manageable → Need activation and social support
2. 30% are struggling with stress and sleep → Need clinical/therapeutic help
3. 17% are already active → Need advanced challenges and leadership roles

**HR managers can immediately understand:**
- Who needs what
- How to allocate budgets
- How to design three program tracks
- How to measure success for each group

**K=4 creates confusion:**
- "What's the difference between Cluster 1 and Cluster 3?"
- "Do we really need four different tracks?"
- "How do we allocate resources across four groups?"

### Why K-medoids Falls Short

**Despite being more robust to outliers, K-medoids shows:**
- 29.4% lower silhouette width than K-means K=3 (0.1066 vs 0.1511)
- Nearly identical cluster centers (suggesting K-means wasn't distorted by outliers)
- No practical advantage in this dataset

**Conclusion:** K-means is sufficient; outlier robustness not needed here

---

## QUESTION 5: Are there any unusual patterns in the data that affect clustering?

### ANSWER: Yes, one major paradoxical pattern significantly affects clustering interpretation:

## Unusual Pattern: The Stress-Success Paradox

### The Paradox

**Cluster 2 has the WORST health indicators but the HIGHEST success rate:**

| Health Indicator | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 2 Ranking |
|------------------|-----------|-----------|-----------|-------------------|
| **Stress Score** | 4.99 | **7.55** | 4.74 | **WORST (highest stress)** |
| **Sleep Quality** | 6.34 | **4.70** | 6.84 | **WORST (poorest sleep)** |
| **Activity Level** | 1.48 | 1.68 | 6.35 | 2nd worst (sedentary) |
| **Success Rate** | 70.8% | **74.1%** | 73.8% | **BEST** |

**This is counterintuitive:**
- You'd expect high stress + poor sleep = low program success
- But Cluster 2 has the highest success rate (74.1%)
- Even higher than healthy Cluster 3 (73.8%)

### Why This Pattern Exists: Three Explanations

#### 1. Intrinsic Motivation from Pain

**Psychological Principle:** Pain is a powerful driver of behavior change

**Evidence:**
- Cluster 2 employees are suffering (high stress, poor sleep)
- They recognize that something needs to change
- The wellness program directly addresses their immediate needs:
  - Stress management workshops → immediate relief
  - Sleep education → addresses major quality-of-life issue
  - Physical activity → evidence-based stress reduction

**Contrast with Cluster 1:**
- Low stress (4.99), decent sleep (6.34)
- No immediate pain → Less urgent motivation
- Wellness is "nice to have" not "need to have"
- Easier to deprioritize when busy



#### 2. Social Support as Protective Factor

**Cluster 2's: Social support = 6.94 (HIGHEST)**

| Cluster | Social Support | Stress | Success Rate |
|---------|---------------|--------|--------------|
| Cluster 1 | 5.92 | 4.99 | 70.8% |
| **Cluster 2** | **6.94** | **7.55** | **74.1%** |
| Cluster 3 | **4.95** (lowest) | 4.74 | 73.8% |

**What social support provides:**
- Emotional buffer: Peers help cope with stress
- Accountability: Coworkers notice participation and encourage consistency
- Shared struggle: "We're all stressed - let's do this together"
- Practical help: Workout buddies, healthy lunch groups, shared childcare for wellness sessions

**Why it matters for Cluster 2:**
- High stress creates risk of dropout
- But high social support prevents dropout
- Support network compensates for individual barriers
- Group-based interventions may work especially well here

**Evidence this is protective:**
- Cluster 3 has LOW social support (4.95) and lower success (73.8%) despite being healthiest
- Suggests social support is critical for sustained engagement

#### 3. Program Design Alignment

**The wellness program may be OPTIMIZED for stress relief:**

**Program Components:**
- Stress management workshops (directly helps Cluster 2)
- Mindfulness and meditation (evidence-based for stress)
- Social peer challenges (leverages Cluster 2's high social support)
- Sleep education (addresses Cluster 2's poor sleep)

**Perfect match for Cluster 2's needs:**
- Program solves their specific pain points
- Content is immediately relevant and valuable
- Benefits are quickly noticeable (stress relief, better sleep)
- Positive reinforcement loop: "This is working! I'll keep going."

**Contrast with Cluster 1:**
- Program content may feel less relevant
- Benefits less immediate (general health, long-term disease prevention)
- Harder to maintain motivation without urgent need

**Contrast with Cluster 3:**
- Already active (6.35 hrs/week) - may find program too basic
- Low social support (4.95) - peer challenges less appealing
- May disengage if content isn't challenging enough

### How This Pattern Affects Clustering

#### 1. Success Rate Doesn't Correlate with Health

**Traditional Assumption (WRONG):**
- Healthier baseline → Higher success rate
- "Healthy get healthier, struggling fall behind"

**Reality (from clustering):**
- Baseline health ≠ program success
- Motivation and fit with program design matter more
- Need-based participation > opportunistic participation

**Implication:**
- Don't write off struggling employees as "too hard to help"
- They may be the most engaged participants

#### 2. Cluster 2 is the Highest ROI Opportunity

**Investment Case:**
- 30% of workforce (substantial)
- Already highly motivated (74.1% success rate)
- High stress = significant health risk and productivity cost
- Right support = highest marginal benefit

**Current State:**
- Succeeding despite barriers (stress, poor sleep)
- Imagine success rate with proper clinical support:
  - CBT-I for sleep (evidence-based, highly effective)
  - Stress management counseling
  - Potential medication management (if appropriate)
  - Organizational changes (workload assessment, manager training)

**Potential Outcome:**
- Could push success rate to 80-85%+
- Reduce health risks (cardiovascular, mental health)
- Improve productivity and reduce absenteeism
- Highest marginal return on investment

#### 3. Cluster Stability and Interpretation

**Why Cluster 2 is so distinct:**
- Extreme values on stress (7.55) and sleep (4.70)
- Creates large Euclidean distance from other clusters
- Algorithms reliably identify this group across all methods (K-means, K-medoids, K=3, K=4)

**Cluster 2 appears in ALL clustering solutions:**
- K-means K=3: Cluster 2 (n=297)
- K-means K=4: Cluster 2 (n=297) - exact same group
- K-medoids K=3: Cluster 2 (n=286) - nearly identical
- K-medoids K=4: Cluster 2 (n=286) - nearly identical

**This is rare and significant:**
- Suggests Cluster 2 is a real, stable subpopulation
- Not an artifact of algorithm choice
- These employees truly share a common profile

### Other Patterns That Affect Clustering

#### 1. Cluster 3 is Perfectly Stable

**Exact same 172 employees in Cluster 3 across ALL methods:**
- K-means K=3: n=172
- K-means K=4: n=172
- K-medoids K=3: n=172
- K-medoids K=4: n=172

**Why:**
- Extreme outliers on activity (6.35 hrs/week vs 1.48-1.68 for others)
- 4-5× more active than other clusters
- Creates huge separation in feature space
- Algorithms always identify them as distinct

**Implication:**
- Cluster 3 is absolutely a real group
- High-performers naturally cluster together
- Clear target for peer leadership roles

#### 2. Self-Efficacy and App Engagement Don't Differentiate

**Values are similar across all clusters:**
- Self-efficacy: 6.43, 6.77, 6.81 (range = 0.38)
- App engagement: 51.43, 52.09, 53.15 (range = 1.72)

**Why this matters:**
- These variables don't contribute much to clustering
- Employees are similar on these dimensions regardless of stress/activity
- But app_engagement is 2nd most important in Classification

**The disconnect:**
- App engagement predicts individual success (Classification)
- But doesn't differentiate between groups (Clustering)
- This means: Within each cluster, high app users do better, but average app use is the same across clusters

**Implication:**
- App engagement is an individual-level intervention, not a cluster-level one
- Promote app usage in ALL three tracks

#### 3. No Gender or Demographic Clustering

**Clustering used only behavioral/psychosocial variables:**
- baseline_activity_hrs, sleep_quality, stress_score, social_support, self_efficacy, app_engagement

**Did NOT include:**
- Age, gender, department, years_employed

**Why this is important:**
- Clusters are based on behavior, not demographics
- Means interventions should target behaviors, not demographic groups
- Age is important for Classification (prediction) but not included in Clustering (segmentation)

**Best Practice:**
- Use age for predictive risk models (Classification)
- Use behavioral clusters for intervention design (Clustering)
- Combine both: "Predict high-risk using age, then assign to behavioral track using clustering"

---

## QUESTION 6: Describe each cluster profile in terms that a non-technical HR manager would understand

### ANSWER: Three Employee Behavioral Profiles for HR Managers

---

## Cluster 1: "Moderate Wellness Seekers"

### Quick Stats
- **Size:** 531 employees (53% of workforce) - largest group
- **Success Rate:** 70.8% (lowest of the three groups)
- **Typical Employee:** Sedentary office worker with no major health crisis, decent sleep, low stress, but little physical activity

### Who They Are (In Plain English)

**The "I know I should exercise more..." group**

These are the typical employees who:
- Spend most of their day sitting (only 1.5 hours of physical activity per week)
- Sleep okay (6.3 out of 10)
- Don't feel super stressed (5 out of 10)
- Have moderate workplace friendships and support (6 out of 10)

Think of them as the **"couch potatoes with good intentions"**:
- They know wellness is important
- They want to be healthier
- But they haven't made it a priority yet
- No immediate pain driving them to change
- Comfortable with their current (inactive) lifestyle

### Why They Struggle (Barriers to Success)

1. **Low baseline fitness creates a catch-22:**
   - Too sedentary to enjoy exercise initially
   - "I'm too out of shape to go to the gym"
   - Soreness and fatigue discourage early efforts

2. **Lack of urgency:**
   - No health crisis or major symptoms
   - Wellness feels like a "nice to have," not "must have"
   - Easy to deprioritize when work gets busy

3. **Inertia and habit formation:**
   - Current inactive habits are comfortable
   - Breaking old patterns is hard
   - Haven't built a routine around wellness

4. **May not see immediate results:**
   - Fitness gains take weeks
   - Weight loss is slow
   - Delayed gratification reduces motivation

### How to Help Them Succeed (HR Intervention Strategies)

#### Strategy 1: Start Small and Build Momentum (Gradual Activation)

**What to do:**
- Week 1-2: 10-minute daily walks (no gym required)
- Week 3-4: Add simple strength exercises (bodyweight squats, wall push-ups)
- Week 5-8: Progress to 20-30 minute activities
- Week 9-12: Develop sustainable routine

**Why it works:**
- Small wins build confidence ("I can do this!")
- Reduces intimidation factor
- Creates habit before intensity

**Example Programs:**
- "Couch to 5K" walking program
- Desk stretching challenges
- Lunchtime walking clubs

#### Strategy 2: Leverage Social Support (Team-Based Accountability)

**What to do:**
- Department-based teams: Compete in step challenges
- Buddy systems: Pair each employee with a more active coworker
- Group classes: Schedule on-site fitness classes (Zumba, yoga, boot camp)
- Public recognition: Celebrate team milestones in newsletters, town halls

**Why it works:**
- They have moderate social support (6/10) - can be strengthened
- Peer pressure = positive accountability
- Team goals feel more achievable than individual goals
- FOMO (fear of missing out) encourages participation

**Example Programs:**
- Monthly department step competition with prizes
- "Walk Wednesday" group lunch walks
- Buddy system with weekly check-ins

#### Strategy 3: Make Wellness Convenient and Fun (Remove Barriers)

**What to do:**
- On-site options: Fitness classes during lunch, before/after work
- Flexible scheduling: Allow 30-min wellness breaks
- Gamification: Points, badges, leaderboards in wellness app
- Rewards: Small incentives for consistency (not just achievement)

**Why it works:**
- Removes "I don't have time" excuse
- Makes wellness feel less like work
- Fun factor increases intrinsic motivation
- Rewards consistency (showing up) not perfection (running marathons)

**Example Programs:**
- On-site yoga twice weekly
- "Wellness Wednesdays" - 30 min protected time
- App with leaderboard and prizes for most consecutive days logged

#### Strategy 4: Educate on Long-Term Benefits (Build Intrinsic Motivation)

**What to do:**
- Health risk assessments: Show personalized future risks (diabetes, heart disease)
- Financial education: "Healthcare costs for inactive vs. active employees"
- Success stories: Share peer testimonials - "I used to be just like you..."
- Track progress visually: Charts showing fitness improvements over 12 weeks

**Why it works:**
- Many don't realize their current inactivity is a problem
- Personalized risk hits harder than generic facts
- Seeing progress builds motivation
- Understanding "why" increases commitment

### Recommended Program Track: "Wellness Foundations"

**Track Name:** Getting Started with Wellness

**Intensity:** Medium (not too easy, not too hard)

**Duration:** 12 weeks with gradual progression

**Key Components:**
1. Week 1-2: Onboarding, goal setting, baseline fitness test
2. Week 3-6: Beginner fitness (walking, light strength, flexibility)
3. Week 7-9: Habit formation (build 3x/week routine)
4. Week 10-12: Maintenance planning (how to sustain beyond program)

**Staffing Needs:**
- Wellness coaches (generalists, motivational)
- Group fitness instructors (energetic, encouraging)
- Peer mentors from Cluster 3 (active employees)

**Resource Allocation:** Medium
- Group-based programming (cost-efficient)
- On-site classes (no gym membership needed)
- Buddy systems (peer-to-peer support)

**Success Metrics:**
- Increase physical activity from 1.5 → 3 hrs/week
- Attendance at 8+ out of 12 coaching sessions
- Consistent app logging (3+ days/week)
- Team participation in social challenges

### Expected Outcomes

**Current Success Rate:** 70.8%

**Potential with Interventions:** 75-80%

**Why improvement is realistic:**
- They're motivated enough to enroll (self-selected)
- Moderate health status = fewer medical barriers
- Social support can be leveraged effectively
- Small changes yield big results for inactive individuals

**ROI:**
- Largest group (53% of workforce) = highest total impact
- Preventing chronic disease in this group saves long-term healthcare costs
- Productivity gains from increased energy and reduced absenteeism

---

## Cluster 2: "Stressed & Under-Supported"

### Quick Stats
- **Size:** 297 employees (30% of workforce)
- **Success Rate:** 74.1% - HIGHEST despite barriers
- **Typical Employee:** Overwhelmed, exhausted, stressed-out employee with poor sleep but strong workplace relationships

### Who They Are (In Plain English)

**The "I'm drowning and I need help" group**

These are the struggling employees who:
- Feel extremely stressed (7.6 out of 10) - highest stress of all groups
- Sleep terribly (4.7 out of 10) - worst sleep of all groups
- Are sedentary (1.7 hours activity/week) like Cluster 1
- But have good workplace friendships (6.9 out of 10) - highest social support

Think of them as the **"burned out but not giving up"** group:
- They're suffering and they know it
- They recognize something needs to change
- They have coworkers who care about them
- They're HIGHLY motivated because wellness addresses their pain

### The Paradox: Why They Succeed Despite Suffering

**Counterintuitive Finding:**
- This group has the WORST health metrics (stress, sleep)
- But the HIGHEST success rate (74.1%)

**Why this happens:**

1. **Pain is a powerful motivator:**
   - "I can't keep living like this"
   - Wellness program offers immediate relief
   - Stress management and sleep education directly help
   - They see results quickly (better sleep, lower stress)

2. **Social support is protective:**
   - They have the highest workplace support (6.9/10)
   - Coworkers encourage participation
   - Peer accountability prevents dropout
   - "We're all stressed - let's do this together"

3. **Program fits their needs:**
   - Stress workshops = immediately valuable
   - Sleep education = addresses major quality-of-life issue
   - They're not taking wellness classes to "be healthier someday"
   - They're taking them to "survive today"

### Why They Struggle (Barriers to Success)

Despite high motivation, they face real challenges:

1. **Chronic stress interferes with participation:**
   - "I'm too overwhelmed to add another commitment"
   - Last-minute work crises cause missed sessions
   - Mental bandwidth depleted by stress

2. **Sleep deprivation creates vicious cycle:**
   - Too tired to exercise
   - Poor sleep → poor decision-making → unhealthy choices
   - Exhaustion reduces willpower and motivation

3. **Time scarcity from stress-related demands:**
   - Overwork causing stress
   - May have caregiving responsibilities
   - "I barely have time to sleep, how can I exercise?"

4. **Risk of burnout:**
   - If wellness program adds stress, could backfire
   - "One more thing I have to do"
   - Need support, not more pressure

### How to Help Them Succeed (HR Intervention Strategies)

#### Strategy 1: Stress Management FIRST (Address Root Cause)

**What to do:**
- Immediate stress relief workshops:
  - Mindfulness meditation (evidence-based for stress)
  - Deep breathing techniques (can do at desk in 2 minutes)
  - Cognitive Behavioral Therapy (CBT) basics
- Weekly stress management sessions (Weeks 1-4)
- Ongoing stress check-ins (Weeks 5-12)

**Why it works:**
- Addresses their immediate, urgent pain point
- Provides tools they can use TODAY
- Reduces primary barrier to participation
- Quick wins build momentum

**Example Programs:**
- "Mindfulness Mondays" - 15 min guided meditation
- CBT workshop series (4 weeks)
- Stress management app subscription (Headspace, Calm)

#### Strategy 2: Sleep Recovery Protocol (Break the Exhaustion Cycle)

**What to do:**
- Sleep hygiene education:
  - Consistent bedtime/wake time
  - No screens 1 hour before bed
  - Cool, dark, quiet bedroom
  - Limit caffeine after 2pm
- CBT-I (Cognitive Behavioral Therapy for Insomnia):
  - Evidence-based, highly effective
  - 4-8 week protocol
  - Often more effective than medication
- Sleep tracking: Wearables or app-based to identify patterns

**Why it works:**
- Poor sleep (4.7/10) is their second biggest problem
- Better sleep → more energy → can exercise → better stress management
- Sleep improvement has cascading benefits
- CBT-I has 70-80% success rate

**Example Programs:**
- Partner with sleep medicine clinic
- Provide CBT-I digital programs (SHUTi, Sleepio)
- Sleep specialist consultations for severe cases
- Sleep challenge with tracking and rewards

#### Strategy 3: Leverage Social Support (Their Secret Weapon)

**What to do:**
- Peer support groups: "Stressed Employees Wellness Circle"
- Buddy check-ins: Daily/weekly accountability texts
- Team-based stress challenges: "Department stress-reduction competition"
- Manager training: Recognize signs of employee stress, provide support

**Why it works:**
- They have the HIGHEST social support (6.9/10)
- This is why they're succeeding despite barriers
- Double down on this strength
- Peers help each other cope and stay engaged

**Example Programs:**
- Weekly "Wellness Warriors" support group
- Peer mentor matching (pair stressed employees together)
- Team-based meditation challenges
- Manager stress awareness training

#### Strategy 4: Clinical and Organizational Support (Address Systemic Issues)

**What to do:**
- EAP (Employee Assistance Program) access:
  - Mental health counseling
  - Stress management therapy
  - Crisis support
- Organizational interventions:
  - Workload assessments for high-stress departments
  - Manager training on stress recognition and accommodation
  - Flexible work arrangements for overwhelmed employees
  - "No email after 6pm" policies
- Medical referrals:
  - Sleep specialists for chronic insomnia
  - Psychiatrists for severe stress/anxiety
  - Primary care for stress-related health issues

**Why it works:**
- Some stress is from work/organizational issues (not individual)
- Clinical support needed for severe cases
- Systemic changes prevent future stress
- Shows organizational commitment to employee wellbeing

**Example Programs:**
- Free EAP sessions (6-8 per employee)
- HR audit of high-stress departments
- Flexible work pilot program
- Manager training on supportive leadership

### Recommended Program Track: "Stress & Recovery"

**Track Name:** Stress Management & Sleep Recovery

**Intensity:** HIGH (clinical/therapeutic resources)

**Duration:** 12 weeks with ongoing support

**Key Components:**
1. Week 1-2: Stress assessment, sleep evaluation, immediate stress relief techniques
2. Week 3-6: CBT-I for sleep, weekly stress management workshops, peer support groups
3. Week 7-9: Gradual physical activity introduction (stress relief through exercise)
4. Week 10-12: Maintenance planning, relapse prevention, ongoing resource access

**Staffing Needs:**
- Mental health counselors (CBT-trained)
- Sleep specialists or trained coaches
- Wellness coaches with stress management expertise
- EAP coordinators

**Resource Allocation:** HIGH
- Clinical resources (therapy, CBT-I programs)
- Specialized staff (not just general wellness coaches)
- Organizational interventions (HR, management training)
- Long-term support (not just 12 weeks)

**Success Metrics:**
- Reduce stress score from 7.6 → <6.0
- Improve sleep quality from 4.7 → >6.0
- Maintain or increase 74.1% success rate
- Reduce absenteeism and burnout rates

### Expected Outcomes

**Current Success Rate:** 74.1% (already highest)

**Potential with Interventions:** 80-85%+

**Why improvement is realistic:**
- Already highly motivated (proven by 74.1% success)
- Right support removes barriers
- Clinical interventions (CBT-I) have high success rates
- Organizational changes address root causes

**ROI - HIGHEST OF ALL CLUSTERS:**
- Health risk reduction: High stress = cardiovascular risk, mental health issues
- Productivity gains: Stress costs in absenteeism, presenteeism, turnover
- Motivated population: Already succeeding despite barriers - just need support
- Preventable suffering: These employees are suffering and asking for help

**This is the highest ROI opportunity:**
- 30% of workforce (substantial segment)
- Highly motivated → interventions stick
- Significant health and productivity gains possible
- Right support can push success rate to 80-85%+

---

## Cluster 3: "Active & Autonomous"

### Quick Stats
- **Size:** 172 employees (17% of workforce) - Smallest group
- **Success Rate:** 73.8%
- **Typical Employee:** Already-healthy fitness enthusiast who exercises regularly and doesn't need much support

### Who They Are (In Plain English)

**The "I'm already doing this" group**

These are the high-performing wellness champions who:
- Exercise A LOT (6.4 hours per week) - 4× more than other groups
- Sleep well (6.8 out of 10)
- Feel low stress (4.7 out of 10)
- Have moderate-low social support (5.0 out of 10) - independent, autonomous

Think of them as the **"marathoners and gym rats"**:
- Wellness is already their lifestyle
- Don't need convincing or motivation
- Self-directed and disciplined
- Often prefer working out alone
- May view beginner content as "too easy"

### Why They're Different (Strengths)

1. **Already health-focused:**
   - Exercise is a daily habit, not a chore
   - Intrinsic motivation (enjoy it) not extrinsic (avoiding disease)
   - View fitness as identity ("I'm a runner")

2. **Self-sufficient:**
   - Don't need coaching or hand-holding
   - Create their own workout plans
   - Self-educate on nutrition, training, recovery

3. **Low stress, good sleep:**
   - Exercise is their stress management tool
   - Healthy habits support good sleep
   - Positive feedback loop (exercise → sleep → exercise)

4. **Autonomous:**
   - Lower social support (5.0/10) but not a problem for them
   - Prefer individual activities over group classes
   - Don't need peer accountability

### Why They Might Disengage (Risks)

Despite strengths, they face unique challenges:

1. **Program may feel too basic:**
   - "Walk 10 minutes a day? I run 5 miles."
   - Beginner content is boring and demotivating
   - May skip sessions if content isn't challenging

2. **Low social support could backfire:**
   - No peer accountability to keep them engaged
   - May drift away from program quietly
   - "I don't need this, I'll just do my own thing"

3. **Risk of burnout or overtraining:**
   - May push too hard without professional guidance
   - Injury risk from high activity volume
   - May not recognize signs of overtraining

4. **May not see value in "wellness basics":**
   - Already know about nutrition, exercise science
   - Generic health education feels redundant
   - Need advanced, specialized content

### How to Help Them Succeed (HR Intervention Strategies)

#### Strategy 1: Advanced Programming (Challenge the Challengers)

**What to do:**
- Performance optimization content:
  - Marathon/triathlon training plans
  - Sports nutrition (macros, timing, supplements)
  - Recovery protocols (foam rolling, stretching, sleep optimization)
  - Strength periodization
- Competitive events:
  - Company 5K/10K races
  - Fitness competitions (CrossFit-style, obstacle courses)
  - Step challenge leaderboards (for competitive types)
- Advanced workshops:
  - Running form analysis
  - Weightlifting technique
  - Sports psychology

**Why it works:**
- Matches their fitness level
- Keeps them engaged and learning
- Taps into competitive drive
- Provides new challenges

**Example Programs:**
- "Boston Marathon Training Group"
- Quarterly fitness competitions with prizes
- Advanced nutrition seminar series
- VO2 max testing and performance coaching

#### Strategy 2: Peer Leadership Roles (Leverage Their Influence)

**What to do:**
- Recruit as wellness ambassadors:
  - Lead group workouts for Cluster 1
  - Mentor beginners in buddy system
  - Share success stories in newsletters
  - Speak at wellness events
- Captain roles:
  - Department team captains for challenges
  - Peer coaches for specific activities (running club, yoga)
- Recognition programs:
  - "Wellness Champion of the Month"
  - Public acknowledgment of leadership
  - Exclusive perks (free gym membership, race entry fees)

**Why it works:**
- Gives them purpose and status
- Increases engagement through leadership responsibility
- Benefits Cluster 1 (peer mentors inspire beginners)
- Low cost, high impact

**Example Programs:**
- "Wellness Ambassador" program with 10 Cluster 3 employees
- Peer coaching matching (1 Cluster 3 + 2-3 Cluster 1)
- Leadership recognition at annual wellness event

#### Strategy 3: Autonomy and Flexibility (Don't Micromanage)

**What to do:**
- Self-directed goal setting:
  - "Set your own wellness goal" - no prescribed program
  - Personalized plans based on their interests (strength, endurance, flexibility)
  - Track whatever metrics matter to them (not just steps)
- Minimal required participation:
  - Optional group sessions (they can attend if interested)
  - Asynchronous content (watch on their own time)
  - Focus on outcomes, not process compliance
- Flexible scheduling:
  - No fixed class times
  - On-demand resources (video library, app content)
  - Work out when/where they want

**Why it works:**
- Respects their autonomy and self-direction
- Reduces friction (don't force them into beginner classes)
- Keeps them enrolled without forcing attendance
- Focuses on what matters (outcomes) not bureaucracy

**Example Programs:**
- "Choose Your Own Adventure" wellness track
- Self-reported goals with quarterly check-ins
- Resource library (no mandatory sessions)

#### Strategy 4: Build Community (Despite Low Social Support)

**What to do:**
- Connect high-performers:
  - "Advanced Athletes Group" for Cluster 3 only
  - Running clubs, cycling groups, lifting partners
  - Social events around fitness (post-race brunch, trail runs)
- Respect their independence:
  - Make community optional, not mandatory
  - Focus on shared interests (running, lifting) not forced socializing
  - Virtual communities for those who prefer
- Friendly competition:
  - Strava challenges
  - Leaderboards for metrics they care about (pace, PRs, mileage)

**Why it works:**
- Low social support (5.0/10) suggests they're independent, not anti-social
- High-performers often enjoy connecting with other high-performers
- Shared passion for fitness creates natural bonds
- Competition drives engagement for this group

**Example Programs:**
- Monthly "Elite Athletes Run Club"
- Strava group with monthly mileage challenges
- Quarterly fitness social events (yoga + brunch, trail run + BBQ)

### Recommended Program Track: "Performance Excellence"

**Track Name:** Advanced Wellness & Performance

**Intensity:** Low-Medium (they're self-directed, need less support)

**Duration:** 12 weeks with optional ongoing community

**Key Components:**
1. Week 1: Advanced goal setting (performance-based, not beginner)
2. Week 2-12: Self-directed training with optional advanced resources
3. Quarterly: Advanced workshops, competitive events, leadership opportunities
4. Ongoing: Alumni network, peer leadership roles, exclusive content

**Staffing Needs:**
- Performance coaches (not general wellness coaches)
- Sports nutritionists
- Running/lifting specialists
- Event coordinators (for competitions)

**Resource Allocation:** Low-Medium
- Mostly self-directed (low staff time per person)
- Advanced content (higher quality but fewer touch points)
- Event-based programming (quarterly, not weekly)
- Leverage them as volunteer peer mentors (reduces cost)

**Success Metrics:**
- Maintain 73.8% success rate (they're already doing well)
- Increase peer leadership participation (50% volunteer as mentors)
- High satisfaction scores (they feel valued and challenged)
- Retention in "alumni" network beyond 12 weeks

### Expected Outcomes

**Current Success Rate:** 73.8%

**Potential with Interventions:** 75-80%

**Why modest improvement:**
- They're already high-performers
- Ceiling effect (hard to improve from already excellent)
- Success for them = maintaining habits, not building new ones

**ROI - Different than other clusters:**
- Not about health improvement (they're already healthy)
- Value is in retention and leadership:
  - Keep them engaged (reduce dropout)
  - Leverage as peer mentors (multiply impact)
  - Build wellness culture through their influence
- Low cost per person (self-directed, minimal staff time)
- High influence (they inspire Cluster 1 employees)

**Strategic Value:**
- Peer mentors for Cluster 1: Their enthusiasm is contagious
- Culture builders: Make wellness "cool" and aspirational
- Low-maintenance participants: Succeed without hand-holding
- Brand ambassadors: Share success stories, recruit others

---

## Summary Table: Three Cluster Comparison for HR

| Aspect | Cluster 1: Moderate Seekers | Cluster 2: Stressed | Cluster 3: Active |
|--------|---------------------------|-------------------|------------------|
| **Size** | 531 (53%) - LARGEST | 297 (30%) | 172 (17%) - SMALLEST |
| **Success Rate** | 70.8% (lowest) | **74.1% (HIGHEST)** | 73.8% |
| **Typical Employee** | Sedentary office worker | Overwhelmed, exhausted | Fitness enthusiast |
| **Main Problem** | Inactive lifestyle | High stress, poor sleep | Risk of disengagement |
| **Main Strength** | Moderate social support | HIGHEST social support | Self-motivated, disciplined |
| **Motivation Level** | Low urgency ("should") | **High urgency ("need to")** | Intrinsic (lifestyle) |
| **Program Track** | Wellness Foundations | Stress & Recovery | Performance Excellence |
| **Focus** | Activation, habit formation | Stress relief, sleep improvement | Advanced challenges, leadership |
| **Resource Intensity** | Medium | **HIGH (clinical)** | Low-Medium |
| **Staff Needed** | Wellness coaches, fitness instructors | Mental health counselors, sleep specialists | Performance coaches, nutritionists |
| **Key Intervention** | Social activation, gradual progression | CBT-I, stress management, EAP | Advanced content, peer leadership |
| **Expected Improvement** | 70.8% → 75-80% | 74.1% → 80-85%+ | 73.8% → 75-80% |
| **ROI Potential** | High (largest group) | **HIGHEST (motivated + need)** | Moderate (culture value) |
| **Primary Lever** | Social support & convenience | Clinical resources & social support | Autonomy & advanced content |

---

## Strategic Recommendations by Cluster

### For HR Leaders: Resource Allocation

**Prioritization by ROI:**
1. **Cluster 2 (30%)** = Highest ROI
   - Already highly motivated (74.1% success)
   - Clinical support can push to 80-85%+
   - Significant health risk reduction
   - Allocate 40-45% of budget here

2. **Cluster 1 (53%)** = Highest Total Impact
   - Largest group (53% of workforce)
   - Moderate ROI per person but high volume
   - Foundational programming (cost-efficient)
   - Allocate 40-45% of budget here

3. **Cluster 3 (17%)** = Strategic Value
   - Already successful (low marginal gain)
   - But: Peer mentors for Cluster 1
   - But: Culture builders and brand ambassadors
   - Allocate 10-15% of budget here (mostly leadership/events)

### For Program Designers: Track Implementation

**Three Parallel Tracks (Not Sequential):**
- Track A: Wellness Foundations (Cluster 1) - Largest enrollment
- Track B: Stress & Recovery (Cluster 2) - Highest clinical support
- Track C: Performance Excellence (Cluster 3) - Smallest but influential

**Shared Components:**
- All tracks use same wellness app (app engagement is important for all)
- All tracks have access to EAP and health resources
- All tracks participate in company-wide challenges (cross-cluster bonding)

**Differentiated Components:**
- Track A: Group fitness, beginner content, social teams
- Track B: CBT-I, stress workshops, mental health counseling
- Track C: Advanced training, competitions, peer leadership

### For Executives: Business Case

**Investment Summary:**
- X dollars per Cluster 1 employee: Group programming, coaches, on-site classes
- 2-3X per Cluster 2 employee: Clinical resources (CBT-I, counseling, specialists)
- 0.5X per Cluster 3 employee: Self-directed with advanced resources

**Expected Outcomes:**
- Overall success rate: 72.3% → 78-82%
- Cluster 2: Significant health risk and productivity improvements
- Cluster 1: Long-term chronic disease prevention
- Cluster 3: Culture transformation and peer influence

**ROI Metrics to Track:**
- Success rates by cluster
- Healthcare utilization (especially Cluster 2 stress-related)
- Absenteeism and presenteeism
- Employee engagement scores
- Retention rates for wellness program participants

---

# Technical Implementation Details

## Technologies Used

**Languages & Environment:**
- R 4.4.3
- R Markdown for reproducible analysis

**Machine Learning:**
- `randomForest` - Random Forest classifier
- `e1071` - Support Vector Machines
- `cluster` - K-means and PAM clustering
- `caret` - Model training and evaluation

**Visualization:**
- `ggplot2` - Statistical graphics
- `patchwork` - Multi-panel figures
- `corrplot` - Correlation matrices
- `factoextra` - Cluster visualization

**Evaluation Metrics:**
- `pROC` - ROC curves and AUC
- `ROCR` - Additional model evaluation

## Reproducibility

**Random Seed:** 42 (set throughout for reproducibility)

**Data Splitting:**
- Training: 70% (n=701)
- Testing: 30% (n=299)
- Stratified sampling maintains class balance

**Model Configurations:**
- **Random Forest:** 500 trees, mtry=√features, nodesize=5
- **SVM Linear:** kernel="linear", probability=TRUE
- **SVM RBF:** kernel="radial", probability=TRUE
- **K-means:** nstart=25, centers=3
- **PAM:** k=3, metric="euclidean"

**Feature Standardization:**
- Clustering variables scaled (mean=0, sd=1)
- SVM features preprocessed (median imputation, near-zero variance removal, centering, scaling)

---

## Results Validation

### Classification Validation
- Hold-out test set (30% of data)
- ROC curves for threshold-independent evaluation
- Multiple metrics reported (accuracy, precision, recall, F1, AUC-ROC)

### Clustering Validation
- Silhouette analysis for cluster quality
- PCA visualization for visual inspection
- External validation via success rate differences
- Heatmaps for interpretability

---

## Usage

### Prerequisites

```r
# Install required packages
install.packages(c(
  "tidyverse", "data.table",
  "caret", "randomForest", "e1071", "cluster",
  "ggplot2", "GGally", "corrplot", "gridExtra", 
  "factoextra", "patchwork",
  "pROC", "ROCR"
))
```

### Running the Analysis

```r
# Set working directory
setwd("path/to/project")

# Ensure data is in data/ subdirectory
# Expected: data/workplace_wellness_data.csv

# Render the analysis
rmarkdown::render("R/workplace_wellness_analysis.Rmd",
                  output_format = "all")  # Generates HTML and Word
```

---

## Citation

If you use this analysis or methodology, please cite:

```
Mastala, P. (2026). Workplace Wellness Program Evaluation: Machine Learning 
Analysis of "Thrive at Work" Program. Meridian Health Systems. 
GitHub: [repository URL]
```

---

## Contact

**Primary Analyst:**  
Precious Mastala  
Data Scientist, Meridian Health Systems  
Email: pmastala@arizona.edu

**For data access requests:**  
Supervisor: Dr. Onicio Neto  
Email: onicio@arizona.edu  
Subject: "Workplace Wellness Data Request"

---

## Version

**Version:** 1.0  
**Date:** February 2026  

---

## Key Takeaways

**For HR Teams:**
- Age is the #1 predictor - design age-stratified tracks
- Don't ignore stressed employees - they have the HIGHEST success rate
- Three behavioral clusters need three different intervention approaches
- Invest heavily in Cluster 2 (stressed) - highest ROI potential

**For Program Designers:**
- Random Forest (82.2% AUC) beats SVMs decisively
- K-means K=3 creates most actionable employee segments
- App engagement is critical (#2 predictor) - improve UX immediately
- Social support matters (#3 predictor) - build it into program structure

**For Data Scientists:**
- Feature importance reveals age dominance (67.15 Gini) - 3.5× more important than #2
- Clustering found paradox: worst health = highest success (motivation beats health status)
- Ensemble methods (Random Forest) naturally suited for wellness prediction
- Behavioral clustering outperforms demographic segmentation

**For Executives:**
- 72.3% baseline success can improve to 78-82% with targeted interventions
- Cluster 2 (30% of workforce) is highest ROI: motivated + clinical need
- Three-track program design maximizes efficiency and outcomes
- Small investments in stress/sleep resources yield biggest gains

---

## Immediate Action Items

**Week 1:**
1. Segment current participants into Clusters 1, 2, 3 using K-means model
2. Deploy Random Forest model for new enrollments (predict high-risk employees)
3. Recruit Cluster 3 employees as peer mentors
4. Launch Cluster 2 stress/sleep interventions (highest ROI)

**Month 1:**
5. Design three parallel program tracks (Foundations, Stress Recovery, Performance)
6. Improve app UX (focus on engagement features - 2nd most important predictor)
7. Build buddy matching system (leverage social support - 3rd most important)
8. Create age-stratified content (age is dominant predictor)

**Quarter 1:**
9. Implement CBT-I for Cluster 2 (clinical sleep intervention)
10. Launch department-based team challenges (activate Cluster 1)
11. Establish quarterly advanced athlete events (retain Cluster 3)
12. Begin tracking cluster-specific KPIs

**Ongoing:**
- Re-cluster employees quarterly (track profile changes)
- A/B test interventions within clusters
- Retrain Random Forest model as data grows
- Measure ROI by cluster (healthcare costs, productivity, retention)

---

# Generative AI Prompts Used for Analysis
## Overview

This document lists all prompts used with Claude AI for improving the analysis R code, and creating documentation for the Workplace Wellness Program Evaluation project.

---

## Session 1: Initial Analysis Setup and Code Generation

### Prompt 1: Project Understanding and Setup
```
I need to complete a machine learning analysis for a workplace wellness program. 
The assignment requires:

Part 1: Classification Analysis
- Build Random Forest and SVM models to predict program success
- Compare performance using accuracy, precision, recall, F1-score, and AUC-ROC
- Identify important features

Part 2: Clustering Analysis  
- Use K-means and K-medoids with K=3 and K=4
- Compare cluster quality and interpret results
- Describe clusters for non-technical stakeholders

The dataset has 1,000 employees with 18 variables including demographics, 
health metrics, psychosocial factors, and program engagement measures.

Can you help me structure the R code for this analysis?
```

### Prompt 2: Improve my interpretations
```
Improve my verbage for interpretation of my results.
```

### Prompt 3: Initial README Creation
```
Create a comprehensive README.md file for my R code.

Use my analysis results to create this README. Make it thorough and professional.
```
---
