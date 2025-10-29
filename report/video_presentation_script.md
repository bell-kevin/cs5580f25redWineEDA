# Video Presentation Script: Exploratory Data Analysis of Red Wine Quality

**Total target duration:** 10 minutes (range 8–12 minutes)

---

## 0:00 – 0:45 · Opening & Agenda
- "Hi everyone, we're Kevin Bell, Angelie Reyes-Sosa, and Dani Lopez, and welcome to our exploratory data analysis of the UCI red wine quality dataset."
- "In the next ten minutes we'll cover why we selected this dataset, the core questions we investigated, the visual and statistical evidence we gathered, and the key takeaways plus next steps."
- Display title slide with project title, name, and agenda bullets.

## 0:45 – 2:00 · Motivation & Dataset Overview
- "We chose this dataset because it easily exceeds the course requirements with 1,599 observations and 11 physicochemical features plus the sensory quality score."
- "It is a staple dataset in the wine analytics literature, which gives us external benchmarks, yet it still leaves room to ask practical winemaking questions."
- "Each row represents a lab analysis of Portuguese *Vinho Verde* red wine. The columns include measurements like fixed and volatile acidity, residual sugar, sulphates, alcohol percentage, and the panel quality rating on a 3-to-8 scale."
- Show table of feature definitions (reuse the table from the report) and briefly define 2–3 features: "Volatile acidity is acetic acid, which causes vinegar aromas; sulphates protect against microbes; alcohol is a proxy for ripeness and body."

## 2:00 – 3:15 · Framing the Three Core Questions
- Display slide listing the three questions with numbers to emphasize priority order.
- Scripted narration:
  - "Our top-priority question asked which chemical attributes best distinguish higher-quality wines. This ties directly to actionable winemaking levers, so we ranked it first."
  - "Second, we examined how acidity profiles interact with sulphur management across quality levels. Acid and sulphates work together to balance freshness and stability."
  - "Third, we looked for latent subgroups—potential wine styles hidden in the chemistry—that could justify segmentation or tailored modelling."
- State assumptions and biases: "We treated the quality score as approximately continuous to enable correlation analyses, and we assumed the lab measurements were unbiased. The main analytical bias is that we focus on chemistry and not viticultural context."
- Hypotheses: "We expected alcohol and sulphates to correlate positively with quality, volatile acidity to have a negative relationship, and density and residual sugar to play smaller roles because most wines here are dry."

## 3:15 – 5:15 · Visualisations & Summary Statistics
- Transition slide titled "Evidence from Descriptive Statistics".
- Highlight summary table: "Table 1 shows alcohol averaging 10.4% with a standard deviation just over 1%, and volatile acidity centered at 0.53 g/dm³ but with a relatively wide spread of 0.18."
- Point out at least three distinct statistical insights:
  1. "Alcohol and sulphates have the highest relative variability, hinting they differentiate wines." (refer to summary table)
  2. "Quality-specific means reveal a monotonic climb in alcohol from 9.96% at quality 3 to 12.09% at quality 8, while volatile acidity drops from 0.89 to 0.42." (show mean-by-quality table)
  3. "Total sulfur dioxide peaks at quality 5, suggesting that moderate additions are helpful but excessive amounts degrade quality."
- Move to visualisations:
  - Scatter plot (Figure 1): "The alcohol vs. quality scatter shows a clear positive slope, especially for scores 7 and 8." Mention overlay of colour-coded quality levels.
  - Correlation bar chart (Figure 2): "Alcohol's correlation with quality is 0.48, sulphates 0.25, citric acid 0.23, while volatile acidity is the strongest negative at -0.39."
  - Box plots (Figure 3): "The volatility box plots tighten as quality increases—higher-quality wines keep volatile acidity within a narrower band."

## 5:15 – 6:15 · Highlighting an Anomaly or Pattern
- "One intriguing pattern involves sulphate-heavy wines. The mean-by-quality table shows sulphates rising with quality up to the mid-tier, but the anomalies we spotted indicate that extreme sulphate levels can coincide with lower quality. These wines might be over-compensating for microbial stability and paying a sensory penalty."
- "We also observed a handful of high-chloride samples with low quality scores—worth flagging for lab re-checks."

## 6:15 – 8:00 · Synthesis of Findings
- "Putting the evidence together, we can characterise high-scoring wines as having: higher alcohol, moderate sulphates, elevated citric acid, and notably lower volatile acidity."
- "These relationships validate our initial hypotheses about alcohol and volatile acidity. Sulphates behave more like a sweet spot variable: moderate boosts quality, but extreme values coincide with outliers."
- "The clustering cues in the scatter plot and box plots suggest at least two segments—average wines rated 5–6 and a higher-quality cluster at 7–8 with distinct chemistry."

## 8:00 – 9:30 · Future Research & Predictive Plan
- Display slide titled "Next Steps" with three bullets.
- Narration:
  - "First, we'll develop predictive models—regularised regression, gradient boosting, and tree ensembles—to estimate quality. We'll include interactions like alcohol with density and sulphates with volatile acidity."
  - "Second, we'll pursue segmentation via Gaussian mixtures or density-based clustering to formalise the latent subgroups." 
  - "Third, we'll simulate experimental adjustments—for example, targeting lower volatile acidity—to estimate potential gains using causal inference tools such as propensity score weighting."
- "We also raised new questions: Are there vintage or producer effects missing here? Could blending strategies mitigate the identified deficiencies?"

## 9:30 – 10:30 · Conclusion & Call to Action
- "To wrap up, this EDA met our objectives: we characterised the data, answered the three core questions with statistics and visuals, and charted next steps."
- "Key takeaways: alcohol and balanced sulphates are friends of quality, volatile acidity is a consistent detractor, and there are actionable subgroups to explore." 
- "Thank you for watching. On behalf of Kevin, Angelie, and Dani, we invite you to check the written report and the repository for reproducible code."
- End slide with contact info and placeholder for video link.

---

## Presenter Notes
- Keep transitions tight—use cross-fades between visuals to stay within the 10-minute target.
- Rehearse the quantitative callouts (e.g., alcohol 10.4%, correlation 0.48) to maintain credibility.
- When showing tables or plots, zoom in briefly to highlight the specific numbers referenced.
- Encourage questions at the end if presenting live, or invite comments if posting asynchronously.
