# Delta for Insights Dashboard

## ADDED Requirements

### Requirement: New ML Scores per Conversation

The dashboard MUST display three new per-conversation scores replacing the legacy TF-IDF sentiment: Sentiment class (negative/neutral/positive via XLM-R), Intent class (one of 9 via XLM-R), and Churn score (0.0–1.0 aggregated).

### Requirement: Churn Distribution Visualization

The dashboard MUST include a histogram of churn scores across all conversations with an adjustable threshold slider (default 0.7). Conversations above the threshold MUST be flagged as "High Churn Risk".

### Requirement: Intent Breakdown Chart

The dashboard MUST include a stacked bar chart showing intent distribution. The chart MUST be filterable by sentiment class (click-to-filter).

### Requirement: High-Risk Conversation Table

The dashboard MUST show a sortable table of conversations with churn score > threshold. Columns MUST include: message preview, sentiment, intent, churn score, and component breakdown.

### Requirement: Side-by-Side Comparison (MAY)

The dashboard MAY include a toggle comparing XLM-R predictions vs legacy TF-IDF/BART predictions for the same conversation.

#### Scenario: Churn threshold filtering

- GIVEN the dashboard is loaded with conversation data
- WHEN the user moves the churn threshold slider from 0.7 to 0.5
- THEN the high-risk table MUST update to include all conversations with churn >= 0.5

#### Scenario: Intent-sentiment cross-filter

- GIVEN the intent distribution chart is visible
- WHEN the user clicks on the "negative" sentiment segment
- THEN the intent chart MUST re-render showing only messages classified as negative
