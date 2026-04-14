# Business Insights Report: Sentiment Analysis

## Executive Summary
This project analyzes a randomized sample (150k downsampled from 1.6M) of Twitter discourse data to uncover user sentiment, emotional drivers, and prevailing trends in user opinions. Our goal is to translate unstructured text data into measurable metrics and actionable insights that can inform marketing, product development, and customer success teams.

## 1. Methodology & Data Understanding
The sentiment pipeline processed the dataset through rigorous cleaning protocols, notably removing URLs, `@mentions`, punctuation, and English stop words. This ensures our models parse only semantically meaningful tokens.

Three robust classification models were evaluated using TF-IDF text vectorization:
- **Logistic Regression**: Accuracy = 76.56%, F1 Score = 76.84% *(Best Performer)*
- **Multinomial Naive Bayes**: Accuracy = 75.24%, F1 Score = 75.01%
- **Linear SVC**: Accuracy = 76.02%, F1 Score = 76.34%

> [!TIP]
> The Logistic Regression model achieved the best predictive performance, signaling that the relationship between linear feature combinations (words/phrases) is well captured by this algorithm. Its ability to provide probability predictions allows for more nuanced "confidence scoring".

## 2. Key Findings & Analytics

### Text Structure & Sentiment Depth
Initial EDA reveals a relatively balanced dataset context:
- The average word count across both positive and negative tweets is largely uniform. Users are generally brief (~7.5 words post-cleanup). Note: extreme verbosity generally doesn't skew heavily into one specific sentiment spectrum over the other.
- While the length is roughly equal, the tone divergence is distinctly identifiable by strong negative emotional modifiers versus positive affirmations.

### Driving Factors of Positive Sentiment
By analyzing the highest weighted coefficients in our best model, we've identified the top phrases that indicate strong positive customer perception.
The presence of gratitude, excitement, and community interactions strongly correlate with positive outcomes.
**Top Positive Tokens:**
1. `thanks` / `thank`
2. `love`
3. `good`
4. `great`
5. `haha` / `lol`

*Business Translation*: Engagements that trigger interpersonal gratitudes (e.g. prompt customer support interactions) immediately uplift brand sentiment.

### Driving Factors of Negative Sentiment
Conversely, negative sentiment drivers provide critical bug-reporting and pain-point flagging insights. 
**Top Negative Tokens:**
1. `sad`
2. `miss` (often implying missed connections or lack of features)
3. `hate`
4. `bad`
5. `sorry`

*Business Translation*: When users express high variance emotional states (e.g., "sad", "hate"), it's often a signal of unfulfilled expectations rather than simple technical glitches. It requires empathetic customer engagement to turn around.

## 3. Actionable Recommendations
Based on the text vectorization modeling and clustering signals, we recommend:

1. **Implement Automated Issue Triage**: Integrating this text classification logic into a ZenDesk/Intercom instance. Tickets flagged automatically as 'Negative' with high confidence (>85%) should bypass standard wait-queues and escalate to senior service agents.
2. **Feature "Misses" Analytics**: Establish a secondary monitoring tool to search for keywords adjacent to "miss" or "wish" (as highlighted in negative correlations). These represent untapped product feature gaps highly desired by the userbase.
3. **Gratitude Loops in Marketing**: Launch campaigns that mimic or respond to the high-performing positive language subsets (e.g. casual "haha", "thanks"). Matching the user's lexigraphic tone improves social-marketing conversions.

## 4. Conclusion
The Sentiment Analysis workflow successfully distills large-scale chaotic text data into reliable structural metrics. By pivoting from pure model metrics to an emphasis on *why* text is scored negatively/positively, stakeholders can preemptively resolve customer friction and better market engaging products. 
