# Employee_sentiment_analysis
# 📊 Employee Sentiment Analysis

This project analyzes employee emails to extract **sentiment patterns**, identify **top positive/negative contributors**, flag **flight risk employees**, and understand the **temporal impact on sentiment** using basic predictive modeling.

---

## 🔍 Summary of Key Findings

### Sentiment Score Distribution
- **Positive** sentiments are the highest, followed by **Neutral** and then **Negative**.
- This suggests a generally **positive working environment** and alignment toward **goal achievement**.

---

## 🏅 Top Performing Employees (May 2010)

### 🔝 Top 3 Positive Employees:
| Employee                     | Month   | Monthly Sentiment Score |
|-----------------------------|---------|--------------------------|
| don.baughman@enron.com      | 2010-05 | 16                       |
| patti.thompson@enron.com    | 2010-05 | 9                        |
| eric.bass@enron.com         | 2010-05 | 6                        |

### 🔻 Top 3 Negative Employees:
| Employee                          | Month   | Monthly Sentiment Score |
|----------------------------------|---------|--------------------------|
| johnny.palmer@enron.com          | 2010-05 | 1                        |
| bobette.riner@ipgdirect.com      | 2010-05 | 2                        |
| john.arnold@enron.com            | 2010-05 | 3                        |

---

## 🚨 Flight Risk Employees
Employees who sent **≥4 negative emails within a 30-day rolling window** were flagged as **potential flight risks**.

**Flagged Employees:**
- bobette.riner@ipgdirect.com  
- don.baughman@enron.com  
- johnny.palmer@enron.com  
- sally.beck@enron.com  

---

## 📈 Predictive Modeling Insights

- **Mean Squared Error (MSE)**: `0.4402`  
- **R² Score**: `-0.0071` → Indicates **poor model performance**, possibly due to limited features or a simplistic model.

### Temporal Insights:
- **Year** had the **most impact** on sentiment trends — potentially reflecting internal changes or shifts in communication culture.
- **Day** and **Month** individually had **minimal impact** on sentiment.


## 📆 Monthly Sentiment Trends

Created a **monthly average sentiment score** to track sentiment over time.

### 📈 Insights:
- The **sentiment trajectory** suggests a **gradual improvement** in workplace tone or employee communication.
- Can help HR teams **identify periods of high morale** or **address low-sentiment phases** effectively.
- The **graph shows seasonal dips and spikes**, indicating that sentiment may be influenced by **internal or external company events**.

---

## 📌 Recommendations

- Enhance sentiment tracking models with **additional features** (e.g., role, department, team structure).
- Use flagged flight-risk indicators for **HR intervention** strategies.
- Extend predictive models to non-linear methods.
  

---

