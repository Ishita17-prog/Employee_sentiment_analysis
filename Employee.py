#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import dateutil.parser
import nltk
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("C:\\Users\\ISHITA GUPTA\\OneDrive\\Desktop\\html work\\Pandas\\Data\\Internship\\test(in).csv",parse_dates=['date'])


# In[3]:


def try_parse_date(val):
    try:
        return dateutil.parser.parse(val)
    except:
        return pd.NaT

df['date'] = df['date'].apply(try_parse_date)
df['month'] = df['date'].dt.to_period('M').astype(str)


# # Data Preprocessing
# 
# We imported the necessary libraries and loaded the dataset. The 'date' column was parsed into datetime format, and a new 'month' column was created for easier analysis.
# 

# In[4]:


df


# In[5]:


df.shape


# # Exploratory Data Analysis (EDA)

# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# # Data Overview
# 
# The dataset has no null values, as confirmed by `df.isnull().sum()`. Also there are no duplicated in the data.
# 

# # Sentiment Labeling

# In[9]:


nltk.download('vader_lexicon')

# Initialize analyzer
sia = SentimentIntensityAnalyzer()

# Sentiment classification
def classify_sentiment(text):
    scores = sia.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply to the 'body' column
df['Sentiment'] = df['body'].apply(classify_sentiment)
df['Sentiment']


# In[10]:


sns.countplot(x='Sentiment', data=df, palette='Set2')
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# We can see that positive sentiments are the highest followed by neutral and negative sentiments. This shows there is a positive working environment and achievement of goals.

# # Employee Score Calculation

# In[11]:


score_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
df['SentimentScore'] = df['Sentiment'].map(score_map)

# Step 2: Ensure 'date' is in datetime format & extract 'month' column
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.to_period('M').astype(str)  # Format: YYYY-MM

# Step 3: Group by employee and month, then sum the scores
monthly_scores = df.groupby(['from', 'month'])['SentimentScore'].sum().reset_index()

# Step 4: Rename column
monthly_scores.rename(columns={'SentimentScore': 'monthly_sentiment_score'}, inplace=True)

# Step 5: Merge back into original DataFrame
df = pd.merge(df, monthly_scores, on=['from', 'month'], how='left')


# In[12]:


df


# # Ranked list of employees

# In[13]:


# Step 6: Get Top 3 Positive and Negative Employees for Each Month
def top_employees_per_month(month_df, top_n=3):
    # Sort for top positive: descending by score, then ascending by name
    top_positive = month_df.sort_values(['monthly_sentiment_score', 'from'], ascending=[False, True]).head(top_n)
    # Sort for top negative: ascending by score, then ascending by name
    top_negative = month_df.sort_values(['monthly_sentiment_score', 'from'], ascending=[True, True]).head(top_n)
    return top_positive, top_negative

# Step 7: Group and apply the function
rankings_by_month = {}
for month in df['month'].unique():
    # Get the distinct employee and their monthly score for that month
    this_month_df = df[df['month'] == month][['from', 'month', 'monthly_sentiment_score']].drop_duplicates()
    top_pos, top_neg = top_employees_per_month(this_month_df)
    rankings_by_month[month] = {
        'Top Positive': top_pos.reset_index(drop=True),
        'Top Negative': top_neg.reset_index(drop=True)
    }

# Step 8: Print Rankings
for month, result in rankings_by_month.items():
    print(f"\n📅 Month: {month}")
    print("\n🔝 Top 3 Positive Employees:")
    print(result['Top Positive'])
    print("\n🔻 Top 3 Negative Employees:")
    print(result['Top Negative'])


# ## Monthly Sentiment Rankings
# 
# We can see the **month-wise top 3 positive and negative sentiment scores**, along with the corresponding employee names.
# 

# # Flight Risk

# In[14]:


# Filter only negative sentiment emails
negative_df = df[df['Sentiment'] == 'Negative'].copy()

# Sort by sender and date
negative_df = negative_df.sort_values(by=['from', 'date'])

# Rolling 30-day negative mail count
flight_risk_employees = set()

# Group by employee
for emp, group in negative_df.groupby('from'):
    group = group[['date']].sort_values('date').reset_index(drop=True)
    for i in range(len(group)):
        # Define rolling window
        window_start = group.loc[i, 'date']
        window_end = window_start + pd.Timedelta(days=30)
        # Count emails within 30-day window
        count_in_window = group[(group['date'] >= window_start) & (group['date'] <= window_end)].shape[0]
        if count_in_window >= 4:
            flight_risk_employees.add(emp)
            break  # No need to check further once flagged

# Output flight risk list
flight_risk_employees = sorted(list(flight_risk_employees))
print("🚨 Flight Risk Employees (≥4 negative mails in 30 days):")
for emp in flight_risk_employees:
    print(emp)


# ## 🚨 Flight Risk Employees
# 
# Based on the analysis, the following employees have been identified as **at risk of leaving**, having sent **4 or more negative emails within a 30-day rolling window** (regardless of sentiment score):
# 
# - bobette.riner@ipgdirect.com  
# - don.baughman@enron.com  
# - johnny.palmer@enron.com  
# - sally.beck@enron.com  
# 
# Urgent need of intervention of HR.
# 
# 
# 

# # Linear regression model

# In[15]:


#Feature: Day of month, month, weekday (seasonality patterns)
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['month_num'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Optional: Count of messages per person per day
df['message_count'] = df.groupby(['from', 'date'])['body'].transform('count')


# In[16]:


# Features and target
X = df[['day', 'weekday', 'month_num', 'year', 'message_count']]
y = df['SentimentScore']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


# Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)


# In[18]:


# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")


# The **MSE of 0.4402** indicates the average squared difference between the actual and predicted sentiment scores. While it's not extremely high, it suggests there's some variance between predictions and true values.
# - The **R² Score of -0.0071** means that the model explains virtually none of the variance in the sentiment scores. In fact, a negative R² indicates that the model performs **worse than a horizontal line (mean prediction)**.  
# - This implies that the **linear regression model is not effectively capturing the patterns** in the data and may be underfitting.

# In[19]:


# Coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
})

print("\nFeature Importance:")
print(coef_df)


# ## Feature Importance – Linear Regression Coefficients
# 
# | Feature         | Coefficient | Interpretation |
# |----------------|-------------|----------------|
# | **day**        | 0.0018      | Sentiment score slightly increases as the day of the month progresses. However, the effect is very small. |
# | **weekday**    | 0.0031      | Slight increase in sentiment based on day of the week (e.g., weekends vs weekdays), though the impact is minimal. |
# | **month_num**  | 0.0105      | Sentiment trends show a small upward movement across months — possibly improving over time. |
# | **year**       | 0.0649      | This is the strongest positive coefficient, suggesting sentiment scores increase more noticeably over different years, indicating a possible **long-term positive trend**. |
# | **message_count** | 0.0080   | Slightly higher sentiment scores are associated with more messages sent per day — **more activity might be linked with positive sentiment**, although the effect is small. |
# 
# > All coefficients are positive but very small, implying that no single feature has a strong impact individually on the sentiment score. However, **"year"** appears to have the most significant influence among them.
# 

# # Employee Ranking Visualization

# In[20]:


# Loop through each month's rankings
for month, result in rankings_by_month.items():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Employee Sentiment Rankings - {month}", fontsize=16, weight='bold')

    # Positive Plot
    sns.barplot(
        ax=axes[0],
        data=result['Top Positive'],
        x='monthly_sentiment_score',
        y='from',
        palette='Greens_d'
    )
    axes[0].set_title('Top 3 Positive Employees')
    axes[0].set_xlabel('Sentiment Score')
    axes[0].set_ylabel('Employee')

    # Negative Plot
    sns.barplot(
        ax=axes[1],
        data=result['Top Negative'],
        x='monthly_sentiment_score',
        y='from',
        palette='Reds_d'
    )
    axes[1].set_title('Top 3 Negative Employees')
    axes[1].set_xlabel('Sentiment Score')
    axes[1].set_ylabel('Employee')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


# # Monthly average sentiment score

# In[21]:


monthly_avg = df.groupby('month')['SentimentScore'].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_avg, x='month', y='SentimentScore', marker='o')
plt.xticks(rotation=45)
plt.title('Monthly Average Sentiment Score')
plt.xlabel('Month')
plt.ylabel('Average Sentiment Score')
plt.tight_layout()
plt.show()


# The sentiment trajectory suggests a gradual improvement in workplace tone or employee communication. This insight can be used to align HR strategies, identify successful morale-boosting periods, or investigate causes of low-sentiment phases.
# 
# The graph shows seasonal dips and spikes, suggesting that employee sentiment can vary month-to-month, possibly influenced by internal or external company events.

# In[ ]:




