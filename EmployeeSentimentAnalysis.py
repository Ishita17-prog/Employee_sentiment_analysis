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


df=pd.read_csv("C:\\Users\\ISHITA GUPTA\\OneDrive\\Desktop\\html work\\Pandas\\Data\\Internship\\test(in).csv")


# In[3]:


df


# In[4]:


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

# In[5]:


df


# In[6]:


df.shape


# # Exploratory Data Analysis (EDA)

# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[12]:


df.duplicated().sum()


# # Data Overview
# 
# The dataset has no null values, as confirmed by `df.isnull().sum()`. Also there are no duplicated in the data.
# 

# # Sentiment Labeling

# In[13]:


nltk.download('vader_lexicon')

# Initialize analyzer
sia = SentimentIntensityAnalyzer()

# Sentiment classification
def classify_sentiment(text):
    scores = sia.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.02:
        return 'Positive'
    elif compound <= -0.02:
        return 'Negative'
    else:
        return 'Neutral'

# Apply to the 'body' column
df['Sentiment'] = df['body'].apply(classify_sentiment)
df['Sentiment']


# In[14]:


df['Sentiment'].value_counts()


# In[15]:


df[df['Sentiment']=='Negative']['body'].values


# - Earlier I set the threshold to (-0.5,0.5) but it led to a large proportion of the corporate email texts being classified as Neutral.However, corporate communication often uses formal and subtle language, where even slightly positive or negative tones carry meaningful sentiment. As a result:Many messages with mild sentiment (e.g., compound scores between -0.05 and +0.05) were getting labeled as Neutral even when they expressed slight concern, approval, or disapproval.
# 
# - But with new threshold (-0.02,0.02) it capture subtle sentiment shifts more accurately.Reduce over-classification into the Neutral category.Make the sentiment analysis more aligned with the formal tone of corporate emails.

# In[16]:


ax=sns.countplot(x='Sentiment', data=df, palette='Set2')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# We can see that positive sentiments are the highest followed by neutral and negative sentiments. This shows there is a positive working environment and achievement of goals.

# # Employee Score Calculation

# In[17]:


def sent_score(str):
    if str=='Positive':
        return 1
    elif str=='Negative':
        return -1
    else:
        return 0
df['Sentiment_score']=df['Sentiment'].apply(sent_score)

# Step 3: Group by employee and month, then sum the scores
monthly_score=df.groupby(['from','month'])['Sentiment_score'].sum().reset_index()

# Step 4: Rename column
monthly_score.rename(columns={'Sentiment_score':'monthly_sentiment_score'},inplace=True)

# Step 5: Merge back into original DataFrame
df=pd.merge(df,monthly_score,on=['from','month'],how='left')
df


# # Ranked list of employees

# In[61]:


def top_employees(this_month):
   month_df=df[df['month']==this_month].drop_duplicates(subset='from')  #to avoid any duplicates as in original we have date column which can have 2-5-11 and 3-5-11
   top_employees=month_df.sort_values(by=['monthly_sentiment_score','from'],ascending=False).head(3)
   top_3_employees=top_employees[['from','monthly_sentiment_score']]
    
   bottom_employees=month_df.sort_values(by=['monthly_sentiment_score','from'],ascending=True).head(3)
   bottom_3_employees=bottom_employees[['from','monthly_sentiment_score']]
    
   print(f"\n📅 Month: {this_month}")
   print("\n🔝 Top 3 Positive Employees:")
   print(top_3_employees)
   print("\n🔻 Top 3 Negative Employees:")
   print(bottom_3_employees)
   
   plt.figure(figsize=(12, 5))

   plt.subplot(1, 2, 1)
   sns.barplot(data=top_3_employees, x='monthly_sentiment_score', y='from', palette='Greens_d')
   plt.title(f"Top 3 Positive - {this_month}")
   plt.xlabel("Sentiment Score")
   plt.ylabel("Employee")

   plt.subplot(1, 2, 2)
   sns.barplot(data=bottom_3_employees, x='monthly_sentiment_score', y='from', palette='Reds_d')
   plt.title(f"Top 3 Negative - {this_month}")
   plt.xlabel("Sentiment Score")
   plt.ylabel("Employee")

   plt.tight_layout()
   plt.show()
 
for i in df['month'].unique():
    top_employees(i)
    
    


# ## Monthly Sentiment Rankings
# 
# We can see the **month-wise top 3 positive and negative sentiment scores**, along with the corresponding employee names.
# 

# # Flight Risk

# In[39]:


# Step 1: Filter negative sentiment emails
negative_df = df[df['Sentiment'] == 'Negative']

# Step 2: Sort by sender and date
negative_df.sort_values(by=['from', 'date'], inplace=True)

# Step 3: Create a flight risk list
flight_risk_employees = []

# Step 4: Loop over each employee
for emp, group in negative_df.groupby('from'):
    group = group.set_index('date').resample('1D').count()  # count per day
    group['rolling_count'] = group['Sentiment'].rolling('30D').sum()

    if group['rolling_count'].max() >= 4:
        flight_risk_employees.append(emp)

# Step 5: Display results
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

# In[40]:


#Feature: Day of month, month, weekday (seasonality patterns)
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['month_num'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Optional: Count of messages per person per day
df['message_count'] = df.groupby(['from', 'date'])['body'].transform('count')


# In[42]:


# Features and target
X = df[['day', 'weekday', 'month_num', 'year', 'message_count']]
y = df['Sentiment_score']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[43]:


# Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)


# In[44]:


# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")


# The **MSE of 0.4402** indicates the average squared difference between the actual and predicted sentiment scores. While it's not extremely high, it suggests there's some variance between predictions and true values.
# - The **R² Score of -0.0071** means that the model explains virtually none of the variance in the sentiment scores. In fact, a negative R² indicates that the model performs **worse than a horizontal line (mean prediction)**.  
# - This implies that the **linear regression model is not effectively capturing the patterns** in the data and may be underfitting.

# In[45]:


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

# # Monthly average sentiment score

# In[47]:


monthly_avg = df.groupby('month')['Sentiment_score'].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_avg, x='month', y='Sentiment_score', marker='o')
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




