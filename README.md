
# Sentiment Analysis of Major News Organizations

**Observation 1:** Fox News has been the most positive news organization, CNN the most negative, over the last 100 tweets<br>
**Observation 2:** The same media organizations that come closest to extreme negative tweets (-1.0) also come the closest to extreme positive tweets (+1.0)<br>
**Observation 3:** The New York Times seems to be the most moderate tweeter (or balance negative with positive), with an average polarity score of -0.01 over the past 100 tweets<br>


```python
#Import Dependencies
import tweepy
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from datetime import datetime

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Get the last 100 tweets for each news org, analyze with VADER, store it in a dataframe
news_orgs = ("@BBCBreaking", "@CBSNews", "@cnnbrk", "@FoxNews", "@nytimes")

oldestTweet = None
News_Sense_df = pd.DataFrame()

for org in news_orgs:
    oldest_tweet = None
   
    sentiments = []
    tweet_times = []
    
    for x in range(1, 6):
        public_tweets = api.user_timeline(org,
                                          page=x, 
                                          result_type="recent", 
                                          tweet_mode="extended",
                                          max_id = oldest_tweet)
        for tweet in public_tweets:
            sent_result = analyzer.polarity_scores(tweet["full_text"])
            sentiments.append(sent_result["compound"])
            tweet_times.append(tweet["created_at"])
            oldest_tweet = tweet["id"] - 1
            
    News_Sense_df[org + "_sents"] = sentiments
    News_Sense_df[org + "_ttime"] = tweet_times
            
```


```python
News_Sense_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>@BBCBreaking_sents</th>
      <th>@BBCBreaking_ttime</th>
      <th>@CBSNews_sents</th>
      <th>@CBSNews_ttime</th>
      <th>@cnnbrk_sents</th>
      <th>@cnnbrk_ttime</th>
      <th>@FoxNews_sents</th>
      <th>@FoxNews_ttime</th>
      <th>@nytimes_sents</th>
      <th>@nytimes_ttime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.6688</td>
      <td>Sat Jul 14 16:23:05 +0000 2018</td>
      <td>-0.8402</td>
      <td>Sun Jul 15 02:03:04 +0000 2018</td>
      <td>-0.6705</td>
      <td>Sun Jul 15 02:02:36 +0000 2018</td>
      <td>-0.4939</td>
      <td>Sun Jul 15 02:15:58 +0000 2018</td>
      <td>0.0000</td>
      <td>Sun Jul 15 02:13:14 +0000 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.0516</td>
      <td>Sat Jul 14 15:52:16 +0000 2018</td>
      <td>0.0000</td>
      <td>Sun Jul 15 01:48:04 +0000 2018</td>
      <td>-0.7964</td>
      <td>Sat Jul 14 22:15:27 +0000 2018</td>
      <td>0.5574</td>
      <td>Sun Jul 15 02:05:17 +0000 2018</td>
      <td>0.5423</td>
      <td>Sun Jul 15 01:56:10 +0000 2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>Sat Jul 14 14:34:06 +0000 2018</td>
      <td>-0.8402</td>
      <td>Sun Jul 15 01:40:07 +0000 2018</td>
      <td>0.7717</td>
      <td>Sat Jul 14 16:52:18 +0000 2018</td>
      <td>0.0000</td>
      <td>Sun Jul 15 02:02:58 +0000 2018</td>
      <td>0.0000</td>
      <td>Sun Jul 15 01:39:40 +0000 2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.4753</td>
      <td>Fri Jul 13 18:47:11 +0000 2018</td>
      <td>-0.6486</td>
      <td>Sun Jul 15 01:33:03 +0000 2018</td>
      <td>0.7717</td>
      <td>Sat Jul 14 16:31:12 +0000 2018</td>
      <td>0.0000</td>
      <td>Sun Jul 15 01:50:28 +0000 2018</td>
      <td>-0.4404</td>
      <td>Sun Jul 15 01:23:20 +0000 2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.7906</td>
      <td>Fri Jul 13 16:46:13 +0000 2018</td>
      <td>-0.6486</td>
      <td>Sun Jul 15 01:18:03 +0000 2018</td>
      <td>0.0000</td>
      <td>Sat Jul 14 16:03:24 +0000 2018</td>
      <td>0.8588</td>
      <td>Sun Jul 15 01:33:49 +0000 2018</td>
      <td>-0.8555</td>
      <td>Sun Jul 15 01:07:13 +0000 2018</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get the average sentiment of the last 100 tweets for each news organization, store in a dataframe
average_sent = []

for org in news_orgs:
    avg_sent = News_Sense_df[org + "_sents"].mean()
    average_sent.append(avg_sent)

news_orgs_present = ("BBC", "CBS", "CNN", "Fox", "NYT")
Averages_df = pd.DataFrame({"Organization" : news_orgs_present,
                           "Average Sentiment": average_sent})
Averages_df = Averages_df[["Organization", "Average Sentiment"]]
Averages_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Organization</th>
      <th>Average Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC</td>
      <td>-0.116790</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBS</td>
      <td>-0.130990</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN</td>
      <td>-0.100812</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fox</td>
      <td>0.066101</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NYT</td>
      <td>-0.010673</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the sentiments chart
plt.figure(figsize=(10,7))
plt.scatter(np.arange(0,100,1),News_Sense_df["@BBCBreaking_sents"], alpha=0.9, facecolors="skyblue", edgecolors="black", label="BBC")
plt.scatter(np.arange(0,100,1),News_Sense_df["@CBSNews_sents"], alpha=0.9, facecolors="green", edgecolors="black", label="CBS")
plt.scatter(np.arange(0,100,1),News_Sense_df["@cnnbrk_sents"], alpha=0.9, facecolors="red", edgecolors="black", label="CNN")
plt.scatter(np.arange(0,100,1),News_Sense_df["@FoxNews_sents"], alpha=0.9, facecolors="blue", edgecolors="black", label="Fox")
plt.scatter(np.arange(0,100,1),News_Sense_df["@nytimes_sents"], alpha=0.9, facecolors="yellow", edgecolors="black", label="New York Times")
plt.xlim(101,-1)
plt.xlabel("Tweets Ago", fontsize=14)
plt.yticks(np.arange(-1, 1.4, step=0.5))
plt.ylim(-1.1,1.1)
plt.ylabel("Tweet Polarity", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
analysis_date = datetime.now()
date = datetime.strftime(analysis_date, '%m/%d/%Y')
plt.title(f"Sentiment Analysis of Media Tweets ({date})")
leg = plt.legend(bbox_to_anchor=(1.22,1), loc="upper right", frameon=1)
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')
leg.set_title("Media Sources", prop = {'size':'12'})
plt.savefig("Media Sentiment Analysis_100 Tweets.png", bbox_inches="tight")
plt.show()
```


![png](main_files/main_5_0.png)



```python
plt.figure(figsize=(10,7))
my_colors = ["skyblue", "green", "red", "blue", "yellow"]
plt.bar(news_orgs_present, average_sent, alpha=1, edgecolor="black", width=1, color=my_colors)
plt.grid(False)
plt.ylim(min(average_sent) - 0.025, max(average_sent) + 0.025)
plt.ylabel("Tweet Polarity", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
for x, y in enumerate(average_sent):
    if y > 0.0:
        plt.text(x-0.15, y + .015, "+" + str(round(y,2)), color='black', fontsize=14)
    else:
        plt.text(x-0.15, y - .015, str(round(y,2)), color='black', fontsize=14) 

plt.title(f"Overall Media Sentiment based on Twitter ({date})")
plt.savefig("Average Media Sentiment.png")
```


![png](main_files/main_6_0.png)

