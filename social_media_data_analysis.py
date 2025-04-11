# import required libraries
import pandas as pd               #for data manipulation and analysis
import numpy as np                #for numerical computations
import matplotlib.pyplot as plt   #for data visualization
import seaborn as sns             #for statistical visualization
import random                     #for random data generation to simulate


#visualization setup
sns.set(style = "whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


#dataset simulation (in case we don't have real data to use)
#Tweets categories list
categories = ["Tech", "Health", "Sports", "Entertainment", "Politics", "Education", "Travel"]


#simulation of 10000 Tweets
np.random.seed(20)
random.seed(20)

data = {
    "tweet_id": np.arange(10000),
    "category": [random.choice(categories) for _ in range(10000)],
    "likes": np.random.poisson(lam = 100, size = 10000),            #engagement simulation
    "comments": np.random.poisson(lam = 35, size = 10000),
    "retweets": np.random.poisson(lam = 60, size = 10000)
}

df = pd.DataFrame(data)


#data information and cleaning
print(df.info())

#missing values check
print("\n\n\nMissing values per column:")
df_checking = df.isnull().sum()
print(df_checking)


#boxplot
plt.figure(figsize = (8, 5))
sns.boxplot(data = df, x = "category", y = "likes")
plt.title("Distribution of likes across categories:")
plt.xticks(rotation = 20)
plt.ylabel("number of likes")
plt.xlabel("tweet category")
plt.tight_layout()
plt.show()


#average likes per category
avg_likes = df.groupby("category")["likes"].mean().sort_values(ascending = False)


#barplot
sns.barplot(data = df, x = avg_likes.index, y = avg_likes.values)
plt.title("Average likes per category:")
plt.ylabel("average number of likes")
plt.xlabel("tweet category")
plt.xticks(rotation = 20)
plt.tight_layout()
plt.show()


#total engagement per category (likes, comments, and retweets)
df["total_engagement"] = df["likes"] + df["retweets"] + df["comments"]

engagement_summary = df.groupby("category")["total_engagement"].mean().sort_values(ascending = False)


#total engagement barplot
sns.barplot(data = df, x = engagement_summary.index, y = engagement_summary.values)
plt.title("Average total engagement per category:")
plt.ylabel("average engagement (likes, retweets, comments)")
plt.xlabel("tweet category")
plt.xticks(rotation = 20)
plt.tight_layout()
plt.show()


#summary of insights
print("\nImportant insights:")
print(f"\n-Most liked category on average: {avg_likes.idxmax()} ({avg_likes.max():.3f} likes.)")
print(f"-Category with highest overall engagement: {engagement_summary.idxmax()} ({engagement_summary.max():.3f} overall interactions.)")
print(f"Dataset includes {df['tweet_id'].nunique()} tweets across {len(categories)} categories.")
