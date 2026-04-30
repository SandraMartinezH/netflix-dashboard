import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# Page settings
st.set_page_config(page_title="Netflix Dashboard", layout="wide")

# Load data
df = pd.read_csv("netflix_titles.csv")

# Title
st.title("Netflix Data Dashboard")
st.write("Analysis and visualizations of Netflix content.")

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Dataset information
st.subheader("Dataset Information")
st.write("Shape:", df.shape)
st.write("Columns:", list(df.columns))

st.subheader("Numerical Summary")
st.dataframe(df.describe())

st.subheader("Categorical Summary")
st.dataframe(df.describe(include="O"))

# -------------------------
# DATA CLEANING
# -------------------------
st.header("Data Cleaning")

st.subheader("Missing Values Before Cleaning")
st.dataframe(df.isnull().sum())

st.write("""
The dataset contains missing values mainly in the columns **director**, **cast**, and **country**.  
This indicates that some information is incomplete, which is common in real-world datasets.
""")

df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["date_added"] = df["date_added"].fillna(df["date_added"].mode()[0])

df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Unknown")
df["country"] = df["country"].fillna(df["country"].mode()[0])
df["rating"] = df["rating"].fillna(df["rating"].mode()[0])
df["duration"] = df["duration"].fillna(df["duration"].mode()[0])

st.subheader("Missing Values After Cleaning")
st.dataframe(df.isnull().sum())

st.write("Duplicate rows:", df.duplicated().sum())

st.write("""
During the data cleaning process, missing values were handled appropriately.  
The **date_added** column was converted into datetime format, and missing values were filled using suitable methods such as the mode or the label **Unknown**.

Overall, the dataset is now clean and ready for analysis.
""")

# New columns
df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month
df["type_numeric"] = df["type"].map({"Movie": 0, "TV Show": 1})

# -------------------------
# DATA VISUALIZATION
# -------------------------
st.header("Data Visualization")

# 1. Content Type Distribution
st.subheader("Content Type Distribution")

fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="type", palette="Set2", edgecolor="black", ax=ax1)

ax1.set_title("Distribution of Movies and TV Shows", fontsize=16, fontweight="bold")
ax1.set_xlabel("Type")
ax1.set_ylabel("Count")

st.pyplot(fig1)

st.write("""
The visualization shows that Netflix has a significantly higher number of **movies** compared to **TV shows**.  
This suggests that the platform focuses more on movie content, although TV shows are also an important part of its catalog.
""")

# 2. Top Countries Producing Content
st.subheader("Top 10 Countries Producing Content")

fig2, ax2 = plt.subplots(figsize=(10, 6))
df["country"].value_counts().head(10).plot(kind="bar", ax=ax2)

ax2.set_title("Top 10 Countries Producing Content", fontsize=16, fontweight="bold")
ax2.set_xlabel("Country")
ax2.set_ylabel("Count")
plt.xticks(rotation=45)

st.pyplot(fig2)

st.write("""
The United States is the leading producer of content on Netflix, followed by other countries.  
From a business perspective, this shows a strong dependence on U.S. content, but it also creates an opportunity to diversify content from other regions.
""")

# 3. Content Ratings Distribution
st.subheader("Content Ratings Distribution")

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.countplot(
    data=df,
    x="rating",
    order=df["rating"].value_counts().index,
    edgecolor="black",
    ax=ax3
)

ax3.set_title("Distribution of Content Ratings", fontsize=16, fontweight="bold")
ax3.set_xlabel("Rating")
ax3.set_ylabel("Count")
plt.xticks(rotation=45)

st.pyplot(fig3)

st.write("""
This chart shows the distribution of content ratings on Netflix.  
It helps identify the main audience segments targeted by the platform, including general audiences and mature viewers.
""")

# 4. Content Growth Over Time
st.subheader("Content Growth Over Time")

fig4, ax4 = plt.subplots(figsize=(10, 6))
df["year_added"].value_counts().sort_index().plot(kind="line", marker="o", ax=ax4)

ax4.set_title("Number of Titles Added Over Time", fontsize=16, fontweight="bold")
ax4.set_xlabel("Year")
ax4.set_ylabel("Count")

st.pyplot(fig4)

st.write("""
This visualization shows how the number of titles added to Netflix has changed over time.  
The growth suggests that Netflix expanded its catalog significantly, likely to remain competitive in the streaming market.
""")

# 5. Top Genres
st.subheader("Top 10 Genres on Netflix")

fig5, ax5 = plt.subplots(figsize=(10, 6))
df["listed_in"].str.split(", ").explode().value_counts().head(10).plot(kind="bar", ax=ax5)

ax5.set_title("Top 10 Genres on Netflix", fontsize=16, fontweight="bold")
ax5.set_xlabel("Genre")
ax5.set_ylabel("Count")
plt.xticks(rotation=45)

st.pyplot(fig5)

st.write("""
The genre analysis shows the most common types of content available on Netflix.  
This insight can help understand audience preferences and guide future content investment decisions.
""")

# 6. Content Type by Country
st.subheader("Content Type by Country")

country_type = pd.crosstab(df["country"], df["type"]).head(10)
st.dataframe(country_type)

st.write("""
This table compares movies and TV shows across the top countries in the dataset.  
It helps identify whether certain countries contribute more movies or TV shows to the Netflix catalog.
""")

# 7. Ratings by Content Type
st.subheader("Ratings Distribution by Type")

fig6, ax6 = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="rating", hue="type", ax=ax6)

ax6.set_title("Ratings Distribution by Type", fontsize=16, fontweight="bold")
ax6.set_xlabel("Rating")
ax6.set_ylabel("Count")
plt.xticks(rotation=45)

st.pyplot(fig6)

st.write("""
This chart compares ratings between movies and TV shows.  
It provides insight into how content type relates to audience targeting.
""")

# 8. Content Added by Month
st.subheader("Content Added by Month")

fig7, ax7 = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="month_added", ax=ax7)

ax7.set_title("Content Added by Month", fontsize=16, fontweight="bold")
ax7.set_xlabel("Month")
ax7.set_ylabel("Count")

st.pyplot(fig7)

st.write("""
This analysis shows whether Netflix adds more content during specific months.  
Some months may show higher activity, which could be related to release strategies or seasonal demand.
""")

# 9. Correlation Analysis
st.subheader("Correlation Analysis")

corr = df[["release_year", "year_added", "type_numeric"]].corr()

fig8, ax8 = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax8)

ax8.set_title("Correlation Including Content Type", fontsize=14, fontweight="bold")

st.pyplot(fig8)

st.write("""
The correlation analysis shows weak relationships between the numerical variables.  
This is expected because most of the Netflix dataset contains categorical information.
""")

# 10. Scatter Plot
st.subheader("Release Year vs Year Added")

fig9, ax9 = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="release_year",
    y="year_added",
    hue="type",
    size="month_added",
    alpha=0.5,
    ax=ax9
)

ax9.set_title("Release Year vs Year Added", fontsize=16, fontweight="bold")
ax9.set_xlabel("Release Year")
ax9.set_ylabel("Year Added")

st.pyplot(fig9)

st.write("""
The scatter plot shows the relationship between a title's release year and the year it was added to Netflix.  
Most content is added within a few years after release, suggesting that Netflix focuses on relatively recent content.  
However, some older titles are also included, showing that the platform maintains a mix of new and older content.
""")

# Final conclusion
st.header("Conclusion")

st.write("""
Overall, this dashboard shows that Netflix has a strong focus on movies, U.S. content, and continuous catalog expansion.  
From a consulting perspective, Netflix could benefit from increasing international content diversity and continuing to align its content strategy with audience preferences.

""")
