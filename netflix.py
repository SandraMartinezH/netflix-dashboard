import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Netflix Dashboard", layout="wide")

# Load data
df = pd.read_csv("netflix_titles.csv")

# Title
st.title("Netflix Data Dashboard")
st.write("Analysis and visualizations of Netflix content.")

# Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Info
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
The dataset contains missing values mainly in the columns **director, cast, and country**.
This is common in real-world datasets and must be handled before analysis.
""")

# Cleaning
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
After cleaning, the dataset is complete and ready for analysis.
""")

# New columns
df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month
df["type_numeric"] = df["type"].map({"Movie": 0, "TV Show": 1})

# -------------------------
# VISUALIZATION
# -------------------------
st.header("Data Visualization")

# 1
st.subheader("Content Type Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x="type", palette="Set2", edgecolor="black", ax=ax1)
st.pyplot(fig1)

st.write("""
Netflix has significantly more **movies than TV shows**, showing a stronger focus on movie content.
""")

# 2
st.subheader("Top Countries Producing Content")
fig2, ax2 = plt.subplots()
df["country"].value_counts().head(10).plot(kind="bar", ax=ax2)
st.pyplot(fig2)

st.write("""
The **United States dominates content production**, followed by other countries.
""")

# 3
st.subheader("Content Ratings Distribution")
fig3, ax3 = plt.subplots()
sns.countplot(data=df, x="rating", order=df["rating"].value_counts().index, ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)

st.write("""
Most content is targeted at a **broad audience**, with some categories focused on mature viewers.
""")

# 4
st.subheader("Content Growth Over Time")
fig4, ax4 = plt.subplots()
df["year_added"].value_counts().sort_index().plot(kind="line", marker="o", ax=ax4)
st.pyplot(fig4)

st.write("""
Netflix has **increased content significantly over time**, especially in recent years.
""")

# 5
st.subheader("Top Genres")
fig5, ax5 = plt.subplots()
df["listed_in"].str.split(", ").explode().value_counts().head(10).plot(kind="bar", ax=ax5)
st.pyplot(fig5)

st.write("""
Some genres are clearly more dominant, reflecting user demand and platform strategy.
""")

# 6
st.subheader("Content Type by Country")
st.dataframe(pd.crosstab(df["country"], df["type"]).head(10))

# 7
st.subheader("Ratings by Content Type")
fig6, ax6 = plt.subplots()
sns.countplot(data=df, x="rating", hue="type", ax=ax6)
plt.xticks(rotation=45)
st.pyplot(fig6)

# 8
st.subheader("Content Added by Month")
fig7, ax7 = plt.subplots()
sns.countplot(data=df, x="month_added", ax=ax7)
st.pyplot(fig7)

st.write("""
Content is added throughout the year, but some months show higher activity.
""")

# 9
st.subheader("Correlation Analysis")
corr = df[["release_year", "year_added", "type_numeric"]].corr()

fig8, ax8 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax8)
st.pyplot(fig8)

st.write("""
There are **weak correlations** between variables, which is expected due to categorical data.
""")

# 10
st.subheader("Release Year vs Year Added")
fig9, ax9 = plt.subplots()
sns.scatterplot(
    data=df,
    x="release_year",
    y="year_added",
    hue="type",
    size="month_added",
    alpha=0.5,
    ax=ax9
)
st.pyplot(fig9)

st.write("""
Most content is added within a few years of release, but older content is also included.
""")
