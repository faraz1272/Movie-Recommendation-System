#%% md
# ##### Data Dictionary:
# 
# ### RATINGS FILE DESCRIPTION
# =========================================================================
# All ratings are contained in the file **"ratings.dat"** and are in the following format:
# `UserID::MovieID::Rating::Timestamp`
# 
# - **UserIDs** range between **1 and 6040**
# - **MovieIDs** range between **1 and 3952**
# - **Ratings** are made on a **5-star scale** (whole-star ratings only)
# - **Timestamp** is represented in **seconds**
# - **Each user has at least 20 ratings**
# 
# ---
# 
# ### USERS FILE DESCRIPTION
# =========================================================================
# User information is in the file **"users.dat"** and is in the following format:
# `UserID::Gender::Age::Occupation::Zip-code`
# 
# - All demographic information is provided voluntarily by the users and is **not checked for accuracy**.
# - Only users who have provided some demographic information are included in this dataset.
# - **Gender** is denoted by:
#   - `"M"` for male
#   - `"F"` for female
# 
# #### Age Groups:
# - `1`: "Under 18"
# - `18`: "18-24"
# - `25`: "25-34"
# - `35`: "35-44"
# - `45`: "45-49"
# - `50`: "50-55"
# - `56`: "56+"
# 
# #### Occupations:
# - `0`: "Other" or not specified
# - `1`: "Academic/Educator"
# - `2`: "Artist"
# - `3`: "Clerical/Admin"
# - `4`: "College/Grad Student"
# - `5`: "Customer Service"
# - `6`: "Doctor/Health Care"
# - `7`: "Executive/Managerial"
# - `8`: "Farmer"
# - `9`: "Homemaker"
# - `10`: "K-12 Student"
# - `11`: "Lawyer"
# - `12`: "Programmer"
# - `13`: "Retired"
# - `14`: "Sales/Marketing"
# - `15`: "Scientist"
# - `16`: "Self-Employed"
# - `17`: "Technician/Engineer"
# - `18`: "Tradesman/Craftsman"
# - `19`: "Unemployed"
# - `20`: "Writer"
# 
# ---
# 
# ### MOVIES FILE DESCRIPTION
# =========================================================================
# Movie information is in the file **"movies.dat"** and is in the following format:
# `MovieID::Title::Genres`
# 
# - Titles are identical to those provided by IMDB (including year of release).
# - Genres are **pipe (`|`) separated** and are selected from the following genres:
# 
#   - **Action**
#   - **Adventure**
#   - **Animation**
#   - **Children's**
#   - **Comedy**
#   - **Crime**
#   - **Documentary**
#   - **Drama**
#   - **Fantasy**
#   - **Film-Noir**
#   - **Horror**
#   - **Musical**
#   - **Mystery**
#   - **Romance**
#   - **Sci-Fi**
#   - **Thriller**
#   - **War**
#   - **Western**
# 
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error
from keras.src.losses import mean_squared_error
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Embedding, Flatten
from keras.layers import dot
from pylab import rcParams
from cmfrec import CMF
from seaborn import color_palette

warnings.filterwarnings('ignore')
#%%
users = pd.read_fwf('/Users/farazahmed/Downloads/drive-download-20250215T114129Z-001/zee-users.dat', encoding="ISO-8859-1")
ratings = pd.read_fwf('/Users/farazahmed/Downloads/drive-download-20250215T114129Z-001/zee-ratings.dat', encoding="ISO-8859-1")
movies = pd.read_fwf('/Users/farazahmed/Downloads/drive-download-20250215T114129Z-001/zee-movies.dat', encoding="ISO-8859-1")
#%%
users.head()
#%%
ratings.head()
#%%
movies.head()
#%%
# Splitting the rows in each table seperated by '::'
delimit = '::'
users = users['UserID::Gender::Age::Occupation::Zip-code'].str.split(delimit, expand=True)
users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
users.head()
#%%
ratings = ratings['UserID::MovieID::Rating::Timestamp'].str.split(delimit, expand=True)
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings.head()
#%%
movies = movies['Movie ID::Title::Genres'].str.split(delimit, expand=True)
movies.columns = ['Movie ID', 'Title', 'Genres']
movies.head()
#%%
# We notice that there are only 7 unique Age values, which represent a range.
users['Age'].value_counts()
#%%
# Binning the age column.
users.replace({'Age':{'1':  "Under 18",
                      '18':  "18-24",
                      '25':  "25-34",
                      '35':  "35-44",
                      '45':  "45-49",
                      '50':  "50-55",
                      '56':  "56 Above"}}, inplace=True)
#%%
# Replacing the occupation values with their corresponding occupation names.
users.replace({'Occupation':{'0': "other",
                             '1': "academic/educator",
                             '2': "artist",
                             '3': "clerical/admin",
                             '4': "college/grad student",
                             '5': "customer service",
                             '6': "doctor/health care",
                             '7': "executive/managerial",
                             '8': "farmer",
                             '9': "homemaker",
                             '10': "k-12 student",
                             '11': "lawyer",
                             '12': "programmer",
                             '13': "retired",
                             '14': "sales/marketing",
                             '15': "scientist",
                             '16': "self-employed",
                             '17': "technician/engineer",
                             '18': "tradesman/craftsman",
                             '19': "unemployed",
                             '20': "writer"}}, inplace=True)
#%%
# Correcting 'MovieID' column name.
movies.columns = ['MovieID', 'Title', 'Genres']
movies.head()
#%%
# Merging movies and ratings.
df1 = pd.merge(movies, ratings, on='MovieID', how='inner')
df1.head()
#%%
# Merging with users.
df2 = pd.merge(df1, users, on='UserID', how='inner')
df2.head()
#%%
df_final = df2.copy(deep=True)
#%%
print("No. of rows: ", df_final.shape[0])
print("No. of columns: ", df_final.shape[1])
#%%
df_final.info()
#%% md
# #### Feature Engineering
#%%
# Converting rating column to integer type.
df_final['Rating'] = df_final['Rating'].astype(int)
#%%
# We have the timestamp given to us in seconds, so converting it to datetime.
df_final['Timestamp'] = pd.to_datetime(df_final['Timestamp'], unit='s')
#%%
df_final.sample(50)
#%%
# Extracting release year from the title and storing it in a new column.
df_final['Release Year'] = df_final['Title'].str.strip().str[-5:-1]
df_final.head()
#%%
# We notice some incorrect year values.
df_final['Release Year'].unique()
#%%
# Replacing incorrect year values where possible.
df_final['Release Year'].replace({'964)': '1964', '995)': '1995', '981)': '1981', '989)': '1989'}, inplace=True)
df_final['Release Year'].unique()
#%%
# Dropping rows where the years do not make sense.

idx_val = df_final[(df_final['Release Year']=='n th') |
               (df_final['Release Year']=='der ') |
               (df_final['Release Year']=='(199') |
               (df_final['Release Year']=='he B') |
               (df_final['Release Year']==' Art') |
               (df_final['Release Year']==') (1') |
               (df_final['Release Year']==' (19') |
               (df_final['Release Year']=='Polar') |
               (df_final['Release Year']=="e d'") |
               (df_final['Release Year']=='olar') |
               (df_final['Release Year']=='(196') |
                (df_final['Release Year']=='(198') |
                (df_final['Release Year']=='nd) ') |
                (df_final['Release Year']=='n) (') |
                (df_final['Release Year']=='dron') |
                (df_final['Release Year']=='e (1')].index

df_final.drop(index=idx_val, inplace=True)
df_final['Release Year'].unique()
#%%
# Converting 'Release Year' to int data type.
df_final['Release Year'] = df_final['Release Year'].astype(int)
#%%
df_final.sample(5)
#%%
# Creating the title column with only the title names.
df_final['Title'] = df_final['Title'].str.strip().str[:-7]
df_final.sample(5)
#%%
# Creating a new column indicating the decade the movie was released.
bins = [1919, 1929, 1939, 1949, 1959, 1969, 1979, 1989, 2000]
labels = ['20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']
df_final['ReleaseDec'] = pd.cut(df_final['Release Year'], bins=bins, labels=labels)
df_final.sample(5)
#%% md
# #### Data Cleaning
#%%
# Checking nulls
df_final.isna().sum()
#%%
df_final[df_final['Genres'].isna()]
#%% md
# #### Exploratory Data Analysis and Visualization
#%%
df_final.sample(5)
#%%
# Distribution of Ratings
plt.figure(figsize=(12, 8))
sns.distplot(df_final['Rating'], color='black', label='Rating')
plt.title('Distribution of Ratings')
plt.show()
#%% md
# ##### From the above distribution we can observe that most of the movies are given rating 4.0
#%%
# Distribution of movies released through the years.
plt.figure(figsize=(12, 8))
sns.distplot(df_final['Release Year'], color='black', label='Release Year')
plt.title('Distribution of movie releases through years')
plt.figure(figsize=(12, 8))
plt.show()
#%% md
# ##### As expected movie releases increase over the years, as validated by the above left skewed distribution.
#%%
plt.figure(figsize=(12, 8))
sns.countplot(df_final['ReleaseDec'], color='green')
plt.title('Number of Movie releases each decade')
plt.show()
#%% md
# ##### From our data, most of the movies are released in the 90s
#%%
plt.figure(figsize=(12, 8))
sns.countplot(x=df_final['Age'])
plt.title('Age distribution')
plt.show()
#%% md
# ##### Media consumption is maximum for ages between 25 to 34, hence the business can focus on creating more tailored content for this age range.
#%%
plt.figure(figsize=(8, 6))
sns.countplot(x=df_final['Gender'])
plt.title('Gender count')
plt.show()
#%% md
# ##### Males consume more content than females
#%%
# Distribution of Occupation.
sorted_counts = df_final['Occupation'].value_counts()
plt.figure(figsize=(12, 8))
sns.countplot(data=df_final, y=df_final['Occupation'], order=sorted_counts.index, hue='Gender')
plt.title('Occupation distribution')
plt.show()
#%% md
# ##### College going students are the major consumers of media, which aligns with our age distribution as well.
#%%
df_final
#%%
# Top 3 Genre by age groups.
df_temp = df_final.groupby(['Age', 'Genres'])['Rating'].mean().reset_index().sort_values(by=['Age', 'Rating'], ascending=[True, False]).groupby('Age').head(3)
df_temp
#%%
# Plots for top 3 genre by age groups.
fig, ax = plt.subplots(figsize=(20, 15), nrows=3, ncols=3)
fig.suptitle('Top 3 Genre by Age Group', fontsize=20)

palette = sns.color_palette("Set2")  # Feel free to change to other palettes like "Paired", "tab10", etc.

# Age 18-24
df_age1 = df_temp[df_temp['Age'] == '18-24']
sns.barplot(data=df_age1, x=df_age1['Genres'], y=df_age1['Rating'], ax=ax[0, 0], palette=palette)
ax[0, 0].set_title('18-24')

# Age 25-34
df_age2 = df_temp[df_temp['Age'] == '25-34']
sns.barplot(data=df_age2, x=df_age2['Genres'], y=df_age2['Rating'], ax=ax[0, 1], palette=palette)
ax[0, 1].set_title('25-34')

# Age 35-44
df_age3 = df_temp[df_temp['Age'] == '35-44']
sns.barplot(data=df_age3, x=df_age3['Genres'], y=df_age3['Rating'], ax=ax[0, 2], palette=palette)
ax[0, 2].set_title('35-44')

# Age 45-49
df_age4 = df_temp[df_temp['Age'] == '45-49']
sns.barplot(data=df_age4, x=df_age4['Genres'], y=df_age4['Rating'], ax=ax[1, 0], palette=palette)
ax[1, 0].set_title('45-49')

# Age 50-55
df_age5 = df_temp[df_temp['Age'] == '50-55']
sns.barplot(data=df_age5, x=df_age5['Genres'], y=df_age5['Rating'], ax=ax[1, 1], palette=palette)
ax[1, 1].set_title('50-55')

# Age 56 Above
df_age6 = df_temp[df_temp['Age'] == '56 Above']
sns.barplot(data=df_age6, x=df_age6['Genres'], y=df_age6['Rating'], ax=ax[1, 2], palette=palette)
ax[1, 2].set_title('56 Above')

# Under 18
df_age7 = df_temp[df_temp['Age'] == 'Under 18']
sns.barplot(data=df_age7, x=df_age7['Genres'], y=df_age7['Rating'], ax=ax[2, 0], palette=palette)
ax[2, 0].set_title('Under 18')

fig.tight_layout()
fig.delaxes(ax[2, 1])
fig.delaxes(ax[2, 2])

plt.show()
#%%
# Top 10 titles by average rating.
df_final.groupby('Title')['Rating'].mean().reset_index().sort_values(by=['Rating'], ascending=False).head(10)
#%%
# Top 10 most rated titles
df_final.groupby('Title')['Rating'].count().reset_index().sort_values(by=['Rating'], ascending=False).head(10)
#%% md
# #### Model Building
# ##### The model will be built using collaborative filtering and we will explore both user-user and item-item based approaches.
#%%
# Taking the mean of rating in case the user gave more than one rating to same title.
df_matrix = df_final.groupby(['UserID', 'Title'])['Rating'].mean().reset_index()
#%%
# Creating interaction table for user, title and rating using pivot table.
interaction_matrix = pd.pivot_table(df_matrix, values='Rating', index='UserID', columns='Title')
interaction_matrix
#%%
# Replacing NaN with zeros.
interaction_matrix.fillna(0, inplace=True)
#%%
interaction_matrix
#%% md
# ##### Recommendation with Pearson correlation
#%% md
# ###### Item based approach
#%%
# Input a movie title.
movie = input('Enter movie title: ')
print(f"Movie title: {movie}")
#%%
# Getting the top 5 movies with the highest correlation.
movie_rating_vector = interaction_matrix[movie]
similar_movies = interaction_matrix.corrwith(movie_rating_vector)
similar_movies_df = pd.DataFrame(similar_movies, columns=['Correlation'])
similar_movies_df.sort_values(by=['Correlation'], ascending=False, inplace=True)
similar_movies_df.iloc[1:6]
#%% md
# ##### We can see above that for a given movie title 'Terminator', which is an Action/Sci-Fi genre, we get movies within similar genre. You can try for other titles!
#%% md
# ##### Recommendation with Cosine Similarity
# ###### Cosine similarity gives an estimate if two vectors are point roughly in the same direction. The values varies between -1 and 1.
# ###### In our case, we will measure the similarity between the movie input by the user with all the movies in the data.
#%%
# Item-Item based similarity
item_sim = cosine_similarity(interaction_matrix.transpose())
item_sim
#%%
item_sim_matrix = pd.DataFrame(item_sim, columns=interaction_matrix.columns, index=interaction_matrix.columns)
item_sim_matrix
#%%
# Testing item-item recommendations
movie = input('Enter movie title: ')
print(f"Movie title: {movie}")
#%%
#We can see similar quality of recommendations for cosine similarity.
item_sim_matrix[movie].sort_values(ascending=False).iloc[1:6]
#%%
df_final['UserID'] = df_final['UserID'].astype(int)
df_final.info()
#%% md
# ##### User-user based approach
#%%
# User-user based recommendation.

user_sim = cosine_similarity(interaction_matrix)
user_sim
#%%
user_sim_matrix = pd.DataFrame(user_sim, columns=interaction_matrix.index, index=interaction_matrix.index)
user_sim_matrix
#%%
# Top 5 users similar to UserID 1
user_sim_matrix['1'].sort_values(ascending=False).iloc[1:6].reset_index()
#%%
# Recommending UserID 1 with the top-rated movies of UserID 5343.
df_final[df_final['UserID'] == 5343].sort_values(by='Rating', ascending=False).head(5)['Title']
#%% md
# #### Recommendation using Nearest Neighbours
#%%
# Initializing and fitting the model.
knn = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
knn.fit(interaction_matrix.transpose())
#%%
movie_name = input('Enter movie title: ')
print(f"Movie title: {movie_name}")
#%%
movie_rating_vector = interaction_matrix[movie_name].values.reshape(1,-1)
distances, indices = knn.kneighbors(movie_rating_vector, n_neighbors=10)
#%%
indices
#%%
distances
#%%
interaction_matrix
#%%
# Getting the top 10 recommendations for the given movie title.
for i in range(0, len(distances.flatten())):
    if i > 0:
        print(f"Movie title: {interaction_matrix.columns[indices.flatten()[i]]}, Distance: {round(distances.flatten()[i], 3)}")
#%% md
# ##### We can compare our recommendations with Google recommendations.
# ##### For 'E.T. the Extra-Terrestrial' we can observe 5 matching recommendations.
#%% md
# <img src="GoogleImage.png" alt="Google Image" width="1000">
#%% md
# <img src="GoogleImage2.png" alt="Google Image" width="1000">
#%% md
# #### Matrix Factorization
#%%
# Converting our matrix into a format that is accepted in the 'Collective Matrix Factorization(CMF)' Model.
df_final1 = df_final.groupby(['UserID', 'MovieID'])['Rating'].mean().reset_index()
df_final1
#%%
# The column names need to be specified in the following format.
df_final1.columns = ['UserId', 'ItemId', 'Rating']
df_final1
#%%
# Initializing and fitting the CMF model.
# We observed best recommending performance at latent factor(k) = 50
model = CMF(k = 50, lambda_ = 0.1, method = 'als', verbose = False, user_bias= True, item_bias=True)
model.fit(df_final1)
#%% md
# ##### Checking the decomposed matrices
#%%
model.A_.shape
#%%
model.B_.shape
#%%
model.A_
#%%
model.B_
#%%
model.glob_mean_
#%%
# Testing recommendations for users (here, user id = 5343)
top_items = model.topN(user = 5343, n = 10)
top_items
#%%
# Checking the recommended movies.
movies.loc[movies.MovieID.isin(top_items)]
#%%
# For comparison with our recommended movies, we get the list of top 10 highly rated movies by this user(5343).
df_final[df_final['UserID'] == 5343].sort_values(by='Rating', ascending=False)[['Title','Genres']].head(10)
#%% md
# ##### We can observe that the recommended genres of the movies is very close to what this user likes.
#%% md
# *-----------------------------------------------------------------------------------------------------------------------------------------------------------------*