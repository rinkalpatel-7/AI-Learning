import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity


credits = pd.read_csv("RecEngine/tmdb_5000_credits.csv")
movies = pd.read_csv("RecEngine/tmdb_5000_movies.csv")


# print(credits.columns)
# print(movies.columns)

new_df = movies.merge(credits,on='title')
# print(new_df.columns)

movies = new_df[['movie_id','title','overview','cast','crew','keywords','genres']]
# string=movies['cast'].iloc[0]
# print(movies.isnull().sum())
movies.dropna(inplace=True)
# print(movies.isnull().sum())
# print(movies.duplicated().sum())

# string=movies.iloc[0].genres
# print(string)


def get_name(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres']=movies['genres'].apply(get_name)
movies['keywords']=movies['keywords'].apply(get_name)
# string=movies.iloc[0].genres
# print(movies.head())

def get_first_3_name(obj):
    L=[]
    cnt=0
    for i in ast.literal_eval(obj):
        if cnt!=3:#picking 3 imp char
            L.append(i['name'])
            cnt+=1
        else:
            break
    return L

movies['cast']=movies['cast'].apply(get_first_3_name)
# print(movies['cast'].iloc[9])

def get_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
    return L

movies['crew']=movies['crew'].apply(get_director)
# print(movies['keywords'].iloc[9])

movies['overview']=movies['overview'].apply(lambda x:x.split())
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags'] = movies['overview']+movies['genres']+movies['cast']+movies['crew']+movies['keywords']

rec_df = movies[['movie_id','title','tags']]
rec_df['tags'] =  rec_df['tags'].apply(lambda x:" ".join(x).lower())

# print(rec_df['tags'][0])

cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(rec_df['tags']).toarray()
print(cv.get_feature_names_out())

ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

rec_df['tags']=rec_df['tags'].apply(stem)

similarity= cosine_similarity(vectors)
# print(sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6])

def recommend(mv):
    mv_ind = rec_df[rec_df['title']==mv].index[0]
    distances = similarity[mv_ind]
    mv_list = sorted(list(enumerate(distances)),key=lambda x:x[1])[1:6]

    for i in mv_list:
        print(rec_df.iloc[i[0]].title)
    
recommend('Avatar')
