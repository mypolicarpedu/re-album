
#Michael Policarpio
##BETA TEST FILE, can ignore for final project purposes!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy




#from scipy.spatial.distance import cdists

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.manifold import TSNE

from scipy.spatial.distance import cdist

import os


import plotly.express as px
#%matplotlib inline

#sp = spotipy.Spotify


def load_data():
    print("Loading data!")
    hello = [("Hello", 10), ("I am bigger", 1500)]

    spotify_data = pd.read_csv('./spotify_dataset.csv')


    return spotify_data



#main function for project functionality
def main():
    print("This is the machine learning project!")
    data = load_data()
    #print(data)

    #sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
    #fig = px.line(data, x='decade', y=sound_features)
    #fig = px.line(data, x='speechiness', y='key')
    #fig.show()

    #f,ax = plt.subplots(figsize=(18, 18))
    #sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
    #plt.show()

    #numeric = data.drop(['track','artist', 'uri'],  axis=1)
    #small = numeric.drop(['tempo','duration_ms','key','loudness','time_signature'], axis=1)
    #sns.set_palette('pastel')
    #small.mean().plot.bar()
    #plt.title('Mean Values of Audio Features')
    #plt.show()

    #fig =  px.scatter(data, x='danceability', y='energy',hover_data=['danceability', 'energy', 'track'])
    #fig.show()

    #data = data.head(1000)

    #drop all but label
    labelcopy = data.copy()
    labelcopy.drop(data.columns.difference(['track']), 1, inplace=True)
    print(labelcopy)

    searchcopy = data.copy()
    searchcopy.drop(data.columns.difference(['track','artist']), 1, inplace=True)
    print(searchcopy)

    printoutcopy = data.copy()
    printoutcopy.drop(data.columns.difference(['track','artist','uri']), 1, inplace=True)
    print(printoutcopy)

    data['decade'].str.replace("s","")
    print(data)
    #data.drop(['track', 'artist', 'uri', 'mode','decade','key','time_signature', 'sections', 'popularity'], axis=1, inplace=True)
    data.drop(['track', 'artist', 'uri', 'mode','decade'], axis=1, inplace=True)
    print(data.shape)
    print(data)

    #embedding with 2 dimensions
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(data)

    proj = pd.DataFrame(columns=['x','y'], data=tsne_result)

    print(tsne_result.shape)
    print(tsne_result)

    print(proj)

    newproj = proj.copy()
    newproj['track'] = labelcopy
    print(newproj)

    fig = px.scatter(newproj, x='x', y='y', hover_data=['x','y','track'])
    fig.show()

    print(tsne_result[1])

    #point = fetch_song_point(searchcopy, 'While I Was Walking', 'Tommy McCook', tsne_result)
    
    point = fetch_song_point(searchcopy, 'Mashed Potato Time', 'Dee Dee Sharp', tsne_result)

    #point = fetch_song_point(searchcopy, 'No Tears Left To Cry', 'Ariana Grande', tsne_result)

    error_point = fetch_song_point(searchcopy, 'Invalid Name', 'Invalid Name', tsne_result)

    print("song tested")
    #point = [(24,24)]
    distances = cdist(point, proj, 'euclidean')
    print(distances)
    index = np.argsort(distances[0])
    print(index)
    print(index[:10])

    shortened = index[:10]
    #shortened = index[:10]
    #print(shortened, 100)

    printoutcopy['distance'] = distances.tolist()[0]

    print(labelcopy.loc[shortened])
    print(printoutcopy.loc[shortened])

    #song_test(proj, labelcopy, printoutcopy)
    """
    for i in range(0, 10):
        temp = index[i]
        print(temp)
        print(labelcopy.loc[temp])
    """

    #distances = cdist
    #fashion_scatter(tsne_result)

    #tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=2))])
    #genre_embedding = tsne_pipeline.fit_transform(X)

    
def song_test(input_data,label_data,print_data):
    print("song tested")
    point = [(24,24)]


    distances = cdist(point, input_data, 'cosine')
    print(distances)
    index = np.argsort(distances[0])
    print(index)
    print(index[:10])

    shortened = index[:10]

    #test = distances.tolist()
    #print(test[0])

    print_data['distance'] = distances.tolist()[0]
    #shortened = index[:10]
    #print(shortened, 100)

    #print(label_data.loc[shortened])
    print(print_data.loc[shortened])

def fetch_song_point(label_data, songname, artistname, tsne_result):
    print("fetching song point")
    if np.where(label_data['track'] == songname):
        if np.where(label_data['artist'] == artistname):
            print("song found!")
            temp = np.where((label_data['track'] == songname) & (label_data['artist'] == artistname))
            print(temp[0])
            result = tsne_result[temp[0]]

            if len(result) == 0:
                print("no song found in set...")
                return "not found"
                #return("Error")
            print(result)
            
            return result
    


  


#run main
if __name__=="__main__":
    main()