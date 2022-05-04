#Michael Policarpio
#Machine Learning Final Project
#Last edit: 5/4/2022
#Version: 1.0

#File handles all the functionality necessary for simple music recommandation system


import numpy as np

#for the dataframes
import pandas as pd

#for plotting data
import matplotlib.pyplot as plt

#not used but could prove very useful
import seaborn as sns
#not used but could be used in future for spotify import data
import spotipy

#import TSNE from sklearn manifold for 2-d fitting
from sklearn.manifold import TSNE

#scipy import for distances between points
from scipy.spatial.distance import cdist

#additional sklearn plugins that could be used for 2-d fitting (not used in this implementation)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#os to read files
import os

#additional plot software
import plotly.express as px

#module for 
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

    #used for searching out the songs for later, only leave track and artist
    searchcopy = data.copy()
    searchcopy.drop(data.columns.difference(['track','artist']), 1, inplace=True)
    print(searchcopy)

    #used for printing out information, the index is used to match up values between copies
    printoutcopy = data.copy()
    printoutcopy.drop(data.columns.difference(['track','artist','uri']), 1, inplace=True)
    print(printoutcopy)

    #actual feature editing can be done here------------
    #an additional example of features not used
    #data.drop(['track', 'artist', 'uri', 'mode','decade','key','time_signature', 'sections', 'popularity'], axis=1, inplace=True)
    data.drop(['track', 'artist', 'uri', 'mode','decade'], axis=1, inplace=True)
    print(data.shape)
    print(data)

    #embedding with 2 dimensions
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(data)

    #create dataframe with xy values
    proj = pd.DataFrame(columns=['x','y'], data=tsne_result)

    #check shape and result
    print(tsne_result.shape)
    print(tsne_result)
    print(proj)

    #make a copy with tracks to add the track names to the plot!
    newproj = proj.copy()
    newproj['track'] = labelcopy
    print(newproj)

    #print out the plot
    fig = px.scatter(newproj, x='x', y='y', hover_data=['x','y','track'])
    fig.show()

    #testa result value to see if its x,y fitted
    print(tsne_result[1])

    #different songs and how to fetch their points
    #point = fetch_song_point(searchcopy, 'While I Was Walking', 'Tommy McCook', tsne_result) 
    point = fetch_song_point(searchcopy, 'Mashed Potato Time', 'Dee Dee Sharp', tsne_result)
    #point = fetch_song_point(searchcopy, 'No Tears Left To Cry', 'Ariana Grande', tsne_result)

    #invaid song test that throws error
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
    #distances = cdist
    #fashion_scatter(tsne_result)

    #tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=2))])
    #genre_embedding = tsne_pipeline.fit_transform(X)

#test the song and find 10 songs most closest related to it based off features
def song_test(input_data,label_data,print_data,point):
    print("song tested")

    #test point value
    #point = [(24,24)]


    distances = cdist(point, input_data, 'cosine')
    print(distances)
    index = np.argsort(distances[0])
    print(index)
    print(index[:10])

    shortened = index[:10]

    print_data['distance'] = distances.tolist()[0]
    #shortened = index[:10]
    #print(shortened, 100)

    #print(label_data.loc[shortened])
    print(print_data.loc[shortened])

#fetch the song location from all the points in the set for usage in similarity checks
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