#Michael Policarpio
#Machine Learning Final Project
#Last edit: 5/4/2022
#Version: 1.0

#File handles all the functionality necessary for simple music recommandation system

#numpy for math functions
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
    print("Loading data from spotify_dataset.csv!")
    print("--------------------------------------")

    #load data from desired data set (provided in the github)
    spotify_data = pd.read_csv('./spotify_dataset.csv')

    return spotify_data



#main function for project functionality
def main():
    print("Welcome to Re-album! Let us recommend songs for you!")
    print("----------------------------------------------------")
    data = load_data()

    #used to cut data and increase runtime for tests
    #data = data.head(1000)

    #drop all but label
    labelcopy = data.copy()
    labelcopy.drop(data.columns.difference(['track']), 1, inplace=True)
    #print(labelcopy)

    #used for searching out the songs for later, only leave track and artist
    searchcopy = data.copy()
    searchcopy.drop(data.columns.difference(['track','artist']), 1, inplace=True)
    #print(searchcopy)

    #used for printing out information, the index is used to match up values between copies
    printoutcopy = data.copy()
    printoutcopy.drop(data.columns.difference(['track','artist','uri']), 1, inplace=True)
    #print(printoutcopy)

    #actual feature editing can be done here------------
    #an additional example of features not used
    #data.drop(['track', 'artist', 'uri', 'mode','decade','key','time_signature', 'sections', 'popularity'], axis=1, inplace=True)
    #additonal optition to remove s from decades for additional feature checking (stil in testing phase)
    #data = data['decade'].str.replace("s","")
    #print(data)
    data.drop(['track', 'artist', 'uri', 'mode','decade'], axis=1, inplace=True)

    #clean printouts for user to know what they are working with
    print("------------------------")
    print("Data shape: ")
    print(data.shape)
    print("Total data features selected for similarity check: ")
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

    #test a result value to see if its x,y fitted properly
    #print(tsne_result[1])

    #different songs and how to fetch their points
    point2 = fetch_song_point(searchcopy, 'While I Was Walking', 'Tommy McCook', tsne_result) 
    point = fetch_song_point(searchcopy, 'Mashed Potato Time', 'Dee Dee Sharp', tsne_result)

    point3 = fetch_song_point(searchcopy, 'No Tears Left To Cry', 'Ariana Grande', tsne_result)

    #invaid song test that throws error
    #error_point = fetch_song_point(searchcopy, 'Invalid Name', 'Invalid Name', tsne_result)

    #test point
    #point = [(24,24)]
    print("Point fetched to cater recommandations to: " + str(point))
    print("Point fetched to cater recommandations to: " + str(point2))

    pointlist = [point,point2]

    averagepoint = average_song_points(pointlist)
    #single run example
    song_test(proj, labelcopy, printoutcopy, point3)
    #average run example
    #song_test(proj, labelcopy, printoutcopy, averagepoint)
    #song_test(proj, labelcopy, printoutcopy, point)
    
    #maybe pipeline for future use with different features
    #tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=2))])
    #genre_embedding = tsne_pipeline.fit_transform(X)

#test the song and find 10 songs most closest related to it based off features
def song_test(input_data,label_data,print_data,point):

    #choose distance algorithm here - euclidean, cosine, etc
    distances = cdist(point, input_data, 'euclidean')
    #print(distances)
    
    #sort distances from least to greatest index
    index = np.argsort(distances[0])
    #print(index)
    #print(index[:10])

    #shorten distances to top 10 indexes - change 10 to wharever n number of recommandations
    shortened = index[:10]
    #shortened = index[:10]
    #print(shortened, 100)

    #append distance data to the list for relevant output
    print_data['distance'] = distances.tolist()[0]

    print("Recommendation album - including the input song (if not an average):")
    print("------------------------------------------------")
    #to pick a different printout type with just labels
    #print(label_data.loc[shortened])
    print(print_data.loc[shortened])

#fetch the song location from all the points in the set for usage in similarity checks
def fetch_song_point(label_data, songname, artistname, tsne_result):
    print("Looking for song in data set to fetch data point...")
    if np.where(label_data['track'] == songname):

        if np.where(label_data['artist'] == artistname):

            temp = np.where((label_data['track'] == songname) & (label_data['artist'] == artistname))

            #print found index
            #print(temp[0])

            result = tsne_result[temp[0]]

            if len(result) == 0:
                print("Song not found in set!")
                return "not found"
            else:
                print("Song found in set!")
            print(result)
            
            return result
    
#function to average out song points, can take any amount of song points
def average_song_points(point_list):
    print("Averaging song points!")
    sumx = 0
    sumy = 0
    count = 0

    for point in point_list:
        sumx = sumx + point[0][0]
        sumy = sumy + point[0][1]
        count = count + 1
    
    if point_list:
        avgx = sumx/count
        avgy = sumy/count

    list = [(avgx, avgy)]

    output = np.array(list)

    print("Average point value found:")
    print(output)
    #print(output.shape())
    #output[0][0] = sumx/count
    #output[0][1] = sumy/count
    #print(output)

    #outputlist = [output]

    return output





  
#run main
if __name__=="__main__":
    main()