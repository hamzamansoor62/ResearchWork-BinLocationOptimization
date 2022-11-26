from tracemalloc import start
from matplotlib.cbook import print_cycles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import math
import time

import sys
from qgis.core import *
qgs = QgsApplication([], False)
QgsApplication.setPrefixPath("/Applications/QGIS.app/Contents/MacOS", True)
qgs.initQgis()
sys.path.append('/Applications/QGIS.app/Contents/Resources/python/plugins')
import processing
from processing.core.Processing import Processing
Processing.initialize()
from QNEAT3.Qneat3Provider import Qneat3Provider
provider = Qneat3Provider()
QgsApplication.processingRegistry().addProvider(provider)


######       Saving Modified Origin-Destination Matrix        #######
def save_od(pathOD, cost):
    with open(pathOD, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['longorigin', 'latorigin', 'longdest', 'latdest', 'cost'])
        for each in cost:
            writer.writerow(each)

######       Function to Save into Csv files        #######
def save_csv(file_name, data, columns):
    with open(file_name, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        index = 0
        for each in data:
            index += 1
            writer.writerow([each[0], each[1], index])

######       Setting Up Origin-Destination Matrix        #######
def get_od(pathOD):
    destination = pd.read_csv('/Users/macbook/Desktop/hamza/data/centroids/centroids.csv')
    origin = pd.read_csv('/Users/macbook/Desktop/hamza/data/Built_Up_Gulberg_Unique.csv')
    od = pd.read_csv('/Users/macbook/Desktop/hamza/data/od/odMatrix.csv')
    
    od['network_cost'] = od['network_cost'].replace(np.nan,od['network_cost'].max())
    od_updated = od.merge(destination,left_on='destination_id', right_on='index').merge(origin,left_on='origin_id', right_on='index').drop(['index_x','index_y'], axis=1)
    cost = od_updated[['Longitude_y','Latitude_y','Longitude_x','Latitude_x','network_cost']].values.tolist()
    save_od(pathOD, cost)


######       Return minimum values for clustering        #######
def min_distance(builtup_n, centroids_k):
    od = pd.read_csv('/Users/macbook/Desktop/hamza/data/od/odMatrix.csv')
    distance = []
    rows = []
    count = 0
    for j in range(builtup_n):
        row = od.values[(j*centroids_k):((j+1)*centroids_k)]
        for each in row:
            if math.isnan(each[3]):
                each[3] = 1000000
            rows.append(each[3])
        distance.append(rows)
        rows = []
        count+=1

    minimumIdx = []
    for each in distance:
        if each:
            minimumIdx.append(np.argmin(each))

    return minimumIdx

######       Incase number of centroids dropped after clustering it keeps non-updated centroids        #######
def compensate(k, minimumIdx, new, cent):
    j = 0
    temp = []
    for i in range(0,k):
        if i in minimumIdx:
            temp.append(new[j])
            j+=1
        else:
            temp.append(cent[i])

    return temp

######       Calculating origin-destination matrix using QGIS plugin        #######
def ODMatrixCalculation():
    parameters = {'INPUT': '/Users/macbook/Desktop/hamza/data/RN/Lahore_RN_Gulberg.shp',
                'FROM_POINT_LAYER': 'delimitedtext://file:///Users/macbook/Desktop/hamza/data/Built_Up_Gulberg_Unique.csv?type=csv&maxFields=10000&detectTypes=yes&xField=Longitude&yField=Latitude&crs=EPSG:4326&spatialIndex=no&subsetIndex=no&watchFile=no',
                'FROM_ID_FIELD': 'index',
                'TO_POINT_LAYER': 'delimitedtext://file:///Users/macbook/Desktop/hamza/data/centroids/centroids.csv?type=csv&maxFields=10000&detectTypes=yes&xField=longitude&yField=latitude&crs=EPSG:4326&spatialIndex=no&subsetIndex=no&watchFile=no',
                'TO_ID_FIELD': 'index', 
                'STRATEGY': 0, 'ENTRY_COST_CALCULATION_METHOD': 0,
                'DIRECTION_FIELD': 'direction', 'VALUE_FORWARD': '1', 'VALUE_BACKWARD': '2', 'VALUE_BOTH': '0',
                'DEFAULT_DIRECTION': 2, 'SPEED_FIELD': '', 'DEFAULT_SPEED': 5, 'TOLERANCE': 0,
                'OUTPUT': '/Users/macbook/Desktop/hamza/data/od/odMatrix.csv'}
    processing.run("qneat3:OdMatrixFromLayersAsTable", parameters)


######       Maps predicted centroids onto roadnetwork      #######
def shortest_path():
    parameters = { 'FIELD' : 'rand_point', 
        'HUBS' : 'delimitedtext://file:///Users/macbook/Desktop/hamza/data/random_points_constrained.csv?type=csv&maxFields=10000&detectTypes=yes&xField=xcoord&yField=ycoord&crs=EPSG:4326&spatialIndex=no&subsetIndex=no&watchFile=no', 
        'INPUT' : 'delimitedtext://file:///Users/macbook/Desktop/hamza/data/od/centroids_od.csv?type=csv&maxFields=10000&detectTypes=yes&xField=Longitude&yField=Latitude&crs=EPSG:4326&spatialIndex=no&subsetIndex=no&watchFile=no', 
        'OUTPUT' : '/Users/macbook/Desktop/hamza/data/shortestpath.csv', 
        'UNIT' : 0 }
    processing.run("qgis:distancetonearesthublinetohub", parameters)
    
    shifts = pd.read_csv('/Users/macbook/Desktop/hamza/data/shortestpath.csv')
    random_set = pd.read_csv('/Users/macbook/Desktop/hamza/data/random_points_constrained.csv')

    mapped_points = []
    for each in shifts.values:
        longMap = float(random_set.loc[random_set['rand_point'] == int(each[3])]['xcoord'])
        latMap = float(random_set.loc[random_set['rand_point'] == int(each[3])]['ycoord'])
        mapped_points.append([longMap, latMap])

    return mapped_points

######       Calculates cost      #######
def calculate_OD_cost(X, centroids, cluster):
    sum = 0
    for i, val in enumerate(X):
        sum += (centroids[int(cluster[i]), 0]-val[0])**2 +(centroids[int(cluster[i]), 1]-val[1])**2
    return sum

######       K-means iteration      #######
def kmeans_OD(X, builtup_n, centroids_k):
    cluster = min_distance(builtup_n, centroids_k)
    new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
    centroids = new_centroids

    return centroids, cluster


######       Main function executing k-means clustering      #######
def kmean_OD(X, builtup_n, centroids_k):
    defaultCentroid = pd.read_csv('/Users/macbook/Desktop/hamza/data/centroids/centroids.csv').iloc[:, [0,1]].values.tolist()
    for j in range(1, 11):
        ODMatrixCalculation()
        centroids_OD, cluster_OD = kmeans_OD(X, builtup_n, centroids_k)
        save_csv('/Users/macbook/Desktop/hamza/data/od/centroids_od.csv', centroids_OD, ['Longitude', 'Latitude', 'index'])
        new_points = shortest_path()

        if len(new_points) < centroids_k:
            new_points = compensate(centroids_k, cluster_OD, new_points, defaultCentroid)

        save_csv('/Users/macbook/Desktop/hamza/data/centroids/centroids.csv', new_points, ['Longitude', 'Latitude', 'index'])

        pathOD = "/Users/macbook/Desktop/hamza/data/centroids/od_cost_{}.csv".format(j)
        get_od(pathOD)

    return centroids_OD, cluster_OD


data = pd.read_csv('/Users/macbook/Desktop/hamza/data/Built_Up_Gulberg_Unique.csv')
x = data.iloc[:,0:2]
X = x.values
clusters = np.zeros(X.shape[0])
centroids = x.sample(n=123).values
save_csv('/Users/macbook/Desktop/hamza/data/centroids.csv', centroids, ['Longitude', 'Latitude', 'index'])

#####     Starting     ###### 
start = time.time()
builtup_n = len(data)
centroids_k = len(centroids)
kmean_OD(X, builtup_n, centroids_k)    ## K-means Clustering
end1 = time.time()
print("Time Taken: ", end1-start)

