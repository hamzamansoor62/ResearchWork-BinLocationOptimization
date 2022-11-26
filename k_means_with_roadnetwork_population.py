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
    vo = pd.read_csv('/Users/macbook/Desktop/hamza/data/od/odMatrix.csv').values.tolist()
    dim = []
    for i in range(builtup_n):
        row = vo[i*centroids_k:(i+1)*centroids_k]
        dim.append(row)
    
    dis = []
    row = []
    for ea in dim:
        for it in ea:
            if math.isnan(it[3]):
                it[3] = 100000
            row.append(it[3])
        dis.append(row)
        row = []

    minIdx = []
    for e in dis:
        minIdx.append(np.argmin(e))

    return minIdx

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

######       Updating centroids with population weights        #######
def PopulationCluster(cluster, pop):
    clusterList = []
    df = pd.DataFrame(pop)
    grouped_df = df.groupby(cluster)
    for key, item in grouped_df:
        clusterList.append([key, grouped_df.get_group(key).values.tolist()])

    centroids = []
    for each in clusterList:
        Longitude = 0
        Latitude = 0
        population = 0
        data = []
        for item in each[1]:
            population += item[2]
        for item in each[1]:
            item[2] = item[2]/population
        for item in each[1]:
            Longitude += (item[0]*item[2])
            Latitude += (item[1]*item[2])
            data.append([item[0], item[1], item[2]])
        centroids.append([Longitude, Latitude])

    return centroids

######       Calculating origin-destination matrix using QGIS plugin        #######
def ODMatrixCalculation():
    parameters = {'INPUT': '/Users/macbook/Desktop/hamza/data/RN/Lahore_RN_Gulberg.shp',
                'FROM_POINT_LAYER': 'delimitedtext://file:///Users/macbook/Desktop/hamza/data/Built_Up_Gulberg_Unique.csv?type=csv&maxFields=10000&detectTypes=yes&xField=Longitude&yField=Latitude&crs=EPSG:4326&spatialIndex=no&subsetIndex=no&watchFile=no',
                'FROM_ID_FIELD': 'index',
                'TO_POINT_LAYER': 'delimitedtext://file:///Users/macbook/Desktop/hamza/centroids.csv?type=csv&maxFields=10000&detectTypes=yes&xField=longitude&yField=latitude&crs=EPSG:4326&spatialIndex=no&subsetIndex=no&watchFile=no',
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
        sum += np.sqrt((centroids[int(cluster[i]), 0]-val[0])**2 +(centroids[int(cluster[i]), 1]-val[1])**2)
    return sum

######       K-means iteration      #######
def kmeans_OD(X, pop, builtup_n, centroids_k):
    cluster = min_distance(builtup_n, centroids_k)
    new_centroids = PopulationCluster(cluster, pop)
    centroids = new_centroids
    
    return centroids, cluster

######       Main function executing k-means clustering      #######
def kmean_OD(X, pop, builtup_n, centroids_k):
    cost_list_OD = []
    defaultCentroid = pd.read_csv('/Users/macbook/Desktop/centroids.csv').iloc[:, [0,1]].values.tolist()
    for j in range(1, 11):
        ODMatrixCalculation()
        centroids_OD, cluster_OD = kmeans_OD(X, pop, builtup_n, centroids_k)

        if len(centroids_OD) < centroids_k:
            updated_points = compensate(centroids_k, cluster_OD, centroids_OD, defaultCentroid)
        else:
            updated_points = centroids_OD
        
        save_csv('/Users/macbook/Desktop/hamza/data/od/centroids_od.csv', updated_points, ['Longitude', 'Latitude', 'index'])
        new_points = shortest_path()
        save_csv('/Users/macbook/Desktop/hamza/centroids.csv', new_points, ['Longitude', 'Latitude', 'index'])

        up_Centroids = pd.DataFrame(new_points).values
        cost = calculate_OD_cost(X, up_Centroids, cluster_OD)
        cost_list_OD.append(cost)

        pathOD = "/Users/macbook/Desktop/hamza/data/centroids/od_cost_{}.csv".format(j)
        get_od(pathOD)

    return centroids_OD, cluster_OD


data = pd.read_csv('/Users/macbook/Desktop/hamza/LUMS/SolidWasteManagementPracticesForUrbanDevelopment/codes/Unique_Build_Obs.csv')
popData = pd.read_csv('/Users/macbook/Desktop/hamza/data/pop_data.csv')
pop = popData.iloc[:,[0,1,5]]
x = pop.iloc[:,0:2]
X = x.values
clusters = np.zeros(X.shape[0])
centroids = x.sample(n=123).values
save_csv('/Users/macbook/Desktop/centroids.csv', centroids, ['Longitude', 'Latitude', 'index'])
save_csv('/Users/macbook/Desktop/hamza/centroids.csv', centroids, ['Longitude', 'Latitude', 'index'])

#####     Starting     ###### 
start = time.time()
builtup_n = len(data)
centroids_k = len(centroids)
kmean_OD(X, pop, builtup_n, centroids_k)  ##  K-means Clustering
end1 = time.time()
print("Time Taken: ", end1-start)
