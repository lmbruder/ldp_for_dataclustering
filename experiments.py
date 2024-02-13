import math
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
from pure_ldp.frequency_oracles.rappor import RAPPORClient, RAPPORServer
import random
import numpy as np
from itertools import permutations
import json 
import statistics
import itertools

# ignore convergence warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def discretize(df, n_bins):
    global bins_age
    global bins_height
    
    # age
    bin_age_width = (70 - 10) / n_bins
    # Generate the bin edges
    bins_age = [10 + i * bin_age_width for i in range(n_bins + 1)]
    
    # height
    bin_heigth_width = (2.0 - 1.4) / n_bins
    # Generate the bin edges
    bins_height = [1.4 + i * bin_heigth_width for i in range(n_bins + 1)]

    discretized_data_age = pd.cut(df['Age'], bins=bins_age, labels=list(range(n_bins)))
    discretized_data_height = pd.cut(df['Height'], bins=bins_height, labels=list(range(n_bins)))

    # Convert the discretized data back to a DataFrame
    df_discretized = pd.DataFrame({
        'Age': discretized_data_age,
        'Height': discretized_data_height
    })

    # cast to int
    df_discretized['Age'] = df_discretized['Age'].astype(int)
    df_discretized['Height'] = df_discretized['Height'].astype(int)
    
    # cell name = value of age bin and value of height bin
    df_discretized['cell'] = df_discretized['Age'].astype(str) + df_discretized['Height'].astype(str)
    df_discretized['cell'] = df_discretized['cell'].astype(int) + 1

    cell_names = sorted(df_discretized['cell'].unique().tolist())
    return df_discretized, cell_names

def rappor_estimation(rappor_params, cell_names, cell_count, df_discretized):
    rappor_estimates = []
    for _ in range(10):
        epsilon = rappor_params['epsilon']
        f = round(1 / (0.5 * math.exp(epsilon / 2) + 0.5), 2)

        server_rappor = RAPPORServer(f=f, num_of_cohorts=64, m=rappor_params['bloom_filter'], k=rappor_params['hash_func_amount'], d=max(cell_names))
        client_rappor = RAPPORClient(f=f, num_of_cohorts=64, m=rappor_params['bloom_filter'], hash_funcs=server_rappor.get_hash_funcs())

        rappor = list(map(client_rappor.privatise, df_discretized['cell'].tolist()))

        server_rappor.aggregate_all(rappor)

        rappor_estimates.append(server_rappor.estimate_all(cell_names, suppress_warnings=True))
    mean_rappor_estimates = np.mean(rappor_estimates, axis=0).tolist()
    return mean_rappor_estimates
    
def generate_rappor_datapoints(mean_rappor_estimates, cell_names):
    # generate new DataFrame with columns Age and Height
    cols = ['Age', 'Height']
    new_coordinates_random_rappor = []
    for index, cell in enumerate(cell_names):
        cell = cell - 1    
        # make sure cell names are properly formatted by adding leading 0 if necessary that got lost when transforming to int for RAPPOR earlier
        if cell < 10: 
            cell = "0" + str(cell)
        else:
            cell = str(cell)

        for _ in range(round(mean_rappor_estimates[index])):
            # get random point in cell
            age = random.uniform(bins_age[int(cell[0])], bins_age[int(cell[0]) + 1])	
            height = random.uniform(bins_height[int(cell[1])], bins_height[int(cell[1]) + 1])	

            # add coordinates as many times as RAPPOR estimates the cell population
            new_coordinates_random_rappor.extend([[age, height]])

    return pd.DataFrame(new_coordinates_random_rappor, columns=cols)

def match_labels(n_clusters, original_centroids, rappor_centroids):
    # all possible pairings
    all_pairings = permutations(range(n_clusters))
    best_pairing = None
    min_total_distance = float('inf')
    
    for pairing in all_pairings:
        current_distance = sum(np.sqrt(np.sum((rappor_centroids[j] - original_centroids[i])**2)) for i, j in enumerate(pairing))
        
        # Update the best pairing if the current pairing has a lower total distance
        if current_distance < min_total_distance:
            min_total_distance = current_distance
            best_pairing = pairing
    mapping = {}
    for i in range(n_clusters):
        mapping[best_pairing[i]] = i
    return mapping

def ldp_for_clustering(n_clusters, cell_count, frac_data, rappor_params):
    # get data from csv and put into dataframe
    df_all = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

    df_all = df_all.drop(df_all.loc[:, 'Weight':'NObeyesdad'].columns, axis=1)
    df_all = df_all.drop(['Gender'], axis=1)

    df_all = df_all.sample(frac=frac_data)

    df = df_all.sample(frac=0.8)
    non_sampled = df_all.drop(df.index)

    # determine bin counts based on cell_count
    n_bins = int(np.sqrt(cell_count))
    # discretize values
    df_discretized, cell_names = discretize(df, n_bins)
    
    mean_rappor_estimates = rappor_estimation(rappor_params, cell_names, cell_count, df_discretized)
    rappor_df = generate_rappor_datapoints(mean_rappor_estimates, cell_names)
   
    # fit model based on 80% of the data
    kmeans_rappor= KMeans(n_init='auto',n_clusters=n_clusters).fit(rappor_df)
    kmeans_original = KMeans(n_init='auto', n_clusters=n_clusters).fit(df)
    
    # predict based on centroids
    original_predict = kmeans_original.predict(non_sampled)
    rappor_predict = kmeans_rappor.predict(non_sampled)
    
    rappor_centroids = kmeans_rappor.cluster_centers_
    original_centroids = kmeans_original.cluster_centers_

    mapping = match_labels(n_clusters, original_centroids, rappor_centroids)

    # Apply the mapping to the original array
    rappor_predict = np.array([mapping[value] for value in rappor_predict])

    count_equal = 0
    for i in range(len(original_predict)):
        if original_predict[i] == rappor_predict[i]:
            count_equal += 1
    
    equal_labels = round(count_equal / len(original_predict),2)
    return equal_labels

# set fixed parameters
bloom_filter = 16
hash_func_amount = 2
#epsilon = 2
n_clusters = 5

# fix one parameter
frac = 1
cell_count = 25

# actual comparisons
possible_fracs = list(map(lambda x: x / 10, range(1, 11)))
possible_cell_counts = list(map(lambda x: x**2, range(2, 8)))
epsilon_range = [1,2,4,8]

current = possible_fracs

equal_labels = {}
for epsilon in epsilon_range:
    equal_labels[epsilon] = {}
    equal_labels[epsilon]['mean'] = []
    equal_labels[epsilon]['median'] = []
    equal_labels[epsilon]['all_values'] = {}

    for i in current:
        print('Current iteration: ', i)
        values_current_run = []
        for j in range(50):
            print(j)
            rappor_params = {'epsilon': epsilon, 'bloom_filter': bloom_filter, 'hash_func_amount': hash_func_amount}
            output = ldp_for_clustering(n_clusters = n_clusters, cell_count = cell_count, frac_data = i, rappor_params = rappor_params)
            values_current_run.append(output)

        equal_labels[epsilon]['all_values'][i] = values_current_run
        mean = statistics.mean(values_current_run)
        median = statistics.median(values_current_run)
        equal_labels[epsilon]['median'].append(median)
        equal_labels[epsilon]['mean'].append(mean)

    print(f"Epsilon {epsilon}: ", equal_labels[epsilon])

# write to file for later analysis
with open('results/big_data_set.json', 'w') as file: 
    file.write(json.dumps(equal_labels))

plt.figure(dpi=1200)
title_str = f"Bloom Filter: {bloom_filter}, Hash Function Amount: {hash_func_amount}, N Clusters: {n_clusters}, Cell count: 25"
plt.title(title_str,fontsize=9)

marker = itertools.cycle(('o', 'v', 'P', 'D')) 
for epsilon in epsilon_range:
    plt.scatter(current, equal_labels[epsilon]['median'], marker = next(marker), label=f"Epsilon {epsilon}")

# plotting the results
plt.xlabel("Fraction of data")
plt.ylabel("Equal labels")
plt.ylim(0, 1)

plt.legend()

#plt.show()
plt.savefig('results/fig.png')
