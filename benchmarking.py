'''
benchmarking.py
purpose: This script takes various different algorithms that can define
         similarity metrics between different CDR3s, and benchmarks their
         performance in terms of their ability to recognise CDR3s that are
         functionally similar (i.e. they respond to the same epitope).
author: Yuta Nagano
version: 1.0.0
'''


# Import required packages and also the various locally implemented algorithms
import pandas as pd
from scipy.stats import spearmanr
import source.benchmarking_algos as algos
from tqdm import tqdm


# Organise the algorithms into a list
algo_list = (
    algos.NegativeLevenshtein,
    algos.AtchleyCs
)


# Helper functions
def get_similarities_from_algo(similarity_func, test_set) -> list:
    '''
    Given a callable similarity function, iterate through all rows in the
    test dataset and call the similarity function on each of the CDR3 pairs.
    Save the result in a list and return that list.
    '''
    # Instantiate list to save results
    similarities = []

    # Iterate through every row and save similarity between the cdr3 pair
    for i, row in tqdm(test_set.iterrows(), total=len(test_set)):
        cdr3_a = row['CDR3_A']
        cdr3_b = row['CDR3_B']
        similarity = similarity_func(cdr3_a, cdr3_b)
        # Save the result in the list
        similarities.append(similarity)

    # Return the list with the results
    return similarities


# Main code
if __name__ == '__main__':
    # Initialise a list to store the benchmarking data
    results = []

    # Load the paired test dataset
    print('Loading dataset...')
    test_set = pd.read_csv('data/labelled_test_paired.csv')
    
    # Create a list representing the paired data's ground truth labels
    ground_truth = test_set['Matched'].map(lambda x: int(x == 'T')).tolist()

    c, p = spearmanr(ground_truth, ground_truth)
    results.append(['Ground Truth', c, p])

    # For each algorithm calculate its performance metric
    for algo in algo_list:
        print(f'Evaluating algorithm: {algo.name}...')
        # Use algorithm to calculate similarity metric between each CDR3 pair
        algo_output = get_similarities_from_algo(algo.similarity_func, test_set)
        # Compare the calculated scores to the ground truth via spearman
        correlation, p_value = spearmanr(algo_output, ground_truth)
        # Add result entry to results dictionary
        results.append([algo.name, correlation, p_value])
    
    # Convert the results dictionary into a dataframe
    print('Collating results...')
    results = pd.DataFrame(results, columns=['Algorithm','Spearman','p'])

    # Save the resulting dataframe as a csv file
    print('Saving results...')
    results.to_csv('benchmark_results/spearman.csv',index=False)
    print('Done!')