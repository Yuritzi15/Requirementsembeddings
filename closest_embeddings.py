from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
class ClosestEmbeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def label_distance(self, similarities, labels,bins):
        labeled = []
        for sim in similarities:
            found = False 
            for i in range(len(labels)):
                #verify if sim is between the bins
                if bins[i] <= sim <= bins[i+1]:
                    labeled.append(labels[i])
                    found = True
                    break
            if not found:
                #out of range if sim is less than the first bin or greater than the last bin
                labeled.append('Out of range')
        return labeled
    


    def find_closest_cosine(self, query_embedding,labels,bins):
        # Compute cosine similarity between the query embedding and all stored embeddings
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        #compute labeled similarities
        labeled_similarities = self.label_distance(similarities, labels, bins)
        print(f"Similarities: {similarities}")
        # print(f"Labeled Similarities: {labeled_similarities}")
        return labeled_similarities, similarities
    
    def euclidean_dist(self, query_embedding):
        # Compute euclidean distance between the query embedding and all stored embeddings
        distances = euclidean_distances([query_embedding], self.embeddings)[0]
        return distances

    def histograms(self, distances, labels,system_name):
        plt.figure(figsize=(10, 6))
        n_bins = len(labels)
        plt.hist(distances, bins=n_bins, color="#330CDD", edgecolor='black')

        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.title("Distribution of distances for system: " + system_name)
        dir="histogramsbertcls12bins"
        if not os.path.exists(dir):
            os.makedirs(dir) 

        safe_system_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in system_name])
        safe_system_name = safe_system_name[:100]
        name=f"euclidean_distances_histogram_{safe_system_name}.png"
        save_path = os.path.join(dir, name)
        plt.savefig(save_path)
        plt.close()
        # plt.show()
        # 
    def histogramsglobal(self, distances, labels,system_name, min_distance, max_distance):
        plt.figure(figsize=(10, 6))
        n_bins = len(labels)
        plt.hist(distances, bins=n_bins, range=(min_distance, max_distance), color="#0CDD97", edgecolor='black')

        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.title("Distribution of distances for system: " + system_name)
        dir="histogramsbertmean12binsminmaxgeneral"
        if not os.path.exists(dir):
            os.makedirs(dir) 

        safe_system_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in system_name])
        safe_system_name = safe_system_name[:100]
        name=f"euclidean_distances_histogram_{safe_system_name}.png"
        save_path = os.path.join(dir, name)
        plt.savefig(save_path)
        plt.close()
        # plt.show()   

    #histogram of distances to see the distribution of all the distances without labels
    def histogram_distances(self,all_distances):
        plt.figure(figsize=(10, 6))
        #histogram and curve of the distribution of distances
        sns.histplot(all_distances, kde=True, color="#780CDD", stat="density", edgecolor='black')

        #density curve
        sns.kdeplot(all_distances, color="#330CDD", linewidth=2,fill=True)

        #plt.hist(all_distances, bins=len(all_distances), color="#330CDD", edgecolor='black')
        plt.xlabel("Distance")
        plt.ylabel("Density")
        plt.title("Distribution of all the distances")
        dir="histogram"
        if not os.path.exists(dir):
            os.makedirs(dir) 
            
        # safe_system_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in system_name])
        # safe_system_name = safe_system_name[:100]
        name=f"euclidean_distances_density_all.png"
        save_path = os.path.join(dir, name)
        plt.savefig(save_path)
        plt.close()
    # plt.show()


    