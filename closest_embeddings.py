from sklearn.metrics.pairwise import cosine_similarity

class ClosestEmbeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def find_closest(self, query_embedding, top_k=5):
        # Compute cosine similarity between the query embedding and all stored embeddings
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get the indices of the top_k closest embeddings
        closest_indices = similarities.argsort()[-top_k:][::-1]

        
        return closest_indices, similarities[closest_indices]


    