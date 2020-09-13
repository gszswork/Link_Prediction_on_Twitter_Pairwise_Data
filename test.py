
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

x = [1,2,3]
x = np.array(x).reshape(1, -1)
y = [1,2,4]
y = np.array(y).reshape(1, -1)

cosine_similarity_result = cosine_similarity(x, y)
print(cosine_similarity_result)