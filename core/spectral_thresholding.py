import cv2
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh


def spectral_thresholding(img, k=3, sigma=20):
     
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Histogram (0-255)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist = hist + 1e-6  # avoid zero issues

    # 2. Build intensity axis
    x = np.arange(256).reshape(-1, 1)

    # 3. Similarity matrix (Gaussian kernel)
    dist2 = (x - x.T) ** 2
    W = np.exp(-dist2 / (2 * sigma ** 2))

    # weight by histogram importance
    W = W * np.outer(hist, hist)

    # 4. Graph Laplacian
    L = csgraph.laplacian(W, normed=True)

    # 5. Eigen decomposition
    eigvals, eigvecs = eigh(L)

    # 6. Take first k eigenvectors (skip trivial one if needed)
    embedding = eigvecs[:, 1:k+1]

    # 7. Simple k-means clustering
    centroids = embedding[np.random.choice(256, k, replace=False)]

    for _ in range(20):
        d = np.linalg.norm(embedding[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(d, axis=1)

        for i in range(k):
            if np.any(labels == i):
                centroids[i] = embedding[labels == i].mean(axis=0)

    # 8. Build segmented image
    output = np.zeros_like(img)

    for i in range(k):
        output[np.isin(img, np.where(labels == i)[0])] = int(255 * i / (k - 1))

    return output
