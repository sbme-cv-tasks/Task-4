import cv2
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh

def spectral_thresholding(img, k=3, sigma=20, random_seed=42):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Compute histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist = hist + 1e-6

    # 2. Build Gaussian similarity matrix weighted by histogram
    x = np.arange(256).reshape(-1, 1)
    dist2 = (x - x.T) ** 2
    W = np.exp(-dist2 / (2 * sigma ** 2))
    W = W * np.outer(hist, hist)

    # 3. Compute normalized graph Laplacian
    L = csgraph.laplacian(W, normed=True)

    # 4. Eigen decomposition — k smallest non-trivial eigenvectors
    eigvals, eigvecs = eigh(L)
    embedding = eigvecs[:, 1:k+1]  # shape (256, k)

    # 5. K-means++ on the spectral embedding
    rng = np.random.default_rng(random_seed)
    first_idx = rng.integers(0, 256)
    centroids = [embedding[first_idx]]

    for _ in range(k - 1):
        dists = np.min(
            [np.linalg.norm(embedding - c, axis=1) ** 2 for c in centroids],
            axis=0
        )
        probs = dists / dists.sum()
        next_idx = rng.choice(256, p=probs)
        centroids.append(embedding[next_idx])
    centroids = np.array(centroids)

    # 6. K-means iterations on embedding
    labels = None
    for _ in range(50):
        d = np.linalg.norm(embedding[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(d, axis=1)
        new_centroids = np.array([
            embedding[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # 7. Sort clusters by mean intensity (dark → bright)
    cluster_means = np.array([
        np.mean(np.where(labels == i)[0]) for i in range(k)
    ])
    sorted_order = np.argsort(cluster_means)
    label_map = np.zeros(k, dtype=int)
    for new_label, old_label in enumerate(sorted_order):
        label_map[old_label] = new_label
    labels = label_map[labels]

    # 8. Build output image directly from labels (no overlap fix)
    output = np.zeros_like(img, dtype=np.uint8)
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = int(255 * labels[i] / (k - 1))
    output = lut[img]

    # Extract boundaries for reference
    boundaries = []
    for i in range(k - 1):
        max_current = int(np.max(np.where(labels == i)[0]))
        min_next    = int(np.min(np.where(labels == i + 1)[0]))
        boundaries.append((max_current + min_next) // 2)

    return output, boundaries