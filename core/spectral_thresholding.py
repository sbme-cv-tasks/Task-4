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

    # 8. Overlap fix — re-assign each intensity to nearest sorted cluster mean
    sorted_cluster_means = np.sort(cluster_means)
    final_labels = np.array([
        np.argmin(np.abs(i - sorted_cluster_means))
        for i in range(256)
    ])

    # 9. Extract boundaries by finding first transition point between classes
    boundaries = []
    for i in range(k - 1):
        for intensity in range(255):
            if final_labels[intensity] == i and final_labels[intensity + 1] == i + 1:
                boundaries.append(intensity)
                break

    # 10. Build output image
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = int(255 * final_labels[i] / (k - 1))
    output = lut[img]

    print(f"k={k}, sorted_means={sorted_cluster_means.round(1)}, boundaries={boundaries}")
    return output, boundaries
    