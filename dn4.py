import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def build_similarity_matrix(features, k, sigma=None):
    features = np.asarray(features, dtype=np.float32)
    n_samples = features.shape[0]
    if n_samples == 0:
        return None, 1.0
    if n_samples == 1:
        value = 1.0 if sigma is None or sigma <= 0 else float(sigma)
        return sparse.identity(1, dtype=np.float32, format='csr'), value # type: ignore
    k = max(1, min(int(k), n_samples - 1))
    neighbor_model = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    neighbor_model.fit(features)
    distances, indices = neighbor_model.kneighbors(features)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    if sigma is None or sigma <= 0:
        base = distances[:, -1]
        base = base[base > 0]
        if base.size == 0:
            sigma = float(np.median(distances)) if distances.size else 1.0
        else:
            sigma = float(np.median(base))
        if sigma <= 0:
            sigma = 1.0
    weights = np.exp(-(distances ** 2) / (2.0 * (sigma ** 2)))
    rows = np.repeat(np.arange(n_samples), indices.shape[1])
    cols = indices.reshape(-1)
    data = weights.reshape(-1)
    adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples), dtype=np.float32)
    adjacency = adjacency.maximum(adjacency.transpose())
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    degrees[degrees == 0] = 1.0
    inv_sqrt = degrees ** -0.5
    scaling = sparse.diags(inv_sqrt.astype(np.float32))
    normalized = scaling @ adjacency @ scaling
    return normalized.tocsr(), sigma


def label_propagation(similarity_matrix, seed_labels, alpha, max_iter, tol):
    if similarity_matrix is None or seed_labels.size == 0:
        return seed_labels
    result = seed_labels.astype(np.float32)
    for _ in range(int(max_iter)):
        updated = alpha * similarity_matrix.dot(result) + (1 - alpha) * seed_labels
        delta = np.linalg.norm(updated - result)
        result = updated
        if delta < tol:
            break
    result = np.maximum(result, 0)
    sums = result.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return result / sums


def aggregate_test_distributions(test_features, support_features, support_distributions, sigma, k_neighbors):
    test_features = np.asarray(test_features, dtype=np.float32)
    support_features = np.asarray(support_features, dtype=np.float32)
    if test_features.size == 0 or support_features.size == 0:
        return np.zeros((test_features.shape[0], support_distributions.shape[1]), dtype=np.float32)
    k = max(1, min(int(k_neighbors), support_features.shape[0]))
    model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    model.fit(support_features)
    distances, indices = model.kneighbors(test_features)
    if sigma <= 0:
        sigma = 1.0
    weights = np.exp(-(distances ** 2) / (2.0 * (sigma ** 2)))
    sums = weights.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    weights = weights / sums
    num_classes = support_distributions.shape[1]
    outputs = np.zeros((test_features.shape[0], num_classes), dtype=np.float32)
    for i in range(test_features.shape[0]):
        idxs = indices[i]
        coeffs = weights[i][:, None]
        outputs[i] = np.sum(coeffs * support_distributions[idxs], axis=0)
    outputs = np.maximum(outputs, 0)
    denom = outputs.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return outputs / denom
