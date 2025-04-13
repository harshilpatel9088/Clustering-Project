from sklearn.cluster import KMeans
from logger import get_logger

logger = get_logger(__name__)

def train_kmeans(data, k):
    logger.info(f"Training KMeans with k={k}")
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(data)
    logger.info("KMeans training complete")
    return model, model.labels_, model.inertia_