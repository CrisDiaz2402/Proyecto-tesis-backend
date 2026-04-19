import threading
from qdrant_client import QdrantClient
import redis
from sentence_transformers import SentenceTransformer
from app.core.config import QDRANT_URL, REDIS_URL, EMBED_MODEL_LOCAL


class _SingletonMeta(type):
    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class QdrantClientSingleton(metaclass=_SingletonMeta):
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)


class RedisClientSingleton(metaclass=_SingletonMeta):
    def __init__(self):
        self.client = redis.from_url(REDIS_URL, decode_responses=True)


class EmbedModelSingleton(metaclass=_SingletonMeta):
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_LOCAL, device="cpu")
