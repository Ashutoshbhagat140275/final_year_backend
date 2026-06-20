from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Auth
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # MongoDB
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "rag_audio_db"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # Ollama / RAG
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"

    # Text embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    query_cache_ttl_seconds: int = 3600
    query_cache_max_per_user: int = 20
    query_cache_similarity_threshold: float = 0.95

    # Audio
    audio_upload_dir: str = "./uploads"
    max_audio_size_mb: int = 50

    # Whisper
    whisper_model_size: str = "small"
    whisper_language: str = "en"

    # Speaker pipeline
    enable_speaker_aware_processing: bool = True

    # Emotion model: prefer the fine-tuned backbone+head if its artifacts exist
    use_finetuned_emotion_model: bool = True

    # User-head storage backend
    use_mongodb_storage: bool = False
    dual_save_mode: bool = False
    mongodb_storage_compression: str = "gzip"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
