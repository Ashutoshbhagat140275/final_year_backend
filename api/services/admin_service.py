"""Admin tooling — list and prune per-user model artifacts (file backend)."""
from datetime import datetime, timedelta, timezone

from api.models.user import User
from api.services.user_head_storage import USER_HEADS_DIR


def list_user_models(db) -> dict:
    models = []
    if USER_HEADS_DIR.exists():
        for path in USER_HEADS_DIR.glob("*.pt"):
            user_id = path.stem
            user = User.get_collection(db).find_one({"user_id": user_id}) if db is not None else None
            st = path.stat()
            models.append({
                "user_id": user_id,
                "feedback_count": (user or {}).get("feedback_count", 0),
                "model_size_kb": round(st.st_size / 1024, 1),
                "last_trained": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            })
    return {"total_count": len(models), "models": models}


def cleanup_user_models(db, min_days_inactive: int | None = None,
                        max_feedback_count: int | None = None,
                        user_ids: list[str] | None = None) -> dict:
    deleted, freed_kb = [], 0.0
    if not USER_HEADS_DIR.exists():
        return {"deleted_count": 0, "deleted_user_ids": [], "total_space_freed_kb": 0.0}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=min_days_inactive)
              if min_days_inactive is not None else None)

    for path in list(USER_HEADS_DIR.glob("*.pt")):
        uid = path.stem
        st = path.stat()
        should = False
        if user_ids and uid in user_ids:
            should = True
        if cutoff and datetime.fromtimestamp(st.st_mtime, tz=timezone.utc) < cutoff:
            should = True
        if max_feedback_count is not None and db is not None:
            user = User.get_collection(db).find_one({"user_id": uid})
            if (user or {}).get("feedback_count", 0) <= max_feedback_count:
                should = True
        if should:
            freed_kb += st.st_size / 1024
            path.unlink(missing_ok=True)
            deleted.append(uid)

    return {"deleted_count": len(deleted), "deleted_user_ids": deleted,
            "total_space_freed_kb": round(freed_kb, 1)}
