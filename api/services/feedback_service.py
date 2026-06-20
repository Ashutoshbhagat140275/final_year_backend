"""
Feedback service — store an emotion correction, increment the user's feedback count,
and trigger per-user training at N>=20 AND N%10==0 (fires at 20, 30, 40, ...).

Owner-safety gate: feedback is only accepted for sessions where the owner's voice was
confidently verified (personalization_trainable == True), so other people's voices
can't corrupt the personal model.
"""
import logging

from api.feature_config import EMOTION_LABELS, MIN_FEEDBACK_FOR_TRAINING

logger = logging.getLogger(__name__)


def should_trigger_training(n: int) -> bool:
    return n >= MIN_FEEDBACK_FOR_TRAINING and n % 10 == 0


def submit_feedback(db, user_id: str, session_id: str, corrected_emotion: str,
                    task_queue=None) -> dict:
    from api.models.audio_session import AudioSession
    from api.models.emotion_analysis import EmotionAnalysis
    from api.models.user import User
    from api.models.user_feedback import UserFeedback

    if corrected_emotion not in EMOTION_LABELS:
        raise ValueError(f"Invalid emotion '{corrected_emotion}'. Allowed: {EMOTION_LABELS}")

    session = AudioSession.get_collection(db).find_one({"session_id": session_id})
    if not session:
        raise ValueError("Session not found")
    if session.get("user_id") != user_id:
        raise PermissionError("You do not own this session")
    if not session.get("personalization_trainable", True):
        raise ValueError("This session is not eligible for personalization "
                         "(owner voice could not be confidently verified).")

    ea = EmotionAnalysis.get_collection(db).find_one({"session_id": session_id})
    if not ea or not ea.get("mfcc_features"):
        raise ValueError("No stored embedding for this session")

    predicted = (session.get("emotion_data") or {}).get("emotion", "neutral")
    UserFeedback.get_collection(db).insert_one(UserFeedback(
        user_id=user_id, session_id=session_id, embedding=ea["mfcc_features"],
        predicted_emotion=predicted, corrected_emotion=corrected_emotion,
    ).to_dict())

    new_count = User.increment_feedback_count(db, user_id)

    triggered = should_trigger_training(new_count)
    job_id = None
    if triggered and task_queue is not None:
        from api.services.training_job_tracker import create_training_job
        from training.train_user_head import train_user_head_async

        job_id = create_training_job(db, user_id)
        task_queue.enqueue(train_user_head_async, user_id, job_id=job_id)

    msg = f"Feedback recorded (corrected to '{corrected_emotion}')."
    if triggered:
        msg += " Personal model training started."
    return {
        "status": "success",
        "feedback_count": new_count,
        "training_triggered": triggered,
        "training_job_id": job_id,
        "message": msg,
    }
