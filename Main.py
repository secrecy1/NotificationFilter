import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-base-v2")

anchors = [
    "urgent message",
    "important reminder",
    "requires immediate attention",
    "something you must act on soon",
    "The world is gonna end",
    "died"
]

notifications = [
    "Meeting starts in 10 minutes",
    "Your pizza delivery is on the way",
    "20% off shoes today!",
    "Reminder: homework due tomorrow",
    "Your mom died",
    "The world is ending please come now or the holocaust will happen again"
]

# --- Encode (normalized so dot = cosine similarity)
anchor_embeddings = model.encode(anchors, normalize_embeddings=True)
notif_embeddings = model.encode(notifications, normalize_embeddings=True)

# --- Dot product between notifications and anchors
# Shape: (num_notifications, num_anchors)
similarities = np.dot(notif_embeddings, anchor_embeddings.T)

# --- Weighted average per notification
weights = np.array([0.8, 0.7, 0.9, 0.6, 1.0, 1.0], dtype=np.float32)
weights = weights / weights.sum()  # normalize to sum to 1

# Compute weighted sum of similarities along anchors axis
weighted_scores = np.dot(similarities, weights)  # shape: (num_notifications,)
# --- Combine notifications + scores
scored = [(n, float(s)) for n, s in zip(notifications, weighted_scores)]

# --- Sort by importance (descending = most important first)
# for ascending order (least important → most important), remove the minus
scored.sort(key=lambda x: -x[1])

# --- Print sorted results
print("Sorted notifications by importance:\n")
for notif, score in scored:
    print(f"{notif:65s} importance: {score:.3f}")