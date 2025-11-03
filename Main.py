import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-base-v2")

anchors = [
    # 🚨 Immediate personal emergencies
    "family emergency",
    "someone is hurt or in danger",
    "hospital emergency",
    "someone has died",

    # ⚠️ Urgent logistics / time critical
    "you need to be somewhere soon",
    "urgent reminder",
    "important appointment or deadline",
    "urgent school or work message",

    # ❤️ Emotional / relationship importance
    "serious personal conversation",
    "someone is upset or crying",
    "breakup or relationship crisis",
    "family needs emotional support",

    # ✅ Everyday important messages
    "important reminder",
    "coordination about plans",
    "question needing timely reply",
    "work or school information",
    "a teacher or coach contacting you",

    # 🙂 Friendly messages
    "friend is reaching out to talk",
    "someone checking in",
    "casual greeting or chat",

    # 💬 Low-priority social
    "jokes or memes",
    "casual conversation",
    "fun or informal message",

    # 🛑 Spam / junk
    "advertisement or promotion",
    "unknown sender spam",
    "scam or phishing text"
]



notifications = [
    "Meeting starts in 10 minutes",
    "Your pizza delivery is on the way",
    "20% off shoes today!",
    "Reminder: homework due tomorrow",
    "Your mom died",
    "The world is ending please come now or the holocaust will happen again",
    "yo whats good bro"
]

# --- Encode (normalized so dot = cosine similarity)
anchor_embeddings = model.encode(anchors, normalize_embeddings=True)
notif_embeddings = model.encode(notifications, normalize_embeddings=True)

# --- Dot product between notifications and anchors
# Shape: (num_notifications, num_anchors)
similarities = np.dot(notif_embeddings, anchor_embeddings.T)

# --- Weighted average per notification
weights = np.array([
    # Urgency
    0.80, 0.75, 0.90, 0.70,

    # Catastrophic
    1.00, 1.00, 1.00, 0.95,

    # Personal tragedy / death
    1.00, 1.00, 0.98, 0.9,

    # Health & safety
    0.95, 0.92, 0.90,

    # Family priority
    0.88, 0.85,

    # Financial / work
    0.82, 0.80, 0.78,

    # General alerts
    0.60, 0.55, 0.9,

    # Low-priority / casual
    0.10, 0.08, 0.05
], dtype=np.float32)

weights = weights / weights.sum()

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