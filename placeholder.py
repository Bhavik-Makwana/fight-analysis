"""Anonymised stand-in frames: the tracked skeletons drawn as mannequins on a drawn ring.

Used with --placeholder (overlay render, labelling sheets, review clips) to produce
screenshots and demos without showing real footage or real people.
"""
import cv2
import numpy as np

# COCO limb pairs, and which keypoints make up the head
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
HEAD = [0, 1, 2, 3, 4]


def ring_background(w, h):
    """A simple 3/4 elevated ringside view: canvas floor, three ropes, two posts."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (42, 40, 38)                                        # dark hall
    floor = np.array([[0, int(h * 0.28)], [w, int(h * 0.22)], [w, h], [0, h]], np.int32)
    cv2.fillPoly(img, [floor], (88, 84, 80))                     # canvas
    for k, y in enumerate((0.30, 0.36, 0.42)):                   # far ropes
        cv2.line(img, (0, int(h * (y + 0.02))), (w, int(h * y)), (30, 30, 30), 2, cv2.LINE_AA)
    for k, y in enumerate((0.62, 0.74, 0.86)):                   # near ropes
        cv2.line(img, (0, int(h * (y + 0.03))), (w, int(h * y)), (25, 25, 25), 3, cv2.LINE_AA)
    for x in (int(w * 0.16), int(w * 0.86)):                     # corner posts
        cv2.rectangle(img, (x - 4, int(h * 0.18)), (x + 4, int(h * 0.45)), (200, 200, 205), -1)
    cv2.putText(img, "placeholder - no real footage", (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.35 * w / 640, (140, 140, 140), 1, cv2.LINE_AA)
    return img


def draw_mannequin(img, k, colour, scale=1.0):
    """One figure from a (17, 3) keypoint array: thick grey limbs, tinted torso, round head."""
    k = np.asarray(k, float)
    if np.isnan(k[[5, 6, 11, 12], :2]).any():
        return
    p = lambda i: (int(k[i, 0] * scale), int(k[i, 1] * scale))
    torso_len = np.linalg.norm(k[5, :2] - k[11, :2]) * scale
    th = max(2, int(torso_len / 5))
    limb = (150, 150, 155)
    for a, b in LIMBS:
        if not np.isnan(k[[a, b], :2]).any():
            cv2.line(img, p(a), p(b), limb, th, cv2.LINE_AA)
    torso = np.array([p(5), p(6), p(12), p(11)], np.int32)
    tint = tuple(int(0.55 * c + 0.45 * 120) for c in colour)
    cv2.fillConvexPoly(img, torso, tint, cv2.LINE_AA)
    head = k[HEAD, :2]
    head = head[~np.isnan(head).any(1)]
    if len(head):
        c = head.mean(0) * scale
    else:
        c = (k[5, :2] + k[6, :2]) / 2 * scale - np.array([0, torso_len * 0.45])
    cv2.circle(img, (int(c[0]), int(c[1])), max(3, int(torso_len * 0.28)), (170, 170, 175), -1, cv2.LINE_AA)


def frame(K, t, w, h, colours, scale=1.0):
    """A full stand-in frame at index t. K: {fighter id: (F, 17, 3)} in source pixels."""
    img = ring_background(w, h)
    for n, arr in K.items():
        if t < len(arr):
            draw_mannequin(img, arr[t], colours.get(n, (200, 200, 200)), scale)
    return img
