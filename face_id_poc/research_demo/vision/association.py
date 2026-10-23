import numpy as np
import uuid
from scipy.optimize import linear_sum_assignment


def associate_faces(heads, faces):
    """Exclusive geometric association; ambiguous matches remain unassigned."""
    if not heads or not faces:
        return {}, set()
    scores = np.full((len(heads), len(faces)), -1.0)
    for i, head in enumerate(heads):
        box = np.asarray(head['bbox'], dtype=float)
        size = box[2:] - box[:2]
        padded = np.r_[box[:2]-size*.15, box[2:]+size*.15]
        for j, face in enumerate(faces):
            fb = np.r_[face[:2], face[:2]+face[2:4]]
            center = (fb[:2]+fb[2:])/2
            if np.any(center < padded[:2]) or np.any(center > padded[2:]):
                continue
            overlap = np.prod(np.maximum(0, np.minimum(padded[2:],fb[2:])-np.maximum(padded[:2],fb[:2])))
            covered = overlap / max(float(np.prod(face[2:4])), 1)
            distance = np.linalg.norm(center-(box[:2]+box[2:])/2) / max(np.linalg.norm(size),1)
            scores[i,j] = covered-.25*distance
    rows, cols = linear_sum_assignment(-scores)
    matches, ambiguous = {}, set()
    for i,j in zip(rows,cols):
        score = scores[i,j]
        if score < .6:
            continue
        competing_heads = np.flatnonzero((scores[:,j] >= .6) & (scores[:,j] >= score-.1))
        competing_faces = np.flatnonzero((scores[i,:] >= .6) & (scores[i,:] >= score-.1))
        if len(competing_heads) > 1 or len(competing_faces) > 1:
            ambiguous.update(int(k) for k in competing_heads)
        else:
            matches[int(i)] = int(j)
    return matches, ambiguous


class TurnAwayCheck:
    def __init__(self):
        self.state = 'idle'
        self.result = None

    def start(self, track_id, now):
        self.test_id = uuid.uuid4().hex
        self.state = 'waiting_for_face_absence'
        self.track_id = track_id
        self.started = now
        self.away_start = None
        self.away_seconds = 0.0
        self.missing_start = None
        self.maximum_head_gap = 0.0
        self.frames = 0
        self.result = None

    def finish(self, state, reason, now):
        self.state = state
        self.result = {'test_id':self.test_id,'status':state, 'reason':reason, 'track_id':self.track_id,
                       'duration_seconds':round(now-self.started,2), 'face_unavailable_seconds':round(self.away_seconds,2),
                       'maximum_head_gap_seconds':round(self.maximum_head_gap,2), 'observed_frames':self.frames,
                       'scope':'Anonymous single-camera head-ID continuity; not identity or biometric accuracy proof'}

    def update(self, track_id, face_usable, head_count, ambiguous, now):
        if self.state not in ('waiting_for_face_absence','waiting_for_face_return'):
            return
        if now-self.started > 30:
            self.finish('incomplete','No completed face-away-and-return interval within 30 seconds',now)
            return
        if head_count > 1 or ambiguous:
            self.finish('interrupted','Multiple heads or ambiguous association',now)
            return
        if track_id is None:
            if self.missing_start is None:
                self.missing_start = now
            self.maximum_head_gap = max(self.maximum_head_gap,now-self.missing_start)
            if self.maximum_head_gap > 1:
                self.finish('failed','Head observation lost for more than one second',now)
            return
        if self.missing_start is not None:
            self.maximum_head_gap = max(self.maximum_head_gap,now-self.missing_start)
            self.missing_start = None
        if self.maximum_head_gap > 1 or track_id != self.track_id:
            self.finish('failed','Head track changed or observation gap exceeded one second',now)
            return
        self.frames += 1
        if not face_usable:
            if self.away_start is None:
                self.away_start = now
            self.away_seconds = now-self.away_start
            if self.away_seconds >= 1:
                self.state = 'waiting_for_face_return'
        elif self.state == 'waiting_for_face_return':
            self.finish('passed','Same anonymous head ID across face-unavailable interval and return',now)
        else:
            self.away_start = None
            self.away_seconds = 0.0

    def snapshot(self):
        if self.state == 'idle':
            return {'status':'idle', 'message':'Not started'}
        if self.result:
            return {**self.result,'message':self.result['reason']}
        return {'status':self.state, 'message':'Waiting for face to be unavailable' if self.state == 'waiting_for_face_absence' else 'Waiting for a clear face to return',
                'track_id':self.track_id,'face_unavailable_seconds':round(self.away_seconds,1)}
