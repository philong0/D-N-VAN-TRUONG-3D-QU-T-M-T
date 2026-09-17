"""Native package reader. No folder discovery and no clinical observations."""
import hashlib
from pathlib import Path
import cv2
import numpy as np

# Sensor OpenGL camera -> OpenCV -> clockwise portrait image axes.
SENSOR_TO_CV = np.diag([1., -1., -1., 1.])
CV_TO_PORTRAIT = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]])
SENSOR_TO_PORTRAIT = CV_TO_PORTRAIT @ SENSOR_TO_CV


def face_to_portrait_camera(pose):
    camera = np.asarray(pose['cameraTransformColumnMajor'], float).reshape(4, 4, order='F')
    face = np.asarray(pose['faceTransformColumnMajor'], float).reshape(4, 4, order='F')
    for matrix in (camera, face):
        if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError('Invalid native rigid transform')
        if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3):
            raise ValueError('Non-rigid native transform')
    return SENSOR_TO_PORTRAIT @ np.linalg.inv(camera) @ face


def package_file(root, name):
    candidate = (Path(root) / name).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError('Package path escapes dataset')
    return candidate


def independent_entries(manifest, root):
    modern = str(manifest.get('schemaVersion', '')).startswith('3')
    if modern and manifest.get('imageOrientation') != 'portrait_cw':
        raise ValueError('Unsupported native image orientation')
    entries = manifest.get('reconstructionFrames') if modern else manifest.get('frames')
    if not isinstance(entries, list):
        raise ValueError('Missing reconstruction frame manifest')
    accepted, rejected, timestamps, hashes = [], [], set(), set()
    for entry in entries:
        filename = entry.get('rgbFileName') or entry['view'] + '.jpg'
        file = package_file(root, filename)
        image = cv2.imread(str(file))
        if image is None:
            raise ValueError('Missing/invalid RGB: ' + filename)
        # Hash decoded pixels as well as timestamp: changing JPEG encoding/name
        # must not manufacture an independent observation.
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        timestamp = entry.get('timestamp')
        if not isinstance(timestamp, (float, int)) or not np.isfinite(timestamp):
            raise ValueError('Missing physical timestamp: ' + filename)
        if timestamp in timestamps or digest in hashes:
            if modern:
                raise ValueError('Duplicate physical reconstruction observation')
            rejected.append({'view': entry['view'], 'reason': 'duplicate physical observation'})
            continue
        timestamps.add(timestamp)
        hashes.add(digest)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi = gray[h//4:3*h//4, w//4:3*w//4]
        sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        exposure = float(roi.mean() / 255.)
        quality = entry.get('quality') or {}
        if entry.get('isTracked', quality.get('isTracked')) is not True or (entry.get('geometry') or {}).get('isTracked') is False:
            rejected.append({'view': entry['view'], 'reason': 'not tracked'})
            continue
        if sharpness < 10 or not .12 <= exposure <= .92:
            rejected.append({'view': entry['view'], 'reason': 'RGB sharpness/exposure'})
            continue
        item = dict(entry, rgbFileName=filename, physicalHash=digest,
                    measuredSharpness=sharpness, measuredExposure=exposure)
        accepted.append(item)
    # Legacy packages may contain 36 sweep observations. Select independently
    # without modifying their raw files, timestamps, geometry or view labels.
    if not modern and len(accepted) > 10:
        yaw = lambda e: float(e.get('yawDeg', (e.get('quality') or {}).get('yawDeg', 0)))
        score = lambda e: np.log1p(e['measuredSharpness']) - 2 * abs(e['measuredExposure'] - .5)
        pool = sorted(accepted, key=score, reverse=True)
        selected = [min(pool, key=lambda e: abs(yaw(e)) - .1 * score(e))]
        while len(selected) < 10:
            remaining = [e for e in pool if e not in selected]
            selected.append(max(remaining, key=lambda e: min(abs(yaw(e)-yaw(s)) for s in selected) + .5*score(e)))
        accepted = sorted(selected, key=lambda e: e['timestamp'])
    return accepted, {'inputEntries': len(entries), 'independentFrames': len(accepted),
                      'rejected': rejected, 'legacy': not modern,
                      'selected': [{k: e[k] for k in ('view', 'timestamp', 'physicalHash',
                                  'measuredSharpness', 'measuredExposure')} for e in accepted]}
