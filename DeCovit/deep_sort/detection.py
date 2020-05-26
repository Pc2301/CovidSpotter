# vim: expandtab:ts=4:sw=4
import numpy as np
from scipy.spatial  import distance as dist

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


class ReferenceObject(object):
    def __init__(self, box, center, reference_width, confidence):
        self.box             = box
        self.center          = center
        self.reference_width = reference_width
        self.confidence      = confidence

class Detection(object):
    """
    This class represents a bounding box detection in a single image.
    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image
    """
    def __init__(self, tlwh, confidence, feature=None, ref_width=1):
        self.tlwh       = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        if feature is not None:
            self.feature = np.asarray(feature, dtype=np.float32)
        else:
            self.feature = None
        self.ref_width              = ref_width
        self.danger_social_distance = False
        self.danger_temperatue      = False
        self.danger_mask            = False
        self.id                     = ''
        self.face_bbox              = None

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e., (top left, bottom right)`. """
        ret      = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret      = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2]  /= ret[3]
        return ret

    def to_four_corners(self):
        ret         = list(self.tlwh.copy())
        orig_X      = ret[0]
        orig_Y      = ret[1]
        width       = ret[2]
        height      = ret[3]

        top_left     = [orig_X, orig_Y]
        top_right    = [orig_X + width, orig_Y ] # Adding the width to X coordinate.
        bottom_left  = [orig_X, orig_Y + height]
        bottom_right = [orig_X + width, orig_Y + height]

        box            = np.asarray([top_left, top_right, bottom_right, bottom_left])
        (tlblX, tlblY) = midpoint(top_left, bottom_left)
        (trbrX, trbrY) = midpoint(top_right, bottom_right)

        # compute the Euclidean distance between the midpoints,
        # then construct the reference object
        D       = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        cX      = np.average(box[:, 0])
        cY      = np.average(box[:, 1])
        refObj  = ReferenceObject(box, (cX, cY), D / self.ref_width, self.confidence)
        return refObj

# import cv2
# import imutils
# from imutils import perspective
# topX_l = 100
# topY_l = 100
# w = 50
# h = 30
# topX_r = topX_l + w
# topY_r = topY_l
#
# bottomX_l = topX_l
# bottomY_l = topY_l + h
#
# bottomX_r = topX_l + w
# bottomY_r = topY_l + h
# cnt = np.asarray([[topX_l, topY_l], [topX_r, topY_r], [bottomX_r, bottomY_r], [bottomX_l, bottomY_l]])
# box = cv2.minAreaRect(cnt)
# box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
# box = perspective.order_points(box)
# box = np.array(box)
# print(box, cnt)

