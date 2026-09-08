# Third-Party Components

InsightFace SCRFD and face alignment Python files are preserved from
`deepinsight/insightface` commit `3e6486942a1be2da0e5b475fac375ea73264bd21` (v0.7).
The MIT license is in `src/camex_faceid/vendor/INSIGHTFACE_LICENSE.txt`.
Source: https://github.com/deepinsight/insightface/tree/3e6486942a1be2da0e5b475fac375ea73264bd21

ByteTrack is attributed separately in `src/camex_faceid/vendor/bytetrack/PROVENANCE.md`
and its adjacent license. The tracker adapter uses head boxes; upstream pedestrian
benchmark scores are not claimed for this implementation.

ONNX Runtime, OpenCV, NumPy, SciPy, scikit-image, Pillow, lap, cython-bbox and
psutil retain their respective upstream licenses.

The supplied/pre-existing pretrained InsightFace weights have separate usage
terms from the MIT Python code. The official project describes its provided
pretrained weights as non-commercial research unless separately authorized:
https://github.com/deepinsight/insightface/tree/master/python-package#license
The custom CrowdHuman head checkpoint also needs provenance/reuse authorization.
This archive is an internal evaluation source package, not a representation that
every weight is cleared for a commercial client deployment.

No employee photographs, biometric vectors, raw footage, or model binaries are
embedded in the source package. They are supplied separately for authorized local
evaluation. First-party source is provided for private review; no public
open-source license is granted by this file.
