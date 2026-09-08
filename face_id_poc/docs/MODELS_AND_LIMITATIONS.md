# Models and Business Limitations

| Stage | Network / method | Function |
| --- | --- | --- |
| Head/body detection | Custom YOLOX-Tiny/CrowdHuman, 416 x 416 | Head boxes and body support |
| Face detection | SCRFD-2.5G, 640 x 640, five landmarks | Locates visible faces |
| Recognition | ArcFace ResNet50/WebFace600K, aligned 112 x 112 | 512-D embeddings compared with enrolled photos |
| Tracking | ByteTrack-based Kalman/assignment with bounded head geometry | Camera-local continuity, not biometric proof |

This is computer vision, not an LLM. The network does not know arbitrary people's
names; labels come from reference photos supplied by the operator.

## Practical Expectations

- Clear, adequately sized, well-lit faces and good enrollment images matter.
  Small faces, blur, overhead/extreme angles, occlusion, masks and lighting changes
  can reduce usable evidence or cause incorrect matches.
- An enrolled employee can still appear unknown. Unknown is not proof that a person
  is an outsider, missing, suspicious or absent from the organization.
- A back-of-head image cannot provide a new face identification. Tracking may carry
  a previous name briefly, but must clear it when continuity is doubtful.
- One reference photo per person is a limited starting point. Validate appearance
  changes and new poses; a future multi-template workflow may need more approved photos.
- Larger galleries increase opportunities for false matches. Validate thresholds
  and top-two ambiguity margins at the intended gallery size.
- Cosine similarity is not probability. Public model benchmarks are not measured
  accuracy on the client's camera footage.
- Head IDs can fragment or switch during crossings/occlusion. Ambiguity handling
  mitigates this; it does not guarantee perfect tracking.
- There is no liveness/anti-spoofing stage. Printed photos and screens are separate
  threats. Do not use this as the sole access-control/security decision; keep human
  confirmation for consequential actions.
- Validate false named matches, unknown rejection, coverage and delay on labeled
  site data. Benchmark performance alongside the platform's other workloads.

## Model-Use Notes

InsightFace's Python code is MIT-licensed, while its provided pretrained weights
are described as non-commercial research assets unless separately authorized.
The custom head checkpoint also needs confirmed provenance/reuse authorization.
A working demo or possession of an ONNX file does not grant commercial deployment
rights. Confirm applicable terms before production use or redistribution.

Do not assume a paid/client commercial PoC is exempt merely because it is called
a demonstration or evaluation. Check whether an existing agreement covers these
exact weights and the intended use; otherwise obtain authorization or replace
them with commercially cleared weights and revalidate/re-enroll. The restriction
concerns the supplied weights and applicable data/model terms, not the ArcFace
algorithm as a general method. Academic training origin alone does not determine
commercial permission. The same distinction applies to standard InsightFace
pretrained packs used in the original application.

Source: [official InsightFace license guidance](https://github.com/deepinsight/insightface/tree/master/python-package#license).
InsightFace's [main repository licensing section](https://github.com/deepinsight/insightface#license)
lists `recognition-oss-pack@insightface.ai` for licensing its public face-recognition
models. Ask about the exact ArcFace and SCRFD assets and the intended client use;
do not assume one permission also clears the independently supplied head model.
See [MODEL_PROVENANCE.md](../MODEL_PROVENANCE.md) and the retained third-party notices.
This is an evaluation/integration preview, not a license clearance or production
accuracy certification.
