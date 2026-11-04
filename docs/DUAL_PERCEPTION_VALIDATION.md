# Dual-perception integration validation — 4 November 2026

Base: AtharvaBhorpe/fr5-board-stacking, `feat/infra-final-integration`, commit
`4ca93b9b6e21179bcc66020ed35759bbf3c7e10f`.
The untouched base was copied to SahilRMane/fr5-board-stacking on the same branch.
Changes are developed on `feat/perception-dual-yolo` in that fork. The teammate's
remote is configured read-only locally; no push or repository change was made there.

## Implemented behavior

- YOLO26 remains the hardware-launch default. Select YOLO11s with
  `pixi run up --perception yolo11` after the documented hardware preparation.
- Both ONNX models are included in Git. SHA256, labels, source checkpoint and
  export versions are in `config/perception_models.json`.
- YOLO26 keeps its original two-table crop inference; YOLO11s uses its trained
  full-frame input. Hand-class predictions are excluded from board output.
- Both use the existing calibrated metric output, world tracking, task controller,
  skills and motion interfaces. Model selection is read-only while running.
- Missing/corrupt weights and incompatible model metadata stop startup. There is
  no silent fallback to another detector.
- The existing gated Phonon speech model is preferred when accessible. Otherwise
  the service reports and uses pinned public Piper voices. Strict Phonon and
  explicit Piper selections are available through `WORKCELL_TTS`.

## Validation executed

Test machine: Linux x86-64, NVIDIA RTX 4070 Laptop with 8 GB VRAM, driver 595.99.02.
Tests used simulation or saved lab images. No real robot was moved.

| Check | Result |
|---|---|
| Locked default and model environment installation | Passed |
| ROS packages and production web build | Passed |
| Original offline and ROS suites (`pixi run test all`) | All passed |
| Final offline runner including new model cases | All passed |
| Launcher lifecycle and model-forwarding tests | 7 passed |
| Additional model/data/calibration tests | 16 passed, with multiple invalid-input subcases |
| Actual YOLO26 and YOLO11s through ROS and the world model | Both passed |
| Full simulated web/text/voice/robot demo | 13 passed, zero failed |
| Voice selection/fallback and actual bilingual audio | 3 passed |
| Desktop/mobile browser rendering | Passed; no page errors or horizontal overflow |
| Six responsive viewport sizes | Passed |
| Gateway authentication, origin, traversal and HTTPS checks | Passed |
| Browser offline controls and no command replay | Passed |
| Microphone PCM conversion, cancelled capture and denied permission | Passed using a synthetic browser microphone |

All original default-environment dependency artifacts are preserved. The only new
model-environment dependency is `piper-tts==1.3.0`. YOLO11s does not add PyTorch to
the ROS environment. On all six supplied lab images, YOLO26 proposals exactly
matched the original integration branch before these changes.

The exported YOLO11s model was compared with its source checkpoint on the same
CPU backend (`rtol=0.001`, `atol=0.001`); predictions and mask prototypes passed.
The largest CUDA-versus-CPU coordinate difference among 44 active proposals was
0.00586 pixels at the network input; the largest class-score difference was 0.00118.

## Independent clean checkout

A separate clone installed both locked environments, built the ROS packages and
production web app, and prepared the cached public voice/ASR/planner models with
`HF_HUB_OFFLINE=1 WORKCELL_TTS=piper`.

Its full `pixi run test all` run passed every suite except an existing skills
test with timing-sensitive assumptions. Investigation found two test-harness
problems: vacuum loss could legitimately drop a board into a zone while the test
always expected an input-table board, and shutdown polling created a new action
result request on every iteration. The test now uses an input-table destination
for that recovery case and waits on one result future. The separate zone-drop
test remains in place. No motion, recovery or shutdown implementation was changed
to accommodate these tests.

The corrected skills suite then passed twice in the clean checkout: **32 checks,
zero failures per run**, with different drop positions and clean process exits.
All other clean-checkout suites, including the 18-check full controller suite and
both actual-model ROS pipelines, had already passed. The earlier failed aggregate
run is distinguished from these successful targeted reruns; it is not reported
as a clean aggregate pass.

Finally, this clean checkout started the complete stack with cached models and
`--sim --no-gui --perception yolo26`. The web/text/voice/controller demo passed
**13 checks, zero failures**, including actual Parakeet transcription and Piper
speech output. Desktop/mobile browser checks reported no page errors or overflow,
12 robot joints, 10 meshes, the simulated camera and an active service worker.
This is a software/simulation result, not physical hardware acceptance.

Failure coverage includes unknown backend names, missing/corrupt weights, paths
with spaces and Unicode, wrong image types/shapes, missing/undecodable images,
changed resolution, invalid ROI coordinates, malformed model tensors, NaN/Inf,
out-of-range confidence, nonpositive boxes, empty detections, invalid calibration
matrices, collinear calibration tags, numeric conversion errors, invalid boolean
flags, malformed board/zone/obstacle data, degenerate or crossed polygons, duplicate
vision sources, and explicit inference-runtime failure.

ROS tests additionally verify fresh snapshot responses, board removal, refusal to
refresh stale observations after source loss, and protection against changing a
model during a running node. Tests execute actual model inference; the full-stack
`--sim` demo uses synthetic observations and is reported separately.

## Practical limits

No software test can guarantee every physical outcome. The new backend has not
been accepted on a physical robot. Camera pose/focus changes require calibration;
gripping, placement, live-camera timing and human-safety behavior require separate
supervised hardware acceptance. Both models retain errors on overlapping boards.
Hand predictions are not a certified safety system. The wrist depth viewer is
preserved; the existing contact-search pick routine is unchanged.

The complete model stack requires the documented Linux/NVIDIA environment and
internet for initial public model downloads. A generic Mac, CPU-only host or
uncalibrated robot cannot reproduce the complete hardware setup merely by cloning.
After model preparation, the cached public models support offline startup.

See the README for setup, model selection and repeatable validation commands.
