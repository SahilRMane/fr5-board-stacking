<div align="center">

# The Right Hand

**Voice-guided board handling with a Fairino FR5.**

See the table. Give a job. Follow its progress.

[Start in simulation](#start-in-simulation) · [See the workcell](#see-the-workcell) · [How it works](#how-it-works) · [Tested status](#tested-status) · [Operations guide](docs/OPERATIONS.md)

<img src="docs/images/workcell.jpg" width="100%" alt="The real Fairino FR5 workcell: robot on an aluminium rail, vacuum tool above the table, and board destination sheets marked with yellow tape." />

*The team's lab workcell. Photo from development, not a simulation.*

</div>

A local assistant for moving boards between an input area, a processing zone and output stacks.
The operator gives jobs through a web app. The controller runs the queue and handles recovery.
Gemma proposes commands. Fixed code validates them before the controller accepts them.

**One Pixi workspace. Local models. One terminal to start the stack.**

This fork builds on the team's `feat/infra-final-integration` branch. It adds a
startup selection between the original **YOLO26s** and our trained **YOLO11s**.
Both feed the same calibrated Vision 1, world model, task state machines, and
motion interfaces. YOLO26 remains the default. This fork's default `main` branch
contains the complete integration; the teammate's repository is unchanged.

## Get this complete version

Use **Linux x86-64 (Ubuntu 24.04 or newer)** with a working NVIDIA driver supporting
CUDA 12.9 and at least 8 GB of GPU memory for the complete local-model stack.
Install [Pixi](https://pixi.sh) and Git first. Allow roughly 30 GB of free disk
space for environments, caches, and model downloads. Initial setup needs internet
access; the two trained vision models are already included in Git.

```bash
git clone --recurse-submodules \
  https://github.com/SahilRMane/fr5-board-stacking.git
cd fr5-board-stacking
pixi install --locked -a
pixi run build
pixi run -e models prepare-models
pixi run up --sim --no-gui
```

Open [localhost:8088](http://localhost:8088) after **All started**. The preparation
command downloads the pinned conversation models before launch, so the first
several-GB download does not hit the launcher's readiness timeout. It also checks
that speech recognition and English/German speech synthesis can load.

Phonon requires approved Hugging Face access. By default, the speech service
prefers Phonon and explicitly reports a **Piper** fallback if Phonon is unavailable.
Piper uses public pinned English/German voices and works offline after preparation.
The model-service health response identifies the active voice backend. To choose
explicitly, set `WORKCELL_TTS=piper` or `WORKCELL_TTS=phonon`; strict Phonon mode
fails if access is unavailable. No Hugging Face token is required for Piper.

## Choose YOLO26 or YOLO11s

| Model | Camera startup option | Input | Classes |
|---|---|---|---|
| Original YOLO26s (default) | `--perception yolo26` | One crop per table | VARa, VARb, VARc |
| Our trained YOLO11s | `--perception yolo11` | One full overhead image | VARa, VARb, VARc; hands excluded from board output |

After completing the real-robot setup and calibration checks in the
[operations guide](docs/OPERATIONS.md#6-start-on-the-real-robot), choose one:

```bash
pixi run up --perception yolo26
# OR, after stopping the previous run:
pixi run up --perception yolo11
```

These two commands use real hardware. For robot-free operation, use `--sim`.
**Simulation uses synthetic board observations**, so choosing a model with
`--sim` does not by itself test that model. To exercise both actual models without
a camera or robot:

```bash
pixi run python tests/perception_models_check.py
ROS_DOMAIN_ID=169 ROS_LOCALHOST_ONLY=1 pixi run python tests/perception_ros_check.py
```

To view either actual model on an included lab image, start only its vision node:

```bash
pixi run vision-yolo26 -p device:=samples/01.jpg
# OR, after Ctrl+C:
pixi run vision-yolo11 -p device:=samples/01.jpg
```

Both publish `/vision/boards` and provide `/vision/detect`. The board packet reports
`perception_backend`; the board positions, zones, units, and downstream interfaces
are shared. Stop the current stack before switching; changes during an active
pick are not supported. Duplicate vision publishers are rejected.

Model files and SHA256 checksums are listed in
[`config/perception_models.json`](config/perception_models.json). A missing,
corrupt, wrongly labelled, or incompatible selected model causes startup to fail;
it does not silently select the other model or classical detection. The YOLO11s
export preserves checkpoint SHA256
`67ee9c19255eea65462b4289ebb6b8f3cf0f88347ff5325e5dded77a3da64a58` and uses
Ultralytics 8.3.228, ONNX 1.19.1, opset 17, and a 960-pixel input.

These models still make detection errors, particularly with overlapping boards.
Neither hand predictions nor a supported outline certify human safety or a unique
topmost board. The supplied calibration belongs to the original lab camera pose;
recalibrate and supervise acceptance tests on a different physical workcell.

## Verify the installation

See the [validation report](docs/DUAL_PERCEPTION_VALIDATION.md) for executed checks,
failure scenarios and hardware limits.

```bash
pixi run test                   # offline logic, real-model decoding, malformed-data checks
pixi run test all               # also isolated ROS, simulated motion, gripper and full controller suites
pixi run -e models python tests/voice_fallback_check.py
```

For complete web/text/voice integration, start `pixi run up --sim` first, then run
the following in a second terminal in the same ROS domain:

```bash
ROS_LOCALHOST_ONLY=1 pixi run python conversation/demo_check.py
```

For browser checks while the simulation is running:

```bash
pixi run npm --prefix web exec -- playwright install chromium
pixi run node web/tests/browser.mjs
pixi run node web/tests/responsive.mjs
pixi run python web/tests/gateway.py
```

Tests use Playwright's installed Chromium by default; optionally set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` to an existing compatible Chrome binary.

## Start in simulation

**No robot or cameras required.** Use the checkout that contains this README.
For a new PC, start with the [installation steps](docs/OPERATIONS.md#4-install-on-a-new-workstation).

### 1. Prepare the workspace once

If this checkout is already installed and built, go to step 2.

```bash
pixi install --locked -a
pixi run build
pixi run -e models prepare-models
```

**Needs:** Linux x86-64, an NVIDIA GPU with 8 GB or more, and a driver that supports CUDA 12.9.
The first model start needs internet access for the downloads. See [model setup](docs/OPERATIONS.md#47-download-the-models).

### 2. Start the stack

```bash
pixi run up --sim
```

Keep this terminal open. Wait for **All started**.
With cached models, allow about 1 minute. The first model download takes longer.

### 3. Give one job

Open **[localhost:8088](http://localhost:8088)**. In the chat, type:

> move the large white board to drop off a

**Look for:** a queued job, motion in the 3D view, and a completion message.
The arm and board movements in this mode are simulated.

<img src="docs/images/web-desktop.png" width="100%" alt="Desktop web app in simulation: a 3D robot view on the left, the simulated overhead scene in the middle, and conversation with text and voice controls on the right." />

*Desktop screenshot from the integration tests. The wrist panels have no feed because this run uses simulation.*

### 4. Stop when finished

Press **Ctrl+C** in the launcher terminal, or run this from another terminal in the same checkout:

```bash
pixi run down
```

To follow the process output in another terminal, use `pixi run logs`.
The launcher uses direct child processes. It does not need tmux.

> **Using the real robot?** Follow the [hardware startup procedure](docs/OPERATIONS.md#6-start-on-the-real-robot) first.
> The software Stop button is not the hardware emergency stop. Shutdown switches the pumps off.

## See the workcell

### A camera view the controller can use

Vision 1 converts the overhead image into board and zone observations.
The world model tracks those observations for the task controller.

<img src="docs/images/vision-overhead.jpg" width="100%" alt="Recorded overhead detection view: two small white boards have green VARb outlines and angles. Tagged templates and size-mismatched candidates have red rejection labels." />

*Recorded lab frame from the vision branch. Green outlines mark accepted candidates. Red labels explain rejection.
The historical overlay labels templates by variant. The current controller uses named processing and drop-off zones.*

| Board | Appearance | Size |
|---|---|---|
| `VARa` | Large white board | 295 × 210 × 5 mm |
| `VARb` | Small white board | 150 × 210 × 5 mm |
| `VARc` | Red board | 135 × 210 × 5 mm |

### The tool. The operator's view.

<table>
<tr>
<td width="50%" align="center"><img src="docs/images/tool-and-wrist.jpg" width="100%" alt="Close-up of the real Schmalz vacuum gripper, green suction cup, and Intel RealSense D405 on a printed wrist bracket." /></td>
<td width="50%" align="center"><img src="docs/images/web-phone.png" width="100%" alt="Phone-width web app screenshot with a scene question, the model's answer, a text box, and the Hold to talk button. This is a browser simulation capture." /></td>
</tr>
<tr>
<td><strong>Real hardware.</strong> Schmalz vacuum tool with a wrist-mounted RealSense D405.</td>
<td><strong>Phone layout.</strong> Text and voice controls from a simulated browser test.</td>
</tr>
</table>

The wrist camera provides colour and depth views in the web app.
The integrated pick routine currently uses contact search, not wrist depth, to find the board surface.

For phone microphone access, follow the [HTTPS pairing steps](docs/OPERATIONS.md#71-connect-a-phone-or-a-tablet).

<details>
<summary><strong>More lab images: overlap cases and the earlier RViz interface</strong></summary>

![Three recorded overhead frames with segmented boards, partly hidden boards, and UNKWN labels.](docs/images/vision-stacks.jpg)

*Historical segmentation results on stacked boards. `UNKWN` and overlapping labels remain visible in the original capture.
These examples show difficult cases, not a guarantee that every stack can be separated.*

![Earlier RViz interface with overhead and wrist images beside the robot model and a depth point cloud.](docs/images/lab-rviz-archive.jpg)

*Archived lab screenshot. This RViz pick panel and its scripts are retired from the integrated branch.
Use the web app shown above. This image documents the earlier camera and robot integration.*

</details>

## How it works

<p align="center">
<img src="docs/images/system.svg" width="680" alt="Two paths feed the task controller: text or voice through Parakeet, Gemma and the command grammar; overhead images through Vision 1 and the world model. The controller calls MoveIt and gripper interfaces to operate the FR5." />
</p>

1. **You request a job.** Parakeet turns speech into text. Typed requests enter as text.
2. **Gemma proposes commands.** A fixed grammar checks them. Direct control commands bypass planning.
3. **Vision supplies observations.** Calibration and tracking give the controller board and zone positions.
4. **The controller executes skills.** MoveIt and the gripper interfaces handle motion and suction. Phonon, or the explicitly reported Piper fallback, provides spoken replies.

The language model has no direct motion access.
Read the [ROS interfaces](docs/OPERATIONS.md#13-ros-interfaces) for topic, service and action names.

<details>
<summary><strong>Find the code for each part</strong></summary>

| Part | Start here |
|---|---|
| Perception | [Vision 1](vision1/) · [Calibration](config/calib.yaml) |
| Task control | [Controller and skills](src/hk_fr5_task/) |
| Motion and suction | [Motion](src/hk_fr5_motion/) · [Gripper](src/hk_fr5_gripper/) |
| Conversation and web | [Local model integration](conversation/) · [Web app](web/) |
| Startup and tests | [Process launcher](tools/stack.py) · [Test runner](tests/run.sh) |

</details>

## Tested status

**Integration checks from 4 November 2026.** Results describe this fork's tested code.

| Check | Result | What the result covers |
|---|---|---|
| End-to-end demo | **13 / 13 passed** | Simulated robot, text planning, real local speech models, pause/resume, stop/cancel and request rejection |
| Launcher lifecycle | **7 / 7 passed** | Startup, failure cleanup, signals, duplicate starts, restart, shutdown ownership and forwarding either model selection |
| Real vision models | **2 / 2 passed** | Saved images through each actual model, calibrated ROS output and world tracking; no physical robot |
| Perception failure cases | **16 tests passed** | Includes malformed types, invalid calibration, missing/corrupt weights, hand exclusion, invalid outputs and geometry |
| Voice fallback | **3 / 3 passed** | Gated-model fallback, strict selection errors and real English/German Piper audio |
| Offline suites | **Passed** | Controller logic, recorded vision samples, command grammar and conversation tests |
| Full-stack shutdown | **No session processes remained** | Simulation started and stopped without tmux |

**Hardware boundary:** the photos show the real lab setup. The current launcher has not been tested on the real robot.
These checks do not establish placement accuracy.

<details>
<summary><strong>Run the checks yourself</strong></summary>

For offline checks:

```bash
pixi run test
```

For the end-to-end demo, start `pixi run up --sim` first. In another terminal, run:

```bash
ROS_LOCALHOST_ONLY=1 pixi run python conversation/demo_check.py
```

This test queues jobs and moves the simulated robot.
See the [full test guide](docs/OPERATIONS.md#11-tests) for ROS suites and web checks.

</details>

### Know these three limits

- Same-colour boards that overlap can remain ambiguous. A rectangle does not prove that only one sheet is present.
- Wrist depth is available for viewing. The integrated pick routine still uses contact search.
- An unusual request can produce a wrong plan. Use explicit variants, counts and destinations.

## Keep the next step small

| You need to… | Open this |
|---|---|
| Install on another PC | [Installation guide](docs/OPERATIONS.md#4-install-on-a-new-workstation) |
| Run the real workcell | [Hardware startup](docs/OPERATIONS.md#6-start-on-the-real-robot) · [Shutdown](docs/OPERATIONS.md#9-stop-the-system) |
| Give jobs or connect a phone | [Web and voice commands](docs/OPERATIONS.md#7-use-the-web-app) · [CLI](docs/OPERATIONS.md#8-use-the-command-line) |
| Fix a failed start | [Troubleshooting](docs/OPERATIONS.md#14-troubleshooting) · [Logs](docs/OPERATIONS.md#10-logs) |
| Change or understand the system | [Configuration](docs/OPERATIONS.md#12-configuration-files) · [Design records](docs/OPERATIONS.md#16-design-records) |

The [operations guide](docs/OPERATIONS.md) keeps the complete setup, commands and recovery instructions.
Photo and screenshot sources are recorded in the [image notes](docs/images/README.md).
