# FoG Detection Pipeline Using MediaPipe Pose and Machine Learning

This pipeline is basically:

**raw walking videos → body-joint coordinates over time → cleaned/normalized leg-motion signals → short time windows → labels from experts → features → ML classifier → probability curve → evaluation against true FoG episodes**

The key idea is that **Freezing of Gait, or FoG, is a time-based event**, not just a single frame. So you need to convert videos into a timeline where each moment has a probability of being FoG.

---

## 1. Frontiers/Figshare FoG Videos

This is your **raw data source**.

You are starting with videos of Parkinson’s patients walking, usually from public datasets hosted through **Frontiers**, **Figshare**, or related supplementary repositories. These videos may include patients doing walking tasks where freezing episodes happen naturally or during provocation tasks like turning, narrow passage walking, or dual-task walking.

At this stage, the data is usually just:

```text
video_001.mp4
video_002.mp4
video_003.mp4
...
```

Each video is a continuous recording over time. It might contain:

```text
normal walking
hesitation
turning
FoG episode
recovery
normal walking again
```

The video itself does not automatically tell your model where FoG happens. It is just visual footage.

The important thing is that many FoG datasets also come with **expert annotations**, such as:

```text
Video 1:
FoG from 12.4 s to 16.8 s
FoG from 31.2 s to 33.0 s

Video 2:
FoG from 8.1 s to 11.5 s
```

Those annotations are your ground truth.

So this first step gives you two things:

```text
Input:
- Raw video files
- Expert labels or annotation files

Output:
- Videos ready to process
- A reference timeline of true FoG/non-FoG
```

The main issues here are video quality, camera angle, frame rate, occlusion, resolution, and whether expert annotations are precise enough.

---

## 2. MediaPipe Pose Landmark Extraction

MediaPipe Pose is a computer vision library that estimates body joint positions from each video frame.

For every frame, MediaPipe tries to identify body landmarks like:

```text
left hip
right hip
left knee
right knee
left ankle
right ankle
left heel
right heel
left foot index
right foot index
shoulders
elbows
wrists
...
```

For your FoG project, the most important landmarks are lower-body landmarks:

```text
left hip
right hip
left knee
right knee
left ankle
right ankle
left heel
right heel
left foot index
right foot index
```

MediaPipe outputs coordinates for each landmark, usually like:

```text
frame_number, time, landmark, x, y, z, visibility
```

Example:

```text
Frame 120, time = 4.00 s

left_hip:   x=0.48, y=0.43, z=-0.12, visibility=0.98
left_knee:  x=0.51, y=0.62, z=-0.08, visibility=0.95
left_ankle: x=0.54, y=0.81, z=-0.03, visibility=0.91
```

The x and y coordinates are often normalized to the image dimensions, meaning x and y may range from 0 to 1 instead of raw pixel values.

This turns the video into a time series of body movement.

Instead of working with pixels, you now have something like:

```text
time → ankle position
time → knee position
time → hip position
time → foot position
```

That is much easier for machine learning.

The output of this step is usually a table:

| frame | time | left_hip_x | left_hip_y | left_knee_x | left_knee_y | left_ankle_x | left_ankle_y | ... |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.00 | 0.48 | 0.42 | 0.50 | 0.61 | 0.53 | 0.80 | ... |
| 1 | 0.03 | 0.48 | 0.42 | 0.50 | 0.61 | 0.53 | 0.80 | ... |
| 2 | 0.07 | 0.48 | 0.43 | 0.51 | 0.62 | 0.54 | 0.81 | ... |

The huge advantage is that you reduce a video from millions of pixels into a compact movement representation.

The main problem is that MediaPipe can make mistakes. If the person turns, gets partially blocked, moves too fast, or is filmed at a weird angle, the landmark positions can jump around.

---

## 3. Lower-Limb Landmark Cleaning

MediaPipe’s output is not perfect, so you clean the landmark data before extracting features.

This matters a lot because FoG detection depends on small changes in leg movement. If the ankle landmark randomly jumps because MediaPipe briefly lost the foot, your model may confuse tracking noise with freezing.

Cleaning usually involves several substeps.

### A. Remove or Flag Low-Confidence Landmarks

MediaPipe gives a `visibility` or confidence score. If the left ankle has low visibility, you should not fully trust its coordinate.

Example:

```text
if left_ankle_visibility < 0.5:
    mark left_ankle_x and left_ankle_y as missing
```

### B. Interpolate Short Missing Gaps

If a landmark disappears for just a few frames, you can estimate its location using surrounding frames.

Example:

```text
Frame 100: left ankle x = 0.52
Frame 101: missing
Frame 102: missing
Frame 103: left ankle x = 0.55
```

You could fill the missing frames smoothly:

```text
Frame 101: 0.53
Frame 102: 0.54
```

### C. Smooth the Landmark Trajectories

Raw pose estimates often jitter frame-to-frame, even when the person is moving smoothly. You might use:

```text
moving average
Savitzky-Golay filter
median filter
low-pass filter
```

The goal is to preserve real movement while reducing random pose noise.

### D. Detect Impossible Jumps

For example, if the ankle suddenly moves halfway across the image in one frame and then returns, that is probably a tracking error.

Example:

```text
Frame 200: left ankle x = 0.48
Frame 201: left ankle x = 0.91
Frame 202: left ankle x = 0.49
```

That middle frame is likely wrong.

After cleaning, each lower-limb landmark should have a smoother and more biologically plausible trajectory.

The output is:

```text
cleaned lower-body landmark time series
```

This step is basically quality control. Garbage pose data will produce garbage FoG predictions.

---

## 4. Coordinate Normalization

Now you normalize the coordinates so that the model focuses on **movement pattern**, not irrelevant differences like camera distance, body size, or video resolution.

Without normalization, the model could accidentally learn things like:

```text
large person in frame = FoG
close camera = FoG
left side of image = non-FoG
```

That would be bad. You want the model to learn gait dynamics.

Common normalization methods include:

### A. Body-Centered Normalization

Instead of using raw image coordinates, you express leg positions relative to the body.

For example, subtract the hip center from all lower-body landmarks:

```text
hip_center_x = (left_hip_x + right_hip_x) / 2
hip_center_y = (left_hip_y + right_hip_y) / 2

normalized_left_ankle_x = left_ankle_x - hip_center_x
normalized_left_ankle_y = left_ankle_y - hip_center_y
```

Now the ankle position is measured relative to the pelvis, not the camera frame.

This helps because the person may move through the video, but their leg motion relative to their hips is what matters.

### B. Scale Normalization

People appear at different sizes depending on camera distance. So you can divide coordinates by a body-length measure.

Common choices:

```text
shoulder width
hip width
torso length
distance from hip center to shoulder center
average leg length
```

Example:

```text
torso_length = distance(hip_center, shoulder_center)

normalized_left_ankle_x = (left_ankle_x - hip_center_x) / torso_length
normalized_left_ankle_y = (left_ankle_y - hip_center_y) / torso_length
```

This makes a short person close to the camera and a tall person far from the camera more comparable.

### C. Time Normalization

If videos have different frame rates, you may resample the pose data to a common rate, such as:

```text
30 Hz
25 Hz
50 Hz
```

This matters because features like velocity and frequency depend on time.

For example, ankle speed is:

```text
change in ankle position / change in time
```

If one video is 60 fps and another is 30 fps, you need consistent timing.

The output of this step is a normalized pose time series:

```text
time → normalized lower-limb coordinates
```

Now different videos are more comparable.

---

## 5. Sliding-Window Segmentation

FoG is not detected from a single frame. You need short chunks of time.

So you split the continuous pose timeline into overlapping windows.

For example, suppose you use:

```text
window length = 2 seconds
step size = 0.5 seconds
frame rate = 30 fps
```

Then each window has:

```text
2 seconds × 30 frames/second = 60 frames
```

The windows might look like:

```text
Window 1: 0.0 s to 2.0 s
Window 2: 0.5 s to 2.5 s
Window 3: 1.0 s to 3.0 s
Window 4: 1.5 s to 3.5 s
...
```

Each window becomes one training example.

This is important because FoG has temporal characteristics:

```text
reduced forward progression
small trembling leg movements
high-frequency leg oscillations
loss of regular step rhythm
sudden decrease in stride amplitude
```

You cannot reliably see those from one frame. You need a time segment.

The window length is a design choice.

A shorter window, like 0.5 seconds, gives better timing precision but may not capture enough gait pattern.

A longer window, like 3 or 4 seconds, captures more context but may blur the exact FoG onset and offset.

A common starting point would be:

```text
1 to 3 second windows
50% to 75% overlap
```

For FoG, I would probably start with something like:

```text
window length: 2 seconds
step size: 0.25 to 0.5 seconds
```

That gives a reasonably smooth timeline.

The output is:

```text
many short windows of normalized pose data
```

Each window will later get a label: FoG or non-FoG.

---

## 6. FoG / Non-FoG Window Labeling From Expert Annotations

Now you need to assign labels to each sliding window based on the expert annotation timeline.

Suppose an expert says:

```text
FoG episode: 10.0 s to 14.0 s
```

And your windows are:

```text
Window A: 8.0 s to 10.0 s
Window B: 8.5 s to 10.5 s
Window C: 9.0 s to 11.0 s
Window D: 10.0 s to 12.0 s
Window E: 12.0 s to 14.0 s
Window F: 14.0 s to 16.0 s
```

You need a rule for labeling.

### Option 1: Any Overlap Rule

If a window overlaps with FoG at all, label it FoG.

```text
Window B overlaps FoG from 10.0 to 10.5
Label = FoG
```

This is sensitive, but it may label transition windows as FoG even if most of the window is normal walking.

### Option 2: Majority Rule

If more than 50% of the window is FoG, label it FoG.

```text
Window B: 0.5 seconds FoG out of 2 seconds = 25%
Label = non-FoG

Window D: 2 seconds FoG out of 2 seconds = 100%
Label = FoG
```

This gives cleaner labels but may miss onset/offset transitions.

### Option 3: Threshold Rule

Label the window FoG only if at least some threshold of the window is FoG.

Example:

```text
FoG if ≥ 30% of the window overlaps with expert FoG
non-FoG if 0% overlap
discard ambiguous windows between 1% and 30%
```

This can improve training quality because transition windows are messy.

### Option 4: Center-Frame Rule

Label a window based on the label at the center time.

Example:

```text
Window: 9.0 to 11.0 s
Center: 10.0 s
If center is inside FoG, label FoG.
```

This is simple and common for timeline prediction.

The output is a training table:

| video | window_start | window_end | label |
|---|---:|---:|---|
| video_001 | 0.0 | 2.0 | non-FoG |
| video_001 | 0.5 | 2.5 | non-FoG |
| video_001 | 9.5 | 11.5 | FoG |
| video_001 | 10.0 | 12.0 | FoG |

This is where your supervised learning dataset is created.

One important issue: FoG is usually rare compared with non-FoG. You may have far more non-FoG windows than FoG windows. That causes class imbalance.

Example:

```text
10,000 non-FoG windows
800 FoG windows
```

If you do nothing, a model can get high accuracy by just predicting non-FoG all the time. So you may need class weighting, balanced sampling, or evaluation metrics like F1 score and recall instead of just accuracy.

---

## 7. Pose-Derived Feature Extraction

Now you turn each sliding window into numeric features.

The classifier does not directly need all raw landmark coordinates. For Random Forest and XGBoost, it is usually better to create interpretable movement features.

Each window gets transformed from:

```text
60 frames × multiple landmarks × x/y/z coordinates
```

into:

```text
one row of features
```

Example row:

```text
mean_left_ankle_speed
std_left_ankle_speed
right_ankle_range_x
left_right_ankle_correlation
stride_frequency
freezing_index
knee_angle_variability
...
```

These features should capture the movement patterns that distinguish FoG from normal gait.

Important feature categories:

### A. Position Range Features

FoG often involves reduced stride amplitude. The feet may stop progressing forward, and leg movement becomes smaller.

You can calculate:

```text
range of left ankle x
range of right ankle x
range of left ankle y
range of right ankle y
range of foot index x
range of foot index y
```

For each window:

```text
left_ankle_x_range = max(left_ankle_x) - min(left_ankle_x)
```

During normal walking, the ankle may move through a larger range.

During FoG, the ankle movement range may shrink.

### B. Velocity Features

Velocity describes how fast landmarks move.

For each landmark:

```text
velocity_x = change in x / change in time
velocity_y = change in y / change in time
speed = sqrt(velocity_x² + velocity_y²)
```

Then summarize within a window:

```text
mean ankle speed
max ankle speed
standard deviation of ankle speed
median ankle speed
```

FoG may show either reduced forward velocity or rapid trembling movements depending on the episode type.

### C. Acceleration and Jerk Features

Acceleration is change in velocity.

Jerk is change in acceleration.

These are useful because FoG can involve irregular, hesitant, or trembling motion.

Features:

```text
mean ankle acceleration
std ankle acceleration
max foot acceleration
jerk variability
```

Normal walking tends to have smoother periodic motion. FoG can produce more erratic micro-movements.

### D. Left-Right Coordination Features

Normal gait has alternating left-right leg movement. During FoG, that rhythm can break down.

You can calculate correlation or phase relationship between the left and right legs.

Examples:

```text
correlation(left_ankle_x, right_ankle_x)
correlation(left_ankle_speed, right_ankle_speed)
difference between left and right ankle speed
left-right symmetry index
```

During normal walking, left and right legs often show a regular alternating pattern.

During FoG, the pattern may become less coordinated.

### E. Joint Angle Features

Instead of only using x/y landmark positions, you can calculate body angles.

Useful lower-body angles:

```text
left hip-knee-ankle angle
right hip-knee-ankle angle
left knee flexion angle
right knee flexion angle
trunk angle
```

A knee angle can be computed using three points:

```text
hip → knee → ankle
```

Then you can extract:

```text
mean knee angle
knee angle range
knee angle variability
knee angular velocity
```

FoG may involve reduced joint excursion, meaning the knee does not flex and extend normally.

### F. Step Rhythm / Frequency Features

FoG often has a characteristic pattern where normal stepping is replaced by trembling or rapid small-amplitude movements.

A classic FoG concept is the **freezing index**, often described as the ratio of high-frequency movement power to normal locomotor-frequency power.

Conceptually:

```text
Freezing Index = power in freezing band / power in locomotor band
```

Usually, normal walking happens at lower frequencies, while freezing trembling may show higher-frequency components.

With pose data, you could estimate this from ankle/foot trajectories using Fourier transform or power spectral density.

Example frequency features:

```text
dominant ankle movement frequency
power from 0.5 to 3 Hz
power from 3 to 8 Hz
ratio of high-frequency power to low-frequency power
spectral entropy
```

For video pose data, this can be noisy, but it is a strong idea if the videos are stable enough.

### G. Progression Features

FoG is literally a failure or interruption of walking progression.

If the camera view allows it, you can estimate whether the person is moving forward.

Features:

```text
hip center displacement
hip center velocity
body center movement range
forward progression speed
```

During FoG, the body may stop moving forward while the legs tremble.

However, this depends heavily on camera angle. If the camera follows the person or the person walks toward the camera, this feature is harder to interpret.

### H. Confidence/Missingness Features

This sounds boring but can help.

If pose tracking quality drops during turning or occlusion, the model may need to know that.

Features:

```text
mean landmark visibility
minimum ankle visibility
number of missing frames
percentage of interpolated points
```

But be careful. You do not want the model to learn “bad video tracking = FoG” unless that genuinely reflects gait behavior.

After feature extraction, your dataset might look like this:

| video | start | end | ankle_speed_mean | ankle_x_range | knee_angle_std | freezing_index | label |
|---|---:|---:|---:|---:|---:|---:|---|
| v1 | 0.0 | 2.0 | 0.13 | 0.22 | 8.4 | 0.7 | non-FoG |
| v1 | 10.0 | 12.0 | 0.04 | 0.05 | 2.1 | 3.8 | FoG |
| v2 | 5.0 | 7.0 | 0.11 | 0.19 | 7.9 | 0.9 | non-FoG |

Now each row is one training example.

---

## 8. Random Forest / XGBoost Classifier

Now you train a model to classify each window as FoG or non-FoG.

You mentioned two model options:

```text
Random Forest
XGBoost
```

Both are tree-based machine learning models. They work well with tabular features and do not require huge datasets.

### Random Forest

A Random Forest is a collection of many decision trees.

Each tree asks questions like:

```text
Is ankle_x_range < 0.06?
Is freezing_index > 2.5?
Is knee_angle_std < 3.0?
Is hip_velocity < 0.02?
```

One tree might be weak, but hundreds of trees voting together can be strong.

Example:

```text
Tree 1: FoG
Tree 2: FoG
Tree 3: non-FoG
Tree 4: FoG
...
Final vote: FoG
```

Random Forest is good because:

```text
easy to train
handles nonlinear relationships
not too sensitive to feature scaling
gives feature importance
works decently with modest datasets
```

It is a great baseline model.

### XGBoost

XGBoost is also tree-based, but instead of building independent trees like Random Forest, it builds trees sequentially.

Each new tree tries to fix the mistakes of the previous trees.

Conceptually:

```text
Tree 1 makes rough predictions.
Tree 2 focuses on examples Tree 1 got wrong.
Tree 3 focuses on remaining errors.
...
```

XGBoost is often stronger than Random Forest on structured/tabular data.

It can handle complex feature interactions like:

```text
low ankle range + high freezing index + low hip velocity = FoG
```

XGBoost is good because:

```text
often very accurate
handles nonlinear patterns
supports class weighting
has strong regularization
works well with engineered features
```

But it needs more careful tuning than Random Forest.

### What the Classifier Learns

The model learns statistical patterns that separate FoG windows from non-FoG windows.

For example, it might learn:

```text
FoG windows often have:
- low ankle displacement
- low hip progression
- high high-frequency ankle power
- irregular left-right coordination
- reduced knee angle range
```

Non-FoG windows often have:

```text
- larger rhythmic ankle movement
- regular left-right alternation
- consistent forward progression
- lower freezing index
```

The classifier output should not just be a hard label. Ideally, it outputs a probability:

```text
Window 1: P(FoG) = 0.03
Window 2: P(FoG) = 0.07
Window 3: P(FoG) = 0.82
Window 4: P(FoG) = 0.91
```

That probability is what lets you build a timeline.

---

## 9. FoG Probability Timeline

After your classifier predicts each sliding window, you convert those window-level predictions back into a time-based curve.

Suppose your windows are:

```text
0.0 to 2.0 s → P(FoG) = 0.05
0.5 to 2.5 s → P(FoG) = 0.08
1.0 to 3.0 s → P(FoG) = 0.12
1.5 to 3.5 s → P(FoG) = 0.76
2.0 to 4.0 s → P(FoG) = 0.88
```

You can plot this as a probability curve over time:

```text
time on x-axis
FoG probability on y-axis
```

It might look like:

```text
0.0s    0.05
0.5s    0.08
1.0s    0.12
1.5s    0.76
2.0s    0.88
2.5s    0.91
3.0s    0.84
3.5s    0.30
4.0s    0.10
```

Then you choose a threshold:

```text
if P(FoG) > 0.5:
    predict FoG
else:
    predict non-FoG
```

But the threshold does not have to be 0.5.

If you care about catching as many FoG episodes as possible, you might use a lower threshold like:

```text
0.3
```

That increases sensitivity, but may create more false positives.

If you care about avoiding false alarms, you might use a higher threshold like:

```text
0.7
```

That improves precision, but may miss subtle FoG episodes.

You may also smooth the probability timeline to remove flickering predictions.

Example raw predictions:

```text
non-FoG, non-FoG, FoG, non-FoG, FoG, FoG, FoG, non-FoG
```

That is noisy.

A smoother version might require FoG to persist for a minimum duration:

```text
Only count FoG if probability stays above threshold for at least 0.5 seconds.
```

This helps avoid random single-window false alarms.

The output is:

```text
Predicted FoG timeline:
FoG from 12.0 to 16.5 s
FoG from 30.8 to 33.2 s
```

Or a continuous probability curve:

```text
time → P(FoG)
```

This is the final model output.

---

## 10. Compare Predicted Timeline to Expert Labels

Finally, you compare your model’s predicted FoG timeline to the expert-annotated timeline.

The expert timeline is the ground truth:

```text
Expert:
FoG from 10.0 to 14.0 s
FoG from 30.0 to 33.0 s
```

Your model might predict:

```text
Model:
FoG from 10.5 to 14.2 s
FoG from 29.7 to 32.5 s
FoG from 45.0 to 46.0 s
```

The first two predictions are good. The third is probably a false positive.

You can evaluate at two levels:

### A. Window-Level Evaluation

Each window has a true label and a predicted label.

You calculate:

```text
accuracy
precision
recall
F1 score
ROC-AUC
PR-AUC
confusion matrix
```

For FoG, accuracy is often misleading because non-FoG is much more common.

Example:

```text
95% of windows are non-FoG
5% are FoG
```

A dumb model that always predicts non-FoG gets 95% accuracy but detects zero FoG.

So more useful metrics are:

```text
FoG recall: Of true FoG windows, how many did you catch?
FoG precision: Of predicted FoG windows, how many were actually FoG?
F1 score: Balance between precision and recall.
PR-AUC: Better than ROC-AUC for imbalanced data.
```

Confusion matrix:

|  | Predicted non-FoG | Predicted FoG |
|---|---:|---:|
| True non-FoG | true negative | false positive |
| True FoG | false negative | true positive |

For FoG detection, false negatives are missed freezing episodes. False positives are false alarms.

### B. Event-Level Evaluation

This is often more clinically meaningful.

Instead of asking whether each window was correct, you ask:

```text
Did the model detect the FoG episode?
How close was the predicted onset?
How close was the predicted offset?
How many false FoG events did it create?
```

Important event-level metrics:

```text
onset error = predicted FoG start time - expert FoG start time
offset error = predicted FoG end time - expert FoG end time
episode detection rate
false positives per minute
mean overlap with expert episode
```

Example:

```text
Expert FoG: 10.0 to 14.0 s
Predicted FoG: 10.5 to 14.2 s

Onset error = +0.5 s
Offset error = +0.2 s
Overlap = strong
```

This is useful because even if the model is off by a few frames, it may still be clinically acceptable.

---

## The Whole Pipeline in One Example

Imagine you have a video:

```text
video_001.mp4
duration: 60 seconds
expert annotation: FoG from 22.0 to 27.5 seconds
```

You run MediaPipe Pose and get landmarks for every frame:

```text
30 fps × 60 seconds = 1800 frames
```

For each frame, you extract lower-limb coordinates:

```text
hips, knees, ankles, heels, foot index
```

Then you clean the data:

```text
remove low-confidence points
interpolate short gaps
smooth jitter
remove impossible jumps
```

Then you normalize:

```text
center coordinates around pelvis
divide by torso length
resample to 30 Hz
```

Then you split into windows:

```text
2-second windows every 0.5 seconds
```

That gives approximately:

```text
117 windows
```

Then you label each window based on expert annotations:

```text
windows overlapping 22.0 to 27.5 s are FoG
others are non-FoG
```

Then you extract features for every window:

```text
ankle speed
foot displacement
knee angle range
left-right coordination
freezing index
hip progression
```

Then you train a model:

```text
Random Forest or XGBoost
```

Then the model predicts:

```text
time 20.0 s: P(FoG) = 0.12
time 21.0 s: P(FoG) = 0.31
time 22.0 s: P(FoG) = 0.71
time 23.0 s: P(FoG) = 0.88
time 24.0 s: P(FoG) = 0.93
time 25.0 s: P(FoG) = 0.90
time 26.0 s: P(FoG) = 0.82
time 27.0 s: P(FoG) = 0.63
time 28.0 s: P(FoG) = 0.20
```

Then you threshold and smooth:

```text
Predicted FoG: 22.0 to 27.5 s
```

Then you compare against expert labels:

```text
Expert FoG:    22.0 to 27.5 s
Predicted FoG: 22.0 to 27.5 s
Good match
```

That is the full workflow.

---

## Why This Pipeline Makes Sense for Your Project

The strongest part of this approach is that it is **interpretable**.

You are not just feeding raw video into a black-box deep learning model. You can say:

```text
The model predicts FoG based on reduced ankle displacement, abnormal leg rhythm, reduced forward progression, high-frequency foot movement, and disrupted left-right coordination.
```

That sounds much more medically and scientifically grounded.

It is also doable on a normal laptop because Random Forest and XGBoost on pose-derived features are much lighter than training a deep video model.

---

## The Biggest Weaknesses to Watch Out For

The main weakness is that MediaPipe was not designed specifically for Parkinson’s gait analysis. It is a general pose estimator. It may struggle with:

```text
turning
occluded feet
side-view vs front-view inconsistency
low-resolution videos
assistive devices
shuffling steps
feet close together
```

Another weakness is that 2D video does not always capture true forward movement, especially if the camera angle changes.

The third weakness is label ambiguity. FoG onset and offset can be hard even for experts to define perfectly. Your model may look “wrong” by 0.5 seconds even if it is clinically reasonable.

The fourth weakness is data leakage. You must split train/test by patient, not by random windows. Otherwise windows from the same person can appear in both training and testing, making performance look artificially high.

Bad split:

```text
random windows from all videos into train/test
```

Better split:

```text
train on some patients
test on completely unseen patients
```

That matters a lot.

---

## Best Version of the Pipeline

A clean version would be:

```text
1. Download FoG videos and expert annotations.
2. Extract MediaPipe Pose landmarks frame by frame.
3. Keep lower-limb landmarks.
4. Remove low-confidence points, interpolate gaps, smooth trajectories.
5. Normalize coordinates relative to hip center and body scale.
6. Segment into overlapping 2-second windows.
7. Label each window based on expert FoG intervals.
8. Extract movement, rhythm, symmetry, angle, and frequency features.
9. Train Random Forest as baseline and XGBoost as stronger model.
10. Generate FoG probability for each window.
11. Smooth probabilities into a continuous FoG timeline.
12. Compare against expert labels using precision, recall, F1, PR-AUC, onset error, and false positives per minute.
```

The final deliverable could be a figure like:

```text
Top: video frame or pose skeleton
Middle: expert FoG label timeline
Bottom: model-predicted FoG probability timeline
```

That would make the project very understandable and impressive.
