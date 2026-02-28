# Software as a Service platform

## Usage

### DataraAI_segmentation.py

```python
python DataraAI_segmentation.py \
    --video_path path/to/video.mp4 \
    --segment "humans"
```

## Setup

This is being used on a Ubuntu 24.03 LTS VM, connected to a GH200 GPU with arm64.

### Dependency Issues

#### SAM3

The decord package is not compatible for arm64 systems, so we're replacing it with cv2's VideoCapture.

```python
# video = VideoReader(video_path, ctx=cpu(0))
cap = cv2.VideoCapture(video_path)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
```
