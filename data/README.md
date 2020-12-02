## Datasets
MS-Celeb-1M part1_test(584K)、YouTube-Faces、DeepFashion

[download data BaiduYun](https://pan.baidu.com/s/1cElauIJjDIM8QRgntFLB6g)(passwd: v06l)

### Data format
The data directory is constucted as follows:
```
.
├── data
|   ├── features
|   |   └── xxx.bin
│   ├── labels
|   |   └── xxx.meta

- `features` currently supports binary file.
- `labels` supports plain text where each line indicates a label corresponding to the feature file.

```

### Feature Extraction
To experiment with your own face pictures, it is required to extracted face features from the pictures.

For training face recognition and feature extraction, you may use any frameworks below, including but not limited to:

[https://github.com/yl-1993/hfsoftmax](https://github.com/yl-1993/hfsoftmax)

[https://github.com/XiaohangZhan/face_recognition_framework](https://github.com/XiaohangZhan/face_recognition_framework)

