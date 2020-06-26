python3 src/main.py -fdm models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \
	-lmm models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
	-hpm models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml \
	-gem models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml \
	-i bin/demo.mp4 \
	-d CPU \
	--print --no_move


python3 src/main.py -fdm models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \
        -lmm models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \
        -hpm models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml \
        -gem models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml \
        -i bin/demo.mp4 \
        -d CPU \
        --print --no_move


python3 src/main.py -fdm models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \
        -lmm models/intel/landmarks-regression-retail-0009/FP16-Int8/landmarks-regression-retail-0009.xml \
        -hpm models/intel/head-pose-estimation-adas-0001/FP16-Int8/head-pose-estimation-adas-0001.xml \
        -gem models/intel/gaze-estimation-adas-0002/FP16-Int8/gaze-estimation-adas-0002.xml \
        -i bin/demo.mp4 \
        -d CPU \
        --print --no_move




.
├── bin
│   ├── demo.mp4
│   └── demo.png
├── gaze-app.log
├── models
├── perf.sh
├── README.md
├── requirements.txt
└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── inference.py
    ├── input_feeder.py
    ├── main.py
    ├── model.py
    ├── mouse_controller.py
    ├── __pycache__
    │   ├── face_detection.cpython-36.pyc
    │   ├── facial_landmarks_detection.cpython-36.pyc
    │   ├── gaze_estimation.cpython-36.pyc
    │   ├── head_pose_estimation.cpython-36.pyc
    │   ├── inference.cpython-36.pyc
    │   ├── input_feeder.cpython-36.pyc
    │   └── mouse_controller.cpython-36.pyc
    └── test.py

