## Realtime Demo Instructions

### 1. Setup

This codebase should run on most standard Linux systems. It is tested with Ubuntu 16.04, pytorch v1.3.1, cuda v10.1, python v3.5.2.

a. This demo uses two external submodules: [EOS](https://pypi.org/project/eos-py/) and
   [HRNET](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) for face and facial landmarks detection, respectively.

If you have already cloned this (`few_shot_gaze`) repository without pulling the submodules, please run:

    git submodule update --init --recursive

Also, download the pre-trained `HR18-WFLW.pth` model for HRNet from [here](https://1drv.ms/u/s!AiWjZ1LamlxzdTsr_9QZCwJsn5U)
   and place it inside the folder:

    mkdir demo/ext/HRNet-Facial-Landmark-Detection/hrnetv2_pretrained

*Please note* that the Python Pip dependencies for the live demo (found under `/demo`) are different to the training/evaluation code of the network. You must install the additional dependencies. This is described in the next step.

b. Create a Python virtual environment:

    cd demo
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt update
    sudo apt install g++-7 -y
    CC=`which gcc-7` CXX=`which g++-7` pip3 install eos-py

### 2. Camera and Monitor calibration
  a. Calibrate your camera:

    python calibrate_camera.py

   This should generate a file named `calib_cam<id>.pkl` inside the `demo` folder.

   b. Calibrate your monitor's orientation and the position of its upper-left corner w.r.t. to the
   camera using the [Mirror-based Calibration](https://computer-vision.github.io/takahashi2012cvpr/) routine and
   update the methods `camera_to_monitor` and `monitor_to_camera` in `monitor.py` for your system appropriately.

   We recommend using the in-built camera in laptops or attaching an external webcam **rigidly** to your monitor.
   If you move your webcam relative to the monitor you will have to calibrate it again.

### 3. Download pre-trained models for FAZE from [here](https://ait.ethz.ch/projects/2019/faze/downloads/demo_weights.zip).
    cd demo
    wget https://ait.ethz.ch/projects/2019/faze/downloads/demo_weights.zip
    unzip demo_weights.zip

   These are slightly updated models that perform better than the originals ones documented in the published ICCV 2019
   paper.

### 4. Run demo
    python run_demo.py

This will collect user calibration data (9-point by default) and fine-tune the gaze network with it. The calibration
targets are the letter 'E' shown on a 3x3 grid on the screen in any of the 4 orientations: up, down, left or right.
The user must press the corresponding arrow key to advance to the next calibration target, otherwise another randomly
oriented target will be shown again at the same screen location. After calibration, the updated gaze network will be
used to continuously compute the user's on-screen point-of-regard and shown on the display.

### Best practices:

* A user should always look directly at the targets when pressing the arrow
keys and not at the keyboard to record accurate calibration data.

* For best results, experiment with the contrast, brightness and sharpness settings of your webcam .
    * see top of `run_demo.py`

* For best results, experiment with the learning rate and number of training steps used for fine-tuning.
    * Adjust the `lr` argument of `fine_tune` as called from `run_demo.py`.

* To change the delay/smoothing of the estimated on-screen point-of-regard modify the Kalman filter settings
in `frame_processor.py`.
