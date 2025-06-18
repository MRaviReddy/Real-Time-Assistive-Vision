# Real-Time Assistive Vision (RealSense or Video Upload)
# Assistive Technology for Blind and Visually Impaired
  (BVI) Individuals Using Deep Learning
/* Blind and Visually Impaired (BVI) individuals (Divyangjan) encounter numerous difficulties in
mobility and performing daily life activities independently. Though assistive technologies have
made significant strides over the past decades, a major issue remains with designs that address the
specific need of the BVI population concerning rare situations, particularly in circumstances of
poor illumination. The recent advances in AI, deep learning, and computer vision technologies
have opened new avenues for consideration to enhance the independence and quality of life of
people with visual disabilities. This literature reviews the most advanced technologies in computer
vision, deep learning models of various approaches as well as assistive devices, targeting the real
problems of the BVI population. Based on a review of some cutting-edge work, the paper
describes several major gaps in the solutions available and proposes an integrated vision for a
system. The suggested system merges novel low-light enhancement techniques, real-time objectdetection algorithms, and natural auditory feedback modalities. These innovative set of solutions
will enable the BVI portion of the population within the region to easily apply various costeffective navigation solutions to enhance mobility with safety and independence in diverse
conditions. It is, therefore, concluded that there is an immense need for a variety of mobility
technologies that should be made easily available to BVI individuals since these will surely
improve their independence and quality of life. */



# Pytorch 
Pytorch implementation of  Real-Time Assistive Vision

# Requirements
```
Name: streamlit
Version: 1.45.1
Summary: A faster way to build and share data apps
Requires: altair, blinker, cachetools, click, gitpython, numpy, packaging, pandas, pillow, protobuf, pyarrow, pydeck, requests, tenacity, toml, tornado, typing-extensions, watchdog
Required-by: streamlit-webrtc
---
Name: torch
Version: 2.7.0+cu126
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Requires: filelock, fsspec, jinja2, networkx, sympy, typing-extensions
Required-by: encodec, torchaudio, torchvision, trainer, TTS, ultralytics, ultralytics-thop
---
Name: torchvision
Version: 0.22.0+cu126
Summary: image and video datasets and models for torch deep learning
Requires: numpy, pillow, torch
Required-by: ultralytics
---
Name: pillow
Version: 11.0.0
Summary: Python Imaging Library (Fork)
Required-by: imageio, matplotlib, scikit-image, streamlit, torchvision, ultralytics
---
Name: gTTS
Version: 2.5.4
Summary: gTTS (Google Text-to-Speech), a Python library and CLI tool to interface with Google Translate text-to-speech API
---
Name: opencv-python
Version: 4.11.0.86
Summary: Wrapper package for OpenCV python bindings.
---
Name: pyrealsense2
Version: 2.55.1.6486
Summary: Python Wrapper for Intel Realsense SDK 2.0.
---
Name: ultralytics
Version: 8.3.146
Summary: Ultralytics YOLO 🚀 for SOTA object detection, multi-object tracking, instance segmentation, pose estimation and image classification.

```

Overview :
├── blind.py
├── lowlight_test.py # testing code
├── lowlight_train.py # training code
├── model.py # Zero-DEC network
├── dataloader.py
├── snapshots
│   ├── Epoch99.pth #  A pre-trained snapshot (Epoch99.pth)
```
### Test: 

cd Zero-DCE_code
```
python lowlight_test.py 
```
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "data". You can find the enhanced images in the "result" folder.

### Train: 
1) cd Zero-DCE_code

2) download the training data <a href="https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view?usp=sharing">google drive</a> or <a href="https://pan.baidu.com/s/11-u_FZkJ8OgbqcG6763XyA">baidu cloud [password: 1234]</a>

3) unzip and put the  downloaded "train_data" folder to "data" folder
```
python lowlight_train.py 
```

```
AFTER MODEL TRAINING 
```
## STREAMLIT RUN
streamlit run blind.py
The python script will run on streamlit and displays a interface of Real Time Assistive Technology and then start the assistive vision on clicking the button.Then their appears the real video ,enhanced video,Yolov8 detection,faster RCNN detection and Real time text to speech generation in multiple languages...

```
References 
``
1. OrCam Technologies. (2021). OrCam MyEye: Assistive Vision Solutions. Whitepaper.
2. Wang, T., Fu, X., Wang, J., Hu, Y., Huang, Q., & Ding, X. (2020). Zero-Reference
Deep Curve Estimation for Low-Light Image Enhancement. Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 1780–1789.
3. Zhang, Q., et al. (2021). Adaptive Assistive Technology for Visually Impaired Navigation
Using Tactile Feedback. IEEE Sensors Journal.
4. Kim, J., & Lee, H. (2021). Real-Time Navigation System for Low-Vision Users. IEEE
Transactions on Consumer Electronics.
5. Su, S., & Zhao, Q. (2020). GAN-Based Real-Time Object Detection in Low-Light
Environments for Navigation Assistance. IEEE Access.
6. Guo, Y., Zhang, L., & Li, Y. (2020). Zero-reference deep curve estimation for low-light
image enhancement. IEEE Transactions on Image Processing, 29, 2821-2832.
7. Microsoft Research. (2020). Soundscape: Using Audio for BVI Navigation. Microsoft
Press.
8. Xu, K., Yang, X., Yin, B., & Lau, R. W. H. (2020). Learning to Restore Low-Light
Images via Decomposition-and-Enhancement. Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2281-2290.
9. Wang, X., et al. (2019). Enhanced Image Enhancement Using Deep Curve Estimation.
CVPR.
10. Paszke, A., Gross, S., Massa, F., Lerer, A., B., Fang, L., Bai, J., & Chintala, S. (2019).
PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in
Neural Information Processing Systems (NeurIPS), 32, 8024–8035
11. Chen, W., & Wei, L. (2019). Low-light image enhancement via a multi-scale deep neural
network. IEEE Transactions on Image Processing, 28(8), 3702-3713.
12. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv
preprint arXiv:1804.02767.
