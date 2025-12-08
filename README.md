# 简介
FlexiVision一个智能分析系统，你可以在此系统上扩展核心业务（安防、交通等），实现你希望的事件上报。该系统支持GB28181/RTSP/ONVIF协议推流，核心算法（Resnet、Yolov8、PaddleOCR等）多平台（Torch、RKNN、TensorRT、onnxRT、Ascend）适配，基于这些核心算法，系统开发了目标追踪、车牌识别、目标匹配、人脸识别等模块。该系统还向外提供API接口、数据库，供web/app端对接。  
  

该系统由C++和Python混合开发，C++主要开发视频解码推流和模型推理，Python主要开发核心业务逻辑和各功能模块以及API接口。  
  
本项目的核心特点是可扩展核心业务、多模型多平台适配。  

*本系统开发中...*  

# 模型适配

## Yolov8
| 平台/任务 | det | seg | obb | pose |
| :---:    | :-: | :-: | :-: | :-: |
| [Torch][1]    | ✅ | ✅ | ✅ | ✅ |
| [TensorRT][2] | ✅ | ✅ | ✅ | ✅ |
| RKNN     | × | × | × | × |
| [Ascend][4]   | ✅ | ✅ | ✅ | ✅ |

- 在单独使用某个包时，注意导入的包或许在其他文件夹中。
- 其中segment任务，由于后处理存在mask计算，计算量较大，除Torch平台(ultralytics官方库)，使用的CUDA加速计算，在其它平台上，segment任务后处理均在CPU上计算，对CPU消耗较大。
- 在进行pose任务时，主要需要将正确的模型关键点数量传入，否则后处理将出错(Torch平台除外)。

## PaddleOCR
| PaddlePaddle | TensorRT | RKNN | Ascend |
| :-: | :-: | :-: | :-: |
| ✅ | × | ✅ | × |

[1]: https://github.com/GIOLZZ/FlexiVision/tree/main/src/python/algo/yolov8/torch
[2]: https://github.com/GIOLZZ/FlexiVision/tree/main/src/python/algo/yolov8/tensorrt
[3]: https://github.com/GIOLZZ/FlexiVision/tree/main/src/python/algo/yolov8/rknn
[4]: https://github.com/GIOLZZ/FlexiVision/tree/main/src/python/algo/yolov8/ascend