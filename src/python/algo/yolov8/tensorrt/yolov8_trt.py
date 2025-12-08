import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from algo.yolov8.enums import ModelTask
from algo.yolov8.results import YoloDetectResults
from algo.yolov8.utils.utils import preprocess
from algo.yolov8.utils.postprocess import postprocess_det, postprocess_seg, postprocess_obb, postprocess_pose


class Yolov8Trt:
    """YoloV8 TensorRT推理"""
    def __init__(self, engine_path :str, task :ModelTask, keypoint_num :int=0, max_batch_size :int=1):
        """
        Args:
            engine_path (str): path
            task (ModelTask): 模型任务.
            keypoint_num (int, optional): 当时用关键点模型时, 关键点数量. Defaults to 0.
            max_batch_size (int, optional): 最大批量大小. Defaults to 1. 目前只支持batch_size=1时
        """
        self.engine_path = engine_path
        self.task = task
        self.keypoint_num = keypoint_num
        self.max_batch_size = max_batch_size

        if not isinstance(self.task, ModelTask):
            raise ValueError("模型任务错误")
        
        # 选择后处理函数
        postprocess_map = {
            ModelTask.DET: postprocess_det,
            ModelTask.SEG: postprocess_seg,
            ModelTask.POSE: postprocess_pose,
            ModelTask.OBB: postprocess_obb
        }
        self.postprocess = postprocess_map[self.task]
        
        # 加载引擎
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        # 分配缓冲区
        self.io_tensors = self._allocate_buffers()
        self.stream = cuda.Stream()
        
        print(f"TensorRT引擎加载成功: {engine_path}")
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            if i == 0:
                self.input_shape = tensor_shape

            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            print(f"  I/O {i}: {tensor_name}, shape={tensor_shape}, dtype={tensor_dtype}, mode={tensor_mode}")

        self.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    def detect(self, image: np.ndarray, conf: float=0.25, iou: float=0.45) -> YoloDetectResults:
        """检测"""
        # 预处理
        input_tensor, origin_img, ratio, pad = preprocess(image, input_size=self.input_shape[2:])

        # 推理
        outputs = self._infer([input_tensor])

        # 后处理
        detections = self.postprocess(
            outputs,
            orig_shape=origin_img.shape[:2],
            conf_thres=conf,
            iou_thres=iou,
            ratio=ratio,
            pad=pad,
            input_shape=self.input_shape[2:],
            keypoint_num=self.keypoint_num
        )

        return detections

    def _load_engine(self):
        """加载序列化引擎"""
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        engine = self.runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError("反序列化引擎失败")
        return engine

    def _allocate_buffers(self):
        """分配主机内存和推理设备缓冲区"""
        io_tensors = {
            'inputs': [],
            'outputs': [],
            'bindings': {}
        }
        
        # 遍历所有I/O张量
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            
            # 处理动态维度（-1），使用max_batch_size
            dynamic_shape = []
            for dim in tensor_shape:
                if dim == -1:
                    dynamic_shape.append(self.max_batch_size)
                else:
                    dynamic_shape.append(dim)
            
            # 内存大小
            size = trt.volume(dynamic_shape)
            dtype = trt.nptype(tensor_dtype)
            
            # 分配页锁定主机内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            
            # 分配推理设备内存
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # 存储张量信息
            tensor_info = {
                'name': tensor_name,
                'host': host_mem,
                'device': device_mem,
                'shape': dynamic_shape,
                'dtype': tensor_dtype,
                'nbytes': host_mem.nbytes
            }
            
            # 按输入/输出分类存储
            if tensor_mode == trt.TensorIOMode.INPUT:
                io_tensors['inputs'].append(tensor_info)
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                io_tensors['outputs'].append(tensor_info)
            
            # 存储绑定关系
            io_tensors['bindings'][tensor_name] = tensor_info
        
        return io_tensors

    def _infer(self, input_data_list: list[np.ndarray]) -> list[np.ndarray]:
        """
        异步推理
        Args:
            input_data_list (list[np.ndarray]): 输入数据列表
        Returns:
            list[np.ndarray]: 推理结果列表
        """
        # 设置输入张量地址和形状
        for i, input_tensor_info in enumerate(self.io_tensors['inputs']):
            # 处理动态输入形状
            actual_shape = tuple(input_data_list[i].shape)
            self.context.set_input_shape(input_tensor_info['name'], actual_shape)
            
            # 拷贝输入数据到主机缓冲区
            np.copyto(input_tensor_info['host'], input_data_list[i].ravel())
            
            # 主机 -> 推理设备
            cuda.memcpy_htod_async(
                input_tensor_info['device'], 
                input_tensor_info['host'], 
                self.stream
            )
            
            # 设置设备内存地址
            self.context.set_tensor_address(
                input_tensor_info['name'], 
                int(input_tensor_info['device'])
            )
        
        # 设置输出张量地址
        for output_tensor_info in self.io_tensors['outputs']:
            self.context.set_tensor_address(
                output_tensor_info['name'], 
                int(output_tensor_info['device'])
            )
        
        # 异步推理
        self.context.execute_async_v3(self.stream.handle)
        
        # 异步传输：推理设备 -> 主机
        for output_tensor_info in self.io_tensors['outputs']:
            cuda.memcpy_dtoh_async(
                output_tensor_info['host'], 
                output_tensor_info['device'], 
                self.stream
            )
        
        # 同步等待完成
        self.stream.synchronize()
        
        # reshape输出数据形状
        output_results = []
        for output_tensor_info in self.io_tensors['outputs']:
            actual_shape = self.context.get_tensor_shape(output_tensor_info['name'])
            reshaped_output = output_tensor_info['host'].reshape(actual_shape)
            output_results.append(reshaped_output)
        
        return output_results
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'stream'):
            self.stream.synchronize()
