import acl
import numpy as np
import cv2
import time

from algo.yolov8.enums import ModelTask
from algo.yolov8.results import YoloDetectResults
from algo.yolov8.utils.utils import preprocess
from algo.yolov8.utils.postprocess import postprocess_det, postprocess_seg, postprocess_obb, postprocess_pose


ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


class YoloAscend:
    def __init__(self, model_path: str, task: ModelTask, keypoint_num :int=0, device_id=0):
        """
        Args:
            model_path (str): path
            task (ModelTask): 模型任务.
            keypoint_num (int, optional): 当时用关键点模型时, 关键点数量. Defaults to 0.
            device_id (int, optional): 设备id. Defaults to 0.
        """
        self.task = task
        self.keypoint_num = keypoint_num
        self.device_id = device_id

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

        # 模型 io shape
        self.input_shape = None
        self.outputs_shape0 = None
        self.outputs_shape1 = None  # seg模型有两个output

        # 初始化昇腾环境
        ret = acl.init()
        if ret == 0:
            print("昇腾环境初始化成功")
        else:
            print(f"昇腾环境初始化失败, code: {ret}")
        ret = acl.rt.set_device(self.device_id)
        if ret == 0:
            print(f"加载设备:{self.device_id} 成功")
        else:
            print(f"加载设备:{self.device_id} 失败, code: {ret}")

        # 加载模型
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        if ret == 0:
            print("模型加载成功")
        else:
            print(f"模型加载失败, code: {ret}")

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret == 0:
            print("get_desc 成功")
        else:
            print(f"get_desc 失败, code: {ret}")

        # 创建输入输出数据
        self.input_dataset, self.input_data = self._prepare_dataset('input')
        self.output_dataset, self.output_data = self._prepare_dataset('output')

        print(f'模型加载成功: {model_path}')
        print('输入数据形状:', self.input_shape)
        print('输出数据形状:', self.outputs_shape0, self.outputs_shape1)

        self.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    
    def detect(self, image: np.ndarray, conf: float=0.25, iou: float=0.45) -> YoloDetectResults:
        """检测"""
        # 预处理
        input_tensor, origin_img, ratio, pad = preprocess(image, input_size=self.input_shape[2:])

        # 推理
        outputs = self._infer(input_tensor)
        if self.task == ModelTask.SEG:
            outputs[0] = outputs[0].reshape(*self.outputs_shape0)
            outputs[1] = outputs[1].reshape(*self.outputs_shape1)
        else:
            outputs[0] = outputs[0].reshape(*self.outputs_shape0)

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
    
    def _infer(self, input):
        """infer"""
        # 拷贝所有输入到设备
        bytes_data = input.tobytes()
        bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
        ret = acl.rt.memcpy(
            self.input_data[0]["buffer"], 
            self.input_data[0]["size"], 
            bytes_ptr, 
            len(bytes_data), 
            ACL_MEMCPY_HOST_TO_DEVICE
        )
        
        # 执行推理
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        # 获取所有输出
        outputs = []
        for i in range(len(self.output_data)):
            buffer_host, _ = acl.rt.malloc_host(self.output_data[i]["size"])
            ret = acl.rt.memcpy(
                buffer_host,
                self.output_data[i]["size"],
                self.output_data[i]["buffer"],
                self.output_data[i]["size"],
                ACL_MEMCPY_DEVICE_TO_HOST
            )
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            data = np.frombuffer(bytes_out, dtype=np.float32)
            outputs.append(data)
            acl.rt.free_host(buffer_host)
            
        return outputs

    def _prepare_dataset(self, io_type):
        """获取输入/输出数量"""
        if io_type == "input":
            io_num = acl.mdl.get_num_inputs(self.model_desc)
            get_size_func = acl.mdl.get_input_size_by_index

            dims = acl.mdl.get_input_dims(self.model_desc, 0)
            if dims[1] != 0:
                print(f"get_input_dims errer code: {dims[1]}")
            self.input_shape = dims[0]['dims']

        else:
            io_num = acl.mdl.get_num_outputs(self.model_desc)
            get_size_func = acl.mdl.get_output_size_by_index
            
            if self.task == ModelTask.SEG:
                dims0 = acl.mdl.get_output_dims(self.model_desc, 0)
                dims1 = acl.mdl.get_output_dims(self.model_desc, 1)
                self.outputs_shape0 = dims0[0]['dims']
                self.outputs_shape1 = dims1[0]['dims']
            else:
                dims = acl.mdl.get_output_dims(self.model_desc, 0)
                self.outputs_shape0 = dims[0]['dims']

        # 创建数据集
        dataset = acl.mdl.create_dataset()
        buffers = []
        for i in range(io_num):
            # 获取内存大小并分配
            buffer_size = get_size_func(self.model_desc, i)

            buffer, _ = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            # 绑定数据缓冲区
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            acl.mdl.add_dataset_buffer(dataset, data_buffer)
            buffers.append({
                "buffer": buffer, 
                "data": data_buffer, 
                "size": buffer_size
            })

        return dataset, buffers

    def __del__(self):
        """释放资源"""
        