import os
import joblib
import torch
import numpy as np
import vtk
import slicer
import qt
import logging
from slicer.ScriptedLoadableModule import *
from slicer import vtkMRMLScalarVolumeNode
from slicer.util import VTKObservationMixin
# import unet

# 设置日志记录
logging.basicConfig(level=logging.DEBUG)

def normalize(slice, bottom=99, down=1):
    """
    对输入切片进行标准化，复现训练时的预处理
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    # 提取非零区域进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp[tmp == tmp.min()] = -9  # 黑色背景区域
        return tmp

def preprocess_prediction(inputs):
    """
    执行标准化，不进行裁剪，保留原始尺寸 (D, 240, 240)
    :param inputs: List of 4 numpy arrays, each of shape (D, H, W)
    :return: Combined preprocessed tensor, shape (1, D, 4, 240, 240)
    """
    normalized = [normalize(img) for img in inputs]  # 标准化
    combined = np.stack(normalized, axis=1)  # 合并成 4 通道，(D, 4, 240, 240)
    combined = combined.astype(np.float32)
    combined = np.expand_dims(combined, axis=0)  # 添加批量维度，(1, D, 4, 240, 240)

    print(f"预处理完成，输出形状: {combined.shape}")
    return combined

# -----------------------
# 逻辑类
# -----------------------
class UnetSegBratsLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super(UnetSegBratsLogic, self).__init__()
        # 加载模型
        self.model = self.load_model()

    def load_model(self):
        # 修改模型路径为训练时保存的文件名称（例如 save_all_model.pth）
        model_path = os.path.join(os.path.dirname(__file__), 'Resources', 'save_all_model.pth')
        logging.info(f"Loading model from path: {model_path}")
        if not os.path.exists(model_path):
            slicer.util.errorDisplay(f"模型文件未找到: {model_path}")
            logging.error(f"模型文件未找到: {model_path}")
            return None
        try:
            # 直接加载整个模型对象，并设置 map_location
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = torch.load(model_path, map_location=device)
            model.eval()
            slicer.util.infoDisplay("模型加载成功。")
            return model
        except Exception as e:
            slicer.util.errorDisplay(f"加载模型时出错: {str(e)}")
            logging.error(f"加载模型时出错: {str(e)}")
            return None

    def predict(self, input_tensor):
        """
        预测单个切片，并返回标签图，确保标签 0,1,2,4 显示。
        """
        if self.model is None:
            slicer.util.errorDisplay("模型未正确加载。")
            return None

        # 获取模型所在设备，并将输入 tensor 移动到该设备
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = self.model(input_tensor)  # 输出形状: [1, 3, H, W]
            # 使用 sigmoid 将 logits 转为概率，然后转到 CPU 并转换为 numpy 数组
            output = torch.sigmoid(output).data.cpu().numpy()[0]

        # 构造一个 5 通道数组，索引对应 [背景, 坏疽, 浮肿, 保留, 增强肿瘤]
        pix_rate = np.stack([
            np.full_like(output[0], 0.5),  # 背景固定权重
            output[0],  # 坏疽
            output[1],  # 浮肿
            np.zeros_like(output[0]),  # 保留0
            output[2]  # 增强肿瘤
        ], axis=0)  # 形状: [5, H, W]

        # 使用 argmax 确定每个像素的最终标签
        label_map = np.argmax(pix_rate, axis=0).astype(np.uint8)

        return label_map

    def run(self, inputVolumes):
        """
        执行预测并输出为 Segmentation 节点
        """
        try:
            logging.info("Run method started.")
            numpy_images = []
            for modality in ['FLAIR', 'T1', 'T1ce', 'T2']:
                volumeNode = inputVolumes.get(modality)
                if volumeNode is None:
                    slicer.util.errorDisplay(f"未找到模态: {modality}")
                    logging.error(f"未找到模态: {modality}")
                    return None
                numpy_images.append(slicer.util.arrayFromVolume(volumeNode))

            # 预处理输入数据（不裁剪）
            preprocessed = preprocess_prediction(numpy_images)  # 形状: (1, D, 4, 240, 240)
            num_slices = preprocessed.shape[1]
            logging.info(f"处理的切片总数: {num_slices}")

            # 初始化3D预测结果数组
            H, W = preprocessed.shape[3], preprocessed.shape[4]
            prediction_volume = np.zeros((num_slices, H, W), dtype=np.uint8)  # (D, 240, 240)

            # 遍历每个切片进行预测
            for i in range(num_slices):
                logging.info(f"预测第 {i + 1}/{num_slices} 切片")
                slice_tensor = preprocessed[0, i, :, :, :]  # 单个切片，形状 (4,240,240)
                slice_tensor = torch.from_numpy(slice_tensor).unsqueeze(0)  # (1, 4, 240, 240)
                prediction = self.predict(slice_tensor)
                if prediction is not None:
                    prediction_volume[i] = prediction  # 保存预测结果
                else:
                    logging.error(f"预测第 {i + 1} 切片时返回 None。")

            # 打印标签分布以供调试
            unique_labels = np.unique(prediction_volume)
            print(f"Prediction volume unique labels: {unique_labels}")
            logging.info(f"Prediction volume unique labels: {unique_labels}")

            # 创建 LabelMapVolume 节点
            labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TumorSegmentation_LabelMap")
            slicer.util.updateVolumeFromArray(labelmapNode, prediction_volume)

            # 设置空间信息
            referenceVolume = inputVolumes['T1']
            labelmapNode.SetSpacing(referenceVolume.GetSpacing())
            labelmapNode.SetOrigin(referenceVolume.GetOrigin())
            ijkToRAS = vtk.vtkMatrix4x4()
            referenceVolume.GetIJKToRASMatrix(ijkToRAS)
            labelmapNode.SetIJKToRASMatrix(ijkToRAS)

            # 创建 Segmentation 节点
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode",
                                                                  "TumorSegmentation_Segmentation")
            segmentationNode.CreateDefaultDisplayNodes()
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolume)

            # 导入 LabelMap 到 Segmentation 节点
            self.numpyToSegmentation(prediction_volume, segmentationNode, referenceVolume)

            # 删除 LabelMapVolume 节点
            slicer.mrmlScene.RemoveNode(labelmapNode)

            slicer.util.infoDisplay("分割完成并已导出为 Segmentation 节点。")
            return segmentationNode

        except Exception as e:
            slicer.util.errorDisplay(f"分割过程中出现错误: {str(e)}")
            logging.error(f"分割过程中出现错误: {str(e)}")
            return None

    def numpyToSegmentation(self, numpyMask, segmentationNode, referenceVolume):
        """
        将Numpy掩码数据导入到Segmentation节点，并设置名称和颜色
        :param numpyMask: 3D Numpy数组，形状 (D, H, W)
        :param segmentationNode: vtkMRMLSegmentationNode
        :param referenceVolume: 参考体积节点，用于空间信息
        """
        try:
            logging.info("Starting numpyToSegmentation.")
            segmentationNode.GetSegmentation().RemoveAllSegments()  # 清空已有的Segments
            logic = slicer.modules.segmentations.logic()

            # 标签到名称的映射，包括背景
            label_to_segment = {
                0: "背景",  # 标签0 -> '背景'
                1: "坏疽",  # 标签1 -> '坏疽'
                2: "浮肿",  # 标签2 -> '浮肿'
                4: "增强肿瘤"  # 标签4 -> '增强肿瘤'
            }

            # # 定义每个标签的颜色（R, G, B）
            # label_colors = {
            #     0: [0, 0, 255],      # 背景 - 蓝色
            #     1: [255, 0, 0],      # 坏疽 - 红色
            #     2: [0, 128, 0],      # 浮肿 - 绿色
            #     4: [255, 255, 0]     # 增强肿瘤 - 黄色
            # }

            # # 定义每个标签的颜色（R, G, B）
            # label_colors = {
            #     0: [0, 0, 255],  # 背景 - 蓝色
            #     1: [254,1,1],  # 坏疽 - 红色
            #     2: [0, 128, 0],  # 浮肿 - 绿色
            #     4: [254,254,1]  # 增强肿瘤 - 黄色
            # }

            # 正确方式（使用0-1范围的浮点值）
            label_colors = {
                0: [0.0, 0.0, 1.0],  # 蓝
                1: [1.0, 0.0, 0.0],  # 红
                2: [0.0, 0.5, 0.0],  # 绿
                4: [1.0, 1.0, 0.0]  # 黄
            }

            # 遍历每个标签，创建二值掩码并导入
            for label, segment_name in label_to_segment.items():
                logging.info(f"处理标签 {label}: {segment_name}")
                binary_mask = (numpyMask == label).astype(np.uint8)

                # 检查是否有对应标签数据
                if not binary_mask.any():
                    logging.info(f"标签 {label} ({segment_name}) 数据为空，跳过。")
                    continue

                # 创建 LabelMapVolume 节点
                labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode",
                                                                  f"{segment_name}_LabelMap")
                slicer.util.updateVolumeFromArray(labelmapNode, binary_mask)

                # 设置空间信息
                labelmapNode.SetSpacing(referenceVolume.GetSpacing())
                labelmapNode.SetOrigin(referenceVolume.GetOrigin())
                ijkToRAS = vtk.vtkMatrix4x4()
                referenceVolume.GetIJKToRASMatrix(ijkToRAS)
                labelmapNode.SetIJKToRASMatrix(ijkToRAS)

                # 导入 LabelMap 到 Segmentation 节点
                logic.ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode)

                # 获取导入后的 SegmentID
                segmentId = segmentationNode.GetSegmentation().GetNthSegmentID(
                    segmentationNode.GetSegmentation().GetNumberOfSegments() - 1)
                if segmentId:
                    # 显式设置 Segment 名称
                    segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
                    segment.SetName(segment_name)
                    # 设置 Segment 颜色
                    # rgbColor = label_colors.get(label, [255, 255, 255])  # 默认白色
                    rgbColor = label_colors.get(label, [1.0, 1.0, 1.0])  # 默认白色
                    segment.SetColor(rgbColor)
                    logging.info(f"成功设置 Segment '{segment_name}' 名称和颜色。")
                else:
                    logging.error(f"导入 Segment '{segment_name}' 失败。")

                # 删除临时节点
                slicer.mrmlScene.RemoveNode(labelmapNode)

            # 设置显示属性（仅保留基本设置，不涉及3D）
            displayNode = segmentationNode.GetDisplayNode()
            if not displayNode:
                displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode")
                segmentationNode.SetAndObserveDisplayNodeID(displayNode.GetID())

            logging.info("Segmentation 数据成功导入并设置完成。")
        except Exception as e:
            slicer.util.errorDisplay(f"转换Numpy数组为Segmentation时出错: {str(e)}")
            logging.error(f"转换Numpy数组为Segmentation时出错: {str(e)}")

# -----------------------
# 界面类
# -----------------------
class UnetSegBratsWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super(UnetSegBratsWidget, self).setup()
        layout = self.layout

        # 创建一个垂直布局框架
        mainLayout = qt.QVBoxLayout()
        self.layout.addLayout(mainLayout)

        # 使用 QFormLayout 来组织标签和选择器
        formLayout = qt.QFormLayout()
        mainLayout.addLayout(formLayout)

        # 定义四个模态
        self.modalities = ['FLAIR', 'T1', 'T1ce', 'T2']
        self.inputSelectors = {}

        for modality in self.modalities:
            # 创建标签
            label = qt.QLabel(f"选择 {modality} 模态：")
            # 创建选择器
            selector = slicer.qMRMLNodeComboBox()
            selector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
            selector.selectNodeUponCreation = False
            selector.addEnabled = False
            selector.removeEnabled = False
            selector.noneEnabled = False
            selector.setMRMLScene(slicer.mrmlScene)
            selector.setToolTip(f"选择 {modality} 模态的影像数据。")
            selector.setWindowTitle(f"选择 {modality} 模态")
            self.inputSelectors[modality] = selector

            # 将标签和选择器添加到表单布局中
            formLayout.addRow(label, selector)

        # 运行分割按钮
        self.runButton = qt.QPushButton("运行分割")
        self.runButton.toolTip = "点击按钮执行U-Net分割。"
        self.runButton.enabled = False
        mainLayout.addWidget(self.runButton)

        # 连接信号
        for selector in self.inputSelectors.values():
            selector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.runButton.connect('clicked(bool)', self.onRun)

        # 分割结果节点
        self.segmentationNode = None

        # 添加垂直分隔
        mainLayout.addStretch(1)

    def onSelect(self):
        # 检查所有模态是否都已选择
        all_selected = all(selector.currentNode() is not None for selector in self.inputSelectors.values())
        self.runButton.enabled = all_selected
        self.segmentationNode = None
        logging.debug("onSelect called. runButton enabled: {}".format(all_selected))

    def onRun(self):
        """
        执行U-Net分割，更新SegmentationNode并设置显示属性。
        """
        logic = UnetSegBratsLogic()  # 创建逻辑类实例
        inputVolumes = {modality: selector.currentNode() for modality, selector in self.inputSelectors.items()}

        # 检查输入是否完整
        if any(node is None for node in inputVolumes.values()):
            slicer.util.errorDisplay("请先选择所有四个模态的输入影像。")
            logging.error("存在未选择的模态影像。")
            return

        # 禁用按钮，开始运行
        self.runButton.setEnabled(False)
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        logging.info("开始运行U-Net分割...")

        try:
            # 执行分割任务
            segmentation = logic.run(inputVolumes)
            if segmentation:
                self.segmentationNode = segmentation  # 保存分割结果
                slicer.util.infoDisplay("分割完成。")
                logging.info("分割完成。")
            else:
                logging.error("分割任务返回 None。")
        except Exception as e:
            slicer.util.errorDisplay(f"分割过程中出现错误: {str(e)}")
            logging.error(f"分割过程中出现错误: {str(e)}")
        finally:
            # 恢复按钮状态
            slicer.app.restoreOverrideCursor()
            self.runButton.setEnabled(True)
            logging.debug("运行分割按钮重新启用。")

# -----------------------
# 模块类
# -----------------------
class UnetSegBrats(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "UnetSegBrats"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["MM"]
        self.parent.helpText = """
        使用预训练的U-Net模型对BRATS20数据集进行脑胶质瘤自动化分割。
        """
        self.parent.acknowledgementText = "this model use unet"

    def setup(self):
        # 创建一个自定义的QWidget
        scriptedLoadableModule = UnetSegBratsWidget()
        scriptedLoadableModule.setup()



