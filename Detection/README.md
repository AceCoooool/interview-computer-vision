# 说明

这部分主要总结各类经典的目标检测方法，主要归纳面试中最可能会问到的问题（这部分需要大家来补充）。

各种方法的实现，以及一些ppt等可以参考：[awesome-object-detection](https://github.com/amusi/awesome-object-detection)

## One Stage

### SSD系列

- [x] SSD
- [x] DSSD：将SSD的基础结构改成了ResNet，并在SSD基础上增加了top-down的结构（只是采用的方式和FPN有些区别，但思想是相近的）
- [x] RFB：引入了RFB结构（在Inception结构基础上增加了不同的dilated的3x3卷积）：主要为了模拟人类视觉感受野存在大小和离心率(应该说成离圆心的距离)之间正相关的信息（即大的卷积kernel应该之后配合大的dilated的卷积--这样就能模拟大的尺寸对应大的离心率）---这篇粗略浏览过去的

> 1. 在SSD系列中：前景背景的划分方式和YOLO不同---每个anchor对应最佳的IoU<0.5为背景，大于0.5为前景(当然，gt对应的anchor强制为对应类别)

### YOLO系列

- [x] YOLOv1
- [x] YOLOv2
- [x] YOLOv3

> 1. 这类方法的正负样本划分：YOLOv2和YOLOv3中，将与gt的IoU最接近的anchor box的偏移作为正样本，将IoU>0.6的无视掉，而其他的作为负样本；而在YOLOv1中则并没有无视的部分
> 2. YOLO一个非常独特的地方在于Confidence，其他方法都是采用别的方式

### Others

- [x] Focal Loss(RetinaNet)：采用"有倾向性"的Focal Loss（"特殊的交叉熵"---即对应正确的分类，使得其损失更小，扩大其与分类不正确的对象损失之间的差距），从而解决One-Stage问题中负样本数量>>正样本的情况。（可以视为与OHEM策略想解决的问题类似，但是采用的方法不同）

## Two Stage

### RCNN系列

- [ ] RCNN
- [ ] Fast RCNN
- [x] Faster RCNN

> 1. RPN中正负样本策略：IoU大于0.7的为前景，IoU小于0.3的为背景
> 2. 最终部分正负样本策略：IoU大于0.5的为前景，IoU介于0.1~0.5的为背景

### R-FCN系列

- [x] R-FCN
- [x] Light-RCNN
- [x] Deformable CNN：给原有的卷积增加了"位置偏移"，使之具有空间形变能力，从能能够使得网络能够将"注意力"全都移到感兴趣的位置的特征上面。

### Others

- [x] FPN：加入top-down的形式，将top的更抽象的语义信息与bottom"位置更精确"的信息进行了融合，从而分别将不同大小的ROI放到不同的层进行预测（大物体放到更高层：feature map size更小的，小物体放到更底层：feature map size更大的）
- [x] relation network：将attention机制引入目标检测领域（建立"候选框"之间的关系）
- [x] IoUNet：引入"定位置信度"IoU预测，从而理由IoU guided NMS解决分类的置信度和定位精度之间的不一致性的问题，以及将IoU预测结合optimization-based来进行定位"修正"。以及新的"Precise RoI Pooling"操作

## 其他改进

- [x] OHEM：选择损失大的候选框框作为训练的ROIs：专挑困难的候选框使得网络更有针对性的学习"如何变强"
- [x] A-Fast-RCNN：基于ROI-Pooling Feature利用生成网络生成掩码和仿射变化参数（专挑能使检测框架"变弱"的地方），对特征层进行"打码"和仿射变化，使得检测框架提升处理遮挡和形变的问题。其实也属于OHEM大类，加大对遮挡和形变的目标的处理

### 结构改进

- [x] DetNet
- [x] RefineDet：将two stage采用RPN进行前后背景划分和"粗"调整的思想引入one stage中，采用anchor refinement Module进行类似RPN的操作，但是同时保持类似SSD的结构形式：采用少量的速度代价换来和two stage相近的精度（可以划分为one-stage）
- [x] PVANet：主要利用CReLU+Inception结构+Hyper Feature Connection特征融合重新设计了一种更简洁的主干网络---加快主干网络的速度（虽然对RPN部分也稍微改变了下），从而提高目标检测网络的速度。

### "细节改进"系列

（下面这两篇文章最终的idea不是那么惊艳，但是其分析问题的方式很值得学习）

- [x] SNIP：主要分析了domain-shift中存在的尺寸差异问题，并采用"统一的多尺度"模型来让物体尽可能和分类网络的输入大小相接近
- [x] Cascade R-CNN：采用级联的思想来克服"IoU阈值选择的问题"，在RCNN中IoU过小容易导致mAP过低，而过高则容易导致overfitting，这篇文章主要是解决这个问题

### NMS改进

- [x] Soft-NMS：将原本NMS的"硬删除"（超过NMS阈值的框框的分数全置为0）,改为了"软抑制"（超过NMS阈值的框框的分数进行减少而不是直接置为0）。能够更有效地解决NMS存在的多删(iou太小)或多保留(iou太大)的问题