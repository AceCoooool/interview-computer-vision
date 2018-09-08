# 实例分割

> 这部分内容主要来自别人的总结

## 1. 基于FCN的分割

> 来自：[Semantic Segmentation using Fully Convolutional Networks over the years](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html)

[基于FCN的分割](semantic_segmentation/segmentation.md)

- [x] FCN：首次将FCN用到语义分割 --- 利用deconvolution作为上采样，以及结合skip connection来"修正"空间信息。
- [x] SegNet：采样unpooling的方式进行上采样
- [x] U-Net："阶梯结构"（也可称为U结构），且采样concat操作
- [x] Fully Convolutional DenseNets：将U-Net里面的基础架构改为DenseNet的Block
- [x] E-Net和Link Net：主要为了消减参数
- [x] Mask R-CNN：引入mask分支---其实针对的是实例分割
- [x] PSPNet：引入Pyramid pooling module --- 整合不同区域的context从而获取全局的context
- [x] RefineNet：通过RefineNet结构（可以允许多个输入进行融合）将粗糙的高层语义特征和细粒度的低层特征融合
- [x] G-FRNet：通过"门控"来"提炼"encoder中对于decoder真正有帮助的信息

## 2. DeepLab系列

[DeepLab系列](deeplab/deeplab.md)：

- [x] DeepLab v1
- [x] DeepLab v2
- [x] DeepLab v3