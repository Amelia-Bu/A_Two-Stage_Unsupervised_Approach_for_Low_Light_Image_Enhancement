# A_Two-Stage_Unsupervised_Approach_for_Low_Light_Image_Enhancement 论文复现

数据集：LOL 
数据集获取链接：链接：https://pan.baidu.com/s/1bEMwcZxmVcwuJzphV7bR4g  提取码：kx55

方法：Retinex理论进行预增强，UNet进行细化调整（主要是去噪）

该实现与论文所述的实验细节还有一些差别：epoch = 10 （论文中为1000），UNet中具体每一层的参数设置等。
该代码距离复现还有很大的距离，没有添加损失函数（无监督的GAN损失等），实现效果不佳。
菜鸟新手尝试的复现第一篇论文。

慢慢更新ing（划掉）
好像不会更新了······

