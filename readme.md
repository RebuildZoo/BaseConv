# Basic CNN Architecture in Classification Task 


## Multi-Class Classification
ImageNet 图像分类大赛评价标准采用 top-5 错误率，或者top-1错误率，即对一张图像预测5个类别，只要有一个和人工标注类别相同就算对，否则算错。
Top-1 = （正确标记 与 模型输出的最佳标记不同的样本数）/ 总样本数；
Top-5 = （正确标记 不在 模型输出的前5个最佳标记中的样本数）/ 总样本数；

top1-----就是你预测的label取最后概率向量里面最大的那一个作为预测结果，如过你的预测结果中概率最大的那个分类正确，则预测正确。否则预测错误

top5-----就是最后概率向量最大的前五名中，只要出现了正确概率即为预测正确。否则预测错误。

| Arch   | ILSVRC(ImageNet) | top-5 error(%) |Accecpted  | Parameters |
|--------|------------------|------------|---------------| -----------|
| Alex   | 2012             |15.3       | NIPS2012   |      12*500W   |
| VGG    | 2014             | 7.3          |            |  3 * 500 W  | 
| googLe | 2014，2015，2016 |6.67->(BN)4.8; 3.5; 3.08|    500 W       |
| Res    | 2015             | 3.75     |                3 * 500 W     |

## Multi-Label Classification

当前使用 VOC 数据集，对单张图片，预测多个相关的标签：指示 image 'has' something 
而不是 image 'is' something. 




