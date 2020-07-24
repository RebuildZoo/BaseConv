# CNN Archs in Classification Task

## Arch
To get a fair comparison, use torchvision.model  For the network architecture provided in, the input unified size is (3, 256, 256), and the number of categories is unified as 100. The parameter quantities of different network architectures are shown in the table below:

| Arch   | ILSVRC(ImageNet) |Accecpted  | Parameters |
|--------|------------------|------------| -----------|
| Alex   | 2012             | NIPS2012   |   61,100,840  |
| vgg11  | 2014             |            |  132,863,336  | 
| vgg16  |                  |            |  138,357,544  |
| vgg16bn|                  |            |  138,365,992  |
| vgg19  |                  |            |  143,667,240  |
| googLe | 2014,2015，2016  |            |   13,004,888  |
| res18  | 2015             |            |   11,689,512  |
| res50  |                  |            |   25,557,032  |
| res101 |                  |            |   44,549,160  |
| dense121 | 2017           |            |    7,978,856  |
| dense161 |                |            |   28,681,000  |
| dense161 |                |            |   28,681,000  |
| dense201 |                |            |   20,013,928  |

## Multi-Class
Here is [wiki definition](https://en.wikipedia.org/wiki/Multiclass_classification). 

To evaluate algorithm performance, 
| Items | Details | 
| -| -|
| Training Criterion  | `nn.CrossEntropyLoss()` | 
| Validating Metric | top-1 error; top-5 error; confusion-mat | 

** top-5 error: If only one of the five categories predicted by an image is the same as the manual annotation category, it will be correct. 

In order to objectively compare the performance of various SOTA network architectures, each of them uses the following benchmark conditions: 
- [MNIST](http://yann.lecun.com/exdb/mnist/); 10 categories, 60K, fixed size 1x 28 x28; all the arch perform beyond 95%, including unlisted LeNet. 
- [ISLVRC2012](http://image-net.org/challenges/LSVRC/2017/). i.e.ImageNet; 1000 categories, 1.2M, resize/crop to 3 x 256 x 256;



## Multi-Label
Here is [wiki definition](https://en.wikipedia.org/wiki/Multi-label_classification). 

To evaluate algorithm performance, 
| Items | Details | 
| -| -|
| Training Criterion  | `nn.multilabelmarginloss()` or `nn.BCE()` | 
| Validating Metric | sklearn, `metrics.hamming_loss()` or `metrics.f1_score()`, or mAP | 

In order to objectively compare the performance of various SOTA network architectures, each of them uses the following benchmark conditions: 

-[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html); 20 categories, resize/crop to 3 x 256 x 256;




