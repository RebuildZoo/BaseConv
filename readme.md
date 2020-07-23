# CNN Archs in Classification Task

# Multi-Class

In order to objectively compare the performance of various SOTA network architectures, each of them uses the following benchmark conditions: 
<!-- - **MNIST**; 10 categories, 60K, fixed size 1x 28 x28;  -->
- **ISLVRC2012**(*The data for the classification and localization tasks will remain unchanged from 2012*) 1000 categories, 1.2M, resize/crop to 3 x 256 x 256;

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


# Multi-Label




