# Dataset Info
The PASCAL Visual Object Classes Challenge 2012 (VOC2012) is to recognize objects from a number of visual object
classes in realistic scenes (i.e. not pre-segmented objects). There are 20
object classes. 

There are 5 main tasks, first 3 is more common: 
| Task | Description | 
|-------|--------|
|Classification | For each of the classes predict the presence/absence of at least one object of that class in a test image. |
|Detection | For each of the classes predict the bounding boxes of each object of that class in a test image. |
|Segmentation |  For each pixel in a test image, predict the class of the object containing that pixel or ‘background’ if the pixel does not belong to one of the 20 specified classes.|
|Action Classiﬁcation |  For each of the action classes predict if a speciﬁed person (indicated by BBX) in a test image is performing the corresponding action. There are 10 action classes| 

For the classification and detection tasks there are 4 sets of images provided:

| Subset | #Images | #Instances |
|-------|--------|--------|
| train | 5717 | 13609 | 
| val | 5823 | 13841 | 
| test | 16135 | ? | 


