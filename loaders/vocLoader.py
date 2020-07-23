import os 
import sys 
import numpy as np 
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt 


def view_tensor(p_img_Tsor):
    # p_img_Tsor = p_img_Tsor / 2 + 0.5     # unnormalize
    img_Arr = p_img_Tsor.numpy()
    plt.imshow(np.transpose(img_Arr, (1, 2, 0)))
    plt.show()

def img_load_helper(pImg_abs_dir):
    return Image.open(pImg_abs_dir)#.convert('RGB')

class voc2012_Classification_Loader(torch.utils.data.Dataset):
    '''
    load the oringinal VOC2012 dataset from folder
    
    '''
    def __init__(self, pImg_dir, pAnno_dir, pMode, pTansfom = None,
                pData_helper = img_load_helper):

        assert os.path.isdir(pImg_dir), "invalid image dir: " + pImg_dir
        assert os.path.isdir(pAnno_dir), "invalid anno dir: " + pAnno_dir

        if not pTansfom:
            self.transform = transforms.Compose([
                # transforms.ToPILImage(), 
                transforms.Resize((224, 224)), 
                transforms.RandomHorizontalFlip(), 
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 
                transforms.ToTensor(), # (0, 255) uint8 HWC-> (0, 1.0) float32 CHW
                # transforms.RandomApply([]),
                
            ])
        else:
            self.transform = pTansfom
        self.data_helper = pData_helper

        self.object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']
        
        if pMode == "train" or pMode == "val":
            pass
        else: 
            pMode = "trainval"
        img_filename_txt = os.path.join(pAnno_dir, pMode + ".txt")

        self.img_filename_Lst = []
        with open(img_filename_txt, 'r') as anno_file:
            for line_i in anno_file.readlines():
                info_i = line_i.strip()
                if not info_i: continue
                cur_absfilename = os.path.join(pImg_dir, info_i + ".jpg")
                assert os.path.isfile(cur_absfilename), "invalid image filename: " + cur_absfilename
                self.img_filename_Lst.append(cur_absfilename)

        # self.anno_Dic = {}
        row_num = len(self.img_filename_Lst); col_num = len(self.object_categories)
        self.anno_Arr = np.zeros((row_num, col_num), np.float32)
        for anno_idx_i, anno_item_i in enumerate(self.object_categories):
            anno_absfilename_i = os.path.join(pAnno_dir, anno_item_i + '_' + pMode + ".txt")
            with open(anno_absfilename_i, 'r') as anno_file:
                for line_idx_j, line_j in enumerate(anno_file.readlines()):
                    info_j = line_j.strip()
                    if not info_i: continue
                    try:
                        filename_j, anno_j = info_j.split(" ") # -1
                    except ValueError:
                        filename_j, anno_j = info_j.split("  ") #  1
                    assert filename_j in self.img_filename_Lst[line_idx_j], "image dimatch :" + filename_j
                    if int(anno_j) == 1 : 
                        #self.anno_Dic[filename_j] = anno_idx_i
                        self.anno_Arr[line_idx_j][anno_idx_i] = 1

    def __len__(self):
        return len(self.img_filename_Lst)
    
    def __getitem__(self, index):
        img_absfilename = self.img_filename_Lst[index]
        img_pil = self.data_helper(img_absfilename)
        img_Tsor = self.transform(img_pil)
        
        # img_absfilename = (os.path.split(img_absfilename)[-1]).split(".")[0]
        # label_idx = self.anno_Dic[img_absfilename]
        label_Vec = self.anno_Arr[index]
        label_Tsor = torch.from_numpy(label_Vec)
        return img_Tsor, label_Tsor
    
    def hot2str(self, label_Tsor):
        label_Arr = label_Tsor.numpy().astype(np.int)
        label_Lst = []
        for label_Vec in label_Arr:
            #label_str_Lst_i = [self.object_categories[i] for i in label_Vec]
            label_str_Lst_i = []
            for label_idx_j, label_j in enumerate(label_Vec):
                if label_j == 1: label_str_Lst_i.append(self.object_categories[label_idx_j])
            label_Lst.append(label_str_Lst_i)
        return label_Lst






if __name__ == "__main__":
    img_dir = r'F:\ZimengZhao_Data\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
    anno_dir = r"F:\ZimengZhao_Data\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Main"

    gm_dataset = voc2012_Classification_Loader(img_dir, anno_dir, "train")

    trainloader = torch.utils.data.DataLoader(dataset = gm_dataset, batch_size =4, 
                        shuffle= False, num_workers = 1)
    
    for i_idx, pac_i in enumerate(trainloader):
        img_Tsor_bacth_i, label_Tsor_bacth_i = pac_i
        print(img_Tsor_bacth_i.shape, label_Tsor_bacth_i.shape)
        print(torch.max(img_Tsor_bacth_i[0]), torch.min(img_Tsor_bacth_i[0]))
        # label_str_Lst = [gm_dataset.object_categories[i] for i in label_Tsor_bacth_i.numpy().tolist()]
        # print(label_Tsor_bacth_i)
        label_str_batch_i = gm_dataset.hot2str(label_Tsor_bacth_i)
        for label_i in label_str_batch_i:
            print(label_i)
        view_tensor(torchvision.utils.make_grid(
                        tensor = img_Tsor_bacth_i, 
                        nrow= 2)
                )
        
        