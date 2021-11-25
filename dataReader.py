import numpy as np
import json
import os
import os.path as osp
from PIL import Image
from PIL import ImageDraw
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os

from utils import parsing_embedding, parsing_embedding_select

# np.random.seed(100)

class DGDataset(data.Dataset):
    """
        Dataset for DGNet.
    """
    def __init__(self, opt):
        super(DGDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode 
        self.stage = opt.stage 
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode) 
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))])
        
        # load data list
        im_names = []
        c_names = []

        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names
        
        def make_dataset(dir):
            images = []
            assert os.path.isdir(dir), '%s is not a valid directory' % dir

            f = dir.split('/')[-1].split('_')[-1]
            # print (dir, f)
            dirs= os.listdir(dir)
            for img in dirs:

                # path = os.path.join(dir, img)
                # print(img)
                images.append(img)
            return images

    
    def name(self):
        return "DGDataset"

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth image & cloth mask
        c = Image.open(osp.join(self.data_path, 'cloth', c_name))
        c = self.transform(c)  # [-1,1]

        cw = Image.open(osp.join(self.data_path, 'warp-cloth', im_name))
        cw = self.transform(cw)  # [-1,1]

        # cloth image & cloth mask
        cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name))
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array) # [0,1]
        cm.unsqueeze_(0)

        # cmw = Image.open(osp.join(self.data_path, 'warp-mask', im_name))
        # cmw = self.transform(cmw)


        # person image 
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = self.transform(im) # [-1,1]

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-label', parse_name))
        im_parse_neck = Image.open(osp.join(self.data_path, 'image-parse-new', parse_name))
        RGB_parse_neck = Image.open(osp.join(self.data_path, 'image-parse-new-vis', parse_name+'/'+ parse_name.split('.')[0]+'_vis.png'))
        im_parse = im_parse_neck
        # 20通道的parse图
        source_parse = parsing_embedding(im_parse, (4, 11, 13))

        seg_shape = parsing_embedding(im_parse, ())

        # seg_shape = self.transform(im_parse.convert("L")) # [-1,1]
        RBG_parse = self.transform(RGB_parse_neck.convert("RGB")) # [-1,1]
        
        # gray = cv2.imread(seg_pth, cv2.IMREAD_GRAYSCALE)
        # .transpose([2, 0, 1])

        # seg_shape = torch.from_numpy(np.array(seg_shape))

        """
        0 -> Background 1 -> Hair 4 -> Upclothes 5 -> Left-shoe  6 -> Right-shoe 7 -> Noise 
        8 -> Pants 9 -> Left_leg 10 -> Right_leg 11 -> Left_arm 12 -> Face 13 -> Right_arm
        """
        
        parse_array = np.array(im_parse)
        parse_array_neck = np.array(im_parse_neck)

        im_mask = Image.open(
            osp.join(self.data_path, 'image-mask', parse_name)).convert('L')
        mask_array = np.array(im_mask)

        # parse_shape = (parse_array > 0).astype(np.float32)  # CP-VTON body shape
        # Get shape from body mask (CP-VTON+)
        parse_shape = (mask_array > 0).astype(np.float32)
        #shape of person
        # parse_shape = (parse_array > 0).astype(np.float32)
        # parse_neck_shape = (parse_array_neck > 0).astype(np.float32)

        #head of person
        
        parse_head = (parse_array == 1).astype(np.float32) + \
            (parse_array == 12).astype(np.float32)

        #cloth person is wearing
        #here try-on is of upper cloth
        parse_cloth = (parse_array == 4).astype(np.float32)

        parse_neck = (parse_array == 20).astype(np.float32)

        parse_arm = (parse_array == 11).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)


        parse_cloth_arm = (parse_array == 4).astype(np.float32) + \
                (parse_array == 11).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)

        parse_cloth_arm_neck = (parse_array == 4).astype(np.float32) + \
                (parse_array == 11).astype(np.float32) + \
                (parse_array == 13).astype(np.float32) + \
                (parse_array == 20).astype(np.float32)
                
        parse_arm_neck = (parse_array == 11).astype(np.float32) + \
                (parse_array == 13).astype(np.float32) + \
                (parse_array == 20).astype(np.float32)
        
        #background in image of person
        parse_background = (parse_array == 0).astype(np.float32)
        
        #texture translation prior required in last stage
        # parse_ttp_tom = (parse_array == 0).astype(np.float32) + \
        #         (parse_array == 1).astype(np.float32) + \
        #         (parse_array == 5).astype(np.float32) + \
        #         (parse_array == 6).astype(np.float32) + \
        #         (parse_array == 7).astype(np.float32) + \
        #         (parse_array == 8).astype(np.float32) + \
        #         (parse_array == 9).astype(np.float32) + \
        #         (parse_array == 10).astype(np.float32) + \
        #         (parse_array == 12).astype(np.float32)

        parse_ttp = (parse_array == 0).astype(np.float32) + \
                (parse_array == 1).astype(np.float32) + \
                (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32) + \
                (parse_array == 8).astype(np.float32) + \
                (parse_array == 9).astype(np.float32) + \
                (parse_array == 10).astype(np.float32) + \
                (parse_array == 12).astype(np.float32)
        parse_ttp_tom = parse_ttp
        
        """
        0 -> Background 1 -> Hair 4 -> Upclothes 5 -> Left-shoe  6 -> Right-shoe 7 -> Noise 
        8 -> Pants 9 -> Left_leg 10 -> Right_leg 11 -> Left_arm 12 -> Face 13 -> Right_arm
        """
     
        
        ptexttp = torch.from_numpy(parse_ttp)
        ptexttp_tom = torch.from_numpy(parse_ttp_tom)
        # im_ttp = im * ptexttp - (1- ptexttp) # [-1,1], fill 0 for other parts
        im_ttp = im * ptexttp_tom - (1- ptexttp_tom) # [-1,1], fill 0 for other parts
        
        im_parse = torch.from_numpy(parse_array) #[0,19]
        # im_parse_loc = parsing_embedding_select(parse_array_neck, (1, 4, 8, 11, 13, 20))
        # im_parse = torch.sum(torch.from_numpy(parse_array_neck*im_parse_loc), dim=0)

        # seg_shape = torch.from_numpy(parse_array_neck*seg_shape)

        # can_bin = parse_ttp.astype(np.float32)
        # can_bin = (mask_array > 0).astype(np.float32)
        can_bin = parse_cloth_arm_neck

        body_shape = (parse_array > 0).astype(np.float32)
        img_label = torch.from_numpy(parse_shape)
        
        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transform(parse_shape) # [-1,1]
        phead = torch.from_numpy(parse_head) # [0,1]


        pcm = torch.from_numpy(parse_cloth) # [0,1]
        pam = torch.from_numpy(parse_arm) # [0,1]
        pne = torch.from_numpy(parse_neck) # [0,1]
        pca = torch.from_numpy(parse_cloth_arm) # [0,1]
        pan = torch.from_numpy(parse_arm_neck) # [0,1]
        


        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        pcm = pcm.unsqueeze(0)
        # pca = pca.unsqueeze(0)
        # upper arms
        im_a = im * pam + (1 - pam) # [-1,1], fill 1 for other parts

        im_ne = im * pne + (1 - pne) # [-1,1], fill 1 for other parts

        im_an = im * pan + (1 - pan) # [-1,1], fill 1 for other parts

        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        # im_cm = pcm - 1 + pcm
        im_cm = pcm
        # im_cm = im_cm.unsqueeze(0)


        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]

        # "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        # "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        # "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        # "LEye": 15, "REar": 16, "LEar": 17, "Background": 18

        # pose_map = torch.zeros(1, self.fine_height, self.fine_width)
        point_line = Image.new('L', (self.fine_width, self.fine_height))
        draw = ImageDraw.Draw(point_line)
        RShoulder_x, RShoulder_y = pose_data[2,0], pose_data[2,1]
        LShoulder_x, LShoulder_y = pose_data[5,0], pose_data[5,1]
        RElbow_x, RElbow_y = pose_data[3,0], pose_data[3,1]
        LElbow_x, LElbow_y = pose_data[6,0], pose_data[6,1]
        RWrist_x, RWrist_y = pose_data[4,0], pose_data[4,1]
        LWrist_x, LWrist_y = pose_data[7,0], pose_data[7,1]

        Neck_x, Neck_y = pose_data[1,0], pose_data[1,1] 

        if RElbow_x>1 and RElbow_y>1 and RWrist_x>1 and RWrist_y>1:
            draw.line((RElbow_x, RElbow_y, RWrist_x, RWrist_y), 'white')
        if LElbow_x>1 and LElbow_y>1 and LWrist_x>1 and LWrist_y>1:
            draw.line((LElbow_x, LElbow_y, LWrist_x, LWrist_y), 'white')

        if RElbow_x>1 and RElbow_y>1 and RShoulder_x>1 and RShoulder_y>1:
            draw.line((RElbow_x, RElbow_y, RShoulder_x, RShoulder_y), 'white')
        if LElbow_x>1 and LElbow_y>1 and LShoulder_x>1 and LShoulder_y>1:
            draw.line((LElbow_x, LElbow_y, LShoulder_x, LShoulder_y), 'white')


        point_line = self.transform(point_line)
        
            

        # just for visualization
        im_pose = self.transform(im_pose)

        # ===================================================================================
        import random
        # hands_size = random.randint(10, 40)
        hands_size = 40
            
        arms_eliminate = Image.new('L', (self.fine_width, self.fine_height))
        draw_arms_eliminate = ImageDraw.Draw(arms_eliminate)
        pointrx = pose_data[4, 0]
        pointry = pose_data[4, 1]
        pointlx = pose_data[7, 0]
        pointly = pose_data[7, 1]
        if pointlx > 1 and pointly > 1:
            draw_arms_eliminate.rectangle((pointlx - hands_size, pointly - hands_size, pointlx + hands_size, pointly + hands_size), 'white', 'white')
        if pointrx > 1 and pointry > 1:
            draw_arms_eliminate.rectangle((pointrx - hands_size, pointry - hands_size, pointrx + hands_size, pointry + hands_size), 'white','white')
        arms_eliminate = self.transform(arms_eliminate)
        part_arms = (arms_eliminate + 1) * 0.5
        part_arms = part_arms*im_a + (1-part_arms)

        # ===================================================================================
        neck_size = 25
            
        neck_eliminate = Image.new('L', (self.fine_width, self.fine_height))
        draw_neck_eliminate = ImageDraw.Draw(neck_eliminate)
        pointnx = pose_data[1, 0]
        pointny = pose_data[1, 1]
        if pointlx > 1 and pointly > 1:
            draw_neck_eliminate.rectangle((pointnx - neck_size, pointny - neck_size, pointnx + neck_size, pointny - 25), 'white', 'white')
        neck_eliminate = self.transform(neck_eliminate)
        part_neck = (neck_eliminate + 1) * 0.5
        part_neck = part_neck*im_ne + (1-part_neck)

        # ===================================================================================

        # cloth-agnostic representation
        agnostic = torch.cat([shape, pose_map, im_h], 0) 

        im_g = Image.open(self.opt.grid_path)
        im_g = self.transform(im_g)


        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            'head': im_h,           # for visualization
            'grid_image': im_g,     # for visualization
            'cloth_mask':  cm,
            'RBG_parse': RBG_parse,
            'ptexttp': ptexttp, # for ground truth
            'can_bin': can_bin,
            'point_line': point_line,
            'parse_model': im_parse, # for ground truth
            'warp_cloth': cw,
            'seg_shape': seg_shape,
            'img_label':img_label,
            'parse_cloth_mask':pcm, # for ground truth
            'texture_t_prior': im_ttp, # for input
            'source_parse': source_parse, # 20 channel
            'im_cloth_mask':  im_cm,
            'pca': pca,
            'pam': pam,
            'im_a': im_a,
            'body_shape':body_shape,
            'part_arms':part_arms,
            'part_neck':part_neck,
            'im_neck': im_ne,
            'im_neck_arms': im_an,
            }


        return result

    def __len__(self):
        return len(self.im_names)
    
    
    
class DGDataLoader(object):
    def __init__(self, opt, dataset):
        super(DGDataLoader, self).__init__()

        self.runmode = opt.runmode
        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            if self.runmode == 'train':
                self.data_iter = self.data_loader.__iter__()
                batch = self.data_iter.__next__()
            if self.runmode == 'test' :
                batch = None
        return batch


def main():
    print("Check the dataset for geometric matching module!")
    from config import parser

    opt = parser()
    dataset = DGDataset(opt)
    # create dataloader
    data_loader = DGDataLoader(opt, dataset)
    

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

if __name__ == "__main__":
    main()