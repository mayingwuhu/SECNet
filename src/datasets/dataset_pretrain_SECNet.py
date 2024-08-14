import os
import random

import numpy as np
import torch
from PIL import Image
import sys
from src.datasets.data_utils import VideoRandomSquareCrop, mask_batch_text_tokens
from src.datasets.dataset_base import AlproBaseDataset, img_collate
from src.datasets.randaugment import TemporalConsistentRandomAugment, RandomAugment
from src.utils.basic_utils import flat_list_of_lists
from src.utils.logger import LOGGER
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms


class PretrainSparseDataset(AlproBaseDataset):
    """
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict {
            "type": "image",
            "filepath": "/abs/path/to/COCO_val2014_000000401092.jpg",
            "text": "A plate of food and a beverage are on a table.",  # should be tokenized and digitized first?
            ...
            }
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    vis_format: str, image or video, used to decide data loading method.
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir, img_db_type, txt_dir,
                video_fmt='.mp4', crop_size=256, resize_size=288, fps=3, num_frm=3, frm_sampling_strategy="rand",
                max_img_size=1000, max_txt_len=20,
                use_itm=True, is_train=True):
        super(PretrainSparseDataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir, 
            img_db_type=img_db_type,
            fps=fps, 
            num_frm=num_frm, 
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, 
            max_txt_len=max_txt_len)
        self.use_itm = use_itm

        self.txt_dir = txt_dir
        self.video_fmt = video_fmt

        self.crop_size = crop_size
        self.video_random_cropper = VideoRandomSquareCrop(crop_size)

        self.resize_size = resize_size

        self.is_train = is_train

        if self.is_train:
            self.randaug = TemporalConsistentRandomAugment(N=2, M=5, augs=['Identity', 'Contrast','Brightness','Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip'])     
        else:
            self.randaug = None

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        start_time = None
        end_time = None

        # fetch video
        num_retries = 10  # skip error videos

        for _ in range(num_retries):
            data_sample = self.datalist.iloc[index]#按行加载数据

            video_id = str(data_sample.video_id)
            txt_len = int(data_sample.txt_len)

            if hasattr(data_sample, 'text'):
                text = data_sample.text.strip()
            else:
                raise NotImplementedError("Un-supported text annotation format.")

            # fetch video
            video_path = os.path.join(self.img_db_dir, video_id + self.video_fmt) #定位视频位置

            # read with retries
            for i in range(3):
                img_array = self._load_video_from_path_decord(video_path, height=self.resize_size, width=self.resize_size)#返回N*3*H*W的视频数据

                if img_array is not None:
                    break

            if img_array is not None:
                t, c, h, w = img_array.shape

            # Select a random video if the current video was not able to access.
            if img_array is None:
                LOGGER.info(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                # square crop
                img_array = self.video_random_cropper(img_array)

                if self.randaug:
                    img_array = self.randaug(img_array.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

                break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        examples = [{'text_str': text, 'itm_label': 1}]

        return dict(
            img=img_array,  # (T, C, H, W)
            examples=examples,
            n_examples=len(examples),  # used to create image feature copies.
            type='video'
        )


class PretrainImageTextDataset(Dataset):
    def __init__(self, datalist, tokenizer, img_lmdb_dir, is_train=True, crop_size=256, resize_size=288,num_frm=4, wh1_num_frm=2, wh2_num_frm=2, where_num_frm=1, when_num_frm=1, max_txt_len=40):
        self.datalist = datalist #json读出来的数据集合
        self.max_txt_len = max_txt_len

        self.crop_size = crop_size
        self.resize_size = resize_size
        self.num_frms = num_frm
        self.img_dir = img_lmdb_dir
        self.wh1_num_frm = wh1_num_frm
        self.wh2_num_frm = wh2_num_frm
        self.where_num_frm = where_num_frm
        self.when_num_frm = when_num_frm
        self.where_class = ["street", "park", "garden", "square", "market", "shop", "school", "campus", "bus stop", "train station",
            "subway station", "commercial building", "cityscape", "amusement park", "playground", "tennis court",
            "football field", "basketball field", "bike lane", "pedestrian street", "sidewalk", "road", "highway",
            "airport", "boathouse","botanical garden", "zoo", "historical site", "beach", "lake", "river", "bridge", "valley", "forest",
            "grassland", "sky","farmland", "farm", "snowfield", "glacier", "ocean", "desert", "cliff", "volcano", "island", "orchard",
            "temple","church", "balcony", "living room", "bedroom", "kitchen", "dining room", "bathroom", "toilet", "study room",
            "office","meeting room", "classroom", "library", "cinema", "theater", "concert hall", "museum", "art gallery",
            "shop","supermarket", "cafe shop", "bar", "hotel", "game room", "gym", "pool", "salon", "hospital", "store",
            "dim light"]
        self.when_class = ["Daytime", "Evening", "Night"]

        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])

        self.is_train = is_train


        # self.transform = transforms.Compose([
        #         transforms.RandomResizedCrop(self.crop_size, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        #         transforms.RandomHorizontalFlip(),
        #         RandomAugment(2,7,isPIL=True,augs=['Identity','Brightness','Sharpness',
        #                                         'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'])
        #     ])
        
    def __len__(self):
        return len(self.datalist)



    def __getitem__(self, index):
        start_time = None
        end_time = None
        img_dir = self.img_dir

        #定义one_hot_where和one_hot_when
        one_hot_where = [0] * len(self.where_class)
        label_dict_where = {label: index for index, label in enumerate(self.where_class)}
        one_hot_when = [0] * len(self.when_class)
        label_dict_when= {label: index for index, label in enumerate(self.when_class)}

        # fetch video
        num_retries = 5  # skip error videos



        for _ in range(num_retries):
            data_sample = self.datalist[index]


            if type(data_sample['caption']) == list:
                text = random.choice(data_sample['caption'])#获得数据中的caption
            else:
                text = data_sample['caption']

            img_When_class = []
            img_When_class.append(data_sample['when'])
            img_Where_class = data_sample['Where']
            img_WH = data_sample['WH']
            img_path = os.path.join(img_dir, data_sample['image'])
            img_arr = Image.open(img_path).convert('RGB')
            if len(img_WH) <= 1:

                img_arr_WH1 = img_arr
                img_arr_WH1 = self.transform(img_arr_WH1)
                img_arr_WH1 = np.asarray(img_arr_WH1, dtype=np.float32)
                img_arr_WH1 = torch.from_numpy(img_arr_WH1).unsqueeze(0)
                img_arr_WH1 = img_arr_WH1.repeat(self.wh1_num_frm, 1, 1, 1)

                img_arr_WH2 = img_arr_WH1



                text_WH1 = text_WH2= "A frame of no moving subject."

            elif len(img_WH) == 2:

                crop_info_W1 = img_WH[0]['W']
                img_arr_W1 = crop_image(img_arr,crop_info_W1)
                img_arr_W1 = self.transform(img_arr_W1)
                img_arr_W1 = np.asarray(img_arr_W1, dtype=np.float32)
                img_arr_W1 = torch.from_numpy(img_arr_W1).unsqueeze(0)

                crop_info_H1 = img_WH[0]['H']
                img_arr_H1 = crop_image(img_arr,crop_info_H1)
                img_arr_H1 = self.transform(img_arr_H1)
                img_arr_H1 = np.asarray(img_arr_H1, dtype=np.float32)
                img_arr_H1 = torch.from_numpy(img_arr_H1).unsqueeze(0)

                img_arr_WH1 = torch.cat([img_arr_W1, img_arr_H1], dim=0)


                crop_info_W2 = img_WH[1]['W']
                img_arr_W2 = crop_image(img_arr, crop_info_W2)
                img_arr_W2 = self.transform(img_arr_W2)
                img_arr_W2 = np.asarray(img_arr_W2, dtype=np.float32)
                img_arr_W2 = torch.from_numpy(img_arr_W2).unsqueeze(0)

                crop_info_H2 = img_WH[1]['H']
                img_arr_H2 = crop_image(img_arr, crop_info_H2)
                img_arr_H2 = self.transform(img_arr_H2)
                img_arr_H2 = np.asarray(img_arr_H2, dtype=np.float32)
                img_arr_H2 = torch.from_numpy(img_arr_H2).unsqueeze(0)

                img_arr_WH2 = torch.cat([img_arr_W2, img_arr_H2], dim=0)

                # 两个的caption
                text_WH1 = img_WH[0]['WH_caption']
                text_WH2 = img_WH[1]['WH_caption']

            img_arr = self.transform(img_arr)
            img_arr = np.asarray(img_arr, dtype=np.float32)
            img_arr = torch.from_numpy(img_arr).unsqueeze(0)
            img_Where = img_When = img_arr.repeat(self.when_num_frm, 1, 1, 1)


            for label in img_Where_class:
                index = label_dict_where[label]
                one_hot_where[index] = 1
            labels_Where = torch.tensor(one_hot_where)

            for label in img_When_class:
                index = label_dict_when[label]
                one_hot_when[index] = 1
            labels_When = torch.tensor(one_hot_when)#tensor([0, 0, 1])

            if img_arr is not None:
                t, c, h, w = img_arr.shape

            # Select a random video if the current video was not able to access.
            if img_arr is None:
                LOGGER.info(f"Failed to load examples with image: {img_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break
        else:
            raise RuntimeError(f"Failed to fetch image after {num_retries} retries.")
        
        examples = [{'text_str': text, 'itm_label': 1}]
        examples_WH1 = [{'text_str_WH1': text_WH1}]
        examples_WH2 = [{'text_str_WH2': text_WH2}]


        return dict(
            img_WH1=img_arr_WH1,# (T, C, H, W)
            img_WH2=img_arr_WH2,
            img_Where=img_Where,
            img_When=img_When,
            labels_Where=labels_Where,
            labels_When=labels_When,
            examples=examples,
            examples_WH1=examples_WH1,
            examples_WH2=examples_WH2,
            n_examples=len(examples),  # used to create image feature copies.
            type='img'
        )

def crop_image(img_arr,crop_info):
    original_width ,original_height = img_arr.size

    left = int(crop_info['x'] * original_width / 100)
    top = int(crop_info['y'] * original_height / 100)
    right = int((crop_info['x'] + crop_info['width']) * original_width / 100)
    bottom = int((crop_info['y'] + crop_info['height']) * original_height / 100)

    cropped_image = img_arr.crop((left, top, right, bottom))


    return cropped_image

class PretrainCollator(object):
    """is_train is kept here if we want to remove
    the randomness during validation of MLM accuracy.
    In that case, instantiate two PretrainCollator
    如果我们想在验证MLM准确性的过程中消除随机性，则此处保留is_train。在这种情况下，实例化两个PretrainCollator
    """
    def __init__(self, tokenizer, 
                 mlm=False, mlm_probability=0.15,
                 patch_size=16,
                 mpm=False,
                 max_length=20, is_train=True):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.is_train = is_train

        self.mpm = mpm
        self.patch_size = patch_size

    def collate_batch(self, batch):
        #通过判断batch中的第一个元素的"img"键是否为torch.Tensor类型，来确定使用哪种方式进行数据的整理，整理后的结果存储在visual_inputs变量中
        if isinstance(batch[0]["img_WH1"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        #根据上面的得到关于WH1,WH2,When,Where的visual_input
        visual_inputs_WH1 = v_collate([d["img_WH1"] for d in batch])# (B, #frm=1 or T, 3, H, W)
        visual_inputs_WH2 = v_collate([d["img_WH2"] for d in batch])
        visual_inputs_When = v_collate([d["img_When"] for d in batch])
        visual_inputs_Where = v_collate([d["img_Where"] for d in batch])
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        text_examples_WH1 = flat_list_of_lists([d["examples_WH1"] for d in batch])
        text_examples_WH2 = flat_list_of_lists([d["examples_WH2"] for d in batch])
        #自己搞得，关于labels_Where和labels_When的
        labels_When = torch.stack([d["labels_When"] for d in batch])  # (B, )
        labels_Where = torch.stack([d["labels_Where"] for d in batch])  # (B, )

        n_examples_list = [d["n_examples"] for d in batch]  # (B, )

        # group elements data
        #Caption
        batch_enc = self.tokenizer.batch_encode_plus(
            [d["text_str"] for d in text_examples],
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_ids_no_mask = text_input_ids.clone()

        # 根据上面的获得关于WH1的caption
        batch_enc_WH1 = self.tokenizer.batch_encode_plus(
            [d["text_str_WH1"] for d in text_examples_WH1],
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        text_input_ids_WH1 = batch_enc_WH1.input_ids  # (B, L)
        text_input_ids_no_mask_WH1 = text_input_ids_WH1.clone()

        # 根据上面的获得关于WH2的caption
        batch_enc_WH2 = self.tokenizer.batch_encode_plus(
            [d["text_str_WH2"] for d in text_examples_WH2],
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        text_input_ids_WH2 = batch_enc_WH2.input_ids  # (B, L)
        text_input_ids_no_mask_WH2 = text_input_ids_WH2.clone()

        if self.mlm:
            text_input_ids, mlm_labels = mask_batch_text_tokens(
                text_input_ids, self.tokenizer,
                is_train=self.is_train)  # make mlm data
        else:
            text_input_ids, mlm_labels = text_input_ids, None
        
        text_input_mask = batch_enc.attention_mask  # (B, L)
        text_input_mask_WH1 = batch_enc_WH1.attention_mask
        text_input_mask_WH2 = batch_enc_WH2.attention_mask


        itm_labels = default_collate(
            [d["itm_label"] for d in text_examples])  # (B, )
        
        #这几句跟mpm有关erase_elems = [random_erase(e, patch_size=self.patch_size) for e in visual_inputs.clone()]

        # if self.mpm:
        #     crop_visual_inputs = v_collate([elems[0] for elems in erase_elems])
        #     mpm_masks = v_collate([elems[1] for elems in erase_elems])
        #     context_visual_inputs = v_collate([elems[2] for elems in erase_elems])

        #     return dict(
        #         crop_visual_inputs=crop_visual_inputs, # (B, #frm=1 or T, H, W, C)
        #         context_visual_inputs=context_visual_inputs,
        #         mpm_mask=mpm_masks,
        #         text_input_ids=text_input_ids_no_mask,
        #         mlm_text_input_ids=text_input_ids,
        #         mlm_labels=mlm_labels,
        #         text_input_mask=text_input_mask, # used to exclude [PAD] token
        #         itm_labels=itm_labels,
        #         n_examples_list=n_examples_list,  # used to create image feature copies.
        #         type=batch[0]['type']
        #     )
        # else:
        #     return dict(
        #         visual_inputs_WH1=visual_inputs_WH1, # (B, #frm=1 or T, H, W, C)
        #         visual_inputs_WH2=visual_inputs_WH2,
        #         visual_inputs_When=visual_inputs_When,
        #         visual_inputs_Where=visual_inputs_Where,
        #         text_input_ids=text_input_ids_no_mask,
        #         text_input_ids_WH1=text_input_ids_no_mask_WH1,
        #         text_input_ids_WH2=text_input_ids_no_mask_WH2,
        #         mlm_text_input_ids=text_input_ids,
        #         mlm_labels=mlm_labels,
        #         text_input_mask=text_input_mask, # used to exclude [PAD] token
        #         text_input_mask_WH1=text_input_mask_WH1,
        #         text_input_mask_WH2=text_input_mask_WH2,
        #         itm_labels=itm_labels,
        #         n_examples_list=n_examples_list,  # used to create image feature copies.
        #         labels_When=labels_When,
        #         labels_Where=labels_Where,
        #         type=batch[0]['type']
        #     )
        return dict(
            visual_inputs_WH1=visual_inputs_WH1,  # (B, #frm=1 or T, H, W, C)
            visual_inputs_WH2=visual_inputs_WH2,
            visual_inputs_When=visual_inputs_When,
            visual_inputs_Where=visual_inputs_Where,
            text_input_ids=text_input_ids_no_mask,
            text_input_ids_WH1=text_input_ids_no_mask_WH1,
            text_input_ids_WH2=text_input_ids_no_mask_WH2,
            mlm_text_input_ids=text_input_ids,
            mlm_labels=mlm_labels,
            text_input_mask=text_input_mask,  # used to exclude [PAD] token
            text_input_mask_WH1=text_input_mask_WH1,
            text_input_mask_WH2=text_input_mask_WH2,
            itm_labels=itm_labels,
            n_examples_list=n_examples_list,  # used to create image feature copies.
            labels_When=labels_When,
            labels_Where=labels_Where,
            type=batch[0]['type']
        )

def random_erase(input_img, patch_size, s_l=0.3, s_h=0.5, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    assert input_img.ndim == 4
    img_t, img_c, img_h, img_w = input_img.shape

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        w = w - w % patch_size
        h = h - h % patch_size

        left = left - left % patch_size
        top = top - top % patch_size

        if left + w <= img_w and top + h <= img_h:
            break

    context_img = input_img.clone()
    context_img[:, :, top: top + h, left: left + w] = 0

    input_img = input_img[:, :, top: top + h, left: left + w]
    pad = (left, img_w - left - w, top, img_h - top - h)
    input_img = torch.nn.functional.pad(input_img, pad=pad, mode='constant', value=0.0)

    img_masks = torch.ones_like(input_img)
    img_masks[:, :, top: top+h, left: left+w] = 0

    img_masks = torch.nn.functional.avg_pool2d(img_masks.float(), kernel_size=(patch_size, patch_size), stride=patch_size)
    img_masks = torch.mean(img_masks, dim=(0, 1))

    return input_img, img_masks, context_img