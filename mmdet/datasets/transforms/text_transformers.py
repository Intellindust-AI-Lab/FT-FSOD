# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import cv2
from PIL import Image

from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes

try:
    from transformers import AutoTokenizer
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

import random
import re

import numpy as np
import torch


def clean_name(name):
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    name = name.lower()
    return name


def check_for_positive_overflow(gt_bboxes, gt_labels, text, tokenizer,
                                max_tokens):
    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    positive_label_list = np.unique(gt_labels).tolist()
    # random shuffule so we can sample different annotations
    # at different epochs
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):

        label_text = clean_name(text[str(label)]) + '. '

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_tokens:
            break
        else:
            kept_lables.append(label)

    keep_box_index = []
    keep_gt_labels = []
    for i in range(len(gt_labels)):
        if gt_labels[i] in kept_lables:
            keep_box_index.append(i)
            keep_gt_labels.append(gt_labels[i])

    return gt_bboxes[keep_box_index], np.array(
        keep_gt_labels, dtype=np.int64), length


def generate_senetence_given_labels(positive_label_list, negative_label_list,
                                    text):
    label_to_positions = {}

    label_list = negative_label_list + positive_label_list

    random.shuffle(label_list)

    pheso_caption = ''

    label_remap_dict = {}
    for index, label in enumerate(label_list):

        start_index = len(pheso_caption)

        pheso_caption += clean_name(text[str(label)])

        end_index = len(pheso_caption)

        if label in positive_label_list:
            label_to_positions[index] = [[start_index, end_index]]
            label_remap_dict[int(label)] = index

        # if index != len(label_list) - 1:
        #     pheso_caption += '. '
        pheso_caption += '. '

    return label_to_positions, pheso_caption, label_remap_dict


@TRANSFORMS.register_module()
class RandomSamplingNegPos(BaseTransform):

    def __init__(self,
                 tokenizer_name,
                 num_sample_negative=85,
                 max_tokens=256,
                 full_sampling_prob=0.5,
                 label_map_file=None,
                 debug_save_dir=None):
        print('called!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_sample_negative = num_sample_negative
        self.full_sampling_prob = full_sampling_prob
        self.max_tokens = max_tokens
        self.label_map = None

        debug_save_whole_dir = 'your_debug_save_dir'
        self.debug_save_dir = debug_save_dir
        if self.debug_save_dir is not None:
            self.debug_save_dir = os.path.join(debug_save_whole_dir, self.debug_save_dir)
            os.makedirs(self.debug_save_dir, exist_ok=True)
        self.debug_save_prob = 0.1

        if label_map_file:
            with open(label_map_file, 'r') as file:
                self.label_map = json.load(file)

    def _save_debug_image(self, results):
        """随机保存图片用于调试数据增强效果"""
        if not self.debug_save_dir or random.random() > self.debug_save_prob:
            return
        
        try:
            # 获取图片数据
            if 'img' in results:
                img = results['img']
                if isinstance(img, np.ndarray):
                    # 如果是numpy数组，转换为PIL图片
                    if img.dtype == np.uint8:
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            # BGR to RGB for PIL
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(img_rgb)
                        else:
                            pil_img = Image.fromarray(img)
                    else:
                        # 归一化到0-255
                        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_normalized)
                else:
                    return  # 无法处理的图片格式
                
                # 生成文件名
                import time
                timestamp = int(time.time() * 1000000)  # 微秒级时间戳
                filename = f"debug_aug_{timestamp}.jpg"
                filepath = os.path.join(self.debug_save_dir, filename)
                
                # 保存图片
                pil_img.save(filepath, 'JPEG', quality=95)
                
                # 保存对应的标注信息
                info_file = filepath.replace('.jpg', '_info.txt')
                with open(info_file, 'w', encoding='utf-8') as f:
                    f.write(f"Text: {results.get('text', 'N/A')}\n")
                    f.write(f"GT Labels: {results.get('gt_bboxes_labels', 'N/A')}\n")
                    f.write(f"GT Bboxes shape: {getattr(results.get('gt_bboxes', []), 'shape', 'N/A')}\n")
                    f.write(f"Tokens Positive: {results.get('tokens_positive', 'N/A')}\n")
                
                print(f"Debug image saved: {filepath}")
        except Exception as e:
            print(f"Failed to save debug image: {e}")
    
    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            return self.vg_aug(results)
        else:
            result = self.od_aug(results)
            # 保存调试图片
            self._save_debug_image(result)
            return result

    def vg_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text'].lower().strip()
        if not text.endswith('.'):
            text = text + '. '

        phrases = results['phrases']
        # TODO: add neg
        positive_label_list = np.unique(gt_labels).tolist()
        label_to_positions = {}
        for label in positive_label_list:
            label_to_positions[label] = phrases[label]['tokens_positive']

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = text
        results['tokens_positive'] = label_to_positions
        return results

    def od_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']

        if 'text' not in results:
            assert self.label_map is not None
            text = self.label_map
        else:
            text = results['text']

        original_box_num = len(gt_labels)
        # If the category name is in the format of 'a/b' (in object365),
        # we randomly select one of them.
        if isinstance(text, (tuple, list)):
                text = {str(i): class_name for i, class_name in enumerate(text)}
        # else:
        for key, value in text.items():
            if '/' in value:
                text[key] = random.choice(value.split('/')).strip()

        gt_bboxes, gt_labels, positive_caption_length = \
            check_for_positive_overflow(gt_bboxes, gt_labels,
                                        text, self.tokenizer, self.max_tokens)

        if len(gt_bboxes) < original_box_num:
            print('WARNING: removed {} boxes due to positive caption overflow'.
                  format(original_box_num - len(gt_bboxes)))

        valid_negative_indexes = list(text.keys())

        positive_label_list = np.unique(gt_labels).tolist()
        full_negative = self.num_sample_negative

        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        outer_prob = random.random()

        if outer_prob < self.full_sampling_prob:
            # c. probability_full: add both all positive and all negatives
            num_negatives = full_negative
        else:
            if random.random() < 1.0:
                num_negatives = np.random.choice(max(1, full_negative)) + 1
            else:
                num_negatives = full_negative

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)

            for i in np.random.choice(
                    valid_negative_indexes, size=num_negatives, replace=False):
                if int(i) not in positive_label_list:
                    negative_label_list.add(i)

        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)
        random.shuffle(negative_label_list)

        negative_max_length = self.max_tokens - positive_caption_length
        screened_negative_label_list = []

        for negative_label in negative_label_list:
            label_text = clean_name(text[str(negative_label)]) + '. '

            tokenized = self.tokenizer.tokenize(label_text)

            negative_max_length -= len(tokenized)

            if negative_max_length > 0:
                screened_negative_label_list.append(negative_label)
            else:
                break
        negative_label_list = screened_negative_label_list
        label_to_positions, pheso_caption, label_remap_dict = \
            generate_senetence_given_labels(positive_label_list,
                                            negative_label_list, text)

        # label remap
        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions

        return results


@TRANSFORMS.register_module()
class GenRandomSamplingNegPos(BaseTransform):
    """Generate random sampling negative and positive samples using O365 classes and dataset classes.
    
    This transform creates negative samples from two sources:
    1. O365 vocabulary (365 classes) - prioritized for diversity
    2. Dataset classes - used as fallback when O365 samples are insufficient
    
    Both positive and negative samples receive real labels and can participate in training
    for contrastive learning, which helps improve model performance.
    
    Features:
    - Intelligent negative sampling from O365 vocabulary
    - Fallback to dataset classes when needed
    - Debug image saving for monitoring data augmentation
    - Real labels for all samples (no fake labels)
    - Token length constraints to fit within model limits
    """
    
    # Object365数据集的365个类别
    O365_CLASSES = [
        # 人物与服饰
        "Person", "Sneakers", "Hat", "Glasses", "Gloves", "Boots", "Belt", "Tie", "Slippers", 
        "Sandals", "High Heels", "Skating and Skiing shoes", "Bow Tie", "Mask",
        
        # 家具与室内物品
        "Chair", "Desk", "Cabinet/shelf", "Bench", "Couch", "Stool", "Bed", "Nightstand",
        "Coffee Table", "Side Table", "Dining Table",
        
        # 电器与设备
        "Lamp", "Monitor.TV", "Speaker", "Air Conditioner", "Refrigerator", "Washing Machine.Drying Machine",
        "Microwave", "Computer Box", "Router.modem",
        
        # 交通工具
        "Car", "SUV", "Van", "Bus", "Motorcycle", "Bicycle", "Truck", "Pickup Truck", "Sports Car",
        "Heavy Truck", "Scooter", "Train", "Airplane", "Boat", "Sailboat",
        
        # 厨房用品
        "Bottle", "Cup", "Plate", "Bowl.Basin", "Wine Glass", "Tea pot", "Pot", "Fork", "Spoon",
        "Knife", "Chopsticks",
        
        # 电子产品
        "Cell Phone", "Camera", "Laptop", "Remote", "Head Phone", "Telephone", "Tablet",
        
        # 食物与饮品
        "Bread", "Cake", "Pizza", "Hot dog", "Hamburger", "French Fries", "Rice", "Pasta",
        
        # 运动用品
        "Baseball Glove", "Soccer", "Basketball", "Volleyball", "Tennis Racket", "Golf Club",
        "Hockey Stick", "Baseball Bat",
        
        # 动物
        "Dog", "Cat", "Horse", "Cow", "Sheep", "Bird", "Duck", "Zebra", "Giraffe", "Elephant",
        
        # 植物
        "Flower", "Potted Plant", "Tree",
        
        # 其余物品按字母顺序排列
        "Awning", "Backpack", "Balloon", "Barrel.bucket", "Baseball", "Basket", "Bathtub",
        "Blackboard.Whiteboard", "Bracelet", "Briefcase", "Broom", "Brush", "Candle",
        "Canned", "Carpet", "CD", "Clock", "Comb", "Cosmetics", "Crane", "Cymbal",
        "Dolphin", "Drum", "Dumbbell", "Eraser", "Extension Cord", "Faucet", "Fire Hydrant",
        "Flag", "Frisbee", "Globe", "Guitar", "Hammer", "Hanger", "Handbag.Satchel",
        "Helmet", "Keyboard", "Kite", "Ladder", "Lantern", "Lifesaver", "Luggage",
        "Medal", "Microphone", "Mirror", "Mouse", "Necklace", "Other Shoes", "Paddle",
        "Pen.Pencil", "Picture.Frame", "Pillow", "Power outlet", "Ring", "Scissors",
        "Sink", "Storage box", "Stuffed Toy", "Sushi", "Tape", "Tennis", "Tent",
        "Toilet", "Toilet Paper", "Towel", "Traffic Light", "Traffic Sign", "Traffic cone",
        "Trash bin Can", "Tripod", "Umbrella", "Vase", "Watch"
    ]

    def __init__(self,
                 tokenizer_name,
                 num_sample_negative=10,
                 max_tokens=256,
                 full_sampling_prob=0.5,
                 label_map_file=None,
                 use_coco_negatives=True,
                 save_debug_images=True,
                 debug_save_dir='./debug_augmented_images',
                 debug_save_prob=0.1,
                 max_label_index=None):
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_sample_negative = num_sample_negative
        self.full_sampling_prob = full_sampling_prob
        self.max_tokens = max_tokens
        self.use_coco_negatives = use_coco_negatives
        self.max_label_index = max_label_index  # Maximum allowed label index to prevent CUDA errors
        self.label_map = None
        if label_map_file:
            with open(label_map_file, 'r') as file:
                self.label_map = json.load(file)
        
        # Debug image saving parameters
        self.save_debug_images = save_debug_images
        self.debug_save_dir = debug_save_dir
        self.debug_save_prob = debug_save_prob
        if self.save_debug_images:
            os.makedirs(self.debug_save_dir, exist_ok=True)

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            return self.vg_aug(results)
        else:
            return self.od_aug(results)

    def vg_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text'].lower().strip()
        if not text.endswith('.'):
            text = text + '. '

        phrases = results['phrases']
        positive_label_list = np.unique(gt_labels).tolist()
        label_to_positions = {}
        for label in positive_label_list:
            label_to_positions[label] = phrases[label]['tokens_positive']

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels
        results['text'] = text
        results['tokens_positive'] = label_to_positions
        return results

    def od_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']


        if 'text' not in results:
            assert self.label_map is not None
            text = self.label_map
        else:
            text = results['text']
            # Convert tuple/list to dict if needed
            if isinstance(text, (tuple, list)):
                text = {str(i): class_name for i, class_name in enumerate(text)}

        original_box_num = len(gt_labels)
        # If the category name is in the format of 'a/b' (in object365),
        # we randomly select one of them.
        for key, value in text.items():
            if '/' in value:
                text[key] = random.choice(value.split('/')).strip()

        gt_bboxes, gt_labels, positive_caption_length = \
            check_for_positive_overflow(gt_bboxes, gt_labels,
                                        text, self.tokenizer, self.max_tokens)

        if len(gt_bboxes) < original_box_num:
            print('WARNING: removed {} boxes due to positive caption overflow'.
                  format(original_box_num - len(gt_bboxes)))

        positive_label_list = np.unique(gt_labels).tolist()
        
        # Get positive class names
        positive_class_names = set()
        for label in positive_label_list:
            class_name = clean_name(text[str(label)]).lower()
            positive_class_names.add(class_name)
        
        # Generate negative samples from O365 classes and dataset classes
        negative_label_list = []
        if self.use_coco_negatives:
            # Filter out O365 classes that are already in positive samples
            # Also avoid classes that have substring relationships
            available_o365_negatives = []
            for cls in self.O365_CLASSES:
                cls_lower = cls.lower()
                # Check if this O365 class conflicts with any positive class
                conflict = False
                for pos_name in positive_class_names:
                    pos_lower = pos_name.lower()
                    # Skip if exact match or if either word contains the other
                    if (cls_lower == pos_lower or 
                        cls_lower in pos_lower or 
                        pos_lower in cls_lower):
                        conflict = True
                        break
                if not conflict:
                    available_o365_negatives.append(cls)
            
            # Also get available negatives from dataset classes
            available_dataset_negatives = []
            for key in text.keys():
                if int(key) not in positive_label_list:
                    available_dataset_negatives.append(key)
            
            # Combine O365 negatives and dataset negatives
            all_available_negatives = []
            
            # Add O365 negatives (prioritized)
            for cls in available_o365_negatives:
                all_available_negatives.append(('o365', cls))
            
            # Add dataset negatives as fallback
            for key in available_dataset_negatives:
                all_available_negatives.append(('dataset', key))
            
            # Randomly sample negative classes
            num_negatives = min(self.num_sample_negative, len(all_available_negatives))
            if num_negatives > 0:
                selected_negatives = random.sample(all_available_negatives, num_negatives)
                
                # Check token length constraints
                negative_max_length = self.max_tokens - positive_caption_length
                
                for neg_type, neg_item in selected_negatives:
                    if neg_type == 'o365':
                        label_text = clean_name(neg_item) + '. '
                        neg_class_name = neg_item
                    else:  # dataset
                        label_text = clean_name(text[str(neg_item)]) + '. '
                        neg_class_name = text[str(neg_item)]
                    
                    tokenized = self.tokenizer.tokenize(label_text)
                    negative_max_length -= len(tokenized)
                    
                    if negative_max_length > 0:
                        if neg_type == 'o365':
                            negative_label_list.append(neg_class_name)
                        else:
                            negative_label_list.append(neg_item)  # dataset key
                    else:
                        break
                        

        else:
            # Use original logic for negative sampling
            valid_negative_indexes = list(text.keys())
            full_negative = self.num_sample_negative
            
            if full_negative > len(valid_negative_indexes):
                full_negative = len(valid_negative_indexes)
            
            outer_prob = random.random()
            if outer_prob < self.full_sampling_prob:
                num_negatives = full_negative
            else:
                if random.random() < 1.0:
                    num_negatives = np.random.choice(max(1, full_negative)) + 1
                else:
                    num_negatives = full_negative
            
            negative_label_set = set()
            if num_negatives != -1:
                if num_negatives > len(valid_negative_indexes):
                    num_negatives = len(valid_negative_indexes)
                
                for i in np.random.choice(
                        valid_negative_indexes, size=num_negatives, replace=False):
                    if int(i) not in positive_label_list:
                        negative_label_set.add(i)
            
            negative_label_list = list(negative_label_set)
            random.shuffle(negative_label_list)
            
            negative_max_length = self.max_tokens - positive_caption_length
            screened_negative_label_list = []
            
            for negative_label in negative_label_list:
                label_text = clean_name(text[str(negative_label)]) + '. '
                tokenized = self.tokenizer.tokenize(label_text)
                negative_max_length -= len(tokenized)
                
                if negative_max_length > 0:
                    screened_negative_label_list.append(negative_label)
                else:
                    break
            negative_label_list = screened_negative_label_list
        


        # Generate caption with O365 negatives
        if self.use_coco_negatives:
            caption_result = self._generate_caption_with_o365_negatives(
                positive_label_list, negative_label_list, text)
            label_to_positions = caption_result['label_to_positions']
            pheso_caption = caption_result['pheso_caption']
            label_remap_dict = caption_result['label_remap_dict']
            
            # 添加负样本信息供调试使用
            if 'neg_labels_info' in caption_result:
                results['neg_labels_info'] = caption_result['neg_labels_info']

        else:
            label_to_positions, pheso_caption, label_remap_dict = \
                generate_senetence_given_labels(positive_label_list,
                                                negative_label_list, text)
         
        # Label remap with safety check
        if len(gt_labels) > 0:
            # Check if all gt_labels exist in label_remap_dict
            missing_labels = []
            for label in gt_labels:
                if int(label) not in label_remap_dict:
                    missing_labels.append(label)
            
            if missing_labels:
                print(f'WARNING: Missing labels in remap dict: {missing_labels}')
                print(f'Available labels in remap dict: {list(label_remap_dict.keys())}')
                print(f'Original gt_labels: {gt_labels.tolist()}')
                # Filter out missing labels
                valid_indices = [i for i, label in enumerate(gt_labels) if int(label) in label_remap_dict]
                if len(valid_indices) > 0:
                    gt_bboxes = gt_bboxes[valid_indices]
                    gt_labels = gt_labels[valid_indices]
                    gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)
                else:
                    # No valid labels, create empty arrays with proper device
                    if hasattr(gt_bboxes, 'device'):
                        device = gt_bboxes.device
                        gt_bboxes = torch.empty((0, 4), dtype=gt_bboxes.dtype, device=device)
                    else:
                        gt_bboxes = gt_bboxes[:0]  # Empty tensor with correct shape
                    gt_labels = np.array([], dtype=np.int64)
                    print('WARNING: No valid labels after filtering, created empty arrays')
            else:
                gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)
        else:
            # Handle case where gt_labels is already empty
            print('INFO: gt_labels is empty, skipping label remapping')
        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels
        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions

        # 随机保存调试图像
        if self.save_debug_images and random.random() < self.debug_save_prob:
            self._save_debug_image(results)



        return results
    
    def _save_debug_image(self, results):
        """随机保存增强后的图像用于调试"""
        try:
            import time
            timestamp = int(time.time() * 1000000)  # 微秒时间戳
            
            # 保存图像
            img = results.get('img', None)
            if img is not None:
                # 处理不同的图像数据格式
                if isinstance(img, np.ndarray):
                    if img.dtype == np.uint8:
                        pil_img = Image.fromarray(img)
                    else:
                        # 归一化到0-255范围
                        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_normalized)
                else:
                    print(f'WARNING: Unsupported image type: {type(img)}')
                    return
                
                # 保存图像文件
                img_filename = f'debug_aug_{timestamp}.jpg'
                img_path = os.path.join(self.debug_save_dir, img_filename)
                pil_img.save(img_path)
                
                # 保存标注信息
                info_filename = f'debug_aug_{timestamp}.txt'
                info_path = os.path.join(self.debug_save_dir, info_filename)
                
                with open(info_path, 'w', encoding='utf-8') as f:
                    f.write(f'Debug Image Info - {timestamp}\n')
                    f.write('=' * 50 + '\n')
                    f.write(f'Text: {results.get("text", "N/A")}\n')
                    f.write(f'GT Labels: {results.get("gt_bboxes_labels", "N/A")}\n')
                    f.write(f'GT Bboxes Shape: {results.get("gt_bboxes", torch.tensor([])).shape}\n')
                    f.write(f'Tokens Positive: {results.get("tokens_positive", "N/A")}\n')
                    
                    # 添加负样本信息
                    if 'neg_labels_info' in results:
                        f.write('\nNegative Labels Info:\n')
                        for neg_info in results['neg_labels_info']:
                            f.write(f'  - Index: {neg_info["index"]}, '
                                   f'Class: {neg_info["class_name"]}, '
                                   f'Source: {neg_info["source"]}\n')
                    
                    f.write(f'\nImage saved as: {img_filename}\n')
                
                print(f'DEBUG: Saved debug image and info to {self.debug_save_dir}')
                
        except Exception as e:
            print(f'WARNING: Failed to save debug image: {e}')
    
    def _generate_caption_with_o365_negatives(self, positive_label_list, 
                                            negative_class_names, text):
        """Generate caption with O365 negative class names.
        
        This method creates a caption that includes both positive labels (from the dataset)
        and negative labels (from O365 vocabulary). Both positive and negative samples
        get real labels and can participate in training for contrastive learning.
        
        The key improvement: limit label indices to stay within the original num_classes
        range to prevent CUDA errors in CdnQueryGenerator's label_embedding layer.
        """
        label_to_positions = {}
        
        # Combine positive labels and negative class names
        all_labels = []
        
        # Add positive labels - ensure we have all positive labels
        for label in positive_label_list:
            if str(label) in text:  # Safety check
                all_labels.append(('positive', label, clean_name(text[str(label)])))
            else:
                print(f'WARNING: Label {label} not found in text dict, skipping')
        
        # Add negative class names from O365 vocabulary and dataset classes
        for neg_item in negative_class_names:
            if isinstance(neg_item, str):
                # O365 class name
                all_labels.append(('negative_o365', neg_item, clean_name(neg_item)))
            else:
                # Dataset class key
                class_name = text[str(neg_item)] if str(neg_item) in text else f'class_{neg_item}'
                all_labels.append(('negative_dataset', neg_item, clean_name(class_name)))
        
        # Shuffle the combined list to randomize order
        random.shuffle(all_labels)
        
        pheso_caption = ''
        label_remap_dict = {}
        neg_labels_info = []  # 存储负样本标签信息供调试使用
        
        # Determine the maximum allowed label index
        if self.max_label_index is not None:
            max_allowed_index = self.max_label_index - 1  # Convert to 0-based index
        else:
            # Fallback: use the number of dataset classes as limit
            max_allowed_index = len(text) - 1
        
        # Track used indices to avoid conflicts
        used_indices = set()
        next_available_index = 0
        
        for caption_index, (label_type, original_label, class_name) in enumerate(all_labels):
            start_index = len(pheso_caption)
            pheso_caption += class_name
            end_index = len(pheso_caption)
            
            # Assign label index based on type and constraints
            if label_type == 'positive':
                # For positive samples, try to use original label if within range
                if int(original_label) <= max_allowed_index:
                    assigned_index = int(original_label)
                else:
                    # Find next available index within range
                    while next_available_index in used_indices and next_available_index <= max_allowed_index:
                        next_available_index += 1
                    if next_available_index <= max_allowed_index:
                        assigned_index = next_available_index
                        next_available_index += 1
                    else:
                        # Reuse an existing index (fallback)
                        assigned_index = 0
                        print(f'WARNING: Reusing index 0 for positive label {original_label}')
                
                label_remap_dict[int(original_label)] = assigned_index
            else:
                # For negative samples, assign next available index within range
                while next_available_index in used_indices and next_available_index <= max_allowed_index:
                    next_available_index += 1
                if next_available_index <= max_allowed_index:
                    assigned_index = next_available_index
                    next_available_index += 1
                else:
                    # Reuse existing indices cyclically when we run out of space
                    assigned_index = len(used_indices) % (max_allowed_index + 1)
                    print(f'WARNING: Reusing index {assigned_index} for negative sample {class_name}')
                
                # Record negative sample info
                neg_info = {
                    'index': assigned_index,
                    'class_name': class_name,
                    'start_pos': start_index,
                    'end_pos': end_index,
                    'is_negative': True,
                    'source': 'o365' if label_type == 'negative_o365' else 'dataset'
                }
                if label_type == 'negative_dataset':
                    neg_info['original_key'] = original_label
                neg_labels_info.append(neg_info)
            
            # Record the assigned index
            used_indices.add(assigned_index)
            label_to_positions[assigned_index] = [[start_index, end_index]]
            
            pheso_caption += '. '
        
        # Ensure all original positive labels have mappings
        for label in positive_label_list:
            if int(label) not in label_remap_dict:
                print(f'WARNING: Creating fallback mapping for missing label {label}')
                # Create a fallback mapping to index 0 or first available index
                if len(label_remap_dict) > 0:
                    label_remap_dict[int(label)] = min(label_remap_dict.values())
                else:
                    label_remap_dict[int(label)] = 0
        
        result_dict = {
            'label_to_positions': label_to_positions,
            'pheso_caption': pheso_caption,
            'label_remap_dict': label_remap_dict
        }
        
        # 添加负样本信息供调试和监控使用
        if neg_labels_info:
            result_dict['neg_labels_info'] = neg_labels_info
        
        return result_dict


@TRANSFORMS.register_module()
class LoadTextAnnotations(BaseTransform):

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            tokens_positive = [
                phrase['tokens_positive']
                for phrase in results['phrases'].values()
            ]
            results['tokens_positive'] = tokens_positive
        else:
            text = results['text']
            results['text'] = list(text.values())
        return results
