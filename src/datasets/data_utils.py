import random
import sys

import torch
import torchvision.transforms as transforms
from torch.nn.functional import interpolate as img_tensor_resize
from torch.nn.functional import pad as img_tensor_pad
from torch.nn.modules.utils import _quadruple
from torchvision.transforms.functional import pad as img_pad
from torchvision.transforms.functional import resize as img_resize
from src.utils.basic_utils import flat_list_of_lists
import numbers
import numpy as np
from PIL import Image
_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def mask_batch_text_tokens(
        inputs, tokenizer, mlm_probability=0.15, is_train=True):
    """ modified from transformers.data.data_collator
    Args:
        inputs: (B, L), 2D torch.Tensor, does not work for 1D. It has already been padded.
        tokenizer:
        mlm_probability: float
        is_train: if True use random masking, else mask tokens at fixed position to remove randomness in evaluation.
    """
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(
            val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(
        special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(
        torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(
        torch.full(labels.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape,
        dtype=torch.long)  # len(tokenizer) == #vocab
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def select_batch_text_pivots(
        inputs, tokenizer, ent2id, mpm_probability=1.0, is_train=True):
    """ Given a input text sequence, generate masks and prototype labels such that:
    1) not to mask special token ([CLS], [SEP], [MASK], [PAD]);
    2) always mask all BPE in a word together.

    Args:
    """
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for as pivots
    probability_matrix = torch.full(labels.shape, mpm_probability)
    # ignore [CLS] [SEP] [MASK] tokens
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(
            val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    # ignore [PAD] tokens
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    
    # create masking indices
    pivot_indices = torch.bernoulli(probability_matrix).bool()
    labels[special_tokens_mask] = -100  # We only compute loss on masked tokens
    labels[~pivot_indices] = -100  # We only compute loss on masked tokens

    # selected pivot positions: (1) non-special token; (2) selected based on mpm probability. 
    text_pivots_pos = (labels > 0).nonzero()
    
    for tpp in text_pivots_pos:

        orig_tpp = tpp.clone()
        
        bth = tpp[0]
        orig_text_pos = orig_tpp[1]

        text_token = tokenizer.convert_ids_to_tokens([inputs[bth][tpp[1]]])[0]
        next_text_token = tokenizer.convert_ids_to_tokens([inputs[bth][tpp[1]+1]])[0] if tpp[1]+1 < inputs.shape[1] else None

        # TODO may consider support encoding beyond sentencepiece. 
        if text_token.startswith('##'):
            # if it is a byte pair, backtrace until we find the prefix
            orig_text_token = ''

            while True:
                if not text_token.startswith('##'):
                    orig_text_token = text_token + orig_text_token

                    break
                else:
                    orig_text_token = text_token[2:] + orig_text_token

                    tpp[1] -= 1

                text_token = tokenizer.convert_ids_to_tokens([inputs[bth][tpp[1]]])[0]
            
            try:
                # assign prototype labels to all the sentencepiece bytes in the pivot word
                labels[bth][tpp[1]: orig_text_pos + 1] = ent2id[orig_text_token]
                pivot_indices[bth][tpp[1]: orig_text_pos + 1] = True
            except KeyError:
                # we do not have this word for prototype
                labels[bth][orig_text_pos] = -100

        elif next_text_token is not None and next_text_token.startswith('##'):
            # if it is a prefix, forward-trace until we find the end of the byte pair
            full_text_token = text_token 

            while True:
                tpp[1] += 1
                text_token = tokenizer.convert_ids_to_tokens([inputs[bth][tpp[1]]])[0]

                if not text_token.startswith('##'):
                    # find the next prefix/word
                    break
                else:
                    # find continuing bytes
                    full_text_token = full_text_token + text_token[2:]

            try:
                # assign prototype labels to all the sentencepiece bytes in the pivot word
                labels[bth][orig_text_pos: tpp[1]] = ent2id[full_text_token]
                pivot_indices[bth][orig_text_pos: tpp[1]] = True
            except KeyError:
                # we do not have this word for prototype
                labels[bth][orig_text_pos] = -100

        else:
            # the word is treated in whole by BERT tokenizer
            try:
                labels[bth][tpp[1]] = ent2id[text_token]
            except KeyError:
                # we do not have this word for prototype
                labels[bth][tpp[1]] = -100

    # restore mask if the word is not in the entity list
    pivot_indices[labels==-100] = False

    return pivot_indices, labels


def image_to_tensor(image: np.ndarray, keepdim: bool = True) -> torch.Tensor:
    """Converts a numpy image to a PyTorch 4d tensor image.
    Args:
        image (numpy.ndarray): image of the form :math:`(H, W, C)`, :math:`(H, W)` or
            :math:`(B, H, W, C)`.
        keepdim (bool): If ``False`` unsqueeze the input image to match the shape
            :math:`(B, H, W, C)`. Default: ``True``
    Returns:
        torch.Tensor: tensor of the form :math:`(B, C, H, W)` if keepdim is ``False``,
            :math:`(C, H, W)` otherwise.
    """
    if not isinstance(image, (np.ndarray,)):
        raise TypeError("Input type must be a numpy.ndarray. Got {}".format(
            type(image)))

    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional array")

    input_shape = image.shape
    tensor: torch.Tensor = torch.from_numpy(image)

    if len(input_shape) == 2:
        # (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)
    elif len(input_shape) == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    elif len(input_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        keepdim = True  # no need to unsqueeze
    else:
        raise ValueError(
            "Cannot process image with shape {}".format(input_shape))

    return tensor.unsqueeze(0) if not keepdim else tensor


def get_padding(image, max_w, max_h, pad_all=False):
    # keep the images to upper-left corner
    if isinstance(image, torch.Tensor):
        h, w = image.shape[-2:]
    else:
        w, h = image.size
    h_padding, v_padding = max_w - w, max_h - h
    if pad_all:
        h_padding /= 2
        v_padding /= 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    else:
        l_pad, t_pad = 0, 0
        r_pad, b_pad = h_padding, v_padding
    if isinstance(image, torch.Tensor):
        padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))
    else:
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class ImagePad(object):
    def __init__(self, max_w, max_h, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.max_w = max_w
        self.max_h = max_h
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        if isinstance(img, torch.Tensor):
            paddings = _quadruple(get_padding(img, self.max_w, self.max_h))
            return img_tensor_pad(
                img, paddings,
                self.padding_mode, self.fill)
        return img_pad(
            img, get_padding(img, self.max_w, self.max_h),
            self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


def get_resize_size(image, max_size):
    """
    Args:
        image: PIL Image or torch.tensor
        max_size:

    Returns:

    Note the height/width order difference
    >>> pil_img = Image.open("raw_img_tensor.jpg")
    >>> pil_img.size
    (640, 480)  # (width, height)
    >>> np_img = np.array(pil_img)
    >>> np_img.shape
    (480, 640, 3)  # (height, width, 3)
    """
    # note the order of height and width for different inputs
    if isinstance(image, torch.Tensor):
        # width, height = image.shape[-2:]
        height, width = image.shape[-2:]
    else:
        width, height = image.size

    if height >= width:
        ratio = width*1./height
        new_height = max_size
        new_width = new_height * ratio
    else:
        ratio = height*1./width
        new_width = max_size
        new_height = new_width * ratio
    size = (int(new_height), int(new_width))
    return size

class VideoRandomSquareCrop(object):
    def __init__(self, crop_size, p=0.5):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        self.p = p

    def __call__(self, video):
        """
        Args:
            img (torch.tensor): video to be cropped.

        Returns:
            torch.tensor: cropped video.
        """
        if isinstance(video, torch.Tensor):
            if len(video.shape) == 4:
                b, t, h, w = video.shape
            else:
                raise RuntimeError('Expecting 4-dimensional tensor of shape (b,t,h,w), got {}'.format(video.shape))

            # if random.uniform(0, 1) < self.p:
            #     video = torch.flip(video, (3,))

            x = random.randint(0, h - self.crop_size)
            y = random.randint(0, w - self.crop_size)

            return video[:, :, x: x + self.crop_size, y: y + self.crop_size]

        else:
            raise NotImplementedError('Support only torch.Tensor as input, got {}'.format(type(video)))


class VideoResizeSquare(object):
    def __init__(self, out_size, interpolation='nearest'):
        assert isinstance(out_size, int)
        self.out_size = out_size
        self.interpolation = interpolation

    def __call__(self, video):
        """
        Args:
            img (torch.tensor): video to be scaled.

        Returns:
            torch.tensor: Rescaled video.
        """
        if isinstance(video, torch.Tensor):
            if len(video.shape) == 4:
                t, c, h, w = video.shape
                assert c == 3, 'Expecting 3-channel color video, got video of shape {}'.format(video.shape)
            else:
                raise RuntimeError('Expecting 4-dimensional tensor of shape (b,t,h,w), got {}'.format(video.shape))

            short_side = h if h < w else w
            # scaling_factor = self.out_size / short_side

            # new_h = int(h * scaling_factor)
            # new_w = int(w * scaling_factor)

            resized_video = img_tensor_resize(video, size=((self.out_size, self.out_size)), mode=self.interpolation)
            
            return resized_video


        else:
            # in other data class, the order of shape might be different.
            raise NotImplementedError('Support only torch.Tensor as input, got {}'.format(type(video)))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.out_size, self.interpolation)


class ImageResize(object):
    """Resize the input image (torch.tensor) to the given size.

    Args:
        max_size (int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, max_size, interpolation=Image.BILINEAR):
        assert isinstance(max_size, int)
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (torch.tensor): Image to be scaled.

        Returns:
            torch.tensor: Rescaled image.
        """
        if isinstance(img, torch.Tensor):
            assert isinstance(self.interpolation, str)
            return img_tensor_resize(
                img, size=get_resize_size(img, self.max_size),
                mode=self.interpolation, align_corners=False)
        return img_resize(
            img, get_resize_size(img, self.max_size), self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


def get_imagenet_transform(min_size=600, max_size=1000):
    """parameters from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    This simply crop the center square from the image
    """
    if min_size != 600:
        import warnings
        warnings.warn(f'Warning: min_size is not used in image transform, '
                      f'setting min_size will have no effect.')
    return transforms.Compose([
        ImageResize(max_size, Image.BILINEAR),  # longer side will be resized to 1000
        ImagePad(max_size, max_size),  # pad to 1000 * 1000
    ])


class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).cuda().view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 1, 3, 1, 1)
        # assert max(std) <= 1 and min(std) >= 0\
        #     or max(mean) <= 1 and min(mean) >= 0,\
        #         "Please provide mean or std within range [0, 1]"

    def __call__(self, img):
        """
        Args:
            img: float image tensors, (B, N, 3, H, W)

        Returns:
            img: normalized float image tensors
        """
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)


def chunk_list(examples, chunk_size=2, pad_to_divisible=True):
    """
    Args:
        examples: iterable, examples grouped by image/video
        chunk_size: int, number of examples in each chunk.
        pad_to_divisible: bool, pad the examples to be divisible by chunk_size.
    >>> test_examples = [3, 4, 5, 6, 7]
    >>> chunk_list(test_examples, chunk_size=2, pad_to_divisible=True)
    [[3, 4], [5, 6], [7, 7]]  # the lst element has some randomness
    >>> chunk_list(test_examples, chunk_size=2, pad_to_divisible=False)
    [[3, 4], [5, 6], [7]]
    """
    n_examples = len(examples)
    remainder = n_examples % chunk_size
    if pad_to_divisible and remainder > 0:
        n_pad = chunk_size - remainder
        pad = random.choices(examples, k=n_pad)  # with replacement
        examples = examples + pad
        n_examples = len(examples)
        remainder = 0
    chunked_examples = []
    n_chunks = int(n_examples / chunk_size)
    n_chunks = n_chunks + 1 if remainder > 0 else n_chunks
    for i in range(n_chunks):
        chunked_examples.append(examples[i*chunk_size: (i+1)*chunk_size])
    return chunked_examples


def mk_input_group(key_grouped_examples, max_n_example_per_group=1, is_train=True,
                   example_unique_key=None):
    """ Re-organize examples into groups. Each input group will have a single image paired
    with X (X=max_n_example_per_img) examples. Images with total #examples > X will be
    split into multiple groups. In the case a group has < X examples, we will copy
    the examples to make the group has X examples.
    Args:
        key_grouped_examples: dict, each key is image/video id,
            each value is a list(example) associated with this image/video
        max_n_example_per_group: int, pair max #examples with each image/video.
           Note that each image can have multiple groups.
        is_train: bool, if True, copy the examples to make sure each input
            group has max_n_example_per_group examples.
        example_unique_key: str, used to make sure no inputs are discarded by matching
            the input and output ids specified by `example_unique_key`
    """
    input_groups = []  # each element is (id, list(example))
    for k, examples in key_grouped_examples.items():
        chunked_examples = chunk_list(examples,
                                      chunk_size=max_n_example_per_group,
                                      pad_to_divisible=is_train)
        for c in chunked_examples:
            # if len(c) == 0:
            #     continue
            input_groups.append((k, c))

    if example_unique_key is not None:
        print(f"Using example_unique_key {example_unique_key} to check whether input and output ids m")
        # sanity check: make sure we did not discard any input example by accident.
        input_question_ids = flat_list_of_lists(
            [[sub_e[example_unique_key] for sub_e in e] for e in key_grouped_examples.values()])
        output_question_ids = flat_list_of_lists(
            [[sub_e[example_unique_key] for sub_e in e[1]] for e in input_groups])
        assert set(input_question_ids) == set(output_question_ids), "You are missing "
    return input_groups



