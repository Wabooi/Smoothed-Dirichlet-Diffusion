import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import click
import numpy as np
import PIL.Image
from tqdm import tqdm


def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)  
    if m:
        return int(m.group(1)), int(m.group(2))  
    raise click.ClickException(f'无法解析元组 {s}')  


def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a


def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]  

 
def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower() 
    return f'.{ext}' in PIL.Image.EXTENSION  


def open_image_folder(source_dir, *, max_images: Optional[int]):

    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]

    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace('\\', '/') for fname in input_images}
    max_idx = maybe_min(len(input_images), max_images)  


    labels = dict()
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            data = json.load(file)['labels']
            if data is not None:
                labels = {x[0]: x[1] for x in data}  

    if len(labels) == 0:
        toplevel_names = {arch_fname: arch_fname.split('/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
        toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))}
        if len(toplevel_indices) > 1:
            labels = {arch_fname: toplevel_indices[toplevel_name] for arch_fname, toplevel_name in toplevel_names.items()}

    def iterate_images():
        for idx, fname in enumerate(input_images):
            img = np.array(PIL.Image.open(fname))  
            yield dict(img=img, label=labels.get(arch_fnames.get(fname)))  
            if idx >= max_idx - 1:  
                break
    return max_idx, iterate_images()


def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
       
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]
        max_idx = maybe_min(len(input_images), max_images)

        
        labels = dict()
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels = {x[0]: x[1] for x in data}

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = np.array(PIL.Image.open(file))  
                yield dict(img=img, label=labels.get(fname))  
                if idx >= max_idx - 1:  
                    break
    return max_idx, iterate_images()


def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  
    import lmdb  

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):  
                try:
                    try:
                    
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode失败')
                        img = img[:, :, ::-1]  
                    except IOError:
                        
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, label=None)  
                    if idx >= max_idx - 1:  
                        break
                except Exception:
                    print(sys.exc_info()[1])  

    return max_idx, iterate_images()


def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        for batch in range(1, 6):  
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')  
            images.append(data['data'].reshape(-1, 3, 32, 32))  
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1])  

    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))  
            if idx >= max_idx - 1:  
                break

    return max_idx, iterate_images()


def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace('-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)  

    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)  

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)

    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))  
            if idx >= max_idx - 1:  
                break

    return max_idx, iterate_images()


def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:

    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:  
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.Resampling.LANCZOS)  
        return np.array(img)


    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])  

        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, 
                 (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        if img.ndim == 2:  
            img = img[:, :, np.newaxis].repeat(3, axis=2)
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS) 
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))  
        if img.shape[1] < width or ch < height:  
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        if img.ndim == 2:  
            img = img[:, :, np.newaxis].repeat(3, axis=2)
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)  
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)  
    if transform == 'center-crop':
        if output_width is None or output_height is None:
            raise click.ClickException('使用center-crop变换时必须指定--resolution=WxH')
        return functools.partial(center_crop, output_width, output_height)  
    if transform == 'center-crop-wide':
        if output_width is None or output_height is None:
            raise click.ClickException('使用center-crop-wide变换时必须指定--resolution=WxH')
        return functools.partial(center_crop_wide, output_width, output_height)  
    assert False, '未知变换类型'



def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        if source.rstrip('/').endswith('_lmdb'):  
            return open_lmdb(source, max_images=max_images)
        else:  
            return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if os.path.basename(source) == 'cifar-10-python.tar.gz':  
            return open_cifar10(source, max_images=max_images)
        elif os.path.basename(source) == 'train-images-idx3-ubyte.gz':  
            return open_mnist(source, max_images=max_images)
        elif file_ext(source) == 'zip':  
            return open_image_zip(source, max_images=max_images)
        else:
            assert False, '未知的压缩包类型'
    else:
        raise click.ClickException(f'找不到输入文件或目录: {source}')

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':  
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)  
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)  
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)  
        return '', zip_write_bytes, zf.close  
    else: 
        
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            raise click.ClickException('--dest 目录必须为空')
        os.makedirs(dest, exist_ok=True)  

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)  
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')  
                fout.write(data)  
        return dest, folder_write_bytes, lambda: None 


@click.command()
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--transform',  help='Input crop/resize mode', metavar='MODE',            type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--resolution', help='Output resolution (e.g., 512x512)', metavar='WxH',  type=parse_tuple)

def main(
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]]
):

    PIL.Image.init()  

    if dest == '':
        raise click.ClickException('--dest 输出路径不能为空')

    num_files, input_iter = open_dataset(source, max_images=max_images)

    archive_root_dir, save_bytes, close_dest = open_dest(dest)


    if resolution is None: 
        resolution = (None, None)

    transform_image = make_transform(transform, *resolution)

    dataset_attrs = None  
    labels = []  


    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'  


        img = transform_image(image['img'])
        if img is None:  
            continue

        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1], 
            'height': img.shape[0], 
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            
            if width != height:
                raise click.ClickException(f'裁剪缩放后的图像必须是正方形。实际尺寸 {width}x{height}')
            
            if dataset_attrs['channels'] not in [1, 3]:
                raise click.ClickException('输入图像必须是RGB或灰度图')
            
            if width != 2 ** int(np.floor(np.log2(width))):
                raise click.ClickException('图像宽高必须是2的幂')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  数据集 {k}/当前图像 {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' 
                  for k in dataset_attrs.keys()]
            raise click.ClickException(f'图像 {archive_fname} 属性必须与数据集一致:\n' + '\n'.join(err))

        img = PIL.Image.fromarray(img, {1: 'L', 3: 'RGB'}[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)  
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        if image['label'] is not None:
            labels.append([archive_fname, image['label']])
        else:
            labels.append(None)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()  


if __name__ == "__main__":
    main()

