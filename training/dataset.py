import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None



class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   
        raw_shape,             
        max_size    = None,    
        use_labels  = False,    
        xflip       = False,   
        random_seed = 0,        
        cache       = False,    
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict()
        self._raw_labels = None
        self._label_shape = None

        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): 
        pass

    def _load_raw_image(self, raw_idx): 
        raise NotImplementedError

    def _load_raw_labels(self): 
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                  
        resolution      = None, 
        use_pyspng      = True, 
        **super_kwargs,        
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] 
        image = image.transpose(2, 0, 1) 
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


class NPYDataset(Dataset):
    def __init__(self,
        path,
        resolution      = None,   
        slice_axis      = None,   
        **super_kwargs,
    ):
        self._path = path
        self._resolution = resolution
        self._slice_axis = slice_axis
        self._items = []
        self._all_fnames = []

        if os.path.isdir(self._path):
            for root, _dirs, files in os.walk(self._path):
                for f in files:
                    if f.lower().endswith('.npy') or f.lower().endswith('.npz'):
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, self._path).replace('\\', '/')
                        self._all_fnames.append(rel)
            if len(self._all_fnames) == 0:
                raise IOError('指定目录下未找到 .npy/.npz 文件')
        elif os.path.isfile(self._path) and (self._file_ext(self._path) in ['.npy', '.npz']):
            self._all_fnames.append(os.path.basename(self._path))
        else:
            raise IOError('Path must point to a directory containing .npy/.npz or a single .npy/.npz file')

        for rel in self._all_fnames:
            fpath = os.path.join(self._path, rel) if os.path.isdir(self._path) else (self._path if rel == os.path.basename(self._path) else None)
            if fpath is None:
                fpath = os.path.join(self._path, rel)
            arr = self._load_array_meta(fpath)
            if arr.ndim == 2:
                self._items.append(('2d', fpath, None))
            elif arr.ndim == 3:
                if (arr.shape[0] in [1, 3]) or (arr.shape[-1] in [1, 3]):
                    self._items.append(('image3d', fpath, None))
                else:
                    axis = self._slice_axis if self._slice_axis is not None else 0
                    for i in range(arr.shape[axis]):
                        self._items.append(('slice', fpath, (axis, i)))
            elif arr.ndim == 4:
                N = arr.shape[0]
                for i in range(N):
                    self._items.append(('batch', fpath, i))
            else:
                raise IOError(f'不支持的数组维度: {arr.ndim} in {fpath}')

        first = self._to_chw_uint8(self._load_item(0))
        C, H, W = first.shape
        if self._resolution is not None and (H != self._resolution or W != self._resolution):
            H = W = self._resolution
        if H != W:
            raise IOError(f'图像必须为正方形，当前为 {H}x{W}')
        if C not in [1, 3]:
            raise IOError(f'通道数必须为1或3，当前为 {C}')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._items), C, H, W]
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_array_meta(self, path):
        data = np.load(path, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            if len(data.files) == 0:
                raise IOError(f'npz 文件为空: {path}')
            arr = data[data.files[0]]
        else:
            arr = data
        return arr

    def _load_item(self, raw_idx):
        kind, fpath, idxinfo = self._items[raw_idx]
        arr = self._load_array_meta(fpath)
        if kind == '2d':
            img = arr
        elif kind == 'image3d':
            if arr.shape[0] in [1, 3] and arr.ndim == 3:
                img = arr
            elif arr.shape[-1] in [1, 3] and arr.ndim == 3:
                img = arr
            else:
                raise IOError(f'无法解析3维图像通道: {fpath}, shape={arr.shape}')
        elif kind == 'slice':
            axis, i = idxinfo
            img = np.take(arr, i, axis=axis)
        elif kind == 'batch':
            img = arr[idxinfo]
        else:
            raise IOError(f'未知条目类型: {kind}')
        return img

    def _to_chw_uint8(self, img):
        if img.ndim == 2:
            img = img[:, :, np.newaxis] 
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            
            img_hwc = img.transpose(1, 2, 0)
        elif img.ndim == 3 and img.shape[-1] in [1, 3]:
            img_hwc = img  
        else:
            raise IOError(f'无法转换图像到 CHW，shape={img.shape}')

        if img_hwc.dtype != np.uint8 or img_hwc.min() < 0 or img_hwc.max() > 255:
            x = img_hwc.astype(np.float32)
            x -= x.min()
            denom = x.max() - x.min() + 1e-8
            x = (x / denom) * 255.0
            img_hwc = x.astype(np.uint8)

        H, W, C = img_hwc.shape
        if self._resolution is not None and (H != self._resolution or W != self._resolution):
            if C == 1:
                im = PIL.Image.fromarray(img_hwc[:, :, 0], mode='L')
                im = im.resize((self._resolution, self._resolution), resample=PIL.Image.LANCZOS)
                img2d = np.array(im)                      
                img_hwc = img2d[:, :, np.newaxis]     
            elif C == 3:
                im = PIL.Image.fromarray(img_hwc, mode='RGB')
                im = im.resize((self._resolution, self._resolution), resample=PIL.Image.LANCZOS)
                img_hwc = np.array(im)
            else:
                raise IOError(f'通道数必须为1或3，当前为 {C}')

        img_chw = img_hwc.transpose(2, 0, 1)
        return img_chw

    def _load_raw_image(self, raw_idx):
        img = self._load_item(raw_idx)
        img = self._to_chw_uint8(img)
        return img

    def _load_raw_labels(self):
        return None
