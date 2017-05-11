class ImageIter(io.DataIter):
    """Image data iterator with a large number of augumentation choices.
    Supports reading from both .rec files and raw image files with image list.
    To load from .rec files, please specify path_imgrec. Also specify path_imgidx
    to use data partition (for distributed training) or shuffling.
    To load from raw image files, specify path_imglist and path_root.
    Parameters
    ----------
    batch_size : int
        Number of examples per batch
    data_shape : tuple
        Data shape in (channels, height, width).
        For now, only RGB image with 3 channels is supported.
    label_width : int
        dimension of label
    path_imgrec : str
        path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec
    path_imglist : str
        path to image list (.lst)
        Created with tools/im2rec.py or with custom script.
        Format: index\t[one or more label separated by \t]\trelative_path_from_root
    imglist: list
        a list of image with the label(s)
        each item is a list [imagelabel: float or list of float, imgpath]
    path_root : str
        Root folder of image files
    path_imgidx : str
        Path to image index file. Needed for partition and shuffling when using .rec source.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration.
        Can be slow for HDD.
    part_index : int
        Partition index
    num_parts : int
        Total number of partitions.
    kwargs : ...
        More arguments for creating augumenter. See mx.image.CreateAugmenter
    """
    def __init__(self, batch_size, data_shape, label_width=1,
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None, **kwargs):
        super(ImageIter, self).__init__()
        assert(path_imgrec or path_imglist or (isinstance(imglist, list)))
        if path_imgrec:
            print('loading recordio...')
            if path_imgidx:
                self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
                self.imgidx = list(self.imgrec.keys)
            else:
                self.imgrec = recordio.MXRecordIO(path_imgrec, 'r')
                self.imgidx = None
        else:
            self.imgrec = None

        if path_imglist:
            print('loading image list...')
            with open(path_imglist) as fin:
                imglist = {}
                imgkeys = []
                for line in iter(fin.readline, ''):
                    line = line.strip().split('\t')
                    label = nd.array([float(i) for i in line[1:-1]])
                    key = int(line[0])
                    imglist[key] = (label, line[-1])
                    imgkeys.append(key)
                self.imglist = imglist
        elif isinstance(imglist, list):
            print('loading image list...')
            result = {}
            imgkeys = []
            index = 1
            for img in imglist:
                key = str(index)
                index += 1
                if isinstance(img[0], numeric_types):
                    label = nd.array([img[0]])
                else:
                    label = nd.array(img[0])
                result[key] = (label, img[1])
                imgkeys.append(str(key))
            self.imglist = result
        else:
            self.imglist = None
        self.path_root = path_root

        assert len(data_shape) == 3 and data_shape[0] == 3
        self.provide_data = [('data', (batch_size,) + data_shape)]
        if label_width > 1:
            self.provide_label = [('softmax_label', (batch_size, label_width))]
        else:
            self.provide_label = [('softmax_label', (batch_size,))]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.label_width = label_width

        self.shuffle = shuffle
        if self.imgrec is None:
            self.seq = imgkeys
        elif shuffle or num_parts > 1:
            assert self.imgidx is not None
            self.seq = self.imgidx
        else:
            self.seq = None

        if num_parts > 1:
            assert part_index < num_parts
            N = len(self.seq)
            C = N/num_parts
            self.seq = self.seq[part_index*C:(part_index+1)*C]
        if aug_list is None:
            self.auglist = CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        self.cur = 0
        self.reset()

    def reset(self):
        if self.shuffle:
            random.shuffle(self.seq)
        if self.imgrec is not None:
            self.imgrec.reset()
        self.cur = 0

    def next_sample(self):
        """helper function for reading in next sample"""
        if self.seq is not None:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.imgrec is not None:
                s = self.imgrec.read_idx(idx)
                header, img = recordio.unpack(s)
                if self.imglist is None:
                    return header.label, img
                else:
                    return self.imglist[idx][0], img
            else:
                label, fname = self.imglist[idx]
                if self.imgrec is None:
                    with open(os.path.join(self.path_root, fname), 'rb') as fin:
                        img = fin.read()
                return label, img
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img

    def next(self):
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s = self.next_sample()
                data = [imdecode(s)]
                if len(data[0].shape) == 0:
                    logging.debug('Invalid image, skipping.')
                    continue
                for aug in self.auglist:
                    data = [ret for src in data for ret in aug(src)]
                for d in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i][:] = nd.transpose(d, axes=(2, 0, 1))
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if not i:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size-i)