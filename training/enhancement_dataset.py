import numpy as np
import time
import pickle
import os
import shutil

from scipy.ndimage import convolve, rank_filter, gaussian_filter
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import cv2

impact_chars = ['�', ' ', 'e', 'a', 'n', 'o', 'i', 'r', 't', 'd', 'l', 's', 'u', 'm', 'c', ',', 'h', 'p', 'g',
                '.', 'v', 'ſ', 'b', 'y', 'k', 'z', 'w', 'j', 'а', 'f', 'о', 'и', 'q', '', 'A', 'т', 'е', 'н',
                'E', 'S', '-', 'I', 'с', 'ъ', 'р', 'C', 'R', 'M', 'D', 'в', 'L', 'N', 'T', 'P', 'O', 'á', 'д',
                'к', 'B', 'л', '', 'é', 'H', "'", 'ѣ', 'V', 'G', '/', 'п', '1', 'м', ':', ';', 'ł', 'x', 'з',
                'K', '2', 'W', '', '0', '', 'г', 'у', 'F', 'б', 'à', 'ч', 'я', 'č', 'ę', 'ż', '3', 'J', 'U',
                'ó', 'ñ', '4', '—', '5', 'ⱥ', '⸗', 'š', 'Z', '8', 'ž', '6', 'ж', 'í', 'Y', '7', 'ﬁ', 'й', '&',
                ')', 'ć', 'ĳ', '(', 'ś', 'щ', 'ѫ', 'ą', 'ɇ', '', 'ш', 'Q', '9', '?', 'х', '„', 'è', 'ò', '!',
                'ц', 'ç', 'ь', 'ß', 'X', '’', 'ź', '', 'ú', '©', '', 'ń', 'С', '“', 'ȧ', 'ﬀ', 'Н', 'ô', 'Т',
                'ê', 'ã', 'ẽ', 'П', 'æ', '"', '', 'õ', 'К', 'И', 'А', 'В', '', '', 'ë', '', '°', 'ю', 'ﬂ',
                'ф', 'Д', 'Б', 'ċ', 'ù', '̄', 'М', '́', 'О', 'Г', 'Р', '”', 'Е', '*', 'û', 'â', 'ṡ', ']', 'ƒ',
                '€', 'З', '', 'Л', 'î', 'ē', 'ì', '̀', 'œ', 'ü', 'ﬃ', 'ѭ', '[', 'Ł', 'ũ', 'ﬆ', 'Č', '»', 'Š',
                'Ж', 'Ч', '½', '|', 'Ц', '☞', 'ï', 'Х', '§', 'Æ', '«', 'Ъ', '£', 'ˮ', '', 'У', 'Ю', '˛', 'Ф',
                'І', '†', 'Ş', 'Я', '', 'Ш', 'Ѕ', 'Ž', 'Щ', 'ě', 'ö', 'ř', 'ä', 'É', '¡', '…', '‘', '=', '¼',
                'Ô', '', '¾', 'ý', 'Ę', 'ṅ', 'і', '\t', 'Ɇ', '̃', 'ā', '⁂', 'Ą', '', 'Ѣ', '', 'Ⱥ', 'ы', 'Ĳ',
                'ū', '']

class EnhancementDataset(object):
    def __init__(self, lines_path, line_height=32, mask_prob=0,
                 noise_prob=0, blur_prob=0, bin_prob=0, max_width=800,
                 max_labels=100, batch_size=16):

        self.transcriptions = []
        self.line_paths = []
        self.line_height = line_height

        self.max_width = max_width
        self.max_labels = max_labels
        self.batch_size = batch_size

        self.chars = impact_chars
        self.from_char, self.to_char = self.get_chars_mapping(self.chars)

        with open(lines_path, 'r') as handle:
            file_lines = handle.read().splitlines()
            for file_line in file_lines:
                line_path, transcription = file_line.split('\t')
                self.line_paths.append(os.path.join(os.path.dirname(lines_path), line_path))
                self.transcriptions.append(transcription)

        self.mask_prob = mask_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.bin_prob = bin_prob

    def normalize(self, data, range=(5,90)):
        a, b = range
        assert a < b and 0<=a<=100 and 0<=b<=100, "Invalid range"
        im = data.astype(np.float32)
        if im.ndim > 2:
            im = np.mean(im, axis=2)
        low,high = np.percentile(im, [a,b])
        if low == high:
            im = np.clip(im, 0, 1)
        else:
            im = np.clip((im-low)/(high-low), 0, 1)
        return im

    def binarize(self, data, gamma=10, range=(0.2,0.4), clip=True):
        im = data**(1/gamma)
        low,high = range
        low = low**(1/gamma)
        high = high**(1/gamma)
        im = (im-low) / (high-low)
        if clip:
            im = np.clip(im, 0, 1)
        return im

    def mask_data(self, data, prob=1, seed=None):
        data_masked = data.copy()
        np.random.seed(seed)
        for i in range(data.shape[0]):
            if np.random.rand() < prob:
                num_masks = np.random.randint(3,5)
                for _ in range(num_masks):
                    mask_length = np.random.randint(data.shape[2]//32, data.shape[2]//8 + 16)
                    mask_start = np.random.randint(0, data.shape[2] - mask_length)
                    data_masked[i, :, mask_start:mask_start+mask_length, :] = 0   # random_img[:data_masked.shape[1], mask_start:mask_start+mask_length, :]
        return data_masked

    def add_noise(self, data, prob=1):
        for i in range(data.shape[0]):
            if np.random.rand() < prob:
                noise_image = gaussian_filter(np.random.randn(data.shape[1],data.shape[2]), np.random.rand()*4)
                noise_image = normalize(noise_image, range=(0,100)) - 0.5
                rng = (0.5*np.random.rand())*noise_image
                for j in range(data.shape[3]):
                    data[i,:,:,j] += rng
        data = np.clip(data, 0, 1)
        return data

    def add_blur(self, data, prob=1):
        for i in range(data.shape[0]):
            if np.random.rand() < prob:
                s = np.random.rand()*2
                for j in range(data.shape[3]):
                    data[i,:,:,j] = gaussian_filter(data[i,:,:,j], s)
        data = np.clip(data, 0, 1)
        return data

    def binarize_data(self, data, prob=1):
        for i in range(data.shape[0]):
            if np.random.rand() < prob:
                range_norm = (0, np.random.randint(50,80))
                range_bin = (0.2, 0.3 + np.random.rand() * 0.3)
                for j in range(data.shape[3]):
                    image = data[i,:,:,j].copy()
                    image = binarize(normalize(image, range=range_norm), range=range_bin)#, gamma=1+np.random.rand()*10, range=(a.min(),a.max())).
                    data[i,:,:,j] = np.clip(image, 0, 1)
        return data

    def degrade(self, images):
        """
        Random degradation of image
        """
        images = self.add_noise(images, prob=self.noise_prob)
        images = self.add_blur(images, prob=self.blur_prob)
        images = self.binarize_data(images, prob=self.bin_prob)
        images = self.mask_data(images, prob=self.mask_prob)

        return images.astype(np.float32)

    def transcriptions_to_labels(self, transcriptions):
        trans = str.maketrans(''.join(self.from_char), ''.join(self.to_char))
        labels = [np.asarray([ord(x) for x in l.translate(trans)], dtype=np.int32) for l in transcriptions]
        return labels

    def transcriptions_to_sequences(self, transcriptions):
        labels = self.transcriptions_to_labels(transcriptions)
        return self.sparse_tuples_from_sequences(labels)

    def sparse_tuples_from_sequences(self, sequences, dtype=np.int32):
        indexes = []
        values = []

        for n, sequence in enumerate(sequences):
            indexes.extend(zip([n] * len(sequence), range(len(sequence))))
            values.extend(sequence)

        indexes = np.asarray(indexes, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indexes).max(0)[1] + 1], dtype=np.int64)
        return indexes, values, shape

    def clip_transcriptions(self, transcriptions):
        new_transcriptions = np.zeros((len(transcriptions), self.max_labels))
        for i, transcription in enumerate(transcriptions):
            new_transcription = np.zeros(self.max_labels)
            if transcription.shape[0] > self.max_labels:
                new_transcription = transcription[:self.max_labels]
            else:
                new_transcription[:transcription.shape[0]] = transcription
            new_transcriptions[i,:] = new_transcription
        return new_transcriptions.astype(np.int32)

    def get_batch(self, degrade=False, show=False):
        images = np.zeros((self.batch_size, self.line_height, self.max_width, 3))
        batch_transcriptions = []

        for i in range(self.batch_size):
            sample_ind = np.random.randint(0, len(self.line_paths)-1)
            line_crop = cv2.imread(self.line_paths[sample_ind])
            images[i, :, :line_crop.shape[1], :] = line_crop / 255.0
            batch_transcriptions.append(self.transcriptions[sample_ind])

        batch_transcriptions = self.transcriptions_to_labels(batch_transcriptions)
        seq_lengths = np.full(self.batch_size, self.max_width // 4, dtype=np.int32) # given by OCR architecture so this should be moved to tf_nets module

        transcriptions = self.clip_transcriptions(batch_transcriptions)
        if degrade:
            images = self.degrade(images)

        ctc_targets = self.sparse_tuples_from_sequences(batch_transcriptions)

        if show:
            canvas = np.zeros((self.batch_size*self.line_height, self.max_width, 3))
            for j in range(self.batch_size):
                canvas[j*self.line_height:(j*self.line_height)+self.line_height, :, :] = images[j, :, :, :]
                s = []
                for k in transcriptions[j]:
                    s.append(self.chars[k])
                print(''.join(s))
            plt.imshow(canvas)
            plt.show()

        return images, transcriptions, ctc_targets, seq_lengths

    def get_chars_mapping(self, chars):
        from_char = []
        to_char = []

        for i, c in enumerate(chars):
            from_char.append(c)
            to_char.append(chr(i))

        return (from_char, to_char)
