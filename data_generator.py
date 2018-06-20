import numpy as np
import sys
import torch


class BouncingMNISTDataHandler(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""
    def __init__(self, batch_size, num_digits):
        self.seq_length_ = 20
        self.batch_size_ = batch_size
        self.image_size_ = 64
        self.num_digits_ = num_digits
        self.max_speed = 0.5  # with canvas size 1
        self.dataset_size_ = 1000000  # The dataset is really infinite. This is just for validation.
        self.digit_size_ = 28
        self.data_ = []  # a list of numpy tensors of shape (1, digit_size_, digit_size)
        for i in range(10):
            try:
                self.data_.append(np.load('mnist/raw digit/{}.npy'.format(i)))
            except:
                print('Please set the correct path to MNIST dataset')
                sys.exit(1)
            else:
                self.data_[i] = self.data_[i].reshape(1,self.digit_size_, self.digit_size_)
                self.indices_ = np.arange(len(self.data_))
                self.row_ = 0
                np.random.shuffle(self.indices_)

    def GetBatchSize(self):
        return self.batch_size_

    def GetDatasetSize(self):
        return self.dataset_size_

    def GetSeqLength(self):
        return self.seq_length_

    def Reset(self):
        pass

    def GetRandomTrajectory(self, batch_size):
        length = self.seq_length_
        canvas_size = self.image_size_ - self.digit_size_
    
        # Initial position uniform random inside the box.
        y = np.random.rand(batch_size)
        x = np.random.rand(batch_size)

        # Choose a random velocity.
        theta = np.random.rand(batch_size) * 2 * np.pi
        # random velocity
        ro = np.random.rand(batch_size) * self.max_speed
        v_y = ro * np.sin(theta)
        v_x = ro * np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))

        for i in range(length):
            # Take a step along velocity.
            y += v_y
            x += v_x

            # Bounce off edges.
            for j in range(batch_size):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
                start_y[i, :] = y
                start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def Overlap(self, a, b):
        """ Put b on top of a."""
        return np.maximum(a, b)
        #return b

    def GetBatch(self):
        start_y, start_x = self.GetRandomTrajectory(self.batch_size_ * self.num_digits_)
    
        # minibatch data
        data = np.zeros((self.seq_length_, self.batch_size_, 1, self.image_size_, self.image_size_), dtype=np.float32)
    
        for j in range(self.batch_size_):
            for n in range(self.num_digits_):
                # get random digit from dataset
                ind = self.indices_[self.row_]
                self.row_ += 1
                if self.row_ == len(self.data_):
                    self.row_ = 0
                    np.random.shuffle(self.indices_)
                digit_image = self.data_[ind]

                # generate video
                for i in range(self.seq_length_):
                    top = start_y[i, j * self.num_digits_ + n]
                    left = start_x[i, j * self.num_digits_ + n]
                    bottom = top + self.digit_size_
                    right = left + self.digit_size_
                    data[i, j, 0, top:bottom, left:right] = self.Overlap(data[i, j, 0, top:bottom, left:right], digit_image)

        data = data.reshape(self.seq_length_, self.batch_size_, 1, self.image_size_, self.image_size_)
        len1 = int(self.seq_length_/2)
        len2 = self.seq_length_-len1
        input_data = data[0:len1,: ,: ,: ,: ].reshape(len1, self.batch_size_, 1, self.image_size_, self.image_size_)
        target_data = data[len1:self.seq_length_,: ,: ,: ,: ].reshape(len2, self.batch_size_, 1, self.image_size_, self.image_size_)
        return (torch.tensor(input_data), torch.tensor(target_data))

    def generate_files(self):
        for i in range(int(self.dataset_size_/self.batch_size_)):
            np.save('mnist/mnist-'+str(self.num_digits_)+'/batch'+str(i)+'.npy',self.GetBatch())

    '''
    def DisplayData(self, data, rec=None, fut=None, fig=1, case_id=0, output_file=None):
        output_file1 = None
        output_file2 = None
    
        if output_file is not None:
            name, ext = os.path.splitext(output_file)
            output_file1 = '%s_original%s' % (name, ext)
            output_file2 = '%s_recon%s' % (name, ext)
    
         # get data
        data = data[case_id, :].reshape(-1, self.image_size_, self.image_size_)
        # get reconstruction and future sequences if exist
        if rec is not None:
            rec = rec[case_id, :].reshape(-1, self.image_size_, self.image_size_)
            enc_seq_length = rec.shape[0]
        if fut is not None:
            fut = fut[case_id, :].reshape(-1, self.image_size_, self.image_size_)
            if rec is None:
                enc_seq_length = self.seq_length_ - fut.shape[0]
            else:
                assert enc_seq_length == self.seq_length_ - fut.shape[0]

        num_rows = 1
        # create figure for original sequence
        plt.figure(2*fig, figsize=(20, 1))
        plt.clf()
        for i in range(self.seq_length_):
            plt.subplot(num_rows, self.seq_length_, i+1)
            plt.imshow(data[i, :, :], cmap=plt.cm.gray, interpolation="nearest")
            plt.axis('off')
            plt.draw()
            if output_file1 is not None:
                print(output_file1)
                plt.savefig(output_file1, bbox_inches='tight')

        # create figure for reconstuction and future sequences
        plt.figure(2*fig+1, figsize=(20, 1))
        plt.clf()
        for i in range(self.seq_length_):
          if rec is not None and i < enc_seq_length:
            plt.subplot(num_rows, self.seq_length_, i + 1)
            plt.imshow(rec[rec.shape[0] - i - 1, :, :], cmap=plt.cm.gray, interpolation="nearest")
          if fut is not None and i >= enc_seq_length:
            plt.subplot(num_rows, self.seq_length_, i + 1)
            plt.imshow(fut[i - enc_seq_length, :, :], cmap=plt.cm.gray, interpolation="nearest")
          plt.axis('off')
        plt.draw()
        if output_file2 is not None:
          print(output_file2)
          plt.savefig(output_file2, bbox_inches='tight')
        else:
          plt.pause(0.1)
    '''
