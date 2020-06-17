import numpy as np


class DataSampler(object):
    def __init__(self, data, output_info, log_frequency):
        self.data = data
        self.model = []
        self.model2 = []
        start = 0
        skip = False
        max_interval = 0
        counter = 0
        for items in output_info:
            if len(items) == 2:
                start += items[0][0] + items[1][0]
                continue
            else:
                assert len(items) == 1
                item = items[0]
                end = start + item[0]
                max_interval = max(max_interval, end - start)
                counter += 1
                self.model.append(np.argmax(data[:, start:end], axis=-1))

                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, start + j])[0])

                self.model2.append(tmp)
                start = end
        assert start == data.shape[1]

        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        start = 0
        self.p = np.zeros((counter, max_interval))
        for items in output_info:
            if len(items) == 2:
                start += items[0][0] + items[1][0]
                continue
            else:
                assert len(items) == 1
                item = items[0]
                end = start + item[0]
                tmp = np.sum(data[:, start:end], axis=0)
                if log_frequency:
                    tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                self.p[self.n_col, :item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                start = end


        self.interval = np.asarray(self.interval)

    def random_choice_prob_index(self, idx):
        a = self.p[idx]
        r = np.expand_dims(np.random.rand(a.shape[0]), axis=1)
        return (a.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch):
        if self.n_col == 0:
            return None

        batch = batch
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec1 = np.zeros((batch, self.n_opt), dtype='float32')
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        mask1[np.arange(batch), idx] = 1
        opt1prime = self.random_choice_prob_index(idx)
        opt1 = self.interval[idx, 0] + opt1prime
        vec1[np.arange(batch), opt1] = 1

        return vec1, mask1, idx, opt1prime

    def sample_zero_condvec(self, batch):
        if self.n_col == 0:
            return None

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        for i in range(batch):
            col = idx[i]
            pick = int(np.random.choice(self.model[col]))
            vec[i, pick + self.interval[col, 0]] = 1

        return vec

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model2[c][o]))

        return self.data[idx]
