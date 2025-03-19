"""DataSampler module."""

import numpy as np

class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""
    
    def __init__(self, data, output_info, log_frequency, weighting_method='log', temperature=1.5):
        """
        Args:
            data (numpy array): The transformed training data.
            output_info (list): List of output info per column.
            log_frequency (bool): Whether to use log-frequency weighting.
            weighting_method (str): The weighting method to use.
                Options:
                  'log'         -> use np.log(freq + 1) (default)
                  'inverse'     -> use inverse frequency, i.e. 1 / (freq + epsilon)
                  'temperature' -> apply temperature scaling to the log-frequency weights.
                  'none'        -> use raw frequencies.
            temperature (float): Temperature value for scaling (only used if weighting_method=='temperature').
        """
        self._data_length = len(data)

        def is_discrete_column(column_info):
            return len(column_info) == 1 and column_info[0].activation_fn == 'softmax'

        n_discrete_columns = sum([1 for column_info in output_info if is_discrete_column(column_info)])
        self._discrete_column_matrix_st = np.zeros(n_discrete_columns, dtype='int32')
        self._rid_by_cat_cols = []  # Store row indices for each category

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficient conditional vector sampling
        max_category = max(
            [column_info[0].dim for column_info in output_info if is_discrete_column(column_info)],
            default=0,
        )

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum([column_info[0].dim for column_info in output_info if is_discrete_column(column_info)])

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                # Compute frequency for each category in the column
                category_freq = np.sum(data[:, st:ed], axis=0)
                
                epsilon = 1e-10  # To avoid division by zero
                
                # Choose weighting scheme:
                if weighting_method == 'log':
                    # Default: use logarithm of frequency (optionally controlled by log_frequency flag)
                    weighted = np.log(category_freq + 1) if log_frequency else category_freq
                elif weighting_method == 'inverse':
                    # Use inverse frequency to boost rare categories
                    weighted = 1 / (category_freq + epsilon)
                elif weighting_method == 'temperature':
                    # Apply temperature scaling to the log frequencies.
                    # Lower temperatures boost differences (i.e. rare categories get relatively more weight)
                    weighted = np.exp(np.log(category_freq + 1) / temperature)
                elif weighting_method == 'hybrid':
                    # Use a blend of log-frequency and inverse-frequency weighting.
                    # 'alpha' is a parameter between 0 and 1 (e.g., 0.7 gives 70% weight to log-frequency).
                    alpha = hybrid_alpha  # Make sure to pass hybrid_alpha as an additional parameter.
                    log_weight = np.log(category_freq + 1)
                    inv_weight = 1 / (category_freq + epsilon)
                    weighted = alpha * log_weight + (1 - alpha) * inv_weight
                elif weighting_method == 'none':
                    # Use raw frequencies
                    weighted = category_freq
                else:
                    raise ValueError("Unsupported weighting_method: choose 'log', 'inverse', 'temperature', 'hybrid', or 'none'.")
                
                # Normalize the weights to create a probability distribution
                category_prob = weighted / np.sum(weighted)
                
                self._discrete_column_category_prob[current_id, : span_info.dim] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])


    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch):
        """Generate the conditional vector for training.
        
        Returns:
            cond (batch x #categories): The conditional vector.
            mask (batch x #discrete columns): A one-hot vector indicating the selected discrete column.
            discrete column id (batch): Integer representation of mask.
            category_id_in_col (batch): Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(np.arange(self._n_discrete_columns), batch)
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = self._discrete_column_cond_st[discrete_column_id] + category_id_in_col
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation using the original frequency."""
        if self._n_discrete_columns == 0:
            return None

        category_freq = self._discrete_column_category_prob.flatten()
        category_freq = category_freq[category_freq != 0]
        category_freq = category_freq / np.sum(category_freq)
        col_idxs = np.random.choice(np.arange(len(category_freq)), batch, p=category_freq)
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        cond[np.arange(batch), col_idxs] = 1

        return cond

    def sample_data(self, data, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.
        
        Args:
            data: The training data.
            n: Number of rows to sample.
            
        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(data), size=n)
            return data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))
        return data[idx]

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        """Generate the condition vector."""
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id_ = self._discrete_column_matrix_st[condition_info['discrete_column_id']]
        id_ += condition_info['value_id']
        vec[:, id_] = 1
        return vec
