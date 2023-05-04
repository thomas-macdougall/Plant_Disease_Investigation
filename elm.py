import numpy as np
import h5py

class ELM(object):
    def __init__(self, input_nodes, hidden_nodes, 
                 n_output_nodes, activation='sigmoid_activation', 
                loss='mse', name=None, beta_init=None, alpha_init=None, bias_init=None):

        self.name = name
        self._input_nodes, self.__hidden_nodes, self.__n_output_nodes = input_nodes, hidden_nodes, n_output_nodes

        self.__beta = beta_init if isinstance(beta_init, np.ndarray) and beta_init.shape == (self.__hidden_nodes, self.__n_output_nodes) \
                    else np.random.uniform(-1.,1.,size=(self.__hidden_nodes, self.__n_output_nodes))

        self.__alpha = alpha_init if isinstance(alpha_init, np.ndarray) and alpha_init.shape == (self._input_nodes, self.__hidden_nodes) \
                    else np.random.uniform(-1.,1.,size=(self._input_nodes, self.__hidden_nodes))

        self.__bias = bias_init if isinstance(bias_init, np.ndarray) and bias_init.shape == (self.__hidden_nodes,) \
                    else np.zeros(shape=(self.__hidden_nodes,))

        self.__activation = self.get_activation_func(activation)
        self.__loss = self.get_loss_func(loss)

    def __call__(self, x):
        #Applies the ELM to input data x and returns the output
        hidden_layer_output = self.__activation(x.dot(self.__alpha) + self.__bias)
        return hidden_layer_output.dot(self.__beta)

    def predict(self, x):
        #Generates predictions for input data x
    
        return list(self(x))

    def fit(self, x, t):
        #Trains the ELM on input data x and targets t
        hidden_layer_output = self.__activation(x.dot(self.__alpha) + self.__bias)
        hidden_layer_pseudoinverse = np.linalg.pinv(hidden_layer_output)
        self.__beta = hidden_layer_pseudoinverse.dot(t)

    def score(self, X, y):
        #Returns the accuracy score for input data X and targets y
        y_pred = self.predict(X)
        accuracy = np.sum(np.argmax(y_pred, axis=-1) == np.argmax(y, axis=-1)) / len(y)
        return accuracy


    def get_activation_func(self, name):
        if name == 'sigmoid_activation':
            return sigmoid_activation
        if name == 'identity_activation':
            return identity_activation
        

    def get_loss_func(self, name):
        if name == 'mse':
            return mse
        if name == 'mae':
            return mae
    
    def set_weights(self, alpha=None, beta=None, bias=None):
        self.__alpha = alpha
        self.__beta = beta
        self.__bias = bias

    @property
    def weights(self):
        return {'alpha': self.__alpha, 'beta': self.__beta, 'bias': self.__bias}

    @property
    def input_shape(self):
        return (self._input_nodes,)

    @property
    def output_shape(self):
        return (self.__n_output_nodes,)

    def _set_property(self, attr, value):
        if hasattr(self, attr):
            setattr(self, attr, value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def _get_property(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    setinput_nodes = lambda self, value: self._set_property('input_nodes', value)
    getinput_nodes = lambda self: self._get_property('input_nodes')
    input_nodes = property(getinput_nodes, setinput_nodes)

    set_hidden_nodes = lambda self, value: self._set_property('hidden_nodes', value)
    get_hidden_nodes = lambda self: self._get_property('hidden_nodes')
    hidden_nodes = property(get_hidden_nodes, set_hidden_nodes)

    set_output_nodes = lambda self, value: self._set_property('output_nodes', value)
    get_output_nodes = lambda self: self._get_property('output_nodes')
    n_output_nodes = property(get_output_nodes, set_output_nodes)

    _set_activation = lambda self, value: self._set_property('activation', self.get_activation_func(value))
    _get_activation = lambda self: self.__get_activation_name(self._activation)
    activation = property(_get_activation, _set_activation)

    _set_loss = lambda self, value: self._set_property('loss', self.get_loss_func(value))
    _get_loss = lambda self: self.__get_loss_name(self._loss)
    loss = property(_get_loss, _set_loss)

    def save(self, filepath):
        with h5py.File(filepath, 'w') as f:
            arc = f.create_dataset('architecture', data=np.array([self.__n_input_nodes, self.__n_hidden_nodes, self.__n_output_nodes]))
            arc.attrs['activation'] = self.__get_activation_name(self.__activation).encode('utf-8')
            arc.attrs['loss'] = self.__get_loss_name(self.__loss).encode('utf-8')
            arc.attrs['name'] = self.name.encode('utf-8')
            f.create_group('weights')
            f.create_dataset('weights/alpha', data=self.__alpha)
            f.create_dataset('weights/beta', data=self.__beta)
            f.create_dataset('weights/bias', data=self.__bias)

def load_model(filepath):
    with h5py.File(filepath, 'r') as f:
        alpha_init = f['weights/alpha'][...]
        beta_init = f['weights/beta'][...]
        bias_init = f['weights/bias'][...]
        arc = f['architecture']
        n_input_nodes = arc[0]
        n_hidden_nodes = arc[1]
        n_output_nodes = arc[2]
        activation = arc.attrs['activation']
        loss = arc.attrs['loss']
        name = arc.attrs['name']
        model = ELM(
            n_input_nodes=n_input_nodes,
            n_hidden_nodes=n_hidden_nodes,
            n_output_nodes=n_output_nodes,
            activation=activation,
            loss=loss,
            alpha_init=alpha_init,
            beta_init=beta_init,
            bias_init=bias_init,
            name=name,
        )
    return model

class ELMEstimatorWrapper(ELM):
    # A wrapper class for the ELM object that implements the get_params() method.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_params(self, deep=True):
        return {
            'input_nodes': self.input_nodes,
            'hidden_nodes': self.hidden_nodes,
            'n_output_nodes': self.n_output_nodes,
            'activation': self.activation,
            'loss': self.loss,
            'name': self.name,
            'beta_init': self.weights['beta'],
            'alpha_init': self.weights['alpha'],
            'bias_init': self.weights['bias'],
        }
    
    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError("Invalid parameter '%s' for estimator %s" % (k, self))
        return self
    


def mse(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred)**2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def sigmoid_activation(x):
    return 1. / (1. + np.exp(-x))

def identity_activation(x):
    return x


    

    
