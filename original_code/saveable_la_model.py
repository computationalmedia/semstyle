import cPickle
import numpy as np
import lasagne as la
from lasagne.utils import floatX

class SaveableModel(object):
    def __init__(self):
        self.net = {}
        self.expr = {}
        self.var = {}

    def _get_params_dictionary(self):
        param_store = {}
        for layer_name in self.net:
            layer_params = self.net[layer_name].get_params(trainable=True)
            layer_params.extend(self.net[layer_name].get_params(saveable=True))
            param_store[layer_name] = layer_params
        return param_store

    def _get_params_values_dictionary(self):
        params_dict = self._get_params_dictionary()
        params_values_dict = {}
        for layer_name in params_dict:
            params_values_dict[layer_name] = []
            for param in params_dict[layer_name]:
                params_values_dict[layer_name].append(param.get_value())
        return params_values_dict

    def save_to_object(self, save_object={}):
        net_param_dict = self._get_params_values_dictionary()
        save_object["net_param_dict"] = net_param_dict
        return save_object

    def _set_all_param_values(self, param_values_dict):
        params_dict = self._get_params_dictionary()
        for layer_name in param_values_dict:
            if layer_name not in params_dict:
                print "Warning: layer '%s' is loaded but current model ignores it" % layer_name
                continue
            print layer_name
            assert len(params_dict[layer_name]) == len(param_values_dict[layer_name])
            for param, value in zip(params_dict[layer_name], param_values_dict[layer_name]):
                if not np.all(param.shape.eval() == value.shape):
                    raise AttributeError("Layer %s was a different shape model: wanted: %s, loaded: %s" % 
                            (layer_name, param.shape.eval(), value.shape))
                param.set_value(value)
        for layer_name in params_dict:
            if layer_name not in param_values_dict:
                print "Warning: for layer '%s' no parameters could be loaded, using default parameters" % layer_name

    def load_from_object(self, save_object):
        assert self.model_is_built
        # once the model has been built we can load from disk
        net_param_dict = save_object["net_param_dict"]
        self._set_all_param_values(net_param_dict)
