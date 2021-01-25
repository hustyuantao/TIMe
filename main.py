import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator

if __name__ == "__main__":
    conf = Configurator("default.properties", default_section="hyperparameters")

    recommender = conf["recommender"]
    
    # Set GPU 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("Not Found GPU Devices.")
    
    dataset = Dataset(conf)
    my_module = importlib.import_module("model.sequential_recommender." + recommender)
    
    MyClass = getattr(my_module, recommender)
    model = MyClass(dataset, conf)

    model.build_graph()
    model.train_model()
