from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from keras.callbacks import Callback
import os

from .models import save_rvpnn
from time import time


def load_tensorboard_scalar(tb_path, scalar_name, extract_values=True):
    event_acc = EventAccumulator(tb_path)
    event_acc.Reload()
    tag_array = event_acc.Scalars(scalar_name)
    if extract_values:
        return [obj.value for obj in tag_array]
    return tag_array


class BestModelCallback(Callback):
    def __init__(self, data, model_root='', start=float('inf'), f=min):
        super(BestModelCallback, self).__init__()
        self.data = data
        self.model_root = model_root
        self.val = start
        self.f = f

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.data
        val = self.model.evaluate(x, y, batch_size=min(1024, x.shape[0]))[-1]
        if self.f(self.val, val) == val:
            os.makedirs(self.model_root, exist_ok=True)
            model_tag = f'{round(val,4)}-{time()}'
            model_path = os.path.join(self.model_root, model_tag)
            save_rvpnn(self.model, model_path)
            self.val = val


def best_accuracy_callback(data, model_root=''):
    return BestModelCallback(data,
                             model_root=model_root,
                             start=0.0, f=max)


def periodic_lr_scheduler(period=100, factor=0.1, start=1):
    def scheduler(epoch, lr):
        if epoch < start or epoch % period != 0:
            return lr
        return lr * factor
    return scheduler
