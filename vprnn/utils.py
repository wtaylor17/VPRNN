from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_scalar(tb_path, scalar_name, extract_values=True):
    event_acc = EventAccumulator(tb_path)
    event_acc.Reload()
    tag_array = event_acc.Scalars(scalar_name)
    if extract_values:
        return [obj.value for obj in tag_array]
    return tag_array
