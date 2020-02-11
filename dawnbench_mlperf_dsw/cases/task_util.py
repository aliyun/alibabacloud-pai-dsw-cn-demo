from resnet50_v1_task.main_task import Resnet50V1Task


def create_task(task_name, height=224, width=224):

  if task_name == 'imagenet':
    task = Resnet50V1Task(height, width)
  else:
    raise ValueError('unrecognized task name' + task_name)
  return task
