"""" 对输入的超参数进行处理 """
import os
import numpy as np
import argparse

""" 设置运行的背景context """
from mindspore import context

""" 对数据集进行预处理 """
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype

""" 构建神经网络 """
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.common.initializer import Normal

""" 训练时对模型参数的保存 """
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor

""" 导入模型训练需要的库 """
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model

# 导入模型参数
from mindspore.train.serialization import load_checkpoint, load_param_into_net



#处理数据集
def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    # 定义数据集
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义所需要操作的map映射
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 使用map映射函数，将数据操作应用到数据集
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # 进行shuffle、batch、repeat操作
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds

class LeNet5(nn.Cell):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_classes, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 创建模型
net = LeNet5()
# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 定义优化器
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
# 构建模型
model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})


def train_network(model, epoch_size, data_path, repeat_size, checkpoint_callback):
    print("============== 开始训练 ==============")
    ds_train = create_dataset(os.path.join(args.data_path, "train"), 32, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[checkpoint_callback, LossMonitor(125), TimeMonitor()],
                dataset_sink_mode=False)
    print("============== 训练结束 ==============")

def train(args):
    global model
    config_checkpoint = CheckpointConfig(save_checkpoint_steps=125, keep_checkpoint_max=15)
    checkpoint_callback = ModelCheckpoint(prefix="checkpoint_lenet", directory=args.model_path,
                                          config=config_checkpoint)

    train_network(model, args.epochs, args.data_path, 1, checkpoint_callback)
    ds_eval = create_dataset(os.path.join(args.data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))



def test(args):
    global model
    # 加载模型参数
    param_dict = load_checkpoint(args.model_path)
    load_param_into_net(net, param_dict)
    ds_eval = create_dataset(os.path.join(args.data_path, "test")).create_dict_iterator()
    data = next(ds_eval)
    # images为测试图片，labels为测试图片的实际分类
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()
    _batch_size = 8
    # 使用函数model.predict预测image对应分类
    output = model.predict(Tensor(data['image']))
    predicted = np.argmax(output.asnumpy(), axis=1)
    # 输出预测分类与实际分类
    for i in range(_batch_size):
        print(f'Predicted: "{predicted[i]}", Actual: "{labels[i]}"')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="Mnist", help="the path of the MNIST dataset")
    args.add_argument("--model_path", default="/home/user_101002/private/p101/wm/image/model", help="the path of the model")
    args.add_argument("--epochs", type=int, default=5, help="the number of epochs")
    args.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="the mode of the code")
    args = args.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    pass
