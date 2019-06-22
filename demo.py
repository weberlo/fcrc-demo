import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from nose.tools import nottest
import numpy as np
import tvm
from tvm.contrib import graph_runtime, util
from tvm import relay
import tvm.micro as micro
from tvm.relay.testing import resnet
import matplotlib.pyplot as plt
import warnings

model_name = "lenet"

# Use GPU if one exists, else use CPU
ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

# MNIST images are 28x28. Total pixels in input layer is 28x28 = 784
num_inputs = 784
# Clasify the images into one of the 10 digits
num_outputs = 10
# 64 images in a batch
batch_size = 64

# Load the training data
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True).transform_first(transforms.ToTensor()),
                                   batch_size, shuffle=True)

def build_lenet(net):
    """Build a simple convolutional network"""
    with net.name_scope():
        # First convolution
        net.add(gluon.nn.Conv2D(channels=5, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Second convolution
        net.add(gluon.nn.Conv2D(channels=5, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Flatten the output before the fully connected layers
        net.add(gluon.nn.Flatten())
        # First fully connected layers with 512 neurons
        net.add(gluon.nn.Dense(32, activation="relu"))
        # Second fully connected layer with as many neurons as the number of classes
        net.add(gluon.nn.Dense(num_outputs))
        return net


def train_model(model):
    """Train a given model using MNIST data"""
    # Initialize the parameters with Xavier initializer
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    # Use cross entropy loss
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    # Use Adam optimizer
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': .001})

    # Train for one epoch
    for epoch in range(1):
        # Iterate through the images and labels in the training data
        for batch_num, (data, label) in enumerate(train_data):
            # get the images and labels
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Ask autograd to record the forward pass
            with autograd.record():
                # Run the forward pass
                output = model(data)
                # Compute the loss
                loss = softmax_cross_entropy(output, label)
            # Compute gradients
            loss.backward()
            # Update parameters
            trainer.step(data.shape[0])

            # Print loss once in a while
            if batch_num % 50 == 0:
                curr_loss = nd.mean(loss).asscalar()
                print("Epoch: %d; Batch %d; Loss %f" % (epoch, batch_num, curr_loss))


def get_sample_point():
    """Grabs a single input/label pair from MNIST"""
    def transform(data, label):
        return data.astype(np.float32)/255.0, label.astype(np.float32)

    # Load ten random images from the test dataset
    sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                           1, shuffle=True)
    for data, label in sample_data:
        img = nd.transpose(data, (1,0,2,3))
        img = nd.reshape(img, (28,28,1))
        imtiles = nd.tile(img, (1,1,3))
        plt.imshow(imtiles.asnumpy())
        plt.savefig("test_input.png")

        data = nd.transpose(data, (0, 3, 1, 2))
        data = data.as_in_context(ctx).asnumpy()
        label = int(label.asnumpy()[0])
        return data, label


# lenet = build_lenet(gluon.nn.HybridSequential())
# lenet.hybridize()
# train_model(lenet)

# lenet.export(model_name, epoch=1)

# Import model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lenet = gluon.nn.SymbolBlock.imports(f"{model_name}-symbol.json", ['data'], f"{model_name}-0001.params", ctx=ctx)

# Convert to Relay
func, params = relay.frontend.from_mxnet(
    lenet, shape={"data": (1, 1, 28, 28)})

# Grab an example
image, label = get_sample_point()
print()
input('check "test_input.png"...')

# Begin a session
import time
start_time = time.time()
with micro.Session("openocd", "riscv64-unknown-elf-", port=6666) as sess:
    # Build the function
    mod = sess.build(func, params=params)
    end_time = time.time()
    print(f'initialization took {end_time - start_time} seconds')

    start_time = time.time()
    # Execute with `image` as the input.
    mod.run(data=image)
    end_time = time.time()
    print(f'model execution took {end_time - start_time} seconds')
    print()

    # Get output
    tvm_output = mod.get_output(0)

    # Check prediction
    print(f'expected label: {label}')
    prediction_idx = np.argmax(tvm_output.asnumpy()[0])
    print(f'actual label: {prediction_idx}')
