import os

import caffe
import cv2
import numpy as np
import onnx
import onnxruntime

dump_path = os.getcwd()
dump_path = os.path.join(dump_path, 'output/dump')

if not os.path.exists(dump_path):
    os.makedirs(dump_path)


def getOnnxInputs(onnx_graph):
    input_tensors = {t.name for t in onnx_graph.initializer}
    inputs = []
    for i in onnx_graph.input:
        if i.name not in input_tensors:
            inputs.append(i.name)
    return inputs


def getOnnxOutputs(onnx_graph):
    outputs = []
    for i in onnx_graph.output:
        outputs.append(i.name)
    return outputs


def load_onnx_model(onnx_path):
    return onnxruntime.InferenceSession(onnx_path)


def load_caffe_model(prototxt_path, caffemodel_path):
    return caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)


# generate input tensor for test
def gen_input(models, in_node):
    np.random.seed(5)
    # get & check input shape
    in_shapes = [get_input_shape_onnx(models[0], in_node),
                 get_input_shape_caffe(models[1], in_node)]
    in_shape = in_shapes[0]
    for shape in in_shapes:
        if shape != in_shape:
            raise Exception("model input shape doesn't match: {} vs {}".format(shape, in_shape))
    # generate tensor of input shape with random value (NCHW)
    # input_tensor = np.random.rand(1, *in_shape).astype(np.float32)
    img = cv2.imread('/media/manu/samsung/pics/students_lt.bmp')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    batch = np.expand_dims(img, 0)
    input_tensor = np.ascontiguousarray(batch)
    input_tensor = input_tensor.astype(np.float32) / 255.
    return input_tensor


def get_input_shape_onnx(onnx_model, in_node):
    # default onnx shape is NCHW
    for node in onnxruntime.InferenceSession.get_inputs(onnx_model):
        if node.name == in_node:
            return node.shape[1:]


def get_input_shape_caffe(caffe_model, in_node):
    # default caffe shape is NCHW
    return list(caffe_model.blobs[in_node].shape)[1:]


# perform network forward
def run_models(models, in_node, out_node, input_tensor):
    net_results = []
    net_results.append(net_forward_onnx(models[0], in_node, out_node, input_tensor))
    net_results.append(net_forward_caffe(models[1], in_node, out_node, input_tensor))
    return net_results


def net_forward_onnx(onnx_model, in_node, out_node, input_tensor):
    result = onnx_model.run(out_node, {in_node: input_tensor})
    return result


def net_forward_caffe(caffe_model, in_node, out_node, input_tensor):
    caffe_model.blobs[in_node].data[...] = input_tensor
    caffe_model.forward()
    result = []
    for node in out_node:
        result.append(caffe_model.blobs[node].data)
    return result


# check output results
def check_results(net_results, onnx_info, caffe_info):
    onnx_results = net_results[0]
    caffe_results = net_results[1]
    for i, result in enumerate(onnx_results):
        print("onnx", result)
        print("caffe", caffe_results[i])

        # check if result are same by cosine distance
        dot_result = np.dot(result.flatten(), caffe_results[i].flatten())
        left_norm = np.sqrt(np.square(result).sum())
        right_norm = np.sqrt(np.square(caffe_results[i]).sum())
        cos_sim = dot_result / (left_norm * right_norm)
        print("cos sim between onnx and caffe models: {}".format(cos_sim))

        # cos_sim = 0.0
        if cos_sim < 0.9999:
            # dump result
            np.savetxt(os.path.join(dump_path, "final_out_onnx.txt"), result.flatten(), fmt='%.18f')
            np.savetxt(os.path.join(dump_path, "final_out_caffe.txt"), caffe_results[i].flatten(), fmt='%.18f')
            from layerComparator import compareLayers
            compareLayers(onnx_info, caffe_info)
            raise Exception("model output different")

    print("models similarity test passed")


def compareOnnxAndCaffe(onnx_path, prototxt_path, caffemodel_path):
    models = [load_onnx_model(onnx_path), load_caffe_model(prototxt_path, caffemodel_path)]

    onnx_graph = onnx.load(onnx_path).graph
    in_node = getOnnxInputs(onnx_graph)
    out_node = getOnnxOutputs(onnx_graph)
    if not len(in_node) == 1:
        raise Exception("only one input is supported, but {} provided: {}"
                        .format(len(in_node), in_node))
    print("input node: {}".format(in_node))
    print("output node: {}".format(out_node))

    # generate input tensor
    # in NCHW format
    input_tensor = gen_input(models, in_node[0])
    dump_input_file = os.path.join(dump_path, "input_{}x{}.txt".format(input_tensor.shape[2], input_tensor.shape[3]))
    np.savetxt(dump_input_file, input_tensor.flatten())
    print("input tensor shape of {}: {}".format(in_node[0], input_tensor.shape))
    print("dump input to {}".format(dump_input_file))

    # feed input tensors for each model and get result
    net_results = run_models(models, in_node[0], out_node, input_tensor)
    for i, node in enumerate(out_node):
        print("output tensor shape of {}: {} for onnx vs {} for caffe"
              .format(node, net_results[0][i].shape, net_results[1][i].shape))

    # check model results
    onnx_info = [onnx_path, in_node[0], dump_input_file, input_tensor.shape]
    caffe_info = [prototxt_path, caffemodel_path, in_node[0], dump_input_file, input_tensor.shape]
    check_results(net_results, onnx_info, caffe_info)


def compareCaffeAndCaffe(prototxt_path_a, caffemodel_path_a, prototxt_path_b, caffemodel_path_b):
    models = [load_caffe_model(prototxt_path_a, caffemodel_path_a),
              load_caffe_model(prototxt_path_b, caffemodel_path_b)]

    in_node = 'images'
    # out_node = ['outputs', '243', '251', '238', '246', '254']
    out_node = ['outputs', '256', '257']
    # out_node = ['outputs', '269', '268']

    # generate input tensor
    # in NCHW format
    input_tensor = np.random.rand(1, 3, 640, 640).astype(np.float32)

    net_results_a = net_forward_caffe(models[0], in_node, out_node, input_tensor)
    net_results_b = net_forward_caffe(models[1], in_node, out_node, input_tensor)

    # check model results
    for i, result in enumerate(net_results_a):
        print("caffe_a", result)
        print("caffe_b", net_results_b[i])

        # check if result are same by cosine distance
        dot_result = np.dot(result.flatten(), net_results_b[i].flatten())
        left_norm = np.sqrt(np.square(result).sum())
        right_norm = np.sqrt(np.square(net_results_b[i]).sum())
        cos_sim = dot_result / (left_norm * right_norm)
        print("cos sim between caffe and caffe models: {}".format(cos_sim))

        if cos_sim < 0.9999:
            # dump result
            np.savetxt(os.path.join(dump_path, "final_out_caffe_a.txt"), result.flatten(), fmt='%.18f')
            np.savetxt(os.path.join(dump_path, "final_out_caffe_b.txt"), net_results_b[i].flatten(), fmt='%.18f')
            raise Exception("model output different")

    print("models similarity test passed")
