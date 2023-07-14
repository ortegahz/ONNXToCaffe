#from __future__ import print_function 
import caffe
import pickle
#GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
#caffe.set_mode_cpu()
#caffe.set_device(GPU_ID)
from datetime import datetime
import numpy as np
import struct
import sys, getopt
import cv2, os, re
import pickle as p
import matplotlib.pyplot as pyplot
import ctypes
import codecs
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as caffe_protobuf

supported_layers=[
"Convolution", "Deconvolution", "Pooling", "InnerProduct", "LRN", "BatchNorm", "Scale", "Bias", "Eltwise", "ReLU", "PReLU", "AbsVal", "TanH", "Sigmoid", "BNLL", "ELU", "LSTM", "RNN", "Softmax", "Exp", "Log", "Reshape", "Flattern", "Split", "Slice", "Concat", "SPP", "Power", "Threshold", "MVN", "Parameter", "Reduction", "Proposal",  "Custom", "Input", "Dropout"]

def isSupportedLayer(layer_type):
    for type in supported_layers:
        if(layer_type == type):
            return True
    return False
	
def enable_cuda():
    flag = 0
    if(False == os.path.exists('C:\\hisilicon\\nnie_sim.ini')):
        return False
    with open('C:\\hisilicon\\nnie_sim.ini', 'rb') as input_file:
         for line in input_file:
             line = line.strip()
             if(str(line).find("[CUDA_CALC_EN]") != -1):
                 flag = 1
             if(1 == flag):
                 if(line.isdigit() and (int(line) == 1)):
                    return True
    return False


def image_to_array(img_file, shape_c_h_w, output_dir):
    result = np.array([])
    print("converting begins ...")
    resizeimage = cv2.resize(cv2.imread(img_file), (shape_c_h_w[2],shape_c_h_w[1]))
    b,g,r = cv2.split(resizeimage )
    height, width, channels = resizeimage.shape
    length = height*width
    #print(channels )
    r_arr = np.array(r).reshape(length)
    g_arr = np.array(g).reshape(length)
    b_arr = np.array(b).reshape(length)
    image_arr = np.concatenate((r_arr, g_arr, b_arr))
    result = image_arr.reshape((1, length*3))
    print("converting finished ...")
    file_path = os.path.join(output_dir, "test_input_img_%d_%d_%d.bin"%(channels,height,width))
    with open(file_path, mode='wb') as f:
         p.dump(result, f)
    print("save bin file success")

def image_to_bin(img_file,shape_c_h_w, output_dir):
    print("converting begins ...")
    #image = cv2.imread(img_file)
    image = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), 1)
    image = cv2.resize(image, (shape_c_h_w[2],shape_c_h_w[1]))
    image = image.astype('uint8')
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    file_path = os.path.join(output_dir, "test_input_img_%d_%d_%d.bin"%(channels,height,width))
    fileSave =  open(file_path,'wb')
    for step in range(0,height):
       for step2 in range (0, width):
          fileSave.write(image[step,step2,2])
    for step in range(0,height):
       for step2 in range (0, width):
          fileSave.write(image[step,step2,1])
    for step in range(0,height):
       for step2 in range (0, width):
          fileSave.write(image[step,step2,0])

    fileSave.close()
    print("converting finished ...")

def bin_to_image(bin_file,shape_c_h_w):
    #fileReader = open(bin_file,'rb', encoding='utf-8')
    fileReader = open(bin_file.encode('gbk'),'rb')
    height = shape_c_h_w[1]
    width = shape_c_h_w[2]
    channel = shape_c_h_w[0]
    imageRead = np.zeros((shape_c_h_w[1], shape_c_h_w[2], shape_c_h_w[0]), np.uint8)
    for step in range(0,height):
       for step2 in range (0, width):
          a = struct.unpack("B", fileReader.read(1))
          imageRead[step,step2,2] = a[0]
    for step in range(0,height):
       for step2 in range (0, width):
          a = struct.unpack("B", fileReader.read(1))
          imageRead[step,step2,1] = a[0]
    for step in range(0,height):
       for step2 in range (0, width):
          a = struct.unpack("B", fileReader.read(1))
          imageRead[step,step2,0] = a[0]
    fileReader.close()
    return imageRead
 
def isfloat(value):
    try:
       float(value)
       return True
    except ValueError:
       return False


def get_float_numbers(floatfile):
    mat = []
    with open(floatfile.encode('gbk'), 'rb') as input_file:
         for line in input_file:
             line = line.strip()
             for number in line.split():
                 if isfloat(number):
                    mat.append(float(number))
    return mat

def isHex(value):
     try:
       int(value,16)
       return True
     except ValueError:
       return False

def isHex_old(value):
    strvalue=str(value)
    length = len(strvalue)
    if length == 0:
        return False
    i = 0
    while(i < length):
        if not (strvalue[i] >= 'a' and strvalue[i] <= 'e' or strvalue[i] >= 'A' and strvalue[i] <= 'E' or strvalue[i] >= '0' and strvalue[i] <= '9'):
            return False
        i += 1
    return True

def get_hex_numbers(hexfile):
    mat = []
    with open(hexfile.encode("gbk")) as input_file:
        for line in input_file:
            line = line.strip()
            for number in line.split():
                if isHex(number):
                    mat.append(1.0*ctypes.c_int32(int(number,16)).value/4096)
    return mat 

def print_CNNfeaturemap(net, output_dir):
    params = list(net.blobs.keys())
    print (params)
    node_name = ''
    for i, pr in enumerate(params[0:]):
        for top_name in net.top_names:
           if pr in net.top_names[top_name]:
               node_name = top_name
               break
        print((i, pr, node_name))   
        res = net.blobs[pr].data[...]
        pr = node_name if not node_name == 'input' else pr
        pr = pr.replace('/', '_')
        print (res.shape)
        for index in range(0,res.shape[0]):
           if len(res.shape) == 4:
              filename = os.path.join(output_dir, "%s_output%d_%d_%d_%d_caffe.linear.float"%(pr,index,res.shape[1],res.shape[2],res.shape[3]))
           elif len(res.shape) == 3:
              filename = os.path.join(output_dir, "%s_output%d_%d_%d_caffe.linear.float"%(pr, index,res.shape[1],res.shape[2]))
           elif len(res.shape) == 2:
              filename = os.path.join(output_dir, "%s_output%d_%d_caffe.linear.float"%(pr,index,res.shape[1]))
           elif len(res.shape) == 1:
              filename = os.path.join(output_dir, "%s_output%d_caffe.linear.float"%(pr,index))
           f = open(filename, 'wb') 

           np.savetxt(f, list(res.reshape(-1, 1)))

def main(argv):
    if(enable_cuda() == True):
        caffe.set_mode_gpu()
        caffe.set_device(0)
    if len(argv) < 6:
        print ('CNN_convert_bin_and_print_featuremap.py -m <model_file> -w <weight_file> -i <img_file or bin_file or float_file> -p <"104","117","123" or "ilsvrc_2012_mean.npy">')
        print ('-m <model_file>:  .prototxt, batch num should be 1')
        print ('-w <weight_file>: .caffemodel')
        print ('-i <img_file>:    .JPEG or jpg or png or PNG or bmp or BMP')
        print ('-i <bin_file>:    test_img_$c_$h_$w.bin')
        print ('-i <float_file>:  %s_output%d_%d_%d_%d_caffe.linear.float')
        print ('-n <norm_type>:   0(default): no process, 1: sub img-val and please give the img path in the parameter p, 2: sub channel mean value and please give each channel value in the parameter p in BGR order, 3: dividing 256, 4: sub mean image file and dividing 256, 5: sub channel mean value and dividing 256') 
        print ('-s <data_scale>:  optional, if not set, 0.003906 is set by default')
        print ('-p <"104", "117", "123" or "ilsvrc_2012_mean.npy" or "xxx.binaryproto">: -p "104", "117", "123" is sub channel-mean-val, -p "ilsvrc_2012_mean.npy" is sub img-val and need a ilsvrc_2012_mean.npy')
        print ('-o <output_dir:   optional, if not set, there will be a directory named output created in current dir>')
        print ('any parameter only need one input')

        sys.exit(2)
    norm_type = 0
    data_scale = 0.003906
    output_dir = 'output/'
    opts, args = getopt.getopt(argv, "hm:w:i:n:s:p:o:")
    for opt, arg in opts:
        if opt == '-h':
            print ('CNN_convert_bin_and_print_featuremap.py -m <model_file> -w <weight_file> -i <img_file or bin_file or float_file> -p <"104","117","123" or "ilsvrc_2012_mean.npy">')
            print ('-m <model_file>:  .prototxt, batch num should be 1')
            print ('-w <weight_file>: .caffemodel')
            print ('-i <img_file>:    .JPEG or jpg or png or PNG or bmp or BMP')
            print ('-i <bin_file>:    test_img_$c_$h_$w.bin')
            print ('-i <float_file>:  %s_output%d_%d_%d_%d_caffe.linear.float')
            print ('-n <norm_type>:   0(default): no process, 1: sub img-val and please give the img path in the parameter p, 2: sub channel mean value and please give each channel value in the parameter p in BGR order, 3: dividing 256, 4: sub mean image file and dividing 256, 5: sub channel mean value and dividing 256')
            print ('-s <data_scale>:  optional, if not set, 0.003906 is set by default')
            print ('-p <"104", "117", "123", "ilsvrc_2012_mean.npy" or "xxx.binaryproto">: -p "104", "117", "123" is sub channel-mean-val, -p "ilsvrc_2012_mean.npy" is sub img-val and need a ilsvrc_2012_mean.npy')
            print ('-o <output_dir:   optional, if not set, there will be a directory named output created in current dir>')
            print ('any parameter only need one input')

            sys.exit()
        elif opt == "-m":
            model_filename = arg
        elif opt == "-w":
            weight_filename = arg
        elif opt == "-i":
            img_filename = arg
        elif opt == "-n":
            norm_type = arg
        elif opt == "-s":
            data_scale = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-p":
            meanfile = arg # default is to divide by 255
            initialway = "sub mean by: " + meanfile
        

    #net = caffe.Net(model_filename.encode('gbk'), caffe.TEST)
    train_net = caffe_pb2.NetParameter()
    f=open(model_filename.encode('gbk'), 'rb')
    train_str = f.read()
    caffe_protobuf.text_format.Parse(train_str, train_net)
    f.close()
    layers = train_net.layer
 
    for layer in layers:
        #print (layer.type)
        if(False == isSupportedLayer(layer.type)):
            print("Layer " + layer.name + " with type " + layer.type + " is not supported, please refer to chapter 3.1.4 and FAQ of \"HiSVP Development Guide.pdf\" to extend caffe!")
            sys.exit(1)
    print ('model file is ', model_filename)
    print ('weight file is ', weight_filename)
    print ('image file is ', img_filename)
    print ('image preprocessing method is ', norm_type) # default is no process
    print ('output dir is ', output_dir)
    print ('data scale is ', data_scale)
    net = caffe.Net(model_filename.encode('gbk'), weight_filename.encode('gbk'), caffe.TEST)
    
    print ('model load success')
    
    if norm_type == '1' or norm_type == '4': 
       if not os.path.isfile(meanfile):
          print("Please give the mean image file path") 
          sys.exit(1)
       if meanfile.endswith('.binaryproto'):
          meanfileBlob = caffe.proto.caffe_pb2.BlobProto()
          meanfileData = open(meanfile.encode('gbk'), 'rb').read()
          meanfileBlob.ParseFromString(meanfileData)
          arr = np.array(caffe.io.blobproto_to_array(meanfileBlob))
          out = arr[0]
          np.save('transMean.npy', out)
          meanfile = 'transMean.npy'
    #if norm_type == '2' or norm_type == '5':
       #print('meanshape:') 
       #lmeanfile=meanfile.split(',')
       #if not isfloat(lmeanfile[0]) or not isfloat(lmeanfile[1]) or not isfloat(lmeanfile[2]): 
       #   print("Please give the channel mean value in BGR order") 
       #   sys.exit(1)

    print ('model file is ', model_filename)
    print ('weight file is ', weight_filename)
    print ('image file is ', img_filename)
    print ('image preprocessing method is ', norm_type) # default is no process
    print ('output dir is ', output_dir)
    print ('data scale is ', data_scale)
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if img_filename.endswith('.jpg') or img_filename.endswith('.png') or img_filename.endswith('.jpeg') or img_filename.endswith('.bmp') or img_filename.endswith('.JPEG') or img_filename.endswith('.PNG') or img_filename.endswith('.JPG') or img_filename.endswith('.BMP'):
       image_to_bin(img_filename, net.blobs['images'].data.shape[1:], output_dir)
       if net.blobs['images'].data.shape[1]==1:
          color = False
       elif net.blobs['images'].data.shape[1]==3:
          color = True
       img = cv2.imdecode(np.fromfile(img_filename, dtype=np.uint8), -1)
       #img = cv2.imread(img_filename, color) # load the image using caffe io
       inputs = img
    elif img_filename.endswith('.bin'):
       fbin = open(img_filename.encode('gbk'))
       data = bin_to_image(img_filename,net.blobs['images'].data.shape[1:])
       inputs = data
    elif img_filename.endswith('.float'):
       data = np.asarray(get_float_numbers(img_filename))
       inputs = data
       inputs= np.reshape(inputs, net.blobs[list(net.blobs.keys())[0]].data.shape)
    elif img_filename.endswith('.hex'):
       data = np.asarray(get_hex_numbers(img_filename))
       inputs = data
       inputs= np.reshape(inputs,net.blobs[list(net.blobs.keys())[0]].data.shape)
    else:
       print("errors: unknown input file!")
       sys.exit(1)

    if len(inputs):
       transformer = caffe.io.Transformer({'images': net.blobs['images'].data.shape})
       if net.blobs['images'].data.shape[1]==3:
          transformer.set_transpose('images', (2,0,1))
       if norm_type == '1' or norm_type == '4' and os.path.isfile(meanfile): # (sub mean by meanfile):  
          if net.blobs['images'].data.shape[1]==3:
              transformer.set_mean('images',np.load(meanfile).mean(1).mean(1))
          elif net.blobs['images'].data.shape[1]==1:
              tempMeanValue = np.load(meanfile).mean(1).mean(1)
              tempa = list(tempMeanValue)
              inputs = inputs - np.array(list(map(float, [tempa[0]])))
       elif norm_type == '2' or norm_type == '5':
          if net.blobs['images'].data.shape[1]==3:
              lmeanfile=meanfile.split(',')
              if len(lmeanfile) != 3:
                  print("Please give the channel mean value in BGR order with 3 values, like 112,113,120") 
                  sys.exit(1)
              if not isfloat(lmeanfile[0]) or not isfloat(lmeanfile[1]) or not isfloat(lmeanfile[2]): 
                  print("Please give the channel mean value in BGR order") 
                  sys.exit(1)
              else:
                  transformer.set_mean('images',np.array(list(map(float,re.findall(r'[-+]?\d*\.\d+|\d+',meanfile)))))
          elif net.blobs['images'].data.shape[1]==1:
              lmeanfile=meanfile.split(',')
              if isfloat(lmeanfile[0]):  # (sub mean by channel)
                  inputs = inputs - np.array(list(map(float, [lmeanfile[0]])))

       elif norm_type == '3':
          inputs = inputs * float(data_scale)
       if img_filename.endswith('.txt') or img_filename.endswith('.float') or img_filename.endswith('.hex'):
          print (inputs.shape)
          data = inputs
       else:
          data = np.asarray([transformer.preprocess('images', inputs)])
       if norm_type == '4' or norm_type == '5':
          data = data * float(data_scale)
    
    data_reshape= np.reshape(data,net.blobs[list(net.blobs.keys())[0]].data.shape)
    net.blobs[list(net.blobs.keys())[0]].data[...] = data_reshape.astype('float')
    out = net.forward()
    
    print_CNNfeaturemap(net, output_dir)
    sys.exit(0)
if __name__=='__main__':
    main(sys.argv[1:])
