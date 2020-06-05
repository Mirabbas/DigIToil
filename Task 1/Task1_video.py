import numpy as np
import cv2 as cv
from openvino.inference_engine import IENetwork, IECore


# Setup network
net = IENetwork('single-image-super-resolution-1033.xml', 'single-image-super-resolution-1033.bin')

# Read an image
cap = cv.VideoCapture(0)
while True:
        ret, img = cap.read()
        inp_h, inp_w = img.shape[0], img.shape[1]
        out_h, out_w = inp_h * 3, inp_w * 3  # Do not change! This is how model works


# Workaround for reshaping bug
        c1 = net.layers['79/Cast_11815_const']
        c1.blobs['custom'][4] = inp_h
        c1.blobs['custom'][5] = inp_w

        c2 = net.layers['86/Cast_11811_const']
        c2.blobs['custom'][2] = out_h
        c2.blobs['custom'][3] = out_w

# Reshape network to specific size
        net.reshape({'0': [1, 3, inp_h, inp_w], '1': [1, 3, out_h, out_w]})

# Load network to device
        ie = IECore()
        exec_net = ie.load_network(net, 'CPU')

# Prepare input
        inp = img.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)
        inp = inp.reshape(1, 3, inp_h, inp_w)
        inp = inp.astype(np.float32)

# Prepare second input - bicubic resize of first input
        resized_img = cv.resize(img, (out_w, out_h), interpolation=cv.INTER_CUBIC)
        resized = resized_img.transpose(2, 0, 1)
        resized = resized.reshape(1, 3, out_h, out_w)
        resized = resized.astype(np.float32)

        outs = exec_net.infer({'0': inp, '1': resized})

        out = next(iter(outs.values()))

        out = out.reshape(3, out_h, out_w).transpose(1, 2, 0)
        out = np.clip(out * 255, 0, 255)
        out = np.ascontiguousarray(out).astype(np.uint8)
        cv.imshow('img', img)
        cv.imshow('super resolution', out)
        k = cv.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
                break
cap.release()
cv.destroyAllWindows()