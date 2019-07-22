import os, glob, argparse
import numpy as np
import scipy.io as io
from utils import img_preprocess
import tensorflow as tf
from yolo_models_feature_extraction import build_yolo_v2_feature_extraction


parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--weights_path", type=str, required=True)
args = parser.parse_args()


img_shape = (608, 608, 3)
weights_path = args.weights_path
inp_path = args.image_dir
save_dir = args.save_dir

os.makedirs(save_dir, exist_ok=True)

# build the model
model_input = tf.placeholder(tf.float32, shape=(None,)+img_shape)
model_output = build_yolo_v2_feature_extraction(model_input)

print('In  shape: '+str(model_input.get_shape())+'\n')
print('Out shape: '+str(model_output.get_shape())+'\n')

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True)))
print('Loading weights...')
saver.restore(sess, weights_path)
print('Done!\n')

all_inps = glob.glob(os.path.join(inp_path, '*.jpg'))
if len(all_inps) == 0:
    print('ERR: No jpg images found in '+inp_path+'\n')
    quit()
print('Found %d images!' % len(all_inps))

# process each image independently
for i, inp in enumerate(all_inps):
    if i % 1000 == 0:
        print('progress: {} / {}'.format(i+1, len(all_inps)))

    inp_feed = np.expand_dims(img_preprocess(inp, shape=img_shape, letterbox=True), 0)
    feed_dict = {model_input: inp_feed}
    out = sess.run(model_output, feed_dict)
    save_file = os.path.join(save_dir, inp.replace('.jpg', '.npy'))
    np.save(save_file, out)
