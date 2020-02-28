
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import logging
import os
import string
import random
from os.path import join, dirname, realpath


from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras import applications

from datetime import datetime

import argparse

import numpy as np
import tensorflow as tf
from flask import *
from werkzeug.utils import secure_filename

app = Flask(__name__,static_url_path = "/static", static_folder = "static")
UPLOAD_FOLDER = '/static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif','jpeg'])
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#学習データを読み込む
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph

#画像ファイルの読み込み
def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  result = sess.run(normalized)

  return result

#ラベルの読み込み
def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


#ファイルを読み込み、結果を返す
def getresult(file_name,model_file,label_file):


  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(
    file_name,
    input_height=input_height,
    input_width=input_width,
    input_mean=input_mean,
    input_std=input_std)

  input_name = "import/" + "Placeholder"
  output_name = "import/" + "final_result"
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.compat.v1.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
      input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)
  result = str(labels[0])

  j = 0
  for i in top_k:
    if j == 0:
      result = str(labels[i])
      break

    print(labels[i], results[i])
    j += 1

  if result == "201":
    result = "国鉄201系"
  elif result == "tobu9000":
    result = "東武9000系"
  elif result == "tobu10030":
    result = "東武10030系"
  elif result == "tobu50000":
    result = "東武50000系"
  elif result == "tobu9000":
    result = "東武9000系"
  elif result == "metro16000":
    result = "東京メトロ16000系"
  elif result == "metro13000":
    result = "東京メトロ13000系"
  elif result == "metro10000":
    result = "東京メトロ10000系"
  elif result == "metro9000":
    result = "営団9000系"
  elif result == "metro6000":
    result = "営団6000系"
  elif result == "e233 5000":
    result = "e233系5000番台"
  elif result == "e233 6000":
    result = "e233系6000番台"
  elif result == "e233 7000":
    result = "e233系7000番台"
  elif result == "e233 8000":
    result = "e233系8000番台"
  return result

#Webアプリ側の処理
@app.route('/show/<filename>')
def uploaded_file(filename):
    filename = 'http://127.0.0.1:8000/uploads/' + filename
    return render_template('index.html', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    model_file = "retrained_graph.pb"
    label_file = "retrained_labels.txt"
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file:
            
            filename = secure_filename(img_file.filename)
              
            
            
            
          
            savedir = os.path.abspath(os.path.dirname(__file__)) + app.config['UPLOAD_FOLDER'] + img_file.filename
            print(savedir,sys.stdout)

            file_name=savedir

            img_file.save(savedir)
            
            
           
            
            return render_template('index.html', traintype="もしかして:"+getresult(file_name, model_file, label_file),filename=img_file.filename)
        else:
            return ''' <p>画像ではありません</p> '''
    else:
        return redirect(url_for('index'))







if __name__ == "__main__":


  app.run(port=8000, debug=True,threaded=True)
















