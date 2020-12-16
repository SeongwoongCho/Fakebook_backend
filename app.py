import os
import sys
import base64
import cv2
import warnings
import numpy as np
import imageio as io
import secrets
import pymysql

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from google.cloud import storage
from config import Config

sys.path.append('./fewshot-face-translation-GAN')
warnings.filterwarnings('ignore')

from pygifsicle import optimize
from models import FaceTranslationGANInferenceModel
from face_toolbox_keras.models.verifier.face_verifier import FaceVerifier
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
from face_toolbox_keras.models.detector.iris_detector import IrisDetector
from utils import utils


app = Flask(__name__)
CORS(app)
config = Config()
models = None
db = pymysql.connect(
    host=       config.DB_HOST,
    port=       config.DB_PORT,
    user=       config.DB_USER,
    passwd=     config.DB_PASSWD,
    db=         config.DB_SCHEMA,
    charset=    config.DB_CHARSET
)


def exec_sql(db, sql):
    with db.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()
        db.commit()
    return result


def load(filepath):
    is_gif = filepath.endswith('.gif') or filepath.endswith('.GIF')
    img_np = []

    if is_gif:
        gif = cv2.VideoCapture(filepath)
        it, frame = gif.read()
        while it:
            it, frame = gif.read()
            if it:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_np.append(frame)
    else:
        img_np.append(cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB))

    return img_np


def init():
    global models

    # Load Models
    model = FaceTranslationGANInferenceModel()
    model.load_weights('./fewshot-face-translation-GAN/weights/')
    fv = FaceVerifier(classes=512)
    fp = face_parser.FaceParser()
    fd = face_detector.FaceAlignmentDetector()
    idet = IrisDetector()
    models = (model, fv, fp, fd, idet)


def inference(models, images, template):
    model, fv, fp, fd, idet = models
    fns_tar = images
    fn_srcs = load(template)
    tar, emb_tar = utils.get_tar_inputs(fns_tar, fd, fv)

    result = []

    for fn_src in fn_srcs:
        try:
            src, mask, aligned_im, (x0, y0, x1, y1), landmarks = utils.get_src_inputs(fn_src, fd, fp, idet)
            out = model.inference(src, mask, tar, emb_tar)
            result_face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))
            result_img = utils.post_process_result(fn_src, fd, result_face, aligned_im, src, x0, y0, x1, y1, landmarks)
            result.append(result_img)
        except:
            result.append(fn_src)
    
    return result


def upload_blob(bucket_name, src, dest):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest)
    blob.upload_from_filename(src)
    print ("File {} uploaded to {}.".format(src, dest))
    pass


@app.route('/generate', methods=['POST'])
def route_generate():
    args = request.json
    name, img_bytes, template = args['name'], args['img_bytes'], args['template']
    print(name, img_bytes, template)

    # Generate a token corresponding to the image
    token = secrets.token_urlsafe(16)

    # Add information to Database
    sql = f"INSERT INTO images(token, name, status) VALUES ('{token}', '{name}', 'converting')"
    exec_sql(db, sql)

    # Image preprocessing
    img_data = base64.b64decode(args['img_bytes'])
    img_np = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_np, flags=1)[...,::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = [img]
    
    template_path = './assets/gif/{}.gif'.format(template)

    # Make inference
    result = inference(models, images=img_list, template=template_path)

    # Save to local file system
    local_path = os.path.join('temp/', token + '.gif')
    with io.get_writer(local_path, mode='I', duration=0.1) as writer:
        for image in result:
            writer.append_data(image)
    optimize(local_path)
    # Upload to Google Cloud Storage
    remote_path = os.path.join('posts/', token + '.gif')
    upload_blob("fakebook-posts", local_path, remote_path)

    # Add information to Database
    sql = f"UPDATE images SET status='converted' WHERE token='{token}'"
    exec_sql(db, sql)

    return jsonify({ 'remote_path': remote_path })


if __name__ == '__main__':
    print('Initializing Web Server...')
    init()
    print('Initializing Done!')
    app.run('0.0.0.0', port=5000)
