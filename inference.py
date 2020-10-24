import sys
import warnings
import numpy as np
import cv2
sys.path.append('./fewshot-face-translation-GAN')
warnings.filterwarnings("ignore")

from models import FaceTranslationGANInferenceModel
from face_toolbox_keras.models.verifier.face_verifier import FaceVerifier
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
from face_toolbox_keras.models.detector.iris_detector import IrisDetector
from utils import utils

def load_img_file(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return im

def load_gif_file(path):
    gif = cv2.VideoCapture(path)
    duration = gif.get(cv2.CAP_PROP_POS_MSEC)
    ret, frame = gif.read()
    result = []
    while ret:
        ret, frame = gif.read()
        if ret:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result.append(frame)
    return result,gif.get(cv2.CAP_PROP_POS_MSEC)
def load(path):
    if path.endswith('.gif') or path.endswith('.GIF'):
        return load_gif_file(path)
    else:
        return load_img_file(path)

def load_models():
    model = FaceTranslationGANInferenceModel()
    model.load_weights('./fewshot-face-translation-GAN/weights/')
    fv = FaceVerifier(classes=512)
    fp = face_parser.FaceParser()
    fd = face_detector.FaceAlignmentDetector()
    idet = IrisDetector()

    return model,fv,fp,fd,idet
def inference(models, images_list, template_path):
    """
    models : loaded models from calling load_models
    images_list : list of target images, length >=1
    template_path : which template to choose

    return list of template
    (if template gif -> len(result) > 1, if template image -> len(result) == 1)
    """

    model,fv,fp,fd,idet = models

    fns_tar = images_list
    fn_srcs,duration = load(template_path)

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

if __name__ == '__main__':
    models = load_models()
    images_list = [load("./datas/tar0-minseong.jpeg")]
    template_path = "./datas/gif/minseong-sexy-body.gif"
    result = inference(models, images_list = images_list,template_path = template_path)

    print(len(result))

    import imageio as io
    import os
    #making animation
    with io.get_writer('./datas/result0.gif', mode='I', duration=0.1) as writer:
        for image in result:
            writer.append_data(image)
    writer.close()