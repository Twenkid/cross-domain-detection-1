import argparse

import chainer
import matplotlib.pyplot as plt
from chainercv import utils
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv.visualizations import vis_bbox

chainer.config.train = False
parser = argparse.ArgumentParser()
parser.add_argument('image', help='path for input image')
parser.add_argument('result', help='path for output image')
"""
parser.add_argument('--load', help='if not specified, use default model \
                                                trained on voc07+12 trainval') #voc doesn't load anymore                                                
parser.add_argument('--load', help='if not specified, the default is watercolor_dt_ssd300 \
                                                trained on voc07+12 trainval is not available online anymore in the address checked by chainercv') #use this or other default
"""                                              
parser.add_argument('--load', help='required, e.g. watercolor_dt_ssd300 \
                                                the original default, trained on voc07+12 trainval is not available online anymore in the address checked by chainercv') #use this or other default
                                        
                                                

parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--score_thresh', type=float, default=0.25)
args = parser.parse_args()

model = SSD300(    
    n_fg_class=len(voc_bbox_label_names, pretrained_model=args.load) #'voc0712' is not available anymore at https://chainercv-models.preferred.jp/ssd300_voc0712_converted_2017_06_06.npz)
    #n_fg_class=len(voc_bbox_label_names, pretrained_model='watercolor_dt_ssd300') #'voc0712' is not available anymore at https://chainercv-models.preferred.jp/ssd300_voc0712_converted_2017_06_06.npz
model.score_thresh = args.score_thresh

"""
# Now it's a required parameter and loaded on creation of SSD300:
if args.load:
    chainer.serializers.load_npz(args.load, model)
"""

if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

img = utils.read_image(args.image, color=True)
bboxes, labels, scores = model.predict([img])
vis_bbox(
    img, bboxes[0], labels[0], scores[0], label_names=voc_bbox_label_names)
plt.axis('off')
plt.tight_layout()
plt.savefig(args.result)
