import logging
import os.path as osp
import time


# EVAL = True: just test, EVAL = False: train and eval
EVAL = False

DATASET = 'MSCOCO'

if DATASET == 'MIRFlickr':
    LABEL_DIR = 'D:/DataSet/mirflickr25k/LAll/mirflickr25k-lall.mat'
    TXT_DIR = 'D:/DataSet/mirflickr25k/YAll/mirflickr25k-yall.mat'
    IMG_DIR = 'D:/DataSet/mirflickr25k/IAll/mirflickr25k-iall.mat'
    NUM_EPOCH = 200
    LR_IMG = 0.005
    LR_TXT = 0.005
    # loss = 1 * loss1 + 1 * loss2 + 1 * loss3 + 1 * loss4 + 1 * loss5 + 1 * loss6 + 2 * loss7 + 0.1 * loss31 + 0.1 * loss32
    EVAL_INTERVAL = 1

if DATASET == 'NUSWIDE':
    LABEL_DIR = 'D:/DataSet/nus-wide-tc10/nus-wide-tc10-lall.mat'
    TXT_DIR = 'D:/DataSet/nus-wide-tc10/nus-wide-tc10-yall.mat'
    IMG_DIR = 'D:/DataSet/nus-wide-tc10/nus-wide-tc10-iall.mat'

    NUM_EPOCH = 200
    LR_IMG = 0.005
    LR_TXT = 0.005
    EVAL_INTERVAL = 10


if DATASET == 'MSCOCO':
    DATA_PATH = 'D:/DataSet/coco_data/COCO.mat'
    TEXT_PATH = 'D:/DataSet/coco_data/COCO_BoW.npy'
    # data = h5py.File(DATA_PATH)  
    #     # FALL_tmp = data['FALL']  # （1，87081）
    #     # XALL_tmp = data['XALL']  # （4096，87081）
    #     # param_tmp = data['param']
    #     # LAll_tmp = data['LAll']  #（91，87081）
    #     # LABEL_DIR = np.array(LAll_tmp).transpose()
    #     # TXT_DIR = np.squeeze(np.load(TEXT_PATH))
    #     # IMG_DIR = data['IAll']  #（87081，3，224，224）
    NUM_EPOCH = 200
    LR_IMG = 0.005
    LR_TXT = 0.005
    EVAL_INTERVAL = 10



K = 1.5
ETA = 0.1
ALPHA = 0.9

BATCH_SIZE = 16
CODE_LEN = 128
MOMENTUM = 0.7
WEIGHT_DECAY = 5e-4

GPU_ID = 0
NUM_WORKERS = 0
EPOCH_INTERVAL = 2

MODEL_DIR = './checkpoint'

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
log_name = now + '_log.txt'
log_dir = './log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info('--------------------------Current Settings--------------------------')
logger.info('EVAL = %s' % EVAL)
logger.info('DATASET = %s' % DATASET)
logger.info('CODE_LEN = %d' % CODE_LEN)
logger.info('GPU_ID =  %d' % GPU_ID)
logger.info('ALPHA = %.4f' % ALPHA)
logger.info('K = %.4f' % K)
logger.info('ETA = %.4f' % ETA)


logger.info('NUM_EPOCH = %d' % NUM_EPOCH)
logger.info('BATCH_SIZE = %d' % BATCH_SIZE)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)


logger.info('LR_IMG = %.4f' % LR_IMG)
logger.info('LR_TXT = %.4f' % LR_TXT)

logger.info('MOMENTUM = %.4f' % MOMENTUM)
logger.info('WEIGHT_DECAY = %.4f' % WEIGHT_DECAY)

logger.info('--------------------------------------------------------------------')
