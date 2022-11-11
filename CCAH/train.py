import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.sparse as sp
import datasets
import settings
from metric import  compress, calculate_top_map, calculate_map, p_topK
from models import ImgNet, GATNet

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class Session:
    def __init__(self):
        self.logger = settings.logger

        torch.cuda.set_device(settings.GPU_ID)


        if settings.DATASET == "MIRFlickr":
            self.train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
            self.test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
            self.database_dataset = datasets.MIRFlickr(train=False, database=True,transform=datasets.mir_test_transform)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=settings.BATCH_SIZE,
                                                        shuffle=True,
                                                        num_workers=settings.NUM_WORKERS,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=settings.BATCH_SIZE,
                                                       shuffle=False,
                                                       num_workers=settings.NUM_WORKERS)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                           batch_size=settings.BATCH_SIZE,
                                                           shuffle=False,
                                                           num_workers=settings.NUM_WORKERS)
        txt_feat_len = datasets.txt_feat_len
        self.CodeNet_I = ImgNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)#,img_feat_len=1024
        self.FeatNet_I = ImgNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)#,img_feat_len=1024
        self.FeatNet_I = ImgNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)

        self.GATNet = GATNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)


        if settings.DATASET == "MIRFlickr" :
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
                                         weight_decay=settings.WEIGHT_DECAY, nesterov=True)

        self.opt_T = torch.optim.SGD(self.GATNet.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM,
                                     weight_decay=settings.WEIGHT_DECAY, nesterov=True)
        self.best = 0

    def train(self, epoch):


        self.FeatNet_I.cuda().eval()

        self.CodeNet_I.cuda().train()
        self.GATNet.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.GATNet.set_alpha(epoch)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for GATNet: %.3f' % (
            epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.GATNet.alpha))

        for idx, (img, txt, labels, _) in enumerate(self.train_loader):
            batch_size = img.size(0)

            #构建图
            """文本部分使用CAT"""
            txt_feature, adjacencyMatrix = self.generate_txt_graph(txt)
            txt = torch.FloatTensor(txt_feature.numpy()).cuda()
            adjacencyMatrix = adjacencyMatrix.cuda()
            """图像为Tensor类型"""
            img = torch.FloatTensor(img).cuda()

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()


            F_I, _, _  = self.FeatNet_I(img)
            F_T = txt
            _, code_I, decoded_t = self.CodeNet_I(img)

            code_T, decoded_i = self.GATNet(txt, adjacencyMatrix)
            # print(decoded_t)     # (16,1386)
            # print(decoded_i)     #(16,1024)

            F_I = F.normalize(F_I)
            S_I = F_I.mm(F_I.t())

            B_F_I = S_I
            S_I = S_I * 2 - 1

            F_T = F.normalize(F_T)
            S_T = F_T.mm(F_T.t())
            B_F_T = S_T
            S_T = S_T * 2 - 1
            decoded_i = F.normalize(decoded_i)
            decoded_t = F.normalize(decoded_t)
            B_decoded_i = decoded_i.mm(decoded_i.t())
            B_decoded_t = decoded_t.mm(decoded_t.t())

            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)

            BI_BI = B_I.mm(B_I.t())
            BT_BT = B_T.mm(B_T.t())
            BI_BT = B_I.mm(B_T.t())

            S_tilde = settings.ALPHA * S_I + (1 - settings.ALPHA) * S_T
            S = settings.K * S_tilde

            loss1 = F.mse_loss(BT_BT, S)
            loss2 = F.mse_loss(BI_BT, S)
            loss3 = F.mse_loss(BI_BI, S)
            loss31 = F.mse_loss(BI_BI, settings.K * S_I)
            loss32 = F.mse_loss(BT_BT, settings.K * S_T)


            diagonal = BI_BT.diagonal()
            all_1 = torch.rand((batch_size)).fill_(1).cuda()
            loss4 = F.mse_loss(diagonal, settings.K * all_1)
            loss5 = F.mse_loss(B_decoded_i, B_F_I)
            loss6 = F.mse_loss(B_decoded_t, B_F_T)
            loss7 = F.mse_loss(BI_BT, BI_BT.t())
            loss = 1 * loss1 + 1 * loss2 + 1 * loss3 + 1 * loss4 + 1 * loss5 + 1 * loss6 + 2 * loss7 + settings.ETA * (loss31 + loss32)
          
            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] '
                    'Loss1: %.4f Loss2: %.4f Loss3: %.4f '
                    'Loss4: %.4f '
                    'Loss5: %.4f Loss6: %.4f '
                    'Loss7: %.4f '
                    'Total Loss: %.4f'
                    % (
                        epoch + 1, settings.NUM_EPOCH, idx + 1,
                        len(self.train_dataset) // settings.BATCH_SIZE,
                        loss1.item(), loss2.item(), loss3.item(),
                        loss4.item(),
                        loss5.item(), loss6.item(),
                        loss7.item(),
                        loss.item()))

    def eval(self, step=0, last=False):

        self.CodeNet_I.eval().cuda()
        self.GATNet.eval().cuda()
        if settings.DATASET == "MIRFlickr":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.GATNet, self.database_dataset, self.test_dataset)
            K = [50,100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

        if settings.EVAL:
            MAP_I2T = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2I = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
            self.logger.info('--------------------------------------------------------------------')
            retI2T = p_topK(qu_BI, re_BT, qu_L, re_L, K)
            retT2I = p_topK(qu_BT, re_BI, qu_L, re_L, K)
            self.logger.info(retI2T)
            self.logger.info(retT2I)
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50 )
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50 )
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
        self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('--------------------------------------------------------------------')
        if MAP_I2T + MAP_T2I > self.best and not settings.EVAL:
            self.save_checkpoints(step=step, best=True)
            self.best = MAP_T2I + MAP_I2T
            self.logger.info("#########is best:%.3f #########" % self.best)

    def save_checkpoints(self, step, file_name='%s_%d_bit_latest.pth' % (settings.DATASET, settings.CODE_LEN),
                         best=False):
        if best:
            file_name = '%s_%d_bit_best_epoch.pth' % (settings.DATASET, settings.CODE_LEN)
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'GATNet': self.GATNet.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='%s_%d_bit_best_epoch.pth' % (settings.DATASET, settings.CODE_LEN)):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.GATNet.load_state_dict(obj['GATNet'])
        self.logger.info('********** The loaded model has been trained for epochs.*********')

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def normalize_adj(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def generate_txt_graph(self, txt):
        """Generate text graph structure."""
        txt_feature = txt
        adj = txt.mm(txt.t())
        adj = torch.sign(adj)
        adj2triad = sp.csr_matrix(adj)
        #adj2triad = adj2triad + adj2triad.T.multiply(adj2triad.T > adj2triad) - adj2triad.multiply(adj2triad.T > adj2triad)
        adj = self.normalize_adj(adj2triad)
        adj = torch.FloatTensor(np.array(adj.todense()))
        #adjacencyMatrix = self.sparse_mx_to_torch_sparse_tensor(adj)
        return txt_feature, adj

def main():
    sess = Session()

    if settings.EVAL == True:
        sess.load_checkpoints()
        sess.eval()

    else:
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval(step=epoch + 1)
            # save the model
        settings.EVAL = True
        sess.logger.info('---------------------------Test------------------------')
        sess.load_checkpoints()
        sess.eval()

if __name__ == '__main__':
    main()
