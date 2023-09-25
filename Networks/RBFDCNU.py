import math
from sys import stdin
from tkinter import W
from importlib_metadata import Pair
import torch
import torch.nn as nn
import numpy as np
from Modules.Interpolation import SpatialTransformer, RadialBasisArbitraryLayer
from Modules.Loss import LOSSDICT, JacobianDeterminantLoss, MaxMinPointDist

from .BaseNetwork import GenerativeRegistrationNetwork
from .ConvBlock import ConvBlock, ConvLayer, InterpolationLayer, FeatureEncoder, PointEncoder

class ContourPointExtractor():

    def __init__(self):
        super(ContourPointExtractor, self).__init__()

        ucp_loc_vectors = [
            torch.linspace(16 // 2, i - 16 // 2,
                            i // 16) for i in [128, 128]
        ] # i // 16
        ucp_loc = torch.meshgrid(ucp_loc_vectors)
        ucp_loc = torch.stack(ucp_loc, 2)[:, :, [1, 0]]
        ucp_loc = torch.flatten(ucp_loc, start_dim=0, end_dim=1).float()
        self.ucp_loc = ucp_loc

    def farthest_point_sample(self, x, y, cpNum):

        distance = np.ones(len(x)) * 1e10
        select_index = [0]
        currentX, currentY = x[0], y[0]
        for i in range(cpNum - 1):
            dist = (x - currentX) ** 2 + (y - currentY) ** 2
            mask = dist < distance
            distance[mask] = dist[mask]
            ind = np.argmax(distance)
            select_index.append(ind)
            currentX, currentY = x[ind], y[ind]
        return x[select_index], y[select_index]


    # 在最小包围盒中找出所有非轮廓点，然后近乎等距取点
    def paddingCP(self, num, x, y, diff, cpNum):
        left = np.min(x)
        right = np.max(x)
        top = np.min(y)
        bottom = np.max(y)
        gapX = right - left + 1
        gapY = bottom - top + 1
        if gapX < 8:
            left = left + gapX - 8
        if gapY < 8:
            top = top + gapY - 8
                
        # 取出最小包围盒中的非轮廓点
        tempX, tempY = np.where(diff[left: right + 1, top: bottom + 1] == 0) 
        tempX = tempX + left
        tempY = tempY + top

        # 填充非轮廓点
        index = np.linspace(0, len(tempX) - 1, cpNum - num, dtype = int)
        pointX = np.append(x, tempX[index])
        pointY = np.append(y, tempY[index])

        return pointX, pointY

    # 提取全部轮廓点
    def contourPoints(self, seg):
        h, w = seg.shape
        seg_h = np.hstack((seg[:, 1: w], np.expand_dims(seg[:, w - 1], 1)))
        diffH = np.abs(seg_h - seg)
        diffH = np.hstack((np.zeros((h, 1)), diffH[:, 1: w])) + diffH
        seg_v = np.vstack((seg[1: h, :], np.expand_dims(seg[h - 1, :], 0)))
        diffV = np.abs(seg_v - seg)
        diffV = np.vstack((np.zeros((1, w)), diffV[1: h, :])) + diffV
        diff = diffH + diffV
        x, y = np.where(diff != 0)

        return x, y, diff

    def getControlPoint(self, segement, cpNum, gap):
        
        segement = segement[:, :, gap: 128 - gap, gap: 128 - gap]

        B = segement.shape[0]
        
        ucp_loc = self.ucp_loc[None, :, :].repeat(B, 1, 1).float().cuda()
        
        for i in range(B):
            seg = segement[i, :, :, :].squeeze().cpu().numpy()

            # 提取全部轮廓点
            x, y, diff = self.contourPoints(seg)

            num = len(x)
            # 轮廓点的数量和cpNum进行比较，轮廓点数量少于cpNum则填充，最后使用FPS采样，采样cp_Num个轮廓点
            # 填充控制点
            if num < cpNum:
                pointX, pointY = self.paddingCP(num, x, y, diff, cpNum)
            else:
                pointX, pointY = self.farthest_point_sample(x, y, cpNum)
             
            if i == 0:
                nucp_loc = np.expand_dims(np.transpose(np.vstack((pointY, pointX))), 0)
            else:
                loc = np.expand_dims(np.transpose(np.vstack((pointY, pointX))), 0)
                nucp_loc = np.concatenate([nucp_loc, loc], 0)
        
        nucp_loc = torch.from_numpy(nucp_loc).float().cuda()
        

        return nucp_loc, ucp_loc
        
class RBFDCNUInterEncoder(nn.Module):
    def __init__(self,
                 gap = 32,
                 dims=[16, 32, 32, 32, 32],
                 num_layers=[1, 1, 1, 1, 1],
                 local_dims=[16, 32, 32, 32],
                 local_num_layers=[1, 1, 1, 1]
                 ):
        super(RBFDCNUInterEncoder, self).__init__()

        self.gap = gap

        # 双线性插值层
        self.Interpolation = InterpolationLayer()

        # uniform branch
        self.cb0 = ConvBlock(num_layers[0], dims[0], 2)  # 128
        self.do0 = ConvLayer(dims[0], dims[1], 3, 2, 1)  # 64

        self.cb1 = ConvBlock(num_layers[1], dims[1], dims[1])
        self.do1 = ConvLayer(dims[1], dims[2], 3, 2, 1)  # 32

        self.cb2 = ConvBlock(num_layers[2], dims[2], dims[2])
        self.do2 = ConvLayer(dims[2], dims[3], 3, 2, 1)  # 16

        self.cb3 = ConvBlock(num_layers[3], dims[3], dims[3])
        self.do3 = ConvLayer(dims[3], dims[4], 3, 2, 1)  # 8

        self.cb4 = ConvBlock(num_layers[4], dims[4], dims[4])

        # uniform alpha
        # 计算控制点的隐变量的均值的卷积层
        self.conv1 = nn.Conv1d(64, 8, 1, stride=1)
        self.uzMean_Layer = nn.Conv1d(8, 2, 1, stride=1)
        # 计算控制点的隐变量的均值的卷积层
        self.conv2 = nn.Conv1d(64, 8, 1, stride=1)
        self.uzVariance_Layer = nn.Conv1d(8, 2, 1, stride=1)
        torch.nn.init.normal_(self.uzVariance_Layer.weight, mean=0.0, std=1e-10)
        torch.nn.init.constant_(self.uzVariance_Layer.bias, -10)
        
        # nonuniform branch
        self.dconv0 = ConvLayer(2, local_dims[0], 3)  # 64

        self.dcb0 = ConvBlock(local_num_layers[0], local_dims[0], local_dims[0])
        self.ddo0 = ConvLayer(local_dims[0], local_dims[1], 3, 2, 1)  # 32

        self.dcb1 = ConvBlock(local_num_layers[1], local_dims[1], local_dims[1])  
        self.ddo1 = ConvLayer(local_dims[1], local_dims[2], 3, 2, 1)  # 16

        self.dcb2 = ConvBlock(local_num_layers[2], local_dims[2], local_dims[2])
        self.ddo2 = ConvLayer(local_dims[2], local_dims[3], 3, 2, 1)  # 8

        self.dcb3 = ConvBlock(local_num_layers[3], local_dims[3], local_dims[3])
        
        # nonuniform alpha
        self.featureEncoder = FeatureEncoder(local_dims[3])
        self.pointEncoder = PointEncoder(2)
        self.dconv1 = nn.Conv1d(1088 * 2, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 64, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        self.nuzMean_Layer = nn.Conv1d(64, 2, 1)
        self.nuzVariance_Layer = nn.Conv1d(64, 2, 1)
        torch.nn.init.normal_(self.nuzVariance_Layer.weight, mean=0.0, std=1e-10)
        torch.nn.init.constant_(self.nuzVariance_Layer.bias, -10)

    def forward(self, src, tgt, nucp_loc, ucp_loc):
        
        # uniform branch
        x_in = torch.cat((src, tgt), 1)

        x0 = self.cb0(x_in)

        x1 = self.cb1(self.do0(x0))

        x2 = self.cb2(self.do1(x1))

        x3 = self.cb3(self.do2(x2))

        x4 = self.cb4(self.do3(x3))

        # uniform alpha
        xi = self.Interpolation(x4, ucp_loc, 16)
        uzMean = self.uzMean_Layer(self.conv1(xi))
        uzVariance = self.uzVariance_Layer(self.conv2(xi))

        # nonuniform branch
        dx0 = self.dconv0(x_in[:, :, self.gap: 128 - self.gap, self.gap: 128 - self.gap])
        
        dx0 = self.dcb0(dx0)
        dx1 = self.ddo0(dx0)

        dx1 = self.dcb1(dx1)  
        dx2 = self.ddo1(dx1)  

        dx2 = self.dcb2(dx2)
        dx3 = self.ddo2(dx2)

        dx4 = self.dcb3(dx3)

        # nonuniform alpha
        dxi = self.Interpolation(dx4, nucp_loc, 8)
        dxfe = self.featureEncoder(dxi)
        dxpe = self.pointEncoder(nucp_loc / (128 - 2 * self.gap))
        dxe = torch.cat([dxfe, dxpe], dim=1)
        dxe = self.relu(self.dbn1(self.dconv1(dxe)))
        dxe = self.relu(self.dbn2(self.dconv2(dxe)))
        dxe = self.relu(self.dbn3(self.dconv3(dxe)))
        dxe = self.relu(self.dbn4(self.dconv4(dxe)))
        nuzMean = self.nuzMean_Layer(dxe)
        nuzVariance = self.nuzVariance_Layer(dxe)

        uzMean = uzMean.transpose(1, 2)
        uzVariance = uzVariance.transpose(1, 2)
        nuzMean = nuzMean.transpose(1, 2)
        nuzVariance = nuzVariance.transpose(1, 2)

        return uzMean, uzVariance, nuzMean, nuzVariance


class RBFDCNUGenerativeNetwork(GenerativeRegistrationNetwork):
    def __init__(self,
                 encoder_param,
                 i_size,
                 c_factor,
                 cpoint_num,
                 nucpoint_num,
                 ucpoint_num,
                 similarity_loss='LCC',
                 similarity_loss_param={},
                 similarity_factor=130000,
                 loss_mode = 3,
                 cropSize = 64,):

        super(RBFDCNUGenerativeNetwork, self).__init__(i_size)

        if cropSize == 64:
            self.gap = 32
        elif cropSize == 96:
            self.gap = 16
        
        self.encoder = RBFDCNUInterEncoder(self.gap, **encoder_param)
        self.nudecoder = RadialBasisArbitraryLayer(i_size, c_factor, nucpoint_num)
        self.udecoder = RadialBasisArbitraryLayer(i_size, c_factor, ucpoint_num)
        self.transformer = SpatialTransformer(i_size, need_grid=True)
        self.CPExtractor = ContourPointExtractor()

        self.i_size = i_size
        self.scale = int(i_size[0] // np.sqrt(cpoint_num))
        self.c_factor = c_factor
        self.cpoint_num = cpoint_num
        self.nucpoint_num = nucpoint_num
        self.ucpoint_num = ucpoint_num

        self.similarity_loss = LOSSDICT[similarity_loss](
            **similarity_loss_param)
        self.nu_cpoint_maxmin = MaxMinPointDist(nucpoint_num)
        self.u_cpoint_maxmin = MaxMinPointDist(ucpoint_num)
        self.similarity_factor = similarity_factor
        self.loss_mode = loss_mode

        # generate a name
        name = str(similarity_loss) + '--'
        for k in similarity_loss_param:
            name += '-' + str(similarity_loss_param[k])

    def sample(self, mu, log_var):
        eps = torch.randn(mu.size(), device=mu.device)
        std = torch.exp(0.5 * log_var)
        return mu + std * eps


    def forward(self, src, tgt, nucp_loc, ucp_loc):
        
        uzMean, uzVariance, nuzMean, nuzVariance = self.encoder(src, tgt, nucp_loc, ucp_loc)

        ualpha = self.sample(uzMean, uzVariance)
        nualpha = self.sample(nuzMean, nuzVariance)

        nucp_loc = nucp_loc + self.gap
        nuflow = self.nudecoder(nucp_loc, nualpha)
        uflow = self.udecoder(ucp_loc, ualpha)
        flow = nuflow + uflow

        warped_src = self.transformer(src, flow)

        zMean = torch.cat([nuzMean, uzMean], 1)
        zVariance = torch.cat([nuzVariance, uzVariance], 1)
        return flow, warped_src, (zMean, zVariance)

    def test(self, src, tgt, segement):

        nucp_loc, ucp_loc = self.CPExtractor.getControlPoint(segement, self.nucpoint_num, self.gap)
        uzMean, uzVariance, nuzMean, nuzVariance = self.encoder(src, tgt, nucp_loc, ucp_loc)
        
        nucp_loc = nucp_loc + self.gap
        nuflow = self.nudecoder(nucp_loc, nuzMean)
        uflow = self.udecoder(ucp_loc, uzMean)
        flow = nuflow + uflow

        warped_src = self.transformer(src, flow)

        zMean = torch.cat([nuzMean, uzMean], 1)
        zVariance = torch.cat([nuzVariance, uzVariance], 1)

        # lossValue = self.computeLossValue(zMean, zVariance, warped_src, nucp_loc, ucp_loc, tgt)
        # return flow, warped_src, (zMean, zVariance), lossValue

        NLLValue = self.computeNLLValue(warped_src, tgt, ucp_loc, nucp_loc, uzMean, uzVariance, nuzMean, nuzVariance)
        return flow, warped_src, (zMean, zVariance), NLLValue

        return flow, warped_src, (zMean, zVariance)

    def computeNLLValue(self, warped_src, tgt, ucp_loc, nucp_loc, uzMean, uzVariance, nuzMean, nuzVariance):
        
        loopTimes = 100
        B = warped_src.shape[0]

        pairNLLList = []
        for i in range(B):
            sampleNLLList = []
            for _ in range(loopTimes):
                KL_loss = self.computeKL(ucp_loc[i].unsqueeze(0), nucp_loc[i].unsqueeze(0), uzMean[i].unsqueeze(0), uzVariance[i].unsqueeze(0), nuzMean[i].unsqueeze(0), nuzVariance[i].unsqueeze(0))
                similarity_loss = self.similarity_loss(warped_src[i].unsqueeze(0), tgt[i].unsqueeze(0))

                ELBO = -(similarity_loss * self.similarity_factor + KL_loss)
                sampleNLLList.append(ELBO)
            pairNLLList.append(torch.min(torch.cat(sampleNLLList)))
        return pairNLLList

        # pairNLLList = []
        # for i in range(B):
        #     # compute distribution of p(z)
        #     epsilonInv_p_uz = self.getEpsilon_p_z(ucp_loc[i].unsqueeze(0))
        #     epsilonInv_p_nuz = self.getEpsilon_p_z(nucp_loc[i].unsqueeze(0))
            
        #     # compute distribution of q(z|x)
        #     epsilonInv_q_uz_x = self.getEpsilon_q_z_x(uzVariance[i])
        #     epsilonInv_q_nuz_x = self.getEpsilon_q_z_x(nuzVariance[i])

        #     sampleNLLList = []
        #     for _ in range(loopTimes):
        #         # compute p(x|z)
        #         ualpha = self.sample(uzMean[i], uzVariance[i])
        #         nualpha = self.sample(nuzMean[i], nuzVariance[i])
        #         logp_p_x_z = self.logp_p_x_z(src[i], tgt[i], ualpha, nualpha, ucp_loc[i], nucp_loc[i])
                
        #         ualpha = torch.flatten(ualpha, start_dim=0)
        #         nualpha = torch.flatten(nualpha, start_dim=0)
        #         u_mu_p_z = torch.zeros_like(ualpha)
        #         nu_mu_p_z = torch.zeros_like(nualpha)
        #         u_mu_q_z_x = torch.flatten(uzMean[i], start_dim=0)
        #         nu_mu_q_z_x = torch.flatten(nuzMean[i], start_dim=0)

        #         logp_p_uz = self.computeLogPro(ualpha, u_mu_p_z, epsilonInv_p_uz)
        #         logp_p_nuz = self.computeLogPro(nualpha, nu_mu_p_z, epsilonInv_p_nuz)
        #         logp_q_uz_x = self.computeLogPro(ualpha, u_mu_q_z_x, epsilonInv_q_uz_x)
        #         logp_q_nuz_x = self.computeLogPro(nualpha, nu_mu_q_z_x, epsilonInv_q_nuz_x)

        #         value = -(logp_p_x_z + logp_p_uz + logp_p_nuz - logp_q_uz_x - logp_q_nuz_x)
        #         sampleNLLList.append(value)
        #     pairNLLList.append(torch.mean(torch.cat(sampleNLLList)))
        # return pairNLLList
    
    def computeKL(self, ucp_loc, nucp_loc, uzMean, uzVariance, nuzMean, nuzVariance):
        
        nu_sigma_term = torch.sum(torch.exp(nuzVariance), dim=[1, 2]) - torch.sum(
            nuzVariance, dim=[1, 2])
        nu_smooth_term, nu_logdet = self.smooth_loss(nuzMean, nucp_loc, self.nucpoint_num)
        nu_KL_loss = (nu_sigma_term + nu_smooth_term - nu_logdet) * 0.5

        u_sigma_term = torch.sum(torch.exp(uzVariance), dim=[1, 2]) - torch.sum(
            uzVariance, dim=[1, 2])
        u_smooth_term, u_logdet = self.smooth_loss(uzMean, ucp_loc, self.ucpoint_num)
        u_KL_loss = (u_sigma_term + u_smooth_term - u_logdet) * 0.5

        KL_loss = nu_KL_loss + u_KL_loss
        return KL_loss
    
    def computeLogPro(self, z, mu, epsilonInv):
        k = z.shape[0]
        term_1 = k * torch.log(torch.tensor(2 * math.pi))

        evals = torch.symeig(epsilonInv)
        term_2 = torch.sum(torch.log(evals.eigenvalues))

        diff = (z - mu).unsqueeze(1)
        term_3 = torch.mm(torch.mm(diff.T, epsilonInv), diff)

        ret = (term_1 - term_2 - term_3) * (-1 / 2)
        return ret

    def logp_p_x_z(self, src, tgt, ualpha, nualpha, ucp_loc, nucp_loc):
        src = src.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
        ualpha = ualpha.unsqueeze(0)
        nualpha = nualpha.unsqueeze(0)
        ucp_loc = ucp_loc.unsqueeze(0)
        nucp_loc = nucp_loc.unsqueeze(0)

        uflow = self.udecoder(ucp_loc, ualpha)
        nuflow = self.nudecoder(nucp_loc, nualpha)
        flow = nuflow + uflow
        warped_src = self.transformer(src, flow)
        loss = self.similarity_loss(warped_src, tgt)
        return -loss

    def getEpsilon_q_z_x(self, log_var):
        var = torch.flatten(torch.exp(log_var), start_dim=0)
        covariance = torch.diag(1 / var)
        return covariance

    def getEpsilon_p_z(self, cp_loc):
        cpoint_num = cp_loc.shape[1]
        if cpoint_num == self.nucpoint_num:
            c = self.nu_cpoint_maxmin(cp_loc) * self.c_factor
        else:
            c = self.u_cpoint_maxmin(cp_loc) * self.c_factor  # b
        c = c.unsqueeze(1).unsqueeze(1)

        cp_loc_ta = cp_loc.unsqueeze(1).repeat(1, cpoint_num, 1, 1)
        cp_loc_tb = cp_loc.unsqueeze(2).repeat(1, 1, cpoint_num, 1)
        dist = torch.norm(cp_loc_ta - cp_loc_tb, dim=3) / c
        mask = dist < 1
        weight = torch.pow(1 - dist, 4) * (4 * dist + 1)
        weight = weight * mask.float()  # b c c
        
        covariance = torch.zeros((cpoint_num * 2, cpoint_num * 2)).cuda()
        covariance[0: cpoint_num, 0: cpoint_num] = weight
        covariance[cpoint_num:, cpoint_num: ] = weight

        return covariance



    def computeLossValue(self, mu, log_var, warped_src, nucp_loc, ucp_loc, tgt):
        
        # nonuniform KL
        nu_mu = mu[:, 0: self.nucpoint_num, :]
        nu_log_var = log_var[:, 0: self.nucpoint_num, :]
        nu_sigma_term = torch.sum(torch.exp(nu_log_var), dim=[1, 2]) - torch.sum(
            nu_log_var, dim=[1, 2])
        nu_smooth_term, nu_logdet = self.smooth_loss(nu_mu, nucp_loc, self.nucpoint_num)
        nu_KL_loss = (nu_sigma_term + nu_smooth_term - nu_logdet) * 0.5
        nu_logSigma = torch.sum(nu_log_var, dim=[1, 2])
        nu_trSigma = torch.sum(torch.exp(nu_log_var), dim=[1, 2])

        # uniform KL
        u_mu = mu[:, self.nucpoint_num :, :]
        u_log_var = log_var[:, self.nucpoint_num :, :]
        u_sigma_term = torch.sum(torch.exp(u_log_var), dim=[1, 2]) - torch.sum(
            u_log_var, dim=[1, 2])
        u_smooth_term, u_logdet = self.smooth_loss(u_mu, ucp_loc, self.ucpoint_num)
        u_KL_loss = (u_sigma_term + u_smooth_term - u_logdet) * 0.5
        u_logSigma = torch.sum(u_log_var, dim=[1, 2])
        u_trSigma = torch.sum(torch.exp(u_log_var), dim=[1, 2])

        # KL
        sigma_term = nu_sigma_term + u_sigma_term
        smooth_term = nu_smooth_term + u_smooth_term
        logdet = nu_logdet + u_logdet
        KL_loss = nu_KL_loss + u_KL_loss
        similarity_loss = self.similarity_loss(warped_src, tgt)
        loss = similarity_loss * self.similarity_factor + KL_loss
        logSigma = nu_logSigma + u_logSigma
        trSigma = nu_trSigma + u_trSigma

        return loss, KL_loss, sigma_term, smooth_term, logdet, logSigma, trSigma


    def testForDraw(self, src, tgt, segement):

        nucp_loc, ucp_loc = self.CPExtractor.getControlPoint(segement, self.nucpoint_num, self.gap)
        uzMean, uzVariance, nuzMean, nuzVariance = self.encoder(src, tgt, nucp_loc, ucp_loc)
        
        nucp_loc = nucp_loc + self.gap
        nuflow = self.nudecoder(nucp_loc, nuzMean)
        uflow = self.udecoder(ucp_loc, uzMean)
        flow = nuflow + uflow

        warped_src = self.transformer(src, flow)

        zMean = torch.cat([nuzMean, uzMean], 1)
        zVariance = torch.cat([nuzVariance, uzVariance], 1)

        cp_loc = torch.cat([nucp_loc, ucp_loc], 1)

        return flow, warped_src, (zMean, zVariance), cp_loc


    def smooth_loss(self, alpha, cp_loc, cpoint_num):
        if cpoint_num == self.nucpoint_num:
            c = self.nu_cpoint_maxmin(cp_loc) * self.c_factor
        else:
            c = self.u_cpoint_maxmin(cp_loc) * self.c_factor  # b
        c = c.unsqueeze(1).unsqueeze(1)
        # b c c 2
        cp_loc_ta = cp_loc.unsqueeze(1).repeat(1, cpoint_num, 1, 1)
        cp_loc_tb = cp_loc.unsqueeze(2).repeat(1, 1, cpoint_num, 1)
        dist = torch.norm(cp_loc_ta - cp_loc_tb, dim=3) / c
        # add mask for r < 1
        mask = dist < 1
        # weight if r<1 weight=(1-r)^4*(4r+1)
        #        else   weight=0
        # Todo: reduce weight size
        weight = torch.pow(1 - dist, 4) * (4 * dist + 1)
        weight = weight * mask.float()  # b c c
        det = torch.det(weight) + 1e-5
        logdet = torch.log(det)
        weight = weight.unsqueeze(3).repeat(1, 1, 1, 2)
        # tile alpha
        alpha_t = alpha.unsqueeze(1).repeat(1, cpoint_num, 1, 1)
        y = torch.sum(alpha_t * weight, dim=2)
        K = torch.sum(y * alpha, dim=[1, 2])
        return K, logdet

    def setHyperparam(self, similarity_loss_param, factor_list):
        '''
        for hyperparam optimization
        '''
        self.similarity_loss.set(**similarity_loss_param)
        self.factor_list = factor_list

    def objective(self, src, tgt, segement):

        nucp_loc, ucp_loc = self.CPExtractor.getControlPoint(segement, self.nucpoint_num, self.gap)
        flow, warped_src, (mu, log_var) = self(src, tgt, nucp_loc, ucp_loc)

        # nonuniform KL
        nucp_loc = nucp_loc + self.gap
        nu_mu = mu[:, 0: self.nucpoint_num, :]
        nu_log_var = log_var[:, 0: self.nucpoint_num, :]
        nu_sigma_term = torch.sum(torch.exp(nu_log_var), dim=[1, 2]) - torch.sum(
            nu_log_var, dim=[1, 2])
        nu_smooth_term, nu_logdet = self.smooth_loss(nu_mu, nucp_loc, self.nucpoint_num)
        nu_KL_loss = (nu_sigma_term + nu_smooth_term - nu_logdet) * 0.5

        # uniform KL
        u_mu = mu[:, self.nucpoint_num :, :]
        u_log_var = log_var[:, self.nucpoint_num :, :]
        u_sigma_term = torch.sum(torch.exp(u_log_var), dim=[1, 2]) - torch.sum(
            u_log_var, dim=[1, 2])
        u_smooth_term, u_logdet = self.smooth_loss(u_mu, ucp_loc, self.ucpoint_num)
        u_KL_loss = (u_sigma_term + u_smooth_term - u_logdet) * 0.5

        # KL
        KL_loss = nu_KL_loss + u_KL_loss
       

        # similarity
        if self.loss_mode == 0:
            similarity_loss = self.similarity_loss(warped_src, tgt)
        elif self.loss_mode == 1:
            similarity_loss = self.similarity_loss(warped_src, tgt) + 2 * self.similarity_loss(warped_src[:, :, 32: 96, 32: 96], tgt[:, :, 32: 96, 32: 96])

        loss = similarity_loss + KL_loss / self.similarity_factor
        ELOB = -1 * loss

        return {
            'similarity_loss': similarity_loss,
            'KL_loss': KL_loss,
            'loss': loss,
            'ELOB': ELOB
        }
