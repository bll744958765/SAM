import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0) #将source和target按列方向合并
        # 将total复制（n+m）份
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        # 高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        # 得到最终的核矩阵
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0]) # 一般默认为源域和目标域的batchsize相同
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

            # 将核矩阵分为4份
            # XX = torch.mean(kernels[:batch_size, :batch_size])
            # YY = torch.mean(kernels[batch_size:, batch_size:])
            # XY = torch.mean(kernels[:batch_size, batch_size:])
            # YX = torch.mean(kernels[batch_size:, :batch_size])
            # loss = torch.mean(XX + YY - XY - YX)
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
            loss = torch.mean(XX + YY - XY - YX)
            return loss


# 注意source和target须为tensor，最终的MMD损失也为tensor。
# 另外，source和target须为一个batch，如果源域和目标域数据为单个样本的话，则需要增加一个维度。
# source和target的样本个数不要求一样，但第二维度须一样。