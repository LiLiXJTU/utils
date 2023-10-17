# 没有激活函数
output = model(data)
outputssoft = torch.softmax(output, dim=1)
mDiceLoss, mDice = MultiClassDiceLoss()(outputssoft, target)
floss = FocalLoss_Ori()(output, target)