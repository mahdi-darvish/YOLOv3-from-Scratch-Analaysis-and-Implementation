import torch.nn as nn
import torch

from model import intersection_over_union


class Yolov3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bcewll = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid_function = nn.Sigmoid()
        self.l_class = 1
        self.l_nj = 10
        self.l_box = 10
        self.l_obj = 1


    def forward_step(self, preds, target, anchors):
        # Check where obj and nj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        nj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bcewll(
            (preds[..., 0:1][nj]), (target[..., 0:1][nj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)

        box_preds = torch.cat([self.sigmoid_function(preds[..., 1:3]), torch.exp(preds[..., 3:5]) * anchors], dim=-1)
        result = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()

        loss_obj = self.mse(self.sigmoid_function(preds[..., 0:1][obj]), result * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        preds[..., 1:3] = self.sigmoid_function(preds[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(preds[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.cross_entropy(
            (preds[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        return (
                self.l_box * box_loss
                + self.l_obj * loss_obj
                + self.l_nj * no_object_loss
                + self.l_class * class_loss
        )
