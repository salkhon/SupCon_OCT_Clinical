import torch
from utils.utils import AverageMeter, warmup_learning_rate, accuracy
import sys
import time
import numpy as np
from config.config_linear import parse_option
from utils.utils import (
    set_loader_new,
    set_model,
    set_optimizer,
    adjust_learning_rate,
    accuracy_multilabel,
)
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    precision_recall_fscore_support,
)
from models.resnet import SupCEResNet, SupCEResNet_Original
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def train_supervised_multilabel(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    device = opt.device
    end = time.time()

    for idx, (image, bio_tensor) in enumerate(train_loader):
        data_time.update(time.time() - end)
        labels = bio_tensor
        images = image.to(device)

        labels = labels.float()

        labels = labels.to(device)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss

        output = model(images)

        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print("Train: [{0}][{1}/{2}]\t".format(epoch, idx + 1, len(train_loader)))

            sys.stdout.flush()

    return losses.avg, top1.avg


def validate_supervised_multilabel(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    label_list = []
    device = opt.device
    out_list = []
    out_list_f = []
    with torch.no_grad():
        end = time.time()
        for idx, (image, bio_tensor) in enumerate(
            val_loader
        ):
            images = image.float().to(device)
            labels = bio_tensor

            labels = labels.float()

            label_list.append(labels.squeeze().detach().cpu().numpy())
            labels = labels.to(device)

            bsz = labels.shape[0]

            # forward
            output = model(images)

            loss = criterion(output, labels)

            out_list.append(torch.sigmoid(output).squeeze().detach().cpu().numpy())
            output = torch.round(torch.sigmoid(output))

            out_list_f.append(output.squeeze().detach().cpu().numpy())
            # update metric
            losses.update(loss.item(), bsz)

    label_array = np.concatenate(label_list, axis=0)
    label_array_report = np.array(label_list)
    out_array_report = np.array(out_list)
    out_array_f = np.concatenate(out_list_f, axis=0)
    out_array_f_report = np.array(out_list_f)
    r = roc_auc_score(label_array_report, out_array_report, average="macro")

    return losses.avg, r


def inference_on_test_images(opt, model):
    # create submission file
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.1706, std=0.2112),
        ]
    )
    submission_df = pd.read_csv(opt.submission_path)
    for idx, row in tqdm(submission_df.iterrows(), total=len(submission_df)):
        img_path = Path(
            opt.submission_img_path,
            row["Path (Trial/Image Type/Subject/Visit/Eye/Image Name)"],
        )
        image = Image.open(img_path).convert("L")
        image = np.array(image)
        image = Image.fromarray(image)
        image = val_transform(image)
        image = image.unsqueeze(0)
        image = image.float().to(opt.device)
        output = model(image)
        output = torch.round(torch.sigmoid(output))
        output = output.squeeze(0)
        for i in range(1, 7):
            submission_df.at[idx, f"B{i}"] = int(output[i - 1])

    submission_df.iloc[:, 1:] = submission_df.iloc[:, 1:].astype(int)

    submission_df.to_csv("/kaggle/working/submission.csv", index=False)


def main_supervised_multilabel():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, test_loader = set_loader_new(opt)

    device = opt.device

    acc_list = []
    r_list = []
    for i in range(0, 1):
        # training routine
        if opt.super == 5:
            model = SupCEResNet_Original(name="resnet50", num_classes=16)
        else:
            model = SupCEResNet_Original(name="resnet101", num_classes=6)
        model = model.to(device)

        # need only encoder weights
        # print("Loading checkpoint weights")
        # ckpt = torch.load(opt.chpt, map_location="cpu")
        # model.load_state_dict(ckpt["model"])

        criterion = torch.nn.BCEWithLogitsLoss()
        criterion = criterion.to(device)
        optimizer = set_optimizer(opt, model)
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()

            loss, acc = train_supervised_multilabel(
                train_loader, model, criterion, optimizer, epoch, opt
            )
            time2 = time.time()
            print(
                "Train epoch {}, total time {:.2f}, loss {:2f}, accuracy:{:.2f}".format(
                    epoch, time2 - time1, loss, acc
                )
            )

        loss, r = validate_supervised_multilabel(test_loader, model, criterion, opt)
        r_list.append(r)

    # df = pd.DataFrame({"AUROC": r_list})
    # excel_name = (
    #     opt.backbone_training
    #     + "_"
    #     + opt.biomarker
    #     + opt.model
    #     + str(opt.percentage)
    #     + "SupervisedmultiAUROC"
    #     + str(opt.patient_split)
    #     + ".csv"
    # )
    # df.to_csv(excel_name, index=False)

    inference_on_test_images(opt, model)
