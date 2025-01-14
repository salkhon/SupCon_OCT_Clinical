import torch
from utils.utils import AverageMeter, warmup_learning_rate
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
    average_precision_score,
    roc_auc_score,
    classification_report,
)
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm


def train_OCT_multilabel(
    train_loader, model, classifier, criterion, optimizer, epoch, opt
):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    device = opt.device
    end = time.time()
    for idx, (image, bio_tensor) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = image.to(device)

        labels = bio_tensor
        labels = labels.float()
        bsz = labels.shape[0]
        labels = labels.to(device)
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():  # FROZEN
            features = model.encoder(images)

        output = classifier(features.detach())

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


def validate_multilabel(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()
    device = opt.device
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    label_list = []
    out_list = []
    with torch.no_grad():
        end = time.time()
        for idx, (image, bio_tensor) in enumerate(val_loader):
            images = image.float().to(device)

            labels = bio_tensor
            labels = labels.float()
            print(idx)
            label_list.append(labels.squeeze().detach().cpu().numpy())
            labels = labels.to(device)
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))

            loss = criterion(output, labels)
            output = torch.round(torch.sigmoid(output))

            out_list.append(output.squeeze().detach().cpu().numpy())
            # update metric
            losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    label_array = np.array(label_list)
    out_array = np.array(out_list)
    out_array = np.concatenate(out_list, axis=0)
    r = roc_auc_score(label_array, out_array, average="macro")

    return losses.avg, r


def inference_on_test_images(opt, model, classifier):
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
        output = model.encoder(image)
        output = classifier(output)
        output = torch.round(torch.sigmoid(output))
        output = output.squeeze(0)
        for i in range(1, 7):
            submission_df.at[idx, f"B{i}"] = int(output[i - 1])

    for i in range(1, 7):
        submission_df[f"B{i}"] = submission_df[f"B{i}"].astype(int)

    submission_df.to_csv("/kaggle/working/submission.csv", index=False)


def main_multilabel():
    best_acc = 0
    opt = parse_option()

    # build data loader
    device = opt.device
    train_loader, test_loader = set_loader_new(opt)

    r_list = []
    # training routine
    for i in range(0, 1):
        model, classifier, criterion = set_model(opt)

        optimizer = set_optimizer(opt, classifier)
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train_OCT_multilabel(
                train_loader, model, classifier, criterion, optimizer, epoch, opt
            )
            time2 = time.time()
            print(
                "Train epoch {}, total time {:.2f}, loss:{:.3f}, accuracy:{:.2f}".format(
                    epoch, time2 - time1, loss, acc
                )
            )

            if epoch % opt.save_freq == 0:
                # todo: save
                pass

        # eval for one epoch
        # loss, r = validate_multilabel(test_loader, model, classifier, criterion, opt)

        # r_list.append(r)

        # save model
        full_model = torch.nn.Sequential(model.encoder, classifier)
        torch.save(full_model.state_dict(), "/kaggle/working/model.pth")

    # df = pd.DataFrame({'AUROC': r_list})
    # excel_name = opt.backbone_training + '_' + opt.biomarker + opt.model + str(opt.percentage) + 'multiAUROC' + str(opt.patient_split) + '.csv'
    # df.to_csv(excel_name, index=False)

    # create submission file
    inference_on_test_images(opt, model, classifier)
