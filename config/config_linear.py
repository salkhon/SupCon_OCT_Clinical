
import argparse
import math
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of training epochs')
    parser.add_argument('--n_cls', type=int, default=2,
                        help='number of training epochs')
    parser.add_argument('--super', type=int, default=0,
                        help='number of training epochs')
    parser.add_argument('--type', type=int, default=0,
                        help='number of training epochs')
    parser.add_argument('--biomarker', type=str, default='fluid_irf')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--patient_lambda', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--parallel', type=int, default=1, help='data parallel')
    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--train_csv_path', type=str, default='train data csv')
    parser.add_argument('--test_csv_path', type=str, default='test data csv')
    parser.add_argument('--train_image_path', type=str, default='/data/Datasets')
    parser.add_argument('--test_image_path', type=str, default='/data/Datasets')
    parser.add_argument('--submission_path', type=str, default='/kaggle/input/olives-vip-cup-2023/2023 IEEE SPS Video and Image Processing (VIP) Cup - Ophthalmic Biomarker Detection/TEST/test_set_submission_template.csv')
    parser.add_argument('--submission_img_path', type=str, default='/kaggle/input/olives-vip-cup-2023/2023 IEEE SPS Video and Image Processing (VIP) Cup - Ophthalmic Biomarker Detection/TEST')
    parser.add_argument('--results_dir_contrastive', type=str, default='/kaggle/working/results.txt')
    parser.add_argument('--save_path', type=str, default='./save/BioMarker/final.pth')
    parser.add_argument('--img_dir', type=str, default='image directory')
    parser.add_argument('--model_type', type=str, default='bcva')
    parser.add_argument('--multi', type=int, default=0)
    parser.add_argument('--noise_analysis', type=int, default=0)
    parser.add_argument('--severity_analysis', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='Prime',
                        choices=['OCT','Biomarker','Prime'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--ford_region',type = int,default = 0,
                        help='Training on 6 region classes or not')
    parser.add_argument('--percentage', type=int, default=100,
                        help='Percentage of Biomarker Training Data Utilized')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--backbone_training', type=str, default='BCVA',
                        help='manner in which backbone was trained')
    parser.add_argument('--patient_split', type=int, default=1,
                        help='choose method')
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'Ford':
        opt.n_cls = 3
    elif opt.dataset == 'Ford_Region':
        opt.n_cls = 3
    elif opt.dataset == 'covid_kaggle':
        opt.n_cls = 4
    elif opt.dataset == 'qu_dataset':
        opt.n_cls = 3
    elif opt.dataset == 'covid_x':
        opt.n_cls = 2
    elif opt.dataset == 'covid_x_A':
        opt.n_cls = 3
    elif opt.dataset == 'OCT':
        opt.n_cls = 4
    elif opt.dataset == 'Prime':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt