from __future__ import print_function
import sys
sys.path.append("/kaggle/working/SupCon_OCT_Clinical")

from config.config_linear import parse_option
from training_linear.training_one_epoch_ckpt import main
from training_linear.training_one_epoch_fusion import main_supervised_fusion
from training_linear.training_one_epoch_supervised import main_supervised
from training_linear.training_one_epoch_supervised_multilabel import (
    main_supervised_multilabel,
)
from training_linear.training_one_epoch_fusion_multilabel import (
    main_supervised_multilabel_fusion,
)
from training_linear.training_one_epoch_ckpt_multi import main_multilabel
from training_linear.training_one_epoch_ckpt_bce import main_bce
from training_linear.training_one_epoch_transformer import main_transformer
from training_linear.training_one_epoch_transformer_multilabel import (
    main_transformer_multilabel,
)
from training_linear.training_one_epoch_ckpt_student_teacher import main_student_teacher

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


if __name__ == "__main__":
    opt = parse_option()
    # opt.super --> Supervised (1) or Not (0) or (2) Fusion Supervised or (3) BCE Loss for AUROC
    # opt.multi --> MultiLabel (1) or Not(0)
    # 0 --> Ckpt Training
    # multi 1 and super 3 --> BCE Individual Biomarkers

    if opt.super == 1 and opt.multi == 0:
        main_supervised()
    elif opt.super == 11:
        main_student_teacher()
    elif opt.super == 4:
        main_transformer()
    elif opt.super == 10:
        main_transformer_multilabel()
    elif opt.super == 3 and opt.multi == 1:
        main_bce()

    elif opt.super == 2 and opt.multi == 0:
        main_supervised_fusion()

    elif opt.multi == 1 and (opt.super == 1 or opt.super == 5):
        print("doing supervised training")
        main_supervised_multilabel()

    elif opt.multi == 1 and opt.super == 2:
        main_supervised_multilabel_fusion()

    elif opt.multi == 1 and (opt.super == 0 or opt.super == 8):
        main_multilabel()

    else:
        main()
