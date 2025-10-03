import os
import sys
import torch
import torch.nn as nn
import numpy as np
import os
import random
import argparse
from datetime import datetime
from tqdm import tqdm

from data.dataset import MorphDataset
from utils.util_fun import load_data_config
from data.augmentation_pipeline import SelfMADppDataset
from utils.train.scheduler import LinearDecayLR
from model.detector import Detector
from utils.logs import log
from eval import evaluate

def main(args) -> None:
    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    cfg = {
        "session_name": args.session_name,
        "train_dataset": args.train_dataset,
        "model": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "model_type": args.model_type
    }
    
    data_cfg = load_data_config("data/data_config.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset=SelfMADppDataset(cfg=data_cfg, phase='train', source_name=args.train_dataset)
    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=cfg['batch_size'] // 2,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn
                        )

    test_dataloader = {}
    for dataset_name in data_cfg["dataset"]["eval"]:
        for morph_type in data_cfg["dataset"]["eval"][dataset_name]['morphs']:
            if not data_cfg["dataset"]["eval"][dataset_name]['morphs'][morph_type]:
                continue
            dataset = MorphDataset(dataset_name=dataset_name, morph_type=morph_type, data_cfg=data_cfg)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            test_dataloader[f"{dataset_name}_{morph_type}"] = dataloader

    model = Detector(model=args.model_type, lr=args.lr)
    n_epoch = cfg['epochs']
    lr_scheduler = LinearDecayLR(model.optimizer, n_epoch, int(n_epoch/4*3))
    
    now = datetime.now()
    save_path = os.path.join(data_cfg["save_path"], f"{args.session_name}_{now.strftime('%m_%d_%H_%M_%S')}/")
    if not args.forget:
        os.mkdir(save_path)
        os.mkdir(save_path+'weights/')
        os.mkdir(save_path+'logs/')
        with open(save_path+"config.txt", "w") as f:
            f.write(str(cfg))
        logger = log(path=save_path+"logs/", file="losses.logs")

    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epoch):
        # ─── TRAIN ───
        np.random.seed(seed + epoch)
        train_loss=0.
        model.train(mode=True)        
        for iter_idx, data in enumerate(tqdm(train_loader, desc="Epoch {}/{}".format(epoch+1, n_epoch))):
            img=data['img'].to(device, non_blocking=True).float()
            mask=data['mask'].to(device, non_blocking=True).long()
            target=data['label'].to(device, non_blocking=True).long()
            out_clf, out_seg = model.training_step(img, target, mask)
            loss = criterion(out_clf, target) + criterion(out_seg, mask[:, 0, :, :])
            train_loss+=loss.item()
            
        lr_scheduler.step()
        log_text="Epoch {}/{} | train loss: {:.4f} |".format(
                        epoch+1,
                        n_epoch,
                        train_loss/len(train_loader),
                        )
        
        # ─── TEST ───
        model.train(mode=False)
        eval_results = {}
        for dataset_name, dataloader in test_dataloader.items():
            results, print_string, _ = evaluate(model, dataloader, device)
            eval_results[dataset_name] = results
            log_text += print_string
        test_eer_total = [results["eer"] for results in eval_results.values()]
        test_eer_total = np.array(test_eer_total)
        log_text += f"Total: mean eer: {np.mean(test_eer_total):.4f}\n"
        
        # ─── SAVE ───
        if not args.forget:
            save_model_path=os.path.join(save_path+'weights/',"epoch_{}.tar".format(epoch+1))
            torch.save({
                    "model":model.state_dict(),
                    "optimizer":model.optimizer.state_dict(),
                    "epoch":epoch,
                    "model_type": cfg["model_type"]
                },save_model_path)
            logger.info(log_text)
        else:
            print(log_text)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-m',dest='model_type', type=str, required=True)
    parser.add_argument('-b', dest='batch_size', type=int, required=True)
    parser.add_argument('-e', dest='epochs', type=int, required=True)
    parser.add_argument('-t', dest='train_dataset', type=str, required=True)
    parser.add_argument('-lr', dest='lr', type=float, required=True)
    parser.add_argument('--forget', dest='forget', action='store_true')
    parser.add_argument('-n', dest='session_name', type=str, required=not '--forget' in sys.argv)
    args = parser.parse_args()
    main(args)
