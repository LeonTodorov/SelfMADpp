import torch
from model.detector import Detector
from data.dataset import MorphDataset
from utils.metrics import performances_compute
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import pickle
from utils.util_fun import load_data_config

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state = torch.load(args.checkpoint_path)
    model = Detector(model = model_state["model_type"])
    model.load_state_dict(model_state["model"], strict=False)
    model.train(mode=False)

    if model_state["model_type"] != "CLIPFuse": # device assignment done in nets.py
        model.to(device)
    
    out_string = ""
    results_total = {}
    data_cfg = load_data_config("data/data_config.yaml")
    for dataset_name in data_cfg["dataset"]["eval"]:
        for morph_type in data_cfg["dataset"]["eval"][dataset_name]["morphs"]:
            if not data_cfg["dataset"]["eval"][dataset_name]["morphs"][morph_type]:
                continue
            
            dataset = MorphDataset(dataset_name=dataset_name, morph_type=morph_type, data_cfg=data_cfg)
            out_string += f"Evaluating {dataset} with morph type {morph_type}\n"    
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            results, print_string, preds = evaluate(model, dataloader, device)
            results_total[f"{dataset_name}_{morph_type}"] = results
            if args.save_preds:
                save_path = args.save_preds
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                preds_path = os.path.join(save_path, f"{dataset_name}_{morph_type}_preds.pkl")
                with open(preds_path, "wb") as f:
                    pickle.dump(preds, f)

            if not args.output:
                print(print_string)
            else:
                out_string += print_string

    mean_eer = sum([results["eer"] for results in results_total.values()]) / len(results_total)
    mean_err_string = f"Mean EER over all datasets: {mean_eer:.4f}\n"
    out_string += mean_err_string
    if args.output:
        if not os.path.exists("eval_out"):
            os.makedirs("eval_out")
        current_time = datetime.now().strftime("%d.%m.%y_%Hh%Mm%Ss")
        file_path = f"eval_out/{args.output}_{current_time}.txt"
        with open(file_path, "w") as f:
            f.write(out_string)
    else:
        print(mean_err_string)


def evaluate(model, dataloader, device, positive_label=None, verbose=True):
    out_lines = ""
    output_dict, target_dict, fnames = [], [], []
    if verbose:
        data_iterator = tqdm(dataloader, desc=f"Evaluating {dataloader.dataset.dataset_name} - {dataloader.dataset.morph_type}")
    for data in data_iterator:
        img = data["img"].to(device, non_blocking=True).float()
        target = data["label"].to(device, non_blocking=True).long()
        fname = data["fname"]
        with torch.no_grad():
            output = model(img)
            if isinstance(output, tuple): # some models output (clf_output, *additional_outputs)
                output = output[0]
        output_dict += output.softmax(1)[:, 1].cpu().data.numpy().tolist()
        target_dict += target.cpu().data.numpy().tolist()
        fnames += fname
    
    val_auc_0, val_eer_0, threshold_APCER_0, threshold_BPCER_0, threshold_ACER_0 = performances_compute(output_dict, target_dict, threshold_type="eer", op_val=0.1, verbose=False, positive_label=0)
    val_auc_1, val_eer_1, threshold_APCER_1, threshold_BPCER_1, threshold_ACER_1 = performances_compute(output_dict, target_dict, threshold_type="eer", op_val=0.1, verbose=False, positive_label=1)
    
    if positive_label is not None:
        if positive_label == 0:
            val_auc, val_eer, threshold_APCER, threshold_BPCER, threshold_ACER = val_auc_0, val_eer_0, threshold_APCER_0, threshold_BPCER_0, threshold_ACER_0
        else:
            val_auc, val_eer, threshold_APCER, threshold_BPCER, threshold_ACER = val_auc_1, val_eer_1, threshold_APCER_1, threshold_BPCER_1, threshold_ACER_1
    else:
        if val_eer_0 < val_eer_1:
            val_auc, val_eer, threshold_APCER, threshold_BPCER, threshold_ACER = val_auc_0, val_eer_0, threshold_APCER_0, threshold_BPCER_0, threshold_ACER_0
            positive_label = 0
        else:
            val_auc, val_eer, threshold_APCER, threshold_BPCER, threshold_ACER = val_auc_1, val_eer_1, threshold_APCER_1, threshold_BPCER_1, threshold_ACER_1
            positive_label = 1
        
    out_lines += "*****************************************************************************************\n"
    out_lines += f"Dataset: {dataloader.dataset.dataset_name}, Method: {dataloader.dataset.morph_type}, Positive Label: {positive_label}\n"
    out_lines += f"AUC: {val_auc:.4f}, EER: {val_eer:.4f}, APCER: {threshold_APCER:.4f}, BPCER: {threshold_BPCER:.4f}, ACER: {threshold_ACER:.4f}\n"
    out_lines += "*****************************************************************************************\n"

    results = {"auc": val_auc, "eer": val_eer, "apcer": threshold_APCER, "bpcer": threshold_BPCER, "acer": threshold_ACER}
    preds = {"raw": {}, "morphs": {}}
    for fname, output, label in zip(fnames, output_dict, target_dict):
        if label == 1:
            preds["morphs"][fname] = output
        else:
            preds["raw"][fname] = output
    return results, out_lines, preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("-p", dest="checkpoint_path", type=str, required=True, help="Path to the model")
    parser.add_argument("-b", dest="batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("-o", dest="output", type=str, default=None, help="Output file to write the results")
    parser.add_argument("-s", dest="save_preds", type=str, metavar="PATH", help="Save the predictions of the model to the specified PATH")

    args = parser.parse_args()
    main(args)
