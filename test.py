from os import PathLike
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from multiprocessing import cpu_count

from .dataset import TestDataset
from . import network


def run_test(test_data: PathLike | pd.DataFrame, 
            model_path: str, output_path: str, 
            input_shape: tuple[int] = (224, 224), 
            features_root: int = 64, 
            batch_size: int = 32):
    
    torch.cuda.empty_cache()
    cudnn.benchmark = True

    if isinstance(test_data, PathLike):
        test_data = pd.read_csv(test_data)
    elif isinstance(test_data, pd.DataFrame):
        test_data = test_data
    else:
        raise ValueError("test_data must be a Path or a DataFrame")

    test_dataset: TestDataset = TestDataset(csv_file=test_data, 
                                            input_shape=input_shape)
    
    test_loader: DataLoader = DataLoader(test_dataset, 
                                         batch_size=batch_size, 
                                         shuffle=False, 
                                         num_workers=64, 
                                         pin_memory=True)

    model = torch.nn.DataParallel(network.AEMAD(in_channels=3, 
                                                features_root=features_root))
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.cuda()
    model.eval()

    mse_criterion = torch.nn.MSELoss(reduction="none").cuda()

    test_scores, gt_labels, test_scores_dict = [], [], []

    with torch.no_grad():

        for data in tqdm(test_loader):
            raw, labels, img_ids = (
                data["images"].cuda(),
                data["labels"],
                data["img_path"],
            )
            _, output_raw = model(raw)

            scores = mse_criterion(output_raw, raw).cpu().data.numpy()
            scores = np.sum(np.sum(np.sum(scores, axis=3), axis=2), axis=1)
            test_scores.extend(scores)
            gt_labels.extend((1 - labels.data.numpy()))
            for j in range(labels.shape[0]):
                l = 'morph' if labels[j].detach().numpy() == 1 else 'bonafide'
                test_scores_dict.append({'Path': img_ids[j], 'Label': l, 'Score': float(scores[j])})

    scores_df = pd.DataFrame(test_scores_dict)
    scores_df.to_csv(output_path, index=False)
    print('Prediction scores write done in', output_path)


# NOT TO CALL -> TODO: provide a better method to run directly

"""
if __name__ == "__main__":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

    if torch.cuda.is_available():
        print("GPU is available")
        torch.cuda.manual_seed(0)
    else:
        print("GPU is not available")
        torch.manual_seed(0)

    import argparse

    parser = argparse.ArgumentParser(description='SPL MAD')

    parser.add_argument("--test_data", required=True, type=str, help="path of data directory including csv files")
    parser.add_argument("--model_path", required=True, type=str, help="model path")
    parser.add_argument("--output_path", default="test.csv", type=str, help="path for output prediction scores")
    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="model input shape")
    parser.add_argument("--features_root", default=64, type=int, help="feature root")
    parser.add_argument("--batch_size", default=32, type=int, help="test batch size")

    args = parser.parse_args()
    run_test(test_data=args.test_data, model_path=args.model_path, output_path=args.output_path,
             input_shape=args.input_shape, features_root=args.features_root, batch_size=args.batch_size)
"""
