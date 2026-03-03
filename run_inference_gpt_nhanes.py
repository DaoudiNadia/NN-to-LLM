"""
Evaluates and compares nhanes predictions from a trained NN
and GPT-5 using raw and guided prompts on the test set.
"""
import re
import os
import json
import numpy as np
from dotenv import load_dotenv
import torch
from sklearn.metrics import accuracy_score
import openai
from utils import prepare_data_nhanes, train_nn, truncate, NN, ensure_dir
from config import  PROMPT_TEMPLATE_RAW_NHANES as template_raw
from config import PROMPT_TEMPLATE_GUIDED_NHANES as template_guided

def print_scores(start_idx, end_idx, x_test, y_test, x_test_orig,
                 feature_orig, nn):
    """
    Evaluates and compares predictions from the NN and GPT-5 RAW and
    GUIDED.

    For each sample in the given range, collects NN predictions and
    sends the sample to GPT-5 twice: once with a raw prompt and once 
    with a prompt guided by NN knowledge. GPT-5 outputs are saved to
    text files and parsed as binary predictions. Accuracy scores for
    all three predictors are printed at the end.

    Parameters:
        start_idx (int): Index of the first test sample to evaluate.
        end_idx (int): Index of the last test sample to evaluate
            (exclusive).

    Returns:
        None, but prints accuracy scores for the NN, raw GPT-5, and
        guided GPT-5 predictions to stdout, and saves GPT-5 outputs to
        the 'predictions/nhanes/' directory.
    """
    y_orig, preds_nn, preds_raw, preds_guided = [], [], [], []
    for sample_id in range(start_idx, end_idx):
        x_sample = torch.from_numpy(x_test[sample_id]).float().unsqueeze(0)
        y_sample = torch.from_numpy(y_test[sample_id]).float().unsqueeze(0)
        y_orig.append(int(y_sample.numpy().item()))
        # Evaluate NN
        nn.eval()
        with torch.no_grad():
            logits = nn(x_sample)
            preds = (torch.sigmoid(logits) > 0.5).float()
        preds_nn.append(int(preds.numpy().item()))

        row = x_test_orig[sample_id]
        sample_dict = {}
        for name, value in zip(feature_orig, row):
            if isinstance(value, (np.generic, np.bool_)):
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode()
            if isinstance(value, (int, float)):
                if float(value).is_integer():
                    value = int(value)
                else:
                    value = truncate(float(value), 1)
            sample_dict[name] = value
        sample = [sample_dict]
        sample_json = json.dumps(sample, indent=2, ensure_ascii=False)
        prompt_raw = template_raw.format(sample=sample_json)
        prompt_guided = template_guided.format(sample=sample_json)

        print("Sending raw prompt to GPT-5...")
        response = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt_raw}],
        )
        gpt_output = response.choices[0].message.content.strip()
        file_raw = f"predictions/nhanes/gpt-5_raw_nhanes{sample_id}.txt"
        with open(file_raw, "w", encoding="utf-8") as f:
            f.write(gpt_output)
        try:
            # try converting directly to int
            prediction = int(gpt_output)
        except ValueError:
            # fallback if output has extra text around it
            match = re.search(r"\b([01])\b", gpt_output)
            prediction = int(match.group(1)) if match else None

        if prediction in [0, 1]:
            preds_raw.append(prediction)
        else:
            print("Failed to parse prediction, retrying...", prediction)

        print("Sending prompt with NN guidance to GPT-5...")
        response = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt_guided}],
        )
        gpt_output = response.choices[0].message.content.strip()
        file_guided = f"predictions/nhanes/gpt-5_guided_nhanes{sample_id}.txt"
        with open(file_guided, "w", encoding="utf-8") as f:
            f.write(gpt_output)
        try:
            # try converting directly to int
            prediction = int(gpt_output)
        except ValueError:
            # fallback if output has extra text around it
            match = re.search(r"\b([01])\b", gpt_output)
            prediction = int(match.group(1)) if match else None

        if prediction in [0, 1]:
            preds_guided.append(prediction)
        else:
            print("Failed to parse prediction, retrying...", prediction)

    print("#"*40)
    acc = accuracy_score(y_orig, preds_nn)
    print("Accuracy NN:", acc)

    acc = accuracy_score(y_orig, preds_raw)
    print("Accuracy GPT raw :", acc)

    acc = accuracy_score(y_orig, preds_guided)
    print("Accuracy GPT guided:", acc)


if __name__ == "__main__":
    load_dotenv(".env")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    NPZ_PATH = "processed_data/nhanes_data_processed.npz"
    if not os.path.exists(NPZ_PATH):
        print("Preprocessed data not found. Preparing data...")
        prepare_data_nhanes()
    else:
        print(f"Data file '{NPZ_PATH}' already exists. Loading directly.")

    data = np.load(NPZ_PATH, allow_pickle=True)

    x_train = data['x_train']
    x_test_processed = data['x_test']
    y_train = data['y_train']
    y_test_processed = data['y_test']
    x_test_orig_all = data['x_test_orig_all']
    meta = data.get('metadata', {})
    meta = meta.item()
    feature_names_orig = meta['feature_names_orig']
    device = torch.device("cpu")
    MODEL_PATH = "trained_nns/nhanes_nn.pth"

    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}, skipping training.")
        input_dim = x_train.shape[1]
        model = NN(input_dim).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        model = train_nn(x_train, y_train, model_path=MODEL_PATH,
                         batch_size=512, epochs=1000, lr=0.003)

    ensure_dir("predictions/nhanes/")
    print_scores(0, 512, x_test_processed, y_test_processed, x_test_orig_all,
                 feature_names_orig, model)
