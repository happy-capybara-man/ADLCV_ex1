# ADLCV Exercise 1: Road Rockfall SD3.5 DreamBooth LoRA

本專案以「道路落石、山路坍方、岩石阻斷道路」作為特定視覺概念，建立一套從資料蒐集、資料清理、影像字幕生成、Stable Diffusion 3.5 Medium DreamBooth LoRA 微調、影像生成，到 CLIP/KID 評估的完整實驗流程。

核心目標是讓 SD3.5 Medium 學會一個可由 trigger token 控制的道路落石事件概念：

```text
zwxrockfall
```

訓練後，模型應能在 prompt 中看到這個 token 時，更穩定地生成「真實攝影風格的道路落石災害場景」，包含大型岩塊、碎石、泥土、山坡、道路阻斷、柏油路面、車道線、護欄或地形脈絡等元素。

## 目前專案狀態

- 使用模型：`stabilityai/stable-diffusion-3.5-medium`
- 微調方法：DreamBooth LoRA for SD3
- 主題概念：road rockfall event / mountain road blocked by rocks
- Trigger token：`zwxrockfall`
- 目前訓練資料：`training_data/`，共 30 張影像
- Caption 格式：Hugging Face imagefolder metadata，欄位為 `file_name` 與 `text`
- 主要訓練腳本：`run_train.sh`
- 批次實驗腳本：`run_experiments.sh`
- 生成腳本：`generate.py`
- 評估腳本：`evaluate.py`
- 實驗輸出、log、生成圖片預設已加入 `.gitignore`，避免大量產物直接進入 git

## 專案架構

```text
ADLCV_ex1/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── crawer.py
├── prepare_data.py
├── caption_training_data.py
│
├── run_train.sh
├── run_experiments.sh
├── train_dreambooth_lora_sd3.py
│
├── generate.py
├── evaluate.py
├── prompts.json
│
├── metadata.json
├── metadata.jsonl
├── training_data/
│   ├── img_0001.jpg
│   ├── ...
│   ├── img_0030.jpg
│   └── metadata.jsonl
│
├── rock/
├── 石頭/
│
├── experiments/          # ignored, batch experiment outputs
├── experiment_logs/      # ignored, console logs
├── lora_output/          # ignored, default LoRA output
├── generated_images/     # ignored for new generated outputs
└── eval_results/         # ignored, evaluation summaries
```

### 重要檔案說明

| 檔案 | 功能 |
| --- | --- |
| `crawer.py` | 使用 DDGS 搜尋並下載道路落石相關圖片，含來源過濾、尺寸檢查、重複檢查與來源 CSV 紀錄。 |
| `prepare_data.py` | 將來源圖片中心裁切成正方形、resize 到 1024x1024，轉成 JPEG，並產生 `metadata.jsonl`。 |
| `caption_training_data.py` | 呼叫 NVIDIA VLM API 為訓練圖片產生 dense captions，並補上 trigger token。 |
| `metadata.json` | Caption 結果的 JSON list 版本，方便人工檢查。 |
| `metadata.jsonl` | Hugging Face imagefolder / diffusers 訓練流程使用的 JSONL 版本。 |
| `training_data/metadata.jsonl` | 實際訓練資料夾內的 metadata copy，讓 `--dataset_name training_data` 可直接讀取。 |
| `run_train.sh` | 單次 SD3.5 DreamBooth LoRA 訓練入口，處理 GPU 選擇、accelerate config、metadata 與訓練參數。 |
| `run_experiments.sh` | 批次執行多組 LoRA rank / learning rate 實驗，並將輸出與 log 分開保存。 |
| `generate.py` | 產生 base model 與 LoRA model 兩組對照影像。 |
| `evaluate.py` | 使用 CLIP Score 與 KID 評估生成影像品質。 |
| `prompts.json` | 評估/生成用 prompt bank，含 trigger token、class prompt、negative prompt 與多個 base prompts。 |
| `train_dreambooth_lora_sd3.py` | Hugging Face diffusers 官方 SD3 DreamBooth LoRA 訓練腳本。若缺少，`run_train.sh` 會自動下載。 |

## 環境與依賴

本專案使用 `uv` 管理 Python 環境。主要依賴列在 `pyproject.toml`：

- PyTorch / TorchVision / TorchAudio, CUDA 12.6 wheel index
- diffusers, transformers, accelerate, peft
- bitsandbytes, tensorboard, datasets
- pillow, requests, python-dotenv
- ddgs, bing-image-downloader
- torch-fidelity
- torchmetrics[multimodal]

建議使用 Python 3.10 以上：

```bash
uv sync
```

SD3.5 Medium 是 gated model，訓練與生成前需要完成 Hugging Face 授權：

```bash
huggingface-cli login
```

或設定 `HF_TOKEN`。同時需要先到 Hugging Face 接受模型授權：

```text
https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
```

若要重新產生 captions，還需要 NVIDIA API key：

```bash
NVIDIA_API_KEY=your_key_here
```

也可以放在 `.env`，`caption_training_data.py` 會用 `python-dotenv` 讀取。

## 完整流程

整個 pipeline 可以分成四個 phase。

### Phase 0: 資料蒐集

`crawer.py` 負責從搜尋引擎抓取道路落石相關圖片。目前搜尋關鍵字包含：

- `rockfall blocking road real photo`
- `highway landslide debris`
- `road rockfall damage`

下載流程包含幾個資料品質控制：

- 避免常見圖庫與浮水印來源，例如 Shutterstock、iStock、Getty Images 等。
- 過濾 URL 中包含 `watermark`、`stock-photo`、`stock-image` 等關鍵字的圖片。
- 檢查圖片尺寸，預設至少 `640x400`。
- 用 SHA-256 hash 避免同一次下載中的重複圖片。
- 針對每個 query 建立資料夾與 `sources.csv`，保存原始圖片 URL、頁面 URL 與標題。

執行方式：

```bash
uv run python crawer.py
```

下載結果預設會放在：

```text
rockfall_dataset/
```

目前專案中另有 `rock/` 與 `石頭/` 兩個來源資料夾，作為 `prepare_data.py` 的預設掃描來源。

### Phase 1: 資料整理

`prepare_data.py` 會掃描來源資料夾，將圖片處理成 SD3.5 訓練較容易使用的格式。

處理步驟：

1. 讀取 `rock/` 與 `石頭/` 下的圖片。
2. 將圖片轉成 RGB。
3. 以中心裁切方式裁成正方形。
4. resize 到 `1024x1024`。
5. 以 JPEG quality 95 輸出。
6. 產生 Hugging Face imagefolder 格式的 `metadata.jsonl`。

執行方式：

```bash
uv run python prepare_data.py
```

目前訓練主流程使用的是 `training_data/`。如果重新跑 `prepare_data.py`，請確認輸出資料夾與 `run_train.sh` 的 `INSTANCE_DATA_DIR` 一致。可以選擇將 `prepare_data.py` 的 `OUTPUT_DIR` 改成 `training_data`，或執行後把整理完成的圖片與 metadata 同步到 `training_data/`。

### Phase 2: Caption 生成與 metadata 設計

`caption_training_data.py` 使用 NVIDIA Chat Completions VLM endpoint 對每張訓練圖片產生 dense caption。預設模型是：

```text
google/gemma-3-27b-it
```

預設 endpoint：

```text
https://integrate.api.nvidia.com/v1/chat/completions
```

設計重點：

- Caption 不是短標籤，而是 70 到 120 字左右的自然語言描述。
- Prompt 要求模型先觀察可見證據，再根據畫面描述岩石大小、形狀、顏色、粗糙紋理、泥土、碎石、道路位置、山坡、天氣、光線與紀實攝影視角。
- 避免 hallucination，不猜測畫面中沒有清楚出現的物件。
- 避免輸出 markdown、bullet points 或逗號 tag list。
- Trigger token 不由 VLM 生成，而是在後處理階段統一加在 caption 開頭。
- 若 API 失敗或 caption 不可用，會重試 fallback prompts；最後仍失敗時會根據圖片亮度、色調與方向建立 generic fallback caption。

輸出格式：

```json
{"file_name": "img_0001.jpg", "text": "zwxrockfall, A substantial rockfall has overwhelmed a section of roadway..."}
```

執行方式：

```bash
uv run python caption_training_data.py \
	--data-dir training_data \
	--output-json metadata.json \
	--output-jsonl metadata.jsonl
```

常用參數：

| 參數 | 說明 |
| --- | --- |
| `--data-dir` | 要產生 caption 的圖片資料夾。 |
| `--output-json` | JSON list 輸出，方便檢查。 |
| `--output-jsonl` | JSONL 輸出，供訓練使用。 |
| `--model` | 覆蓋預設 NVIDIA VLM 模型。 |
| `--trigger-token` | 覆蓋預設 trigger token。 |
| `--force` | 重新產生已存在的 caption。 |
| `--raw-captions` | 不加 SD3.5 enrichment，直接保存 VLM 原始 caption。 |
| `--no-stream` | 使用一般 JSON response 而不是 SSE streaming。 |

### Phase 3: SD3.5 DreamBooth LoRA 訓練

`run_train.sh` 是主要訓練入口。它包住 diffusers 官方 `train_dreambooth_lora_sd3.py`，並加入本專案需要的安全檢查與預設參數。

執行方式：

```bash
./run_train.sh
```

預設訓練設定：

| 設定 | 預設值 |
| --- | --- |
| Base model | `stabilityai/stable-diffusion-3.5-medium` |
| Instance data | `training_data` |
| Output dir | `lora_output` |
| Trigger token | `zwxrockfall` |
| Resolution | `1024` |
| Batch size | `1` |
| Gradient accumulation | `4` |
| Effective batch size | `4` |
| Learning rate | `1e-4` |
| LR scheduler | cosine |
| Warmup steps | `100` |
| Max steps | `1000` |
| LoRA rank | `16` |
| Checkpoint interval | `200` steps |
| Mixed precision | `fp16` |
| Seed | `42` |

GPU 設計：

- 預設 `GPU_ID=0`。
- 腳本會設定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，減少 GPU index 對不上 `nvidia-smi` 的問題。
- 腳本會用 `CUDA_VISIBLE_DEVICES` mask 成單張 GPU，讓 accelerate 看到 logical GPU 0。
- 預設要求至少 `20 GiB` VRAM，避免誤跑到較小 GPU 造成 OOM。
- 可以用 `MIN_GPU_MEMORY_GIB` 覆蓋最低 VRAM 檢查。

常用覆蓋方式：

```bash
GPU_ID=1 ./run_train.sh
```

```bash
RANK=8 LR=1e-4 MAX_STEPS=1000 OUTPUT_DIR=experiments/exp_rank8 ./run_train.sh
```

```bash
METADATA_JSONL=metadata.jsonl OUTPUT_DIR=lora_output ./run_train.sh
```

快速 sanity check：

```bash
MIN_DATA=1 ./run_train.sh
```

`MIN_DATA=1` 會建立很小的 `training_data_min/`，只跑 15 steps，主要用來確認環境、資料讀取、LoRA 訓練腳本與 GPU 設定都能正常工作。

Validation 設計：

- 預設 validation prompt：`zwxrockfall, large boulder blocking mountain highway, real photograph`
- 預設每 10 steps 取 2 個 batch 計算 validation loss。
- 預設跳過中途影像 validation，以降低 VRAM 使用。
- 訓練結束時仍保留 final validation。

TensorBoard：

```bash
tensorboard --logdir lora_output/logs
```

### Phase 4: 影像生成

`generate.py` 用來建立 base model 與 LoRA model 的對照組。

執行方式：

```bash
uv run python generate.py --setting both
```

可選設定：

```bash
uv run python generate.py --setting a
uv run python generate.py --setting b
uv run python generate.py --setting both
```

兩組設定：

| Setting | 模型 | Prompt |
| --- | --- | --- |
| A | SD3.5 Medium base model | `prompts.json` 中的 base prompts，不加 trigger token |
| B | SD3.5 Medium + LoRA | 在每個 base prompt 前加上 `zwxrockfall` |

生成設計：

- Prompt 數量：10 個
- 每個 prompt 生成 seed 數量：30 個
- 每組 setting 目標：300 張圖
- 圖片大小：`1024x1024`
- Inference steps：`28`
- Guidance scale：`7.0`
- 檔名格式：`img_{prompt_index:02d}_{seed:03d}.jpg`
- 支援 resume：已存在的圖片會跳過，不會重複生成

輸出目錄：

```text
generated_images/setting_a/
generated_images/setting_b/
```

`generated_images/` 目前已加入 `.gitignore`，新的生成結果不會自動進入 git。

### 最終提交版本重現

本次最後決定提交的版本不是早期的 `setting_b`，而是以下這條實驗分支：

- 訓練輸出：`experiments/prior_textenc_bf16_rank16_lr1e-4_te5e-6_steps600`
- Class image 設定：`class_data/custom_rockfall_v1_generation_config.json`
- 生成 prompt：`old_prompt_dirty_realistic_short.json`
- 生成 LoRA scale：`0.8`
- 正式輸出目錄：`generated_images/experiments/prior_textenc_bf16_rank16_lr1e-4_te5e-6_steps600_dirty_realistic_short_lora0.8_seed0to29`

需要注意的是，這個最終版本有開啟 text encoder fine-tuning，而目前 `run_train.sh` 沒有把 `--train_text_encoder` 與 `--text_encoder_lr` 暴露成 shell 參數；因此若要**精確重現**最後提交版本，應直接呼叫 `train_dreambooth_lora_sd3.py`。

1. 啟動環境

```bash
cd /home/willyfr4214/hw/ADLCV/ADLCV_ex1
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

2. 訓練最終提交版本

```bash
accelerate launch train_dreambooth_lora_sd3.py \
	--pretrained_model_name_or_path=stabilityai/stable-diffusion-3.5-medium \
	--dataset_name=training_data \
	--caption_column=text \
	--output_dir=experiments/prior_textenc_bf16_rank16_lr1e-4_te5e-6_steps600 \
	--mixed_precision=bf16 \
	--instance_prompt="zwxrockfall, road blocked by rockfall debris, real photograph, outdoor" \
	--resolution=512 \
	--train_batch_size=1 \
	--gradient_accumulation_steps=4 \
	--gradient_checkpointing \
	--learning_rate=1e-4 \
	--text_encoder_lr=5e-6 \
	--train_text_encoder \
	--lr_scheduler=cosine \
	--lr_warmup_steps=100 \
	--max_train_steps=600 \
	--rank=16 \
	--checkpointing_steps=200 \
	--seed=42 \
	--with_prior_preservation \
	--class_data_dir=class_data \
	--class_prompt="realistic photograph of a rockfall event on an asphalt mountain road, fallen boulders and small rock fragments scattered on the driving lane, visible lane markings, rocky hillside, outdoor daylight" \
	--class_negative_prompt="empty road, clean road, no rocks on road, rocks only on roadside, rocks only on hillside, traffic cone, construction cone, people, vehicle, cartoon, anime, painting, 3d render, cgi, blurry, low quality, text, watermark" \
	--num_class_images=100 \
	--prior_loss_weight=1.0 \
	--sample_batch_size=1 \
	--prior_generation_precision=bf16 \
	--validation_prompt="zwxrockfall, large boulder blocking mountain highway, real photograph" \
	--num_validation_images=1 \
	--validation_num_inference_steps=20 \
	--validation_epochs=20 \
	--validation_loss_steps=10 \
	--validation_loss_num_batches=2 \
	--skip_intermediate_validation \
	--final_validation_cpu_offload \
	--logging_dir=logs \
	--report_to=tensorboard
```

最終訓練設定摘要：

| 項目 | 值 |
| --- | --- |
| Base model | `stabilityai/stable-diffusion-3.5-medium` |
| Method | DreamBooth LoRA + text encoder fine-tuning |
| Mixed precision | `bf16` |
| Resolution | `512` |
| Rank | `16` |
| Transformer LR | `1e-4` |
| Text encoder LR | `5e-6` |
| Max steps | `600` |
| Checkpoint steps | `200` |
| Prior preservation | enabled |
| Class prompt | `realistic photograph of a rockfall event on an asphalt mountain road, fallen boulders and small rock fragments scattered on the driving lane, visible lane markings, rocky hillside, outdoor daylight` |
| Class negative prompt | `empty road, clean road, no rocks on road, rocks only on roadside, rocks only on hillside, traffic cone, construction cone, people, vehicle, cartoon, anime, painting, 3d render, cgi, blurry, low quality, text, watermark` |
| Class data config | `class_data/custom_rockfall_v1_generation_config.json` |
| Class images | `100` |
| Prior loss weight | `1.0` |
| Train batch size | `1` |
| Gradient accumulation | `4` |
| Seed | `42` |

3. 先做單 seed 檢查圖

```bash
./.venv/bin/python generate.py \
	--lora_dir experiments/prior_textenc_bf16_rank16_lr1e-4_te5e-6_steps600 \
	--output_dir generated_images/experiments/prior_textenc_bf16_rank16_lr1e-4_te5e-6_steps600_dirty_realistic_short_lora0.8_seed0 \
	--prompts_file old_prompt_dirty_realistic_short.json \
	--n_per_prompt 1 \
	--lora_scale 0.8
```

4. 生成正式提交用 300 張圖

```bash
./.venv/bin/python generate.py \
	--lora_dir experiments/prior_textenc_bf16_rank16_lr1e-4_te5e-6_steps600 \
	--output_dir generated_images/experiments/prior_textenc_bf16_rank16_lr1e-4_te5e-6_steps600_dirty_realistic_short_lora0.8_seed0to29 \
	--prompts_file old_prompt_dirty_realistic_short.json \
	--n_per_prompt 30 \
	--lora_scale 0.8
```

這個最終版本的生成設定如下：

- Prompt 數量：`10`
- 每個 prompt seeds：`30`
- 總輸出張數：`300`
- Prompt file：`old_prompt_dirty_realistic_short.json`
- Negative prompt：使用 `old_prompt_dirty_realistic_short.json` 內的 `negative_prompt`
- LoRA scale：`0.8`
- Inference steps：`28`
- Guidance scale：`7.0`
- Output size：`1024x1024`

5. 評估最終提交版本

```bash
./.venv/bin/python evaluate.py \
	--generated_dir generated_images/experiments/prior_textenc_bf16_rank16_lr1e-4_te5e-6_steps600_dirty_realistic_short_lora0.8_seed0to29 \
	--real_dir training_data
```

`evaluate.py` 會自動從 `generation_config.json` 讀回 prompt file，因此這一步通常不需要另外指定 `--prompts_file`。

### Phase 5: 評估

`evaluate.py` 會比較 base model 與 LoRA model 的生成結果。

執行方式：

```bash
uv run python evaluate.py
```

評估項目：

| 指標 | 說明 | 方向 |
| --- | --- | --- |
| CLIP Score | 衡量生成圖片與對應 prompt 的語意一致性。 | 越高越好 |
| KID | Kernel Inception Distance，衡量生成圖片分布與真實訓練圖片分布的距離。 | 越低越好 |

CLIP Score 設定：

- 使用 `openai/clip-vit-large-patch14`
- batch size 預設為 `16`
- 圖片與 prompt 會根據檔名中的 prompt index 對齊

KID 設定：

- real images：`training_data/`
- fake images：`generated_images/setting_a/` 或 `generated_images/setting_b/`
- `kid_subset_size = min(n_real, n_fake, 100)`

輸出：

```text
eval_results/results.json
```

終端機也會印出 comparison table，方便直接比較 Setting A 與 Setting B。

## 實驗設計

目前的實驗主要在回答三個問題：

1. LoRA 是否能讓 SD3.5 更穩定地生成道路落石概念？
2. LoRA rank 對畫面細節與概念吸收程度有何影響？
3. Learning rate 對訓練穩定性與生成品質有何影響？

### 單次 baseline 設定

`run_train.sh` 的預設設定即為 baseline：

| 項目 | 值 |
| --- | --- |
| Rank | `16` |
| LR | `1e-4` |
| Max steps | `1000` |
| Checkpoint interval | `200` |
| Resolution | `1024` |
| Effective batch size | `4` |

Rank 16 被視為目前資料規模下的平衡點：參數量足以吸收道路落石概念與岩石紋理，但又不至於過度增加訓練與儲存成本。

### 批次實驗矩陣

`run_experiments.sh` 目前定義四組實驗：

| 實驗名稱 | Rank | LR | Steps | 目的 |
| --- | ---: | ---: | ---: | --- |
| `baseline` | 16 | `1e-4` | 1000 | 主 baseline，作為其他實驗比較基準。 |
| `exp1_rank8` | 8 | `1e-4` | 1000 | 測試較低 rank 是否仍能學到概念，觀察容量下降的影響。 |
| `exp2_lr5e-5` | 16 | `5e-5` | 1000 | 測試較低 learning rate 是否更穩定、是否降低過擬合。 |
| `exp3_lr8e-5` | 16 | `8e-5` | 1000 | 測試介於 `5e-5` 與 `1e-4` 之間的折衷。 |

執行方式：

```bash
./run_experiments.sh
```

預設輸出：

```text
experiments/<exp_name>_rank<rank>_lr<lr>_steps<steps>/
experiment_logs/<timestamp>_<exp_name>_rank<rank>_lr<lr>_steps<steps>.log
```

可以用環境變數覆蓋批次設定：

```bash
GPU_ID=1 CKPT_STEPS=100 SKIP_EXISTING=1 ./run_experiments.sh
```

`SKIP_EXISTING=1` 時，若某個實驗目錄已經存在最終 `pytorch_lora_weights.safetensors`，該實驗會被跳過，方便中斷後續跑。

### Checkpoint 比較

每組實驗預設每 200 steps 保存 checkpoint。這使同一組實驗內可以比較：

- 200 steps：概念是否已開始出現
- 400 steps：構圖與 rockfall token 是否穩定
- 600 steps：岩石材質、道路阻斷與地形是否更具體
- 800 steps：是否開始過擬合到訓練圖特徵
- 1000 steps：最終模型品質

這個設計讓實驗不只比較不同 hyperparameters，也可以觀察訓練過程中概念被吸收的速度。

## Prompt 設計

`prompts.json` 包含：

- `trigger_token`: `zwxrockfall`
- `class_prompt`: prior/class image 生成用的通用道路落石描述
- `negative_prompt`: 排除乾淨道路、車輛、人物、標誌、coastal cliff 與過度鮮豔風格
- `base_prompts`: 多個道路落石相關生成 prompt

Prompt bank 的設計目標：

- 每個 prompt 都描述「真實紀實攝影」風格，而不是插畫或概念圖。
- 場景涵蓋山路、峽谷、公路、隧道口、海岸道路、河邊山路等不同地理脈絡。
- 反覆強調 natural rocks、boulders、gravel、mud、wet asphalt、collapsed slope 等核心視覺元素。
- 多數 prompt 明確排除人、車、標誌、錐筒或文字，降低模型生成不必要物件的機率。
- Base setting 不使用 trigger token，LoRA setting 使用 trigger token，讓比較更清楚。

範例 LoRA prompt：

```text
zwxrockfall, a realistic documentary photo of an empty two-lane mountain road completely blocked by natural irregular gray and tan boulders, jagged rock fragments, muddy gravel, wet asphalt, collapsed steep slope, no people, no vehicles, no signs
```

## 模型與資料設計理念

### 為什麼使用 DreamBooth LoRA

道路落石是一個具體但資料量有限的視覺概念。直接 full fine-tuning SD3.5 成本高、需要更多 VRAM，且容易破壞 base model 原本能力。LoRA 的優點是：

- 訓練成本低。
- 只保存小型 adapter 權重。
- 可用 trigger token 控制概念是否啟用。
- 適合少量資料學習特定 visual concept。
- 可以保留 base model 的一般生成能力。

### 為什麼使用 dense captions

若只使用短 caption，例如 `rockfall on road`，模型可能只學到「有石頭、有道路」，但難以學到岩石大小、道路阻斷、泥土、山坡、光線與紀實攝影脈絡。因此本專案使用 dense captions，把每張圖的視覺證據寫得更細。

Dense caption 讓訓練資料同時提供：

- 主體概念：road rockfall event
- 物件細節：large boulders、jagged blocks、gravel、mud、guardrail、lane markings
- 空間關係：rocks blocking both lanes、debris on shoulder、slope above road
- 視覺風格：real photograph、documentary perspective、natural outdoor lighting
- 環境脈絡：mountain road、cliff face、river、tunnel、overcast weather

### 為什麼用 trigger token

Trigger token 是把特定概念綁定到 LoRA 的控制介面。訓練時所有 caption 都以 `zwxrockfall` 開頭，生成時只要在 prompt 前加上這個 token，就能提示 LoRA 啟動該概念。

這樣的設計讓比較實驗更乾淨：

- Setting A：沒有 trigger token，測 base model 原始能力。
- Setting B：有 trigger token 並載入 LoRA，測微調後能力。

### 為什麼評估 CLIP Score 與 KID

這兩個指標衡量的是不同面向：

- CLIP Score 看 prompt-image alignment，適合檢查生成圖是否符合文字描述。
- KID 看生成圖分布與真實資料分布距離，適合檢查 LoRA 是否讓生成結果更像訓練資料域。

兩者一起看比單一指標更有意義。例如 LoRA 可能提高 KID 表現，代表更接近真實落石資料；但如果 CLIP Score 下降，可能表示生成圖雖然像訓練集，卻沒有精準遵守 prompt。

## 常用指令

安裝依賴：

```bash
uv sync
```

登入 Hugging Face：

```bash
huggingface-cli login
```

產生 captions：

```bash
uv run python caption_training_data.py --data-dir training_data
```

單次訓練：

```bash
./run_train.sh
```

指定 GPU 訓練：

```bash
GPU_ID=1 ./run_train.sh
```

快速測試訓練流程：

```bash
MIN_DATA=1 ./run_train.sh
```

跑完整實驗矩陣：

```bash
./run_experiments.sh
```

只生成 LoRA 組：

```bash
uv run python generate.py --setting b
```

生成 base 與 LoRA 對照組：

```bash
uv run python generate.py --setting both
```

評估：

```bash
uv run python evaluate.py
```

查看 TensorBoard：

```bash
tensorboard --logdir experiments
```

## 輸出與版本控制策略

以下目錄屬於實驗產物或大型生成結果，已加入 `.gitignore`：

```text
experiments/
experiment_logs/
lora_output/
lora_output_min/
generated_images/
generated_preview_lora/
eval_results/
*.log
```

這樣設計的原因：

- LoRA checkpoint、TensorBoard events、生成圖片與評估結果會快速變大。
- 實驗結果通常可重跑，不一定適合直接放進 git。
- Repo 保留程式、metadata、prompt 設計與可重現流程即可。

如果需要保存重要結果，建議另外建立 release artifact、雲端儲存、或只挑選少量代表圖與結果表格加入報告。

## 已知注意事項

- SD3.5 Medium 需要 Hugging Face 授權，未接受 license 會導致模型下載失敗。
- 訓練需要高 VRAM GPU；目前腳本預設檢查至少 20 GiB。
- `generate.py` 使用 `torch.bfloat16` 並將模型放到 CUDA，沒有 GPU 時無法正常生成。
- 若要生成最終提交版本的 `1024x1024`、`10 prompts x 30 seeds`，請先確認 GPU 沒有其他大型 CUDA 程序佔用；否則 `generate.py` 在把整個 SD3.5 pipeline 搬到 CUDA 時可能直接 OOM。
- `evaluate.py` 的 CLIP Score 與 KID 預設使用 CUDA。
- 目前主訓練資料夾是 `training_data/`；若重新整理資料，請確認圖片與 metadata 都在同一個資料夾中。
- `generated_images/` 已被 ignore，但若歷史 commit 中已追蹤過某些生成圖，`.gitignore` 不會自動把已追蹤檔案移出版本控制。

## 建議報告撰寫方向

在實驗報告中，可以依照以下順序整理：

1. 問題定義：讓 SD3.5 學會道路落石事件概念。
2. 資料處理：來源、清理、裁切、caption、trigger token。
3. 方法：DreamBooth LoRA、SD3.5 Medium、rank、learning rate、steps。
4. 實驗設計：baseline、rank8、lr5e-5、lr8e-5、checkpoint 比較。
5. 生成設定：base vs LoRA、10 prompts、30 seeds。
6. 評估：CLIP Score、KID、定性視覺比較。
7. 討論：LoRA 是否改善道路落石概念、是否有過擬合、哪組 hyperparameters 最穩定。
