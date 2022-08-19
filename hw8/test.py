import torch
import torch.nn as nn
from Sequence2sequence import EN2CNDataset
from Sequence2sequence import build_model
from Sequence2sequence import tokens2sentence
from Sequence2sequence import computebleu
import torch.utils.data as data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判斷是用 CPU 還是 GPU 執行運算
def test(model, dataloader, loss_function):
  model.eval()
  loss_sum, bleu_score= 0.0, 0.0
  n = 0
  result = []
  for sources, targets in dataloader:
    sources, targets = sources.to(device), targets.to(device)
    batch_size = sources.size(0)
    outputs, preds = model.inference(sources, targets)
    # targets 的第一個 token 是 <BOS> 所以忽略
    outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
    targets = targets[:, 1:].reshape(-1)

    loss = loss_function(outputs, targets)
    loss_sum += loss.item()

    # 將預測結果轉為文字
    targets = targets.view(sources.size(0), -1)
    preds = tokens2sentence(preds, dataloader.dataset.int2word_cn)
    sources = tokens2sentence(sources, dataloader.dataset.int2word_en)
    targets = tokens2sentence(targets, dataloader.dataset.int2word_cn)
    for source, pred, target in zip(sources, preds, targets):
      result.append((source, pred, target))
    # 計算 Bleu Score
    bleu_score += computebleu(preds, targets)

    n += batch_size

  return loss_sum / len(dataloader), bleu_score / n, result

def test_process(config):
  # 準備測試資料
  test_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'testing')
  test_loader = data.DataLoader(test_dataset, batch_size=1)
  # 建構模型
  model, optimizer = build_model(config, test_dataset.en_vocab_size, test_dataset.cn_vocab_size)
  print ("Finish build model")
  loss_function = nn.CrossEntropyLoss(ignore_index=0)
  model.eval()
  # 測試模型
  test_loss, bleu_score, result = test(model, test_loader, loss_function)
  # 儲存結果
  with open(f'./test_output.txt', 'w') as f:
    for line in result:
      print (line, file=f)

  return test_loss, bleu_score
class configurations(object):
  def __init__(self):
    self.batch_size = 60
    self.emb_dim = 256
    self.hid_dim = 512
    self.n_layers = 3
    self.dropout = 0.5
    self.learning_rate = 0.00005
    self.max_output_len = 50              # 最後輸出句子的最大長度
    self.num_steps = 12000                # 總訓練次數
    self.store_steps = 300                # 訓練多少次後須儲存模型
    self.summary_steps = 300              # 訓練多少次後須檢驗是否有overfitting
    self.load_model = False               # 是否需載入模型
    self.store_model_path = "./ckpt"      # 儲存模型的位置
    self.load_model_path = None           # 載入模型的位置 e.g. "./ckpt/model_{step}"
    self.data_path = "./cmn-eng"          # 資料存放的位置
    self.attention = False                # 是否使用 Attention Mechanism
# 在執行 Test 之前，請先行至 config 設定所要載入的模型位置
if __name__ == '__main__':
  config = configurations()
  print ('config:\n', vars(config))
  test_loss, bleu_score = test_process(config)
  print (f'test loss: {test_loss}, bleu_score: {bleu_score}')