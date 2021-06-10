import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from utils.global_variables import Global
from model.layers import embedding, crf, outputLayer
from transformers import (
    BertPreTrainedModel,
    BertModel,
    BertConfig,
)

class Crf(nn.Module):
    def __init__(self, config):
        super(Crf, self).__init__()
        self.config = config
        if not self.config.has_option("data", "BERT"):
            self.embedding = embedding.Embedding(config)
            self.rnn = DynamicRNN(config)
        else:
            bert_config = BertConfig.from_pretrained('bert-base-uncased')
            self.bert = MYBERT.from_pretrained('bert-base-uncased', config=bert_config)
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))
        self.hidden2tag = nn.Linear(in_features=config.getint("model", "hidden_size"),
                                    out_features=config.getint("runtime", "num_class") + 2,
                                    bias=True)
        self.pad_label_id = config.getint("data", "pad_label_id")
        if Global.device == torch.device("cpu"):
            self.crf = crf.CRF(tagset_size=config.getint("runtime", "num_class"))
        else:
            self.crf = crf.CRF(tagset_size=config.getint("runtime", "num_class"), use_gpu=Global.device)

    def forward(self, data, **params):
        """
        :param data: 这一轮输入的数据
        :param params: 存放任何其它需要的信息
        """
        mode = params["mode"]
        tokens = data["tokens"]         # [B, L]
        labels = data["labels"]         # [B, L]
        lengths = data["lengths"]       # [B, ]
        flags = data["flags"]
        attention_masks = data["masks"] # [B, L]
        if self.config.has_option("data", "BERT"):
            attention_bert_masks = data["attention_masks"]

        if self.config.has_option("data", "BERT"):
            token_type_ids = data['token_type_ids']

        if not self.config.has_option("data", "BERT"):
            prediction = self.embedding(tokens)     # [B, L, E]
            prediction = self.dropout(prediction)
            # 如果这里换成bert，则实际上输出的就是[B, L, 768]
            prediction = self.rnn(prediction, lengths)  # [B, L, H]
            # 这里多出的两个num_class到底对应的是什么类以及是什么含义？
            prediction = self.hidden2tag(prediction)    # [B, L, N+2]
        else:
            prediction = self.bert(input_ids=tokens, attention_mask=attention_bert_masks, token_type_ids=token_type_ids)
            prediction = self.hidden2tag(prediction)

        pad_masks = (labels != self.pad_label_id)
        loss_masks = ((attention_masks == 1) & pad_masks)

        if params["crf_mode"] == "train":
            crf_labels, crf_masks = self.to_crf_pad(labels, loss_masks)
            crf_logits, _ = self.to_crf_pad(prediction, loss_masks)
            loss = self.crf.neg_log_likelihood(crf_logits, crf_masks, crf_labels)
            return {"loss": loss,
                    "prediction": None,
                    "labels": None}

        elif params["crf_mode"] == "test":
            masks = (attention_masks == 1)
            crf_logits, crf_masks = self.to_crf_pad(prediction, masks)
            crf_masks = crf_masks.sum(axis=2) == crf_masks.shape[2]
            best_path = self.crf(crf_logits, crf_masks)
            temp_labels = (torch.ones(loss_masks.shape) * self.pad_label_id).to(torch.long)
            prediction = self.unpad_crf(best_path, crf_masks, temp_labels, masks)
            return {"loss": None,
                    "prediction": self.normalize(prediction, flags, lengths),
                    "labels": self.normalize(labels, flags, lengths)} if mode != "test" else {
                        "prediction": self.normalize(prediction, flags, lengths)
                    }

        else:
            raise NotImplementedError

    def normalize(self, logits, flags, lengths):
        results = []
        logits = logits.tolist()
        lengths = lengths.tolist()
        for logit, flag, length in zip(logits, flags, lengths):
            result = []
            for i in range(length):
                if flag[i] == 1:
                    assert logit[i] != self.pad_label_id
                    result.append(Global.id2label[str(logit[i])])
            results.append(result)
        return results

    def to_crf_pad(self, org_array, org_mask):
        # 实际上是把org_array当中的mask为true的部分进行截取，然后使用-100进行pad??
        crf_array = [aa[bb] for aa, bb in zip(org_array, org_mask)]
        crf_array = pad_sequence(crf_array, batch_first=True, padding_value=self.pad_label_id)
        crf_pad = (crf_array != self.pad_label_id)
        # 把pad_label_id全替换为0
        crf_array[~crf_pad] = 0
        return crf_array, crf_pad

    def unpad_crf(self, returned_array, returned_mask, org_array, org_mask):
        out_array = org_array.clone().detach().to(Global.device)
        out_array[org_mask] = returned_array[returned_mask]
        return out_array


class DynamicRNN(nn.Module):
    def __init__(self, config):
        super(DynamicRNN, self).__init__()
        # embedding_size = 100
        self.embedding_size = config.getint("runtime", "embedding_size")
        self.sequence_length = config.getint("runtime", "sequence_length")
        # num_layers = 1
        self.num_layers = config.getint("model", "num_layers")
        self.hidden_size = config.getint("model", "hidden_size")
        self.rnn = nn.LSTM(input_size=self.embedding_size,
                           hidden_size=self.hidden_size // 2,
                           num_layers=self.num_layers,
                           bias=True,
                           # batch * seg_len * input_size
                           batch_first=True,
                           dropout=0,
                           bidirectional=True)

    def forward(self, inputs, lengths):
        embedding_packed = nn.utils.rnn.pack_padded_sequence(input=inputs,
                                                             lengths=lengths,
                                                             batch_first=True,
                                                             enforce_sorted=False)
        # 没有提供最开始的h0和c0，故传入None
        outputs, _ = self.rnn(embedding_packed, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(sequence=outputs,
                                                      batch_first=True,
                                                      padding_value=0.0,
                                                      total_length=self.sequence_length)
        return outputs

class MYBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        outputs =self.bert(
            # 对于bert模型而言，实际上只用到了前三个参数
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        return outputs[0]
