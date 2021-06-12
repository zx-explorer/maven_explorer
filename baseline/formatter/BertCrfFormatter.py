import torch
from utils.global_variables import Global

class BertCrfFormatter(object):
    def __init__(self, config):
        self.config = config
        self.pad_label_id = config.getint("data", "pad_label_id")

    def process(self, data, mode):
        """
        :param data: [{"tokens": list(int), "labels": list(int)}, ...]
        :param mode: train/valid/test
        :return: {"tokens": LongTensor,
                  "lables": LongTensor,
                  "masks": LongTensor,
                  "lengths": LongTensor}
        """
        tokens, token_type_ids, canids, labels, flags, masks, lengths, docids = [], [], [], [], [], [], [], []

        sequence_length = self.config.getint("runtime", "sequence_length")

        for item in data:
            docid = item["docids"]
            token_info = Global.tokenizer.encode_plus(
                item["tokens"], add_special_tokens=True, max_length=sequence_length, return_token_type_ids=True
            )
            token = token_info['input_ids']
            token_type = token_info["token_type_ids"]
            canid_ = item["canids"]
            length = len(token)
            if mode != "test":
                label = item["labels"]
            else:
                label = [0] * len(token)
            if "flags" in item:
                flag = item['flags']
            else:
                flag = [1] * len(token)
            if len(label) > sequence_length - 2:
                label = label[:sequence_length - 2]
                canid_ = canid_[:sequence_length - 2]
                flag = flag[:sequence_length - 2]
            label.insert(0, 0)
            label.insert(len(label), 0)
            canid_.insert(0, '')
            canid_.insert(len(canid_), '')
            flag.insert(0, 0)
            flag.insert(len(flag), 0)

            token += [0] * (sequence_length - length)
            label += [self.pad_label_id] * (sequence_length - length)
            canid = []
            for i in range(len(flag)):
                if flag[i] == 1:
                    canid.append(canid_[i])
            # 将候选词的id都提前了
            flag += [0] * (sequence_length - length)
            for i in range(sequence_length):
                if i < length and flag[i] == 1:
                    assert label[i] != self.pad_label_id
            token_type += [0] * (sequence_length - length)
            token_type_ids.append(token_type)
            docids.append(docid)
            tokens.append(token)
            canids.append(canid)
            labels.append(label)
            flags.append(flag)
            # 问题在于这个mask是什么？
            masks.append([1] * length + [0] * (sequence_length - length))
            lengths.append(length)
            for i in range(length):
                assert labels[-1][i] != self.pad_label_id

        tlt = lambda t: torch.LongTensor(t)
        tt = lambda t: torch.Tensor(t)

        tokens = tlt(tokens)
        token_type_ids = tlt(token_type_ids)
        labels = tlt(labels)
        masks = tlt(masks)
        lengths = tlt(lengths)

        return {"tokens": tokens,
                "token_type_ids": token_type_ids,
                "labels": labels,
                "flags": flags,
                "masks": masks,
                "lengths": lengths,
                "canids": canids,
                "docids": docids}