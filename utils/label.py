import torch


class DoNothingLabelTransformer(object):
    def __init__(self):
        pass

    def parse_target(self, targets):
        return targets

    def parse_prediction(self, pred, to_string=True, top_k=1):
        return pred


class MultiLabelTransformer(object):
    def __init__(self, label_file='label.txt', encoding='UTF-8'):
        """
        :param label_file: file that include all label characters
        """
        self.alphabet = ''
        with open(label_file, 'r', encoding=encoding) as labeltxt:
            for line in labeltxt.readlines():
                self.alphabet += line.strip()
        # see method parse_target, to parse blank
        self.alphabet += '-'
        self.dict = {}
        for i, char in enumerate(self.alphabet):
            self.dict[char] = i + 1

    def parse_target(self, targets):
        retval = []
        for batch in targets:
            row = []
            for pos in batch:
                row.append(self.alphabet[pos - 1])
            retval.append(row)
        return retval

    def parse_prediction(self, pred, to_string=True, top_k=1):
        """
        :param pred: model returned shape like (W, N, C*H)
        :param to_string: determine if need to convert to string
        :param top_k: top_k
        :return:
        """
        val, pred = pred.topk(top_k, 2, True, True)
        pred = pred.permute(1, 2, 0)
        col_length = pred.size(2)
        convert_pred = []
        for row in pred:
            row_no_blank = []
            for col in row:
                col_no_blank = []
                for idx, item in enumerate(col):
                    if item != 0 and (idx == 0 or (idx > 0 and item != col[idx - 1])):
                        if to_string:
                            col_no_blank.append(self.alphabet[item.item() - 1])
                        else:
                            col_no_blank.append(item.item())
                for _ in range(0, col_length - len(col_no_blank)):
                    col_no_blank.append(0)

                row_no_blank.append(col_no_blank)
            convert_pred.append(row_no_blank)
        return convert_pred


if __name__ == '__main__':
    transformer = MultiLabelTransformer(label_file='../label.txt', encoding='GB18030')
    # last line of the test.txt
    line1 = '279 89 54 1591 517 14 23 24 98 96'.split(' ')
    line1 = [int(i) for i in line1]
    print(transformer.parse_target([line1]))
    preds = torch.randn(3, 4, 5)
    print(transformer.parse_prediction(preds, top_k=3))
