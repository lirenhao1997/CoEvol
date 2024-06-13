from mylogging import my_log
import jsonlines
import json
import os


def json_load(path):
    with open(path, 'r') as file:
        tmp = json.load(file)
    return tmp


def jsonl_load(f):
    data = []
    with jsonlines.open(f, 'r') as reader:
        for item in reader:
            data.append(item)
    return data


class SFTDataLoader(object):
    def __init__(self, args) -> None:
        self.file_path = os.path.join(
            args.root_path, args.dataset_name)
        self.format=args.dataset_format
        if self.format=="alpaca":
            self.data = self.__load_alpaca_data()
        elif self.format=="sharegpt":  
            self.data = self.__load_sharegpt_data()
        else:
            raise NotImplementedError
        self.indx = args.start_indx
        self.end_indx = len(self.data) if args.end_indx is None \
            else args.end_indx
        if "id" not in self.data[0]:
            self.__assign_sample_id()

    def __load_alpaca_data(self) -> tuple[list[dict], set]:
        my_log.info("Loading data with alpaca format...")
        try:
            return json_load(self.file_path)
        except Exception as e:
            return jsonl_load(self.file_path)
    
    def __load_sharegpt_data(self) -> tuple[list[dict], set]:
        my_log.info("Loading data with sharegpt format...")
        try:
            return json_load(self.file_path)
        except Exception as e:
            return jsonl_load(self.file_path)

    def __assign_sample_id(self):
        for indx, sample in enumerate(self.data):
            tmp = {"id" : indx}
            tmp.update(sample)
            self.data[indx]=tmp

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.indx < self.end_indx:
            cur_sample = self.data[self.indx]
            self.indx += 1
            return self.indx, cur_sample
        else:
            raise StopIteration
