import numpy as np


class LabelParser:
    def __init__(self, label_format):
        self.format_dict = self.get_attribute_idx(label_format)

    def parse_label(self, label_path, idx_key=None, prediction=False, csv=False):
        """
        :param prediction: if prediction also fetch score (required)
        :return:
        """
        if idx_key is None:
            idx_key = self.format_dict
        return self.new_label_from_txt(label_path, idx_key, prediction, csv)

    @staticmethod
    def new_label_from_txt(label_path, idx_key, pred, csv):
        classes = []
        score = []
        x, y, z, r = [], [], [], []
        l, w, h = [], [], []
        with open(label_path, "r") as f:
            labels = f.read().split("\n")
            for label in labels:
                if not label:
                    continue
                if csv:
                    label = label.replace(" ", "")
                    label = label.split(",")
                else:
                    label = label.replace(",", "")
                    label = label.split(" ")
                if 'class' in idx_key:
                    classes.append(label[idx_key['class']])
                else:
                    classes.append(['Car']) # assume if no class is specified, its a car
                if 'x' in idx_key:
                    x.append(label[idx_key['x']])
                else:
                    x.append(0)
                if 'y' in idx_key:
                    y.append(label[idx_key['y']])
                else:
                    y.append(0)
                if 'z' in idx_key:
                    z.append(label[idx_key['z']])
                else:
                    z.append(0)
                if 'r' in idx_key:
                    r.append(label[idx_key['r']])
                else:
                    r.append(0)
                if 'l' in idx_key:
                    l.append(label[idx_key['l']])
                else:
                    l.append(0)
                if 'w' in idx_key:
                    w.append(label[idx_key['w']])
                else:
                    w.append(0)
                if 'h' in idx_key:
                    h.append(label[idx_key['h']])
                else:
                    h.append(0)
                if pred:
                    if 'score' in idx_key:
                        score.append(label[idx_key['score']])

        final_array = np.hstack((
            np.array(classes).reshape(-1, 1),
            np.array(x).reshape(-1, 1),
            np.array(y).reshape(-1, 1),
            np.array(z).reshape(-1, 1),
            np.array(l).reshape(-1, 1),
            np.array(w).reshape(-1, 1),
            np.array(h).reshape(-1, 1),
            np.array(r).reshape(-1, 1)
        ))
        if pred:
            final_array = np.hstack((final_array, np.array(score).reshape(-1, 1)))
        return final_array

    def get_attribute_idx(self, conversion, verbose=False):
        idx_dict = {}
        conversion = conversion.split(" ")
        self.check_conversion(conversion)
        if verbose:
            print("-- Your conversion key --")
        for i, attribute in enumerate(conversion):
            if verbose:
                print(i, attribute)
            if attribute == 'class':
                idx_dict['class'] = i
            elif attribute == 'x':
                idx_dict['x'] = i
            elif attribute == 'y':
                idx_dict['y'] = i
            elif attribute == 'z':
                idx_dict['z'] = i
            elif attribute == 'l':
                idx_dict['l'] = i
            elif attribute == 'w':
                idx_dict['w'] = i
            elif attribute == 'h':
                idx_dict['h'] = i
            elif attribute == 'r':
                idx_dict['r'] = i
            elif attribute == 'score':
                idx_dict['score'] = i
        return idx_dict

    @staticmethod
    def check_conversion(conversion_array):
        assert 'x' in conversion_array
        assert 'y' in conversion_array
        assert 'l' in conversion_array
        assert 'w' in conversion_array
        assert 'score' in conversion_array
