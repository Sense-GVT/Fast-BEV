import numpy as np


class KMMatcher:

    def __init__(self):
        """
        Initialization Function

        """
        self.matched = None
        self.info_matrix = None
        self.label_x = None
        self.label_y = None
        self.vis_x = None
        self.vis_y = None
        self.slack = None
        self.transposed = False
        self.num_x = 0
        self.num_y = 0

    def setInformationMatrix(self, matrix):
        """
        Interface for setting information matrix

        """
        matrix = np.array(matrix)
        self.info_matrix = matrix
        self.num_x = matrix.shape[0]
        self.num_y = matrix.shape[1]
        self.transposed = False

        if self.num_x > self.num_y:
            self.transposed = True
            self.info_matrix = self.info_matrix.transpose()
            matrix = matrix.transpose()
            tmp = self.num_y
            self.num_y = self.num_x
            self.num_x = tmp

        # print(self.transposed)
        self.label_x = np.full((matrix.shape[0]), float("-inf"))
        for i in range(self.num_x):
            for j in range(self.num_y):
                if self.label_x[i] < self.info_matrix[i][j]:
                    self.label_x[i] = self.info_matrix[i][j]
        self.label_y = np.zeros((matrix.shape[1]))

        self.vis_x = np.zeros_like(self.label_x)
        self.vis_y = np.zeros_like(self.label_y)

        self.matched = np.full((matrix.shape[1]), -1)
        self.slack = np.zeros(matrix.shape[1])

    def __perfectMatch(self, x):
        """
        Finding perfect matching in its equal sub-graph

        """
        self.vis_x[x] = True
        for y in range(self.num_y):
            if self.vis_y[y]:
                continue

            delta = self.label_x[x] + self.label_y[y] - self.info_matrix[x][y]
            if abs(delta) < 1e-3:
                self.vis_y[y] = True
                if self.matched[y] == -1 or self.__perfectMatch(self.matched[y]):
                    self.matched[y] = x
                    return True
            elif self.slack[y] > delta:
                self.slack[y] = delta

        return False

    def processKM(self):
        """
        Processing Kuhn Munkres matching algorithm

        """
        for x in range(self.num_x):
            self.slack = np.full((self.num_y), float("+inf"))
            while True:
                self.vis_x = np.zeros(self.num_x)
                self.vis_y = np.zeros(self.num_y)
                if self.__perfectMatch(x):
                    break
                else:
                    idx_array_nvisy = [i for i in range(
                        self.num_y) if not self.vis_y[i]]
                    idx_array_visy = [i for i in range(
                        self.num_y) if self.vis_y[i]]
                    idx_array_visx = [i for i in range(
                        self.num_x) if self.vis_x[i]]

                    if len(idx_array_nvisy) != 0:
                        delta = np.min(self.slack[idx_array_nvisy])
                        self.slack[idx_array_nvisy] -= delta
                    if len(idx_array_visx) != 0:
                        self.label_x[idx_array_visx] -= delta
                    if len(idx_array_visy) != 0:
                        self.label_y[idx_array_visy] += delta

        return True

    def getMatchedResult(self):
        """
        Getting KM algorithm matched results

        """
        if not self.transposed:
            return self.matched.astype(np.int16)
        else:
            tmp_matched = np.zeros((self.num_x))
            for i in range(self.matched.size):
                idx = self.matched[i]
                if idx == -1:
                    continue
                tmp_matched[idx] = i
            return tmp_matched.astype(np.int16)
