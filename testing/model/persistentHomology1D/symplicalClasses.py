import numpy as np


class Vertex:
    def __init__(self, *args):
        if len(args) == 2:
            self.xidx = args[0]
            self.yidx = args[1]
            self.mapXCoord = args[0] * 2
            self.mapYCoord = args[1] * 2
        else:
            self.xidx = -1
            self.yidx = -1
            self.mapXCoord = -1
            self.mapYCoord = -1
    def testing(self):
        print(self.xidx, self.yidx)
        print(self.mapXCoord, self.mapYCoord)

class Edge:
    def __init__(self, *args):
        if len(args) == 4:
            self.v1_order = min(args[0], args[1])
            self.v2_order = max(args[0], args[1])
            self.mapXCoord = min(args[2], args[3])
            self.mapYCoord = max(args[2], args[3])
        else:
            self.v1_order = -1
            self.v2_order = -1
            self.mapXCoord = -1
            self.mapYCoord = -1

    def testing(self):
        print(self.v1_order, self.v2_order)
        print(self.mapXCoord, self.mapYCoord )

class Triangle:
    def __init__(self, *args):
        if len(args) == 10:
            self.v1_order, self.v2_order, self.v3_order, self.v4_order = np.sort( args[0:4])
            self.e1_order, self.e2_order, self.e3_order, self.e4_order = np.sort(args[4:8])
            self.mapXCoord, self.mapYCoord = np.sort(args[8:10])
            # self.v1_order = args[0]
            # self.v2_order = args[1]
            # self.v3_order = args[2]
            # self.v4_order = args[3]
            # self.e1_order = args[4]
            # self.e2_order = args[5]
            # self.e3_order = args[6]
            # self.e4_order = args[7]
            # self.mapXCoord = args[8]
            # self.mapYCoord = args[9]

        else:
            self.v1_order = -1
            self.v2_order = -1
            self.v3_order = -1
            self.v4_order = -1
            self.e1_order = -1
            self.e2_order = -1
            self.e3_order = -1
            self.e4_order = -1
            self.mapXCoord = -1
            self.mapYCoord = -1
    def testing(self):
        print(self.v1_order, self.v2_order, self.v3_order, self.v4_order)
        print(self.e1_order, self.e2_order, self.e3_order, self.e4_order)
        print(self.mapXCoord, self.mapYCoord)


vertex = np.dtype([('xidx', int), ('yidx', int), ('mapXCoord', int), ('mapYCoord', int)])
vertex_density = np.dtype([('xidx', int), ('yidx', int), ('mapXCoord', int), ('mapYCoord', int), ('density', np.float32)])
edge = np.dtype([('v1_order', int), ('v2_order', int), ('mapXCoord', int), ('mapYCoord', int)])
triangle = np.dtype([('v1_order', int), ('v2_order', int), ('v3_order', int), ('v4_order', int),
                     ('e1_order', int), ('e2_order', int), ('e3_order', int), ('e4_order', int),
                     ('mapXCoord', int), ('mapYCoord', int)])
# EdgeTrigPair = np.dtype([('low', int), ('i', int), ('tmp_rob', np.float32),
#                          ('tmp_birth', np.float32),('tmp_death', np.float32)])

VertEdgeTrigPair = np.dtype([('low', int), ('i', int), ('tmp_rob', np.float32),
                         ('tmp_birth', np.float32),('tmp_death', np.float32)])

VertEdgeTrigPairSimple = np.dtype([('tmp_birth', np.float32),('tmp_death', np.float32)])

def trigComp(a, b):
    if a['e4_order'] != b['e4_order']:
        return a['e4_order'] - b['e4_order']
    if a['e3_order'] != b['e3_order']:
        return a['e3_order'] - b['e3_order']
    if a['e2_order'] != b['e2_order']:
        return a['e2_order'] - b['e2_order']
    else:
        return a['e1_order'] - b['e1_order']

def eComp(a, b):
    if a['v2_order'] != b['v2_order']:
        return a['v2_order'] - b['v2_order']
    else:
        return a['v1_order'] - b['v1_order']

def list_sym_diff(a, b):
    # a = [11, 9, 3, 4, 5, 6]
    # b = [4, 5, 6, 7, 8]
    a = set(a)
    b = set(b)
    # c = a.symmetric_difference(b)
    # print(type(a))
    # print(type(b))
    # print(type(c))
    # print(c)
    temp = list(a.symmetric_difference(b))
    temp.sort()
    # return temp
    return temp

def show(arg):
    n = arg.shape[0]
    for i in range(n):
        for j in range(arg[i].shape[0]):
            print(arg[i, j],end='\t')
        print()
def showboundary(arg):
    n = len(arg)
    for i in range(n):
        for j in range(len(arg[i])):
            print(arg[i][j],end='\t')
        print()

def show_List(arg):
    n = arg.shape[0]
    for i in range(n):
        for j in range(len(arg[i])):
            print(arg[i][j],end='\t')
        print()

# def show_List(arg, img):
#     n = arg.shape[0]
#     for i in range(n):
#         for j in range(len(arg[i])):
#             print(arg[i][j],end='\t')
#         print(img[arg[i][j-3],arg[i][j-2]], end='\t')
#         print()


def main():

    img = plt.imread('../13.png')*255 - 125 # 可以读取tif文件
    img = img[:,:,1].astype(np.float32)
    plt.imshow(img, cmap='gray', vmin=0, vmax=125)
    plt.show()
    print(type(img))
    print(img.shape)
    # np.set_printoptions(linewidth=np.inf)  # show each row in each line.
    print(img)
    # t = Triangle(11,2,3,4,52,6,71,8,91,10)
    # # t = Triangle()
    # t.testing()
    #
    # # e = Edge(2,1,7,3)
    # e = Edge()
    # e.testing()
    #
    # v = Vertex(2, 1)
    # v.testing()

    # a = np.ones(5, dtype=int) * -1
    # b = np.ones((5,), dtype=int) * -1
    # c = np.ones((5), dtype=int) * -1
    # d = np.ones(5, dtype=int) * -1
    # print(a)
    # print(b)
    # print(c)
    # print(d)

if __name__ == '__main__':
    main()