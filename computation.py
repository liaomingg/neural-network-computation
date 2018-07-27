# python2.7 
# @\author: liaoming

def unpack(item):
    '''
    @function: unpack parameter
    '''
    if list == type(item):
        if len(item) == 2:
            return item
        return item*2
    return [item, item]

def shape2str(shape):
    if len(shape) > 0:
        shape_str = "{}".format(shape[0])
        for each in shape[1:]:
            shape_str += "x{}".format(each)
        return shape_str
    else:
        return "missing shape!"

class blob:
    def __init__(self, height = 512, width = 512, channel = 3, name = "blob"):
        self.height = height
        self.width = width
        self.channel = channel
        self.type = "blob"
        self.name = name
    @property
    def size(self):
        return self.height * self.width * self.channel
    @property
    def shape_str(self):
        return "{}x{}x{}".format(self.height, self.width, self.channel)
    @property
    def shape(self):
        return [self.height, self.width, self.channel]

class kernel:
    def __init__(self, kernel_h = 3, kernel_w = 3, name = "kernel"):
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.type = "kernel"
        self.name = "kernel"
    @property
    def shape_str(self):
        return "{}x{}".format(self.kernel_h, self.kernel_w)
    @property
    def shape(self):
        return [self.kernel_h, self.kernel_w]
    @property
    def size(self):
        return self.kernel_h * self.kernel_w

class pad:
    def __init__(self, pad_h = 0, pad_w = 0, name = "pad"):
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.type = "pad"
        self.name = "pad"
    @property
    def shape_str(self):
        return "{}x{}".format(self.pad_h, self.pad_w)
    @property
    def shape(self):
        return [self.pad_h, self.pad_w]
    @property
    def size(self):
        return self.pad_h * self.pad_w

class stride:
    def __init__(self, stride_h = 0, stride_w = 0, name = "stride"):
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.type = "stride"
        self.name = stride
    @property
    def shape_str(self):
        return "{}x{}".format(self.stride_h, self.stride_w)
    @property
    def shape(self):
        return [self.stride_h, self.stride_w]

class data:
    def __init__(self, img, name = "data"):
        self.top = img 
        self.name = name 
        self.type = "img data"
    @property
    def shape_str(self):
        return self.top.shape_str
    @property
    def shape(self):
        return self.shape 
    @property
    def output(self):
        return self.top
    @property
    def computation(self):
        return 0

class conv:
    def __init__(self, bottom, kernel, pad, stride, output, name = "convolution"):
        self.bottom_shape = bottom.top.shape
        self.bottom_name = bottom.name 
        self.kernel = kernel 
        self.pad = pad
        self.stride = stride
        self.output_c = output
        self.output_h = (self.bottom_shape[0] + 2 * self.pad.pad_h - self.kernel.kernel_h) // self.stride.stride_h + 1
        self.output_w = (self.bottom_shape[1] + 2 * self.pad.pad_w - self.kernel.kernel_w) // self.stride.stride_w + 1
        self.top = blob(self.output_h, self.output_w, self.output_c)
        self.type = "convolution"
        self.name = name
    @property
    def output(self):
        return self.top
    @property
    def computation(self):
        return self.output_h * self.output_w * self.bottom_shape[2] * self.kernel.size * self.output_c

class deconv:
    def __init__(self, bottom, kernel, pad, stride, output, name = "deconvolution"):
        self.bottom_shape = bottom.top.shape
        self.bottom_name = bottom.name 
        self.kernel = kernel 
        self.pad = pad
        self.stride = stride
        self.output_c = output
        self.output_h = self.stride.stride_h * (self.bottom_shape[0] - 1) + self.kernel.kernel_h - 2 * self.pad.pad_h
        self.output_w = self.stride.stride_w * (self.bottom_shape[1] - 1) + self.kernel.kernel_w - 2 * self.pad.pad_w 
        self.top = blob(self.output_h, self.output_w, self.output_c)
        self.type = "deconvolution"
        self.name = name 
    @property
    def output(self):
        return self.top
    @property
    def computation(self):
		return self.bottom_shape[0] * self.bottom_shape[1] * self.bottom_shape[2] * self.kernel.size * self.output_c


class batch_norm:
    def __init__(self,bottom, name = "batch norm"):
        self.bottom_shape = bottom.top.shape  
        self.bottom_name = bottom.name 
        self.type = "batch norm"
        self.name = name 
        self.top = blob(self.bottom_shape[0], self.bottom_shape[1], self.bottom_shape[2])
    @property
    def output(self):
        return self.top
    @property
    def computation(self):
        return self.top.size

class scale:
    def __init__(self, bottom, name = "scale"):
        self.bottom_shape = bottom.top.shape 
        self.bottom_name = bottom.name 
        self.type = "scale"
        self.name = name 
        self.top = blob(self.bottom_shape[0], self.bottom_shape[1], self.bottom_shape[2])
    @property
    def output(self):
        return self.top 
    @property
    def computation(self):
        return self.top.size

class max_pool:
    def __init__(self, bottom, kernel, pad, stride, name = "max pooling"):
        self.bottom_shape = bottom.top.shape
        self.bottom_name = bottom.name 
        self.kernel = kernel
        self.pad = pad
        self.stride = stride
        self.output_h = (self.bottom_shape[0] + 2 * self.pad.pad_h + 1) // (self.stride.stride_h)
        self.output_w = (self.bottom_shape[1] + 2 * self.pad.pad_w + 1) // (self.stride.stride_w)
        self.top = blob(self.output_h, self.output_w, self.bottom_shape[2])
        self.type = "max pooling"
        self.name = name 
    @property
    def output(self):
    	return self.top
    @property
    def computation(self):
		return self.output_h * self.output_w * (self.kernel.size - 1)

class relu:
    def __init__(self, bottom, name = "relu"):
        self.bottom_shape = bottom.top.shape
        self.type = "relu"
        self.name = name
        self.top = bottom.top 
    @property
    def output(self):
        return self.top 
    @property
    def computation(self):
        return self.top.size
	
class concat:
    def __init__(self, name = "concat", *bottoms):
        self.top_shape = bottoms[0].top.shape
        self.bottom_shapes = [bottoms[0].top.shape]
        for each in bottoms[1:]:
            self.top_shape[2]  = self.top_shape[2] + each.top.channel
            if (self.top_shape[0] != each.top.height or self.top_shape[1] != each.top.width):
                print "bottom blob's shape mismach"
            self.bottom_shapes.append(each.top.shape)
        self.top = blob(self.top_shape[0], self.top_shape[1], self.top_shape[2])
        self.type = "concat"
        self.name = name
    @property
    def output(self):
		return self.top
    @property
    def computation(self):
		return 0

def info_str(layer):
    if layer.type in ["convolution", "deconvolution", "max pooling"]:
        print "|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|" \
            .format(layer.name, shape2str(layer.bottom_shape), layer.kernel.shape_str,
            layer.pad.shape_str, layer.stride.shape_str, layer.output.shape_str,
            layer.computation)
    elif layer.type in ["concat"]: # multi bottom
        for i, each in enumerate(layer.bottom_shapes):
            if (i == len(layer.bottom_shapes) // 2):
                print "|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|" \
                    .format(layer.name, shape2str(each), "",
                    "", "", layer.output.shape_str, layer.computation)
            else:
                 print "|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|" \
                    .format("", shape2str(each), "", "", "", "", "")
    elif layer.type in ["img data"]:
        print "|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|" \
            .format(layer.name, "", "", "", "", layer.output.shape_str, layer.computation)
    elif layer.type in ["scale", "batch norm", "relu"]:
        print "|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|" \
            .format(layer.name, shape2str(layer.bottom_shape), "", "", "", layer.output.shape_str, layer.computation)

class net:
    '''
    @\ define network
    '''
    def __init__(self, name = "neural network"):
        self.layers = []
        self.name = name
    def add(self, layer):
        self.layers.append(layer)
    def layer(self, name):
        for lay in self.layers:
            if name == lay.name:
                return lay
        return None
    @property
    def layers(self):
        return self.layers 
    def remove(self, name):
        r = self.layer(name) # return
        if None == r:
            print "there are no layer named {} in this net.".format(name)
        else:
            self.layers.remove(r)
    def info(self):
        print "net name: {}".format(self.name)
        if len(self.layers) < 1:
            print "net is empty.\n"
        else:
            total_computation = 0
            print "{:-^94}".format("")
            print "|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|" \
                .format("name", "bottom", "kernel", "pad", "stride", "top", "computation")
            print "{:-^94}".format("")
            for each in self.layers:
                total_computation = total_computation + each.computation
                info_str(each)
            print "{:-^94}".format("")
            print "total computation:", total_computation
    def info_write(self, filename):
        with open(filename, 'w') as f:
            f.writelines("net name: {}\n".format(self.name))
            if len(self.layers) < 1:
                f.writelines("net is empty.\n")
            else:
                total_computation = 0
                f.writelines("{:-^94}\n".format(""))
                f.writelines("|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|\n" \
                    .format("name", "bottom", "kernel", "pad", "stride", "top", "computation"))
                f.writelines("{:-^94}\n".format(""))
                for layer in self.layers:
                    total_computation = total_computation + layer.computation
                    if layer.type in ["convolution", "deconvolution", "max pooling"]:
                        f.write("|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|\n" \
                            .format(layer.name, shape2str(layer.bottom_shape), layer.kernel.shape_str,
                            layer.pad.shape_str, layer.stride.shape_str, layer.output.shape_str,
                            layer.computation))
                    elif layer.type in ["concat"]: # multi bottom
                        for i, each in enumerate(layer.bottom_shapes):
                            if (i == len(layer.bottom_shapes) // 2):
                                f.write("|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|\n" \
                                    .format(layer.name, shape2str(each), "",
                                    "", "", layer.output.shape_str, layer.computation))
                            else:
                                f.write("|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|\n" \
                                    .format("", shape2str(each), "", "", "", "", ""))
                    elif layer.type in ["img data"]:
                        f.writelines("|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|\n" \
                            .format(layer.name, "", "", "", "", layer.output.shape_str, layer.computation))
                    elif layer.type in ["scale", "batch norm", "relu"]:
                        f.writelines("|{:^16}|{:^14}|{:^8}|{:^7}|{:^8}|{:^14}|{:>20}|\n" \
                            .format(layer.name, shape2str(layer.bottom_shape), "", "", "", layer.output.shape_str, layer.computation))
                f.writelines("{:-^94}\n".format(""))
                f.writelines("total computation: {}\n".format(total_computation))


# constants
k3x3 = kernel(3, 3)
k2x2 = kernel(2, 2)
k1x1 = kernel(1, 1)
s1x1 = stride(1, 1)
s2x2 = stride(2, 2)
s2x2 = stride(2, 2)
p1x1 = pad(1, 1)
p0x0 = pad(0 ,0)
