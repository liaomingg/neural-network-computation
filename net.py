# python2.7
#@\author: liaoming

from computation import *

img = blob(65, 33, 3)
base = 4

fb = net(name="foregrand-background-person")
fb.add(data(img, name="data"))

fb.add(conv(fb.layer("data"), k3x3, p1x1, s1x1, base, name="conv1_1"))
fb.add(batch_norm(fb.layer("conv1_1"), name="conv1_1/bn"))
fb.add(scale(fb.layers[-1], name="conv1_1/scale"))
fb.add(relu(fb.layers[-1], name="conv1_1/relu"))

fb.add(max_pool(fb.layer("conv1_1/relu"), k2x2, p0x0, s2x2, name="max_pool1"))

fb.add(conv(fb.layer("max_pool1"), k3x3, p1x1, s1x1, base*2, name="conv2_1"))
fb.add(batch_norm(fb.layers[-1], name="conv2_1/bn"))
fb.add(scale(fb.layers[-1], name="conv2_1/scale"))
fb.add(relu(fb.layers[-1], name="conv2_1/relu"))

fb.add(max_pool(fb.layer("conv2_1"), k2x2, p0x0, s2x2, name= "max_pool2"))

fb.add(conv(fb.layer("max_pool2"), k3x3, p1x1, s1x1, base*4, name="conv3_1"))
fb.add(batch_norm(fb.layers[-1], name="conv3_1/bn"))
fb.add(scale(fb.layers[-1], name="conv3_1/scale"))
fb.add(relu(fb.layers[-1], name="conv3_1/relu"))

fb.add(conv(fb.layer("conv3_1"), k3x3, p1x1, s1x1, base*4, name="conv3_2"))
fb.add(batch_norm(fb.layers[-1], name="conv3_2/bn"))
fb.add(scale(fb.layers[-1], name="conv3_2/scale"))
fb.add(relu(fb.layers[-1], name="conv3_2/relu"))

fb.add(max_pool(fb.layer("conv3_2"), k2x2, p0x0, s2x2, name="max_pool3"))

fb.add(conv(fb.layer("max_pool3"), k3x3, p1x1, s1x1, base*8, name="conv4_1"))
fb.add(batch_norm(fb.layers[-1], name="conv4_1/bn"))
fb.add(scale(fb.layers[-1], name="conv4_1/scale"))
fb.add(relu(fb.layers[-1], name="conv4_1/relu"))

fb.add(conv(fb.layer("conv4_1"), k3x3, p1x1, s1x1, base*8, name="conv4_2"))
fb.add(batch_norm(fb.layers[-1], name="conv4_2/bn"))
fb.add(scale(fb.layers[-1], name="conv4_2/scale"))
fb.add(relu(fb.layers[-1], name="conv4_2/relu"))

fb.add(max_pool(fb.layer("conv4_2"), k2x2, p0x0, s2x2, name="max_pool4"))

fb.add(conv(fb.layer("max_pool4"), k3x3, p1x1, s1x1, base*8, name="conv5_1"))
fb.add(batch_norm(fb.layers[-1], name="conv5_1/bn"))
fb.add(scale(fb.layers[-1], name="conv5_1/scale"))
fb.add(relu(fb.layers[-1], name="conv5_1/relu"))

fb.add(conv(fb.layer("conv5_1"), k3x3, p1x1, s1x1, base*8, name="conv5_2"))
fb.add(batch_norm(fb.layers[-1], name="conv5_2/bn"))
fb.add(scale(fb.layers[-1], name="conv5_2/scale"))
fb.add(relu(fb.layers[-1], name="conv5_2/relu"))

fb.add(max_pool(fb.layer("conv5_2"), k2x2, p0x0, s2x2, name="max_pool5"))

fb.add(conv(fb.layer("max_pool5"), k3x3, p1x1, s1x1, base*8, name="conv6_1"))
fb.add(batch_norm(fb.layers[-1], name="conv6_1/bn"))
fb.add(scale(fb.layers[-1], name="conv6_1/scale"))
fb.add(relu(fb.layers[-1], name="conv6_1/relu"))

fb.add(deconv(fb.layer("conv6_1"), k3x3, p1x1, s2x2, base*4, name = "deconv7_1"))
fb.add(batch_norm(fb.layers[-1], name="deconv7_1/bn"))
fb.add(scale(fb.layers[-1], name="deconv7_1/scale"))
fb.add(relu(fb.layers[-1], name="deconv7_1/relu"))

fb.add(concat("concat5", fb.layer("deconv7_1"), fb.layer("conv5_2")))

fb.add(conv(fb.layer("concat5"), k3x3, p1x1, s1x1, base*8, name="conv7_1"))
fb.add(batch_norm(fb.layers[-1], name="conv7_1/bn"))
fb.add(scale(fb.layers[-1], name="conv7_1/scale"))
fb.add(relu(fb.layers[-1], name="conv7_1/relu"))

fb.add(deconv(fb.layer("conv7_1"), k3x3, p1x1, s2x2, base*4, name="deconv8_1"))
fb.add(batch_norm(fb.layers[-1], name="deconv8_1/bn"))
fb.add(scale(fb.layers[-1], name="deconv8_1/scale"))
fb.add(relu(fb.layers[-1], name="deconv8_1/relu"))

fb.add(concat("concat4", fb.layer("deconv8_1"), fb.layer("conv4_2")))

fb.add(conv(fb.layer("concat4"), k3x3, p1x1, s1x1, base*8, name="conv8_1"))
fb.add(batch_norm(fb.layers[-1], name="conv8_1/bn"))
fb.add(scale(fb.layers[-1], name="conv8_1/scale"))
fb.add(relu(fb.layers[-1], name="conv8_1/relu"))

fb.add(deconv(fb.layer("conv8_1"), k3x3, p1x1, s2x2, base*2, name="deconv9_1"))
fb.add(batch_norm(fb.layers[-1], name="deconv9_1/bn"))
fb.add(scale(fb.layers[-1], name="deconv9_1/scale"))
fb.add(relu(fb.layers[-1], name="deconv9_1/relu"))

fb.add(concat("concat3", fb.layer("deconv9_1"), fb.layer("conv3_2")))

fb.add(conv(fb.layer("concat3"), k3x3, p1x1, s1x1, base*4, name="conv9_1"))
fb.add(batch_norm(fb.layers[-1], name="conv9_1/bn"))
fb.add(scale(fb.layers[-1], name="conv9_1/scale"))
fb.add(relu(fb.layers[-1], name="conv9_1/relu"))

fb.add(deconv(fb.layer("conv9_1"), k3x3, p1x1, s2x2, base, name="deconv10_1"))
fb.add(batch_norm(fb.layers[-1], name="deconv10_1/bn"))
fb.add(scale(fb.layers[-1], name="deconv10_1/scale"))
fb.add(relu(fb.layers[-1], name="deconv10_1/relu"))

fb.add(concat("concat2", fb.layer("deconv10_1"), fb.layer("conv2_1")))

fb.add(conv(fb.layer("concat2"), k3x3, p1x1, s1x1, base*2, name="conv11_1"))
fb.add(batch_norm(fb.layers[-1], name="conv11_1/bn"))
fb.add(scale(fb.layers[-1], name="conv11_1/scale"))
fb.add(relu(fb.layers[-1], name="conv11_1/relu"))

fb.add(deconv(fb.layer("conv11_1"), k3x3, p1x1, s2x2, base, name="deconv11_1"))
fb.add(batch_norm(fb.layers[-1], name="deconv11_1/bn"))
fb.add(scale(fb.layers[-1], name="deconv11_1/scale"))
fb.add(relu(fb.layers[-1], name="deconv11_1/relu"))

fb.add(concat("concat1", fb.layer("deconv11_1"), fb.layer("conv1_1")))

fb.add(conv(fb.layer("concat1"), k3x3, p1x1, s1x1, 2, name="conv_cls"))
print "fb_base{}:".format(base)
fb.info()
fb.info_write("fb_base{}_computation.txt".format(base))


fb2 = net()
fb2.add(conv(fb.layer("data"), k3x3, p1x1, s1x1, 32, name = "conv1"))
fb2.add(conv(fb2.layer("conv1"), k3x3, p1x1, s1x1, 32, name="conv2"))
fb2.add(conv(fb2.layer("conv2"), k3x3, p1x1, s1x1, 32, name = "conv3"))
fb2.add(conv(fb2.layer("conv3"), k3x3, p1x1, s1x1, 32, name = "conv4"))
fb2.add(conv(fb2.layer("conv4"), k3x3, p1x1, s1x1, 2, name="conv_cls"))
print "\nfb_ref:"
fb2.info()

