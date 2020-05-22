from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
import glob, pydicom

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


import tensorflow as tf

TRAIN_PATH = "/data/scratch/mselim/dcgan/dataset/bl_br_separate/npy_dataset_bl"
#TEST_PATH = "/data/scratch/mselim/labeled_image/val/combined"
log_dir="/data/scratch/mselim/dcgan/d2d512_wnidow_training_2/logs/"
checkpoint_dir = '/data/scratch/mselim/dcgan/d2d512_wnidow_training_2/training_checkpoints'

#
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512

OUTPUT_CHANNELS = 3
LAMBDA = 100   

EPOCHS = 50



#OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result



def Generator(shapes=[IMG_WIDTH,IMG_WIDTH,3]):
  inputs = tf.keras.layers.Input(shape=shapes)
  base_pretrained_model = tf.keras.applications.VGG19(input_shape =shapes, include_top = False, weights = 'imagenet')
  base_pretrained_model.trainable = False
  #base_pretrained_model.summary()


  from collections import defaultdict, OrderedDict
  #from keras.models import Model
  layer_size_dict = defaultdict(list)
  inputs = []
  for lay_idx, c_layer in enumerate(base_pretrained_model.layers):
      if not c_layer.__class__.__name__ == 'InputLayer':
          layer_size_dict[c_layer.get_output_shape_at(0)[1:3]] += [c_layer]
      else:
          inputs += [c_layer]
  # freeze dict
  layer_size_dict = OrderedDict(layer_size_dict.items())
  #for k,v in layer_size_dict.items():
  #    print(k, [w.__class__.__name__ for w in v])

  pretrained_encoder = tf.keras.Model(inputs = base_pretrained_model.get_input_at(0), 
                            outputs = [v[-1].get_output_at(0) for k, v in layer_size_dict.items()])
  pretrained_encoder.trainable = True

  in_t0 = tf.keras.layers.Input(shape=shapes)
  wrap_encoder = lambda i_layer: {k: v for k, v in zip(layer_size_dict.keys(), pretrained_encoder(i_layer))}

  t0_outputs = wrap_encoder(in_t0)

  #print(t0_outputs)


  lay_dims = sorted(t0_outputs.keys(), key = lambda x: x[0])
  skip_layers = 2
  last_layer = None
  for k in lay_dims[skip_layers:]:
      cur_layer = t0_outputs[k]
      channel_count = cur_layer.shape[-1]
      cur_layer = tf.keras.layers.Conv2D(channel_count//2, kernel_size=(3,3), padding = 'same', activation = 'linear')(cur_layer)
      cur_layer = tf.keras.layers.BatchNormalization()(cur_layer) # gotta keep an eye on that internal covariant shift
      cur_layer = tf.keras.layers.Activation('relu')(cur_layer)
      
      if last_layer is None:
          x = cur_layer
      else:
          last_channel_count = last_layer.shape[-1]
          x =  tf.keras.layers.Conv2D(last_channel_count//2, kernel_size=(3,3), padding = 'same')(last_layer)
          x = tf.keras.layers.UpSampling2D((2, 2))(x)
          x = tf.keras.layers.concatenate([cur_layer, x])
      last_layer = x

  #initializer = tf.random_normal_initializer(0., 0.02)
  #last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
  #                                       strides=2,
  #                                       padding='same',
  #                                       kernel_initializer=initializer,
  #                                       activation='tanh') # (bs, 256, 256, 3)

  final_output =  tf.keras.layers.Conv2D(3, kernel_size=(1,1), padding = 'same', activation = 'tanh')(last_layer)
  #crop_size = 20
  #final_output = tf.keras.layers.Cropping2D((crop_size, crop_size))(final_output)
  #final_output = tf.keras.layers.ZeroPadding2D((crop_size, crop_size))(final_output)
  unet_model = tf.keras.Model(inputs = [in_t0],
                    outputs = [final_output])
  return unet_model
  
  
  
generator = Generator()
#tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)




#gen_output = generator(inp[tf.newaxis,...], training=False)
#plt.imshow(gen_output[0,...])


#LAMBDA = 100


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) #tf.reduce_mean(tf.keras.losses.MSE(target, gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss
  


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[IMG_WIDTH,IMG_WIDTH,3], name='input_image')
  tar = tf.keras.layers.Input(shape=[IMG_WIDTH,IMG_WIDTH,3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down0 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down1 = downsample(64, 4, False)(down0) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)
  
  
discriminator = Discriminator()
#tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)


#disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
#plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
#plt.colorbar()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss
  
  
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
								 

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  
  
@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)
    #print(gen_total_loss)
	
  
def fit(train_ds, epochs):
  for epoch in range(epochs):
    start = time.time()

    #display.clear_output(wait=True)

    #for example_input, example_target in test_ds.take(1):
    #  generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      #print('.', end='')
      #if (n+1) % 100 == 0:
      #  print(".")
      train_step(input_image, target, epoch)
    print(".", end='')

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 5 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  #checkpoint.save(file_prefix = checkpoint_prefix)




############################# Get Training dataset ##############################
############### get data files list ################
x_train = []
y_train = []
list = ["1", "3", "15", "5"]
root = "/data/scratch/mselim/data/old_data/DICOM_5_patients/"

for thickness in list :
	source  = root+"PATIENT1/BI57_"+thickness
	target  = root+"PATIENT1/BI64_"+thickness
	s_dicom_files = sorted(glob.glob(os.path.join(source, "*.IMA")))#sorted(os.listdir(source))
	t_dicom_files = sorted(glob.glob(os.path.join(target, "*.IMA")))#sorted(os.listdir(target))
	#print(type(s_dicom_files))
	x_train.extend(s_dicom_files)
	y_train.extend(t_dicom_files)
    
	source  = root+"PATIENT1/BR40_"+thickness
	target  = root+"PATIENT1/BI64_"+thickness
	s_dicom_files = sorted(glob.glob(os.path.join(source, "*.IMA")))#sorted(os.listdir(source))
	t_dicom_files = sorted(glob.glob(os.path.join(target, "*.IMA")))#sorted(os.listdir(target))
	x_train.extend(s_dicom_files)
	y_train.extend(t_dicom_files)


  
source  = root+"PATIENT3/bl57"
target  = root+"PATIENT3/bl64"
s_dicom_files = sorted(glob.glob(os.path.join(source, "*.IMA")))#sorted(os.listdir(source))
t_dicom_files = sorted(glob.glob(os.path.join(target, "*.IMA")))#sorted(os.listdir(target))
x_train.extend(s_dicom_files)
y_train.extend(t_dicom_files)

source  = root+"PATIENT3/br40"
target  = root+"PATIENT3/bl64"
s_dicom_files = sorted(glob.glob(os.path.join(source, "*.IMA")))#sorted(os.listdir(source))
t_dicom_files = sorted(glob.glob(os.path.join(target, "*.IMA")))#sorted(os.listdir(target))
x_train.extend(s_dicom_files)
y_train.extend(t_dicom_files)

source  = root+"PATIENT4/bl57"
target  = root+"PATIENT4/bl64"
s_dicom_files = sorted(glob.glob(os.path.join(source, "*.IMA")))#sorted(os.listdir(source))
t_dicom_files = sorted(glob.glob(os.path.join(target, "*.IMA")))#sorted(os.listdir(target))
x_train.extend(s_dicom_files)
y_train.extend(t_dicom_files)

source  = root+"PATIENT4/br40"
target  = root+"PATIENT4/bl64"
s_dicom_files = sorted(glob.glob(os.path.join(source, "*.IMA")))#sorted(os.listdir(source))
t_dicom_files = sorted(glob.glob(os.path.join(target, "*.IMA")))#sorted(os.listdir(target))
x_train.extend(s_dicom_files)
y_train.extend(t_dicom_files)

source  = root+"PATIENT5/bl57"
target  = root+"PATIENT5/bl64"
s_dicom_files = sorted(glob.glob(os.path.join(source, "*.IMA")))#sorted(os.listdir(source))
t_dicom_files = sorted(glob.glob(os.path.join(target, "*.IMA")))#sorted(os.listdir(target))
x_train.extend(s_dicom_files)
y_train.extend(t_dicom_files)

source  = root+"PATIENT5/br40"
target  = root+"PATIENT5/bl64"
s_dicom_files = sorted(glob.glob(os.path.join(source, "*.IMA")))#sorted(os.listdir(source))
t_dicom_files = sorted(glob.glob(os.path.join(target, "*.IMA")))#sorted(os.listdir(target))
x_train.extend(s_dicom_files)
y_train.extend(t_dicom_files)


root = "/data/scratch/mselim/data/CHESTPHANOM/"
bl57_list = ["0009", "0012", "0015", "0018", "0021", "0024", "0027"]
bl64_list = ["0010", "0013", "0016", "0019", "0022", "0025", "0028"]
br40_list = ["0011", "0014", "0017", "0020", "0023", "0026", "0029"]
list = zip(bl57_list , bl64_list, br40_list)
bl57 = root+"CHEST_WITHOUT_1_0_BL57_3_"
bl64 = root+"CHEST_WITHOUT_1_0_BL64_3_"
br40 = root+"CHEST_WITHOUT_1_0_BR40_3_"

for (f57, f64, f40) in list :
	source  = bl57+f57
	target  = bl64+f64
	s_dicom_files = sorted(glob.glob(os.path.join(source, "*.IMA")))#sorted(os.listdir(source))
	t_dicom_files = sorted(glob.glob(os.path.join(target, "*.IMA")))#sorted(os.listdir(target))
	#print(type(s_dicom_files))
	x_train.extend(s_dicom_files)
	y_train.extend(t_dicom_files)
    
	source  = br40+f40
	target  = bl64+f64
	s_dicom_files = sorted(glob.glob(os.path.join(source, "*.IMA")))#sorted(os.listdir(source))
	t_dicom_files = sorted(glob.glob(os.path.join(target, "*.IMA")))#sorted(os.listdir(target))
	x_train.extend(s_dicom_files)
	y_train.extend(t_dicom_files)

train_set = zip(x_train, y_train)
marker = 0
def read_npy_file(item, marker):
	filename = item.numpy()
	#data = np.load(item.numpy())
	dataset = pydicom.dcmread(filename)
	file = dataset.pixel_array.copy()
	if marker == 0:
		#data[(data<=500) & (data>1500)] = 0
		file = np.clip((file)/255, 0, 1).astype(np.float32)
	elif marker == 1:
		file = np.clip((file)/511, 0, 1).astype(np.float32)
	elif marker == 2:
		file = np.clip((file)/1023, 0, 1).astype(np.float32)
	else:
		file = np.clip((file)/1500, 0, 1).astype(np.float32)
        
	data = file * 2.0 - 1.0 # [0, 4000] ==> [0, 1] ==>> [-1, 1]
    
	hight, width = data.shape
	image= np.zeros([hight, width, 3])
	image[:,:,0] = data
	image[:,:,1] = data
	image[:,:,2] = data
	#print(filename)
	return image


def load_train_data(item):
    image  = tf.py_function(read_npy_file, [item[0], marker], [tf.float32,])
    target = tf.py_function(read_npy_file, [item[1], marker], [tf.float32,])
    image = tf.squeeze(image)
    target = tf.squeeze(target)
    return image, target

marker = 0
train_dataset = tf.data.Dataset.from_tensor_slices(train_set)
train_dataset = train_dataset.map(load_train_data)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

fit(train_dataset, 10)

marker = 1
train_dataset = tf.data.Dataset.from_tensor_slices(train_set)
train_dataset = train_dataset.map(load_train_data)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

fit(train_dataset, 10)


marker = 2
train_dataset = tf.data.Dataset.from_tensor_slices(train_set)
train_dataset = train_dataset.map(load_train_data)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

fit(train_dataset, 10)


marker = 3
train_dataset = tf.data.Dataset.from_tensor_slices(train_set)
train_dataset = train_dataset.map(load_train_data)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

fit(train_dataset, 10)
