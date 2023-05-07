import os
import zipfile
from os.path import basename

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from math import floor
from io import BytesIO
import base64
import tempfile

from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, AveragePooling2D, GaussianNoise
from keras.layers import Reshape, UpSampling2D, Activation, Dropout, Flatten, Conv2DTranspose
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam

import cv2

import math
from skimage import transform

import tensorflow as tf
import tensorflow_hub as hub
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

import settings


def zero():
    return np.random.uniform(0.0, 0.01, size=[1])


def one():
    return np.random.uniform(0.99, 1.0, size=[1])


def noise(n):
    return np.random.uniform(-1.0, 1.0, size=[n, 4096])


class GAN(object):

    def __init__(self):

        # Models
        self.D = None
        self.G = None

        self.OD = None

        self.DM = None
        self.AM = None

        # Config
        self.LR = 0.0001
        self.steps = 1

    def discriminator(self):

        if self.D:
            return self.D

        self.D = Sequential()

        # add Gaussian noise to prevent Discriminator overfitting
        self.D.add(GaussianNoise(0.2, input_shape=[256, 256, 3]))

        # 256x256x3 Image
        self.D.add(Conv2D(filters=8, kernel_size=3, padding='same'))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 128x128x8
        self.D.add(Conv2D(filters=16, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 64x64x16
        self.D.add(Conv2D(filters=32, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 32x32x32
        self.D.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 16x16x64
        self.D.add(Conv2D(filters=128, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 8x8x128
        self.D.add(Conv2D(filters=256, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        # 4x4x256
        self.D.add(Flatten())

        # 256
        self.D.add(Dense(128))
        self.D.add(LeakyReLU(0.2))

        self.D.add(Dense(1, activation='sigmoid'))

        return self.D

    def generator(self):

        if self.G:
            return self.G

        self.G = Sequential()

        self.G.add(Reshape(target_shape=[1, 1, 4096], input_shape=[4096]))

        # 1x1x4096
        self.G.add(Conv2DTranspose(filters=256, kernel_size=4))
        self.G.add(Activation('relu'))

        # 4x4x256 - kernel sized increased by 1
        self.G.add(Conv2D(filters=256, kernel_size=4, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 8x8x256 - kernel sized increased by 1
        self.G.add(Conv2D(filters=128, kernel_size=4, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 16x16x128
        self.G.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 32x32x64
        self.G.add(Conv2D(filters=32, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 64x64x32
        self.G.add(Conv2D(filters=16, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 128x128x16
        self.G.add(Conv2D(filters=8, kernel_size=3, padding='same'))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        # 256x256x8
        self.G.add(Conv2D(filters=3, kernel_size=3, padding='same'))
        self.G.add(Activation('sigmoid'))

        return self.G

    def DisModel(self):

        if self.DM == None:
            self.DM = Sequential()
            self.DM.add(self.discriminator())

        self.DM.compile(optimizer=Adam(learning_rate=self.LR * (0.85 ** floor(self.steps / 10000))), loss='binary_crossentropy')

        return self.DM

    def AdModel(self):

        if self.AM == None:
            self.AM = Sequential()
            self.AM.add(self.generator())
            self.AM.add(self.discriminator())

        self.AM.compile(optimizer=Adam(learning_rate=self.LR * (0.85 ** floor(self.steps / 10000))), loss='binary_crossentropy')

        return self.AM

    def sod(self):

        self.OD = self.D.get_weights()

    def lod(self):

        self.D.set_weights(self.OD)


class Model_GAN(object):

    def __init__(self):

        self.GAN = GAN()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.generator = self.GAN.generator()

    def train(self, batch=16):

        (a, b) = self.train_dis(batch)
        c = self.train_gen(batch)

        print(f"D Real: {str(a)}, D Fake: {str(b)}, G All: {str(c)}")

        if self.GAN.steps % 500 == 0:
            self.save(floor(self.GAN.steps / 1000))
            self.evaluate()

        if self.GAN.steps % 5000 == 0:
            self.GAN.AM = None
            self.GAN.DM = None
            self.AdModel = self.GAN.AdModel()
            self.DisModel = self.GAN.DisModel()

        self.GAN.steps = self.GAN.steps + 1

    def train_gen(self, batch):

        self.GAN.sod()

        label_data = []
        for i in range(int(batch)):
            # label_data.append(one())
            label_data.append(zero())

        g_loss = self.AdModel.train_on_batch(noise(batch), np.array(label_data))

        self.GAN.lod()

        return g_loss

    def save(self, num):
        print("NUM: ", num)
        gen_json = self.GAN.G.to_json()
        dis_json = self.GAN.D.to_json()

        with open("./models/gen.json", "w+") as json_file:
            json_file.write(gen_json)

        with open("./models/dis.json", "w+") as json_file:
            json_file.write(dis_json)

        self.GAN.G.save_weights("./models/gen" + str(num) + ".h5")
        self.GAN.D.save_weights("./models/dis" + str(num) + ".h5")

        print(f"Model number {str(num)} Saved!")

    def load(self, num):
        steps1 = self.GAN.steps

        self.GAN = None
        self.GAN = GAN()

        # Generator
        gen_file = open("./models/gen.json", 'r')
        gen_json = gen_file.read()
        gen_file.close()

        self.GAN.G = model_from_json(gen_json)
        self.GAN.G.load_weights("./models/gen" + str(num) + ".h5")

        # Discriminator
        dis_file = open("./models/dis.json", 'r')
        dis_json = dis_file.read()
        dis_file.close()

        self.GAN.D = model_from_json(dis_json)
        self.GAN.D.load_weights("./models/dis" + str(num) + ".h5")

        # Reinitialize
        self.generator = self.GAN.generator()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()

        self.GAN.steps = steps1



def heightmap_generation():
    model = Model_GAN()
    model.load(75000)

    hmap_vector = model.generator.predict(noise(1))
    hmap_vector = np.squeeze(hmap_vector, axis=0)  # remove the batch dimension
    hmap_image = Image.fromarray(np.uint8(hmap_vector * 255))

    hmap_buffer = BytesIO()
    hmap_image.save(hmap_buffer, format='png')

    hmap_bytes = hmap_buffer.getvalue()
    hmap_b64 = base64.b64encode(hmap_bytes).decode('utf-8') # The bytes from the image are converted to bytes in base64, and then it's converted to srings.

    return hmap_b64

## ENHANCING THE IMAGE


def preprocess_image(hmap_b64):
    hmap_bytes = base64.b64decode(hmap_b64.encode('utf-8'))
    hr_image = tf.image.decode_image(hmap_bytes)
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def super_resolution(hmap_b64):
    hr_image = preprocess_image(hmap_b64)
    model_enh = hub.load(settings.SAVED_MODEL_PATH)
    sr_image = model_enh(hr_image)
    sr_image = tf.squeeze(sr_image)
    sr_image = Image.fromarray(np.uint8(sr_image))

    sr_buffer = BytesIO()
    sr_image.save(sr_buffer, format='png')
    sr_bytes = sr_buffer.getvalue()
    sr_b64 = base64.b64encode(sr_bytes).decode('utf-8')
    return sr_b64

## TEXTURE GENERATION

def define_texture(texture, lower_bound, upper_bound, position):
    lower_bound = int(lower_bound)
    upper_bound = int(upper_bound)

    # list of textures
    dict_text = {
              'flat rock': './textures/rock_1.jpg',
              'big rock': './textures/rock_2.jpg',
              'rock with snow': './textures/rock_snow.jpg',
              'mars': './textures/mars.jpg',
              'grass': './textures/grass.jpg',
              'clay and moss': './textures/clay_moss.jpg',
              'clay': './textures/clay.jpg',
              'mud': './textures/mud.jpg',
              'snow': './textures/snow.jpg'
    }

    # textures and its limits
    ranges = []

    lower_bound = math.floor(lower_bound * 255 / 100)
    upper_bound = math.ceil(upper_bound * 255 / 100)
    bounds_tuple = (lower_bound, upper_bound)

    if position == 'all':
        hor_pos_tuple = tuple((0, 1023))
        vert_pos_tuple = tuple((0, 1023))
        # Create a tuple with the range values and file path
        new_tuple = (bounds_tuple, hor_pos_tuple, vert_pos_tuple, dict_text[texture])
        # Print the new tuple
        ranges.append(new_tuple)
    else:
        if position == 'up_left':
            hor_pos_tuple = tuple((0, 512))
            vert_pos_tuple = tuple((0, 512))
            # Create a tuple with the range values and file path
            new_tuple = (bounds_tuple, hor_pos_tuple, vert_pos_tuple, dict_text[texture])
            # Print the new tuple
            ranges.append(new_tuple)
        if position == 'up_right':
            hor_pos_tuple = tuple((512, 1023))
            vert_pos_tuple = tuple((0, 512))
            # Create a tuple with the range values and file path
            new_tuple = (bounds_tuple, hor_pos_tuple, vert_pos_tuple, dict_text[texture])
            # Print the new tuple
            ranges.append(new_tuple)
        if position == 'down_left':
            hor_pos_tuple = tuple((0, 512))
            vert_pos_tuple = tuple((512, 1023))
            # Create a tuple with the range values and file path
            new_tuple = (bounds_tuple, hor_pos_tuple, vert_pos_tuple, dict_text[texture])
            # Print the new tuple
            ranges.append(new_tuple)
        if position == 'down_right':
            hor_pos_tuple = tuple((512, 1023))
            vert_pos_tuple = tuple((512, 1023))
            # Create a tuple with the range values and file path
            new_tuple = (bounds_tuple, hor_pos_tuple, vert_pos_tuple, dict_text[texture])
            # Print the new tuple
            ranges.append(new_tuple)

    return ranges

def apply_texture(ranges, hmap_img):
    hmap_bytes = base64.b64decode(hmap_img.encode('utf-8'))
    hmap_img = tf.image.decode_image(hmap_bytes)

    hmap_img = hmap_img.numpy()
    hmap_img = hmap_img.astype(np.uint8)
    result = np.zeros_like(hmap_img)
    edges = np.zeros_like(hmap_img[:, :, 0]).astype(np.uint8)

    hmap_img = hmap_img.astype(np.uint8)

    result_hmaps = []
    result_seps = []

    for r in ranges:
        (height_range, vert_pos_range, hor_pos_range, texture_path) = r

        # Create a mask of the pixels that are within the height and position ranges
        mask = np.zeros_like(hmap_img)
        mask[(hmap_img[:, :, 0] >= height_range[0]) & (hmap_img[:, :, 0] < height_range[1]) &
             (np.indices(hmap_img[:, :, 0].shape)[0] >= hor_pos_range[0]) & (np.indices(hmap_img[:, :, 0].shape)[0] < hor_pos_range[1]) &
             (np.indices(hmap_img[:, :, 0].shape)[1] >= vert_pos_range[0]) & (np.indices(hmap_img[:, :, 0].shape)[1] < vert_pos_range[1])] = 255
        mask = mask.astype(np.uint8)

        # Load the texture image
        texture_img = cv2.imread(texture_path)
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)
        texture_img = cv2.resize(texture_img, mask.shape[:2][::-1])

        # Replace the pixels in the range with the corresponding texture pixels
        result[mask == 255] = texture_img[mask == 255]
        # NEW: SACAR LA TEXTURA EN LA PARTE BLANCA DE LA MÁSCARA Y NADA EN LA PARTE NEGRA.
        result_sep = np.zeros_like(hmap_img)
        result_sep[mask == 255] = texture_img[mask == 255]

        result_sep_pil = Image.fromarray(result_sep)
        result_sep_buffer = BytesIO()
        result_sep_pil.save(result_sep_buffer, format='png')
        result_sep_bytes = result_sep_buffer.getvalue()
        result_sep_b64 = base64.b64encode(result_sep_bytes).decode('utf-8')

        result_seps.append(result_sep_b64)

        # Separate the heightmap and texture per textures.
        result_hmap = hmap_img.copy()
        result_hmap[mask == 0] = 0

        result_hmap_pil = Image.fromarray(result_hmap)
        result_hmap_buffer = BytesIO()
        result_hmap_pil.save(result_hmap_buffer, format='png')
        result_hmap_bytes = result_hmap_buffer.getvalue()
        result_hmap_b64 = base64.b64encode(result_hmap_bytes).decode('utf-8')

        result_hmaps.append(result_hmap_b64)

        edges = cv2.add(edges, cv2.Canny(mask, 100, 200))

    # Define the structuring element (a 3x3 matrix with all ones)
    kernel = np.ones((5,5), np.uint8)

    # Dilate the image
    dilated_image = cv2.dilate(edges, kernel)

    # Replace the remaining pixels (those outside all ranges) with white pixels
    result[result.sum(axis=2) == 0] = 255

    # Apply a Gaussian blur to only those parts of the original image that correspond to white pixels in the mask
    blurred_image = cv2.GaussianBlur(result, ksize=(0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    result[dilated_image == 255] = blurred_image[dilated_image == 255]


    # Define the parameters for each swirl effect
    swirl_params = [
        {'rotation': 0, 'strength': 2, 'radius': 100, 'center': (i*64, result.shape[0]/2)} for i in range(0, 16)
    ]

    swirl_params.extend([
        {'rotation': 0, 'strength': 2, 'radius': 100, 'center': (result.shape[0]/2, i*64)} for i in range(0, 16)
    ])

    # Apply the swirl effects to the image
    for params in swirl_params:
        result = transform.swirl(result, **params)

    result = Image.fromarray(np.uint8(result * 255))

    result_buffer = BytesIO()
    result.save(result_buffer, format='png')
    result_bytes = result_buffer.getvalue()
    result_b64 = base64.b64encode(result_bytes).decode('utf-8')  # Images are converted to bytes and from bytes we obtain strings
    return result_b64, result_hmaps, result_seps

# BLENDER
# Step 1 - Erase everything
def clear_page(bpy):
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    return

def create_plane(bpy):
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0), enter_editmode=True)
    plane_obj = bpy.data.objects[0]
    plane_obj.name = 'MyPlane'
    return plane_obj

# Step 3 - Subdivide
def subdivide(num_cuts, bpy):
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=num_cuts)
    bpy.ops.object.mode_set(mode='OBJECT')
    return

# Step 4 - Material (Color layer)
def create_material(plane_obj, texture_b64, bpy):
    plane_material_obj = bpy.data.materials.new(plane_obj.name + '-Material')
    plane_material_obj.use_nodes = True

    bsdf = plane_material_obj.node_tree.nodes["Principled BSDF"]
    texImage = plane_material_obj.node_tree.nodes.new('ShaderNodeTexImage')

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        hmap_bytes = base64.b64decode(texture_b64.encode('utf-8'))
        f.write(hmap_bytes)
        temp_filepath = f.name

    texImage.image = bpy.data.images.load(temp_filepath)
    plane_material_obj.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    plane_obj.data.materials.append(plane_material_obj)
    return

# Step 5 - Texture (Heightmap)

def add_modifiers(plane_obj, hmap_b64, bpy):
    # Add Modifiers that will help us give shape
    # Step 5.1 - Subdivision Surface modifier
    subsurf = plane_obj.modifiers.new(name='Subdivision Surface', type='SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 5
    subsurf.subdivision_type = 'SIMPLE'

    # Step 5.2 - Displacer modifier
    mod = plane_obj.modifiers.new(name='Displace', type='DISPLACE')
    tex = bpy.data.textures.new(name='Hmap', type='IMAGE')

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        hmap_bytes = base64.b64decode(hmap_b64.encode('utf-8'))
        f.write(hmap_bytes)
        temp_filepath = f.name

    tex.image = bpy.data.images.load(temp_filepath)
    mod.texture = tex
    mod.mid_level = 0.0
    mod.strength = 0.5
    return

def render_blender(hmap_b64, texture_b64, buffer=None, name="MyTerrain_Complete"):
    import bpy

    clear_page(bpy)
    plane_obj = create_plane(bpy)
    subdivide(100, bpy)
    create_material(plane_obj, texture_b64, bpy)
    add_modifiers(plane_obj, hmap_b64, bpy)
    file_path = "./test/MyTerrain.obj"
    bpy.ops.export_scene.obj(filepath=file_path, use_selection=True)

    buffer = BytesIO() if buffer is None else buffer
    with zipfile.ZipFile(buffer, 'a') as my_terrain:
        my_terrain.write(file_path, name + '/' + basename(file_path))
        my_terrain.write(file_path.replace('.obj', '.mtl'), name + '/' + basename(file_path.replace('.obj', '.mtl')))

    return buffer

def render_result(hmap_b64, texture_b64):
    import bpy

    clear_page(bpy)
    plane_obj = create_plane(bpy)
    subdivide(100, bpy)
    create_material(plane_obj, texture_b64, bpy)
    add_modifiers(plane_obj, hmap_b64, bpy)
    file_path = "./test/MyTerrain.gltf"

    # Export the scene to a BytesIO buffer.
    bpy.ops.export_scene.gltf(
        filepath=file_path,
        use_selection=True,
        export_format='GLTF_EMBEDDED',
        export_apply=True
    )

    return open(file_path, 'rb')
