#import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import TextVectorization
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.layers import LayerNormalization,Dense,Embedding,MultiHeadAttention,Dropout,Layer
from tensorflow.keras import Model

import os
import random
from pycocotools.coco import COCO
from PIL import Image

# === Paths ===
data_dir = 'data'
img_dir = os.path.join(data_dir, 'train2017','train2017')
ann_file = os.path.join(data_dir,'annotations_trainval2017', 'annotations', 'captions_train2017.json')

# === Load COCO captions ===
coco = COCO(ann_file)
img_ids = coco.getImgIds()

# === Choose 100 random image IDs ===
selected_img_ids = random.sample(img_ids, 100)

samples = []

for img_id in selected_img_ids:
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    file_path = os.path.join(img_dir, file_name)

    # Load the first caption (or more if needed)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    caption = anns[0]['caption'] if anns else ""

    # Load image using PIL (or use tf.io.read_file if you prefer tensors)
    image = Image.open(file_path).convert('RGB')

    samples.append({
        'image': image,
        'caption': caption,
        'file_path': file_path,
    })

# ✅ Now 'samples' is a list of 100 (image, caption) dictionaries
import tensorflow as tf

def load_image_tensor(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

# Convert to tf.data.Dataset if needed
file_paths = [sample['file_path'] for sample in samples]
captions = [sample['caption'] for sample in samples]

dataset = tf.data.Dataset.from_tensor_slices((file_paths, captions))

def process(path, caption):
    image = load_image_tensor(path)
    return {'image': image, 'captions': {'text': caption}}

dataset = dataset.map(process)
#image model (encoder)
encoder = EfficientNetB3(include_top=False,pooling='avg')
encoder.trainable = False
max_token = 10000
max_length = 30
vectorizer = TextVectorization(max_tokens=max_token, output_sequence_length=max_length+1)
captions = dataset.map(lambda x: x['captions']['text'])
vectorizer.adapt(captions)
vocab_size = vectorizer.vocabulary_size()

#decoder Layer
class DecoderLayer(Layer):
    def __init__(self,model_dim,self_attention_num,feed_forward_dim,Rate = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_heads=self_attention_num,key_dim=model_dim)
        self.cross_attention = MultiHeadAttention(num_heads=self_attention_num,key_dim=model_dim)
        self.feed_forward_nn = tf.keras.Sequential([Dense(feed_forward_dim, activation="relu"), Dense(model_dim)])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.norm3 = LayerNormalization()
        self.drop1 = Dropout(Rate)
        self.drop2 = Dropout(Rate)
        self.drop3 = Dropout(Rate)
    def call(self, x, img_feature, training, look_ahead_mask, padding_mask):
        att1 = self.self_attention(x,x, attention_mask=look_ahead_mask)
        att1 = self.drop1(att1,training=training)
        out1 = self.norm1(x+att1)

        att2 = self.cross_attention(out1,img_feature, attention_mask=padding_mask)
        att2 = self.drop2(att2,training=training)
        out2 = self.norm2(out1+att2)

        ffn = self.feed_forward_nn(out2)
        ffn = self.drop3(ffn,training=training)
        out3 = self.norm3(out2+ffn)

        return out3


        

#decoder
class TransformerDcoder(Layer):
    def __init__(self,model_dim,num_layers,vocabulary_size,positional_encoding_size,feed_forward_dim,self_attention_num,Rate = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.embedding = Embedding(vocabulary_size,model_dim)
        self.pos_embedding = Embedding(positional_encoding_size,model_dim)
        self.decoder_layers = [DecoderLayer(model_dim=model_dim,self_attention_num=self_attention_num,Rate=Rate,feed_forward_dim=feed_forward_dim) for _ in range(num_layers)]
        self.dropout = Dropout(Rate)

    def call(self,target,img_feature,training,look_ahead_mask, padding_mask):
        seq_len = tf.shape(target)[1]
        x = self.embedding(target) * tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        positions = tf.range(start=0, limit=seq_len, delta=1)
        x += self.pos_embedding(positions)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x,img_feature,training,look_ahead_mask,padding_mask)
        return x

#encoder

def ImgEncoder():
    
    base_model = EfficientNetB3(include_top=False,weights='imagenet')
    base_model.trainable = False

    output = base_model.output
    output = tf.keras.layers.Reshape((-1, output.shape[-1]))(output)
    
    encoder = Model(base_model.input,output)
    return encoder

#main model
class MyModel(Model):
    def __init__(self,ImgEncoder,decoder,vocab_size):
        super().__init__()
        self.encoder = ImgEncoder
        self.decoder = decoder
        self.dense = Dense(vocab_size)
    
    def create_masks(self, tar):
        seq_len = tf.shape(tar)[1]
        look_ahead = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padding_mask = None
        return look_ahead[tf.newaxis, tf.newaxis, :, :], padding_mask

    def call(self,input,target,training):
        img_feature = self.encoder(input)
        look_ahead_mask,padding_mask = self.create_masks(tar=target)
        decoder_output = self.decoder(target, img_feature, training, look_ahead_mask, padding_mask)
        return self.dense(decoder_output)

model_dim = 512
num_layers = 4
vocab_size = 10000
pos_encoding_size = 1000
dff = 2048
num_heads = 8
dropout_rate = 0.1
decoder = TransformerDcoder(model_dim=model_dim,num_layers=num_layers,vocabulary_size=vocab_size
                            ,positional_encoding_size=pos_encoding_size,feed_forward_dim=dff
                            ,self_attention_num=num_heads,Rate=dropout_rate)
encoder = ImgEncoder()


def process(path, caption):
    image = load_image_tensor(path)
    image = tf.image.resize(image, (300, 300))  # EfficientNetB3 uses 300x300
    image = preprocess_input(image)
    return {'image': image, 'captions': {'text': caption}}

model = MyModel(ImgEncoder=encoder,vocab_size=vocab_size,decoder=decoder)
