# Class imbalance / Class weighting
# Ref: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
# Ref: https://datascience.stackexchange.com/a/41752/88261

# Transfer learning
# Ref: https://www.tensorflow.org/tutorials/images/transfer_learning

# Loading images
# Ref: https://www.tensorflow.org/tutorials/load_data/images

# Converting dataframe to tensorflow dataset
# Ref: https://www.tensorflow.org/tutorials/load_data/pandas_dataframe

import tensorflow as tf
import pandas as pd


import imageio
import numpy as np
import os
import time

from itertools import chain
from operator import itemgetter



from sklearn.model_selection import train_test_split

from summan import SummaryManager
summary_manager = SummaryManager()

import argparse
from tqdm import tqdm

class Classifier:
    def __init__(self,
                 summary_manager,
                 validation_split=.5,
                 base_learning_rate=0.0001,
                 checkpoint_filepath='checkpoint',
                 save_checkpoint_filepath=None,
                 base_network='mobilenetv2',
                 training_mode=True,
                 fine_tune=0,
                 use_dataaug=False,
                 custom_classifier=None,
                 ssdliteclassifier=False):
        if not summary_manager.is_summary_loaded():
            raise Exception("Summary not loaded yet!")

        self.dl_model = None
        self.image_preprocessor = None
        self.model_checkpoint_callback = None

        self.summary_manager = summary_manager
        self.validation_split = validation_split
        self.base_learning_rate = base_learning_rate
        self.checkpoint_filepath = checkpoint_filepath
        self.save_checkpoint_filepath = save_checkpoint_filepath
        self.base_network = base_network
        self.fine_tune = fine_tune
        self.use_dataaug = use_dataaug
        self.custom_classifier = custom_classifier
        self.ssdliteclassifier = ssdliteclassifier

        self.model_ready = False
        
        self._set_image_preprocessor()

        
        self.class_names = self.summary_manager.labels[2:]
        # Each class correspond to a combination of the labels
        # e.g. If an image has the following labels: (0,1,0) then its categorical class
        # would be '5', that is, this image has powerlines but has not neither trees nor
        # intersections.
        self.classes_categorical = [i for i in range(2 ** len(self.class_names))]

        # getting the labeled data from the model
        self.dataframe_labeled_samples = self.summary_manager.get_labeled_samples()

        # inserting a new column as a proxy for the labels
        self.dataframe_labeled_samples.insert(
            len(self.dataframe_labeled_samples.columns),
            'categorical_labels', 0)
        # populating the new column with a computed value (categorized) for the labels
        # it is important to use a auxiliar variable to avoid reference errors from the DataFrame
        aux = np.apply_along_axis(
                lambda x: np.sum([x[i] * 2 ** i for i in range(len(x))]),
                1,
                self.dataframe_labeled_samples[self.class_names].astype('int').values)
        self.dataframe_labeled_samples.loc[:, 'categorical_labels'] = aux

        self.dataset_size = len(self.dataframe_labeled_samples)

        # samples_per_class is a dict whose key is a categorical label (i.e. from 0 up to 5).
        # Theoretically with 3 labels there are 8 possibilities of classes, but since here
        # the 'Intersection' class depends on the presence of the other labels 'Tree' and
        # 'Powerline' then there are only 5 valid classes.
        #
        # The values of each key in the dict samples_per_class
        # correspond to the indexes of images in the class
        # of the key.
        self.samples_per_class = {}
        for class_label in self.classes_categorical:
            self.samples_per_class[class_label] = \
                self.dataframe_labeled_samples[
                self.dataframe_labeled_samples\
                    ['categorical_labels'] == class_label
            ].index.values

        aux = []
        for key in self.samples_per_class.keys():
            if len(self.samples_per_class[key]) == 0:
                aux.append(key)
                self.classes_categorical.remove(key)
        # To avoid issues during the composition of balanced batches,
        # if a given class has no samples then it is simply removed.
        for key in aux:
            del self.samples_per_class[key]

    def _set_image_preprocessor(self):
        if self.base_network == 'mobilenetv2':
            self.image_preprocessor = tf.keras.applications.mobilenet_v2.preprocess_input
        elif self.base_network == 'resnet50v2':
            self.image_preprocessor = tf.keras.applications.resnet_v2.preprocess_input
        elif self.base_network == 'resnet101v2':
            self.image_preprocessor = tf.keras.applications.resnet_v2.preprocess_input
        pass

    def _define_base_model(self):
        IMG_SIZE = 640  # depends on the maximum size of the network, currently it should be 224
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

        if self.base_network == 'mobilenetv2':
            base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                           include_top=False,
                                                           weights='imagenet')
        elif self.base_network == 'resnet50v2':
            base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                                          include_top=False,
                                                          weights='imagenet')
        elif self.base_network == 'resnet101v2':
            base_model = tf.keras.applications.ResNet101V2(input_shape=IMG_SHAPE,
                                                           include_top=False,
                                                           weights='imagenet')
        else:
            validOptions = [
                'mobilenetv2',
                'resnet50v2',
                'resnet101v2'
            ]
            raise Exception("Invalid base model. Valid options are: " + ",".join(validOptions))
        return base_model

    def get_image_paths_batch_linear(self, batch_size=1, start_at_idx=0):
        """
        Based on get_images_batch.
        In contrast with get_images_batch this function returns
        an iterator with batches that together forms the whole dataset.

        Notice that this function returns the path to the images, not the images themselves!

        Notice that the last element can be a smaller batch in case that
        the dataset size is not a multiple of the batch_size.
        :param batch_size: Number of samples per batch
        :param start_at_idx: Start getting samples at this index onwards
        :return: An iterator with references to each batch of images paths and labels
        """
        images_paths = []
        labels = []
        for idx in self.dataframe_labeled_samples.index.values:
            images_paths.append(idx)
            labels.append(
                tuple(self.dataframe_labeled_samples.loc[idx][self.class_names].values.astype('float')))

        labels = np.asarray(labels, 'int')

        samples_left = len(self.dataframe_labeled_samples.index.values) - start_at_idx
        current_sample_idx = start_at_idx
        batches = []
        while samples_left > 0:
            if samples_left > batch_size:
                upper_index = current_sample_idx + batch_size
                batches.append((images_paths[current_sample_idx:upper_index], labels[current_sample_idx:upper_index]))
                current_sample_idx += batch_size
                samples_left -= batch_size
            else:
                batches.append((images_paths[current_sample_idx:], labels[current_sample_idx:]))
                current_sample_idx += samples_left
                samples_left = 0
        return batches

    def get_images_batch(self, batch_size):
        """
        Gets a balanced and random batch of samples.
        :param batch_size: Number of samples per batch
        :return: return a tuple with two numpy arrays, the first with samples,
        and the second with the respective (categorical) labels
        """
        images = []
        labels = []
        num_classes = len(self.samples_per_class.keys())
        if batch_size < num_classes:
            raise Exception("Batch smaller than the number of classes!")
        rest = batch_size % num_classes
        idxs = []
        if rest == 0:
            num_samples_per_class = batch_size // num_classes
            for key in self.samples_per_class.keys():
                idxs = np.hstack((
                    idxs,
                    np.random.choice(self.samples_per_class[key], num_samples_per_class)
                ))
        else:
            num_samples_per_class = np.hstack((
                np.full(rest, 1 + (batch_size // num_classes)),
                np.full(num_classes - rest, batch_size // num_classes)
            ))
            for ikey, key in enumerate(self.samples_per_class):
                idxs = np.hstack((
                    idxs,
                    np.random.choice(self.samples_per_class[key], [num_samples_per_class[ikey]])
                ))
        for idx in idxs:
            imgFilename = os.path.join(os.path.dirname(
                self.summary_manager.current_labelgui_summary_filepath),
                idx)
            images.append(self.image_preprocessor(imageio.imread(imgFilename)))
            labels.append(
                tuple(self.dataframe_labeled_samples.loc[idx][self.class_names].values.astype('float')))

        images = np.asarray(images)
        labels = np.asarray(labels, 'int')
        return images, labels

    def _create_network(self, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        base_model = self._define_base_model()
        base_model.trainable = False

        if self.fine_tune > 0:
            # Fine-tune from this layer onwards
            fine_tune_at = self.fine_tune
            # Unfreeze all the layers before the `fine_tune_at` layer
            for layer in base_model.layers[-fine_tune_at:]:
                layer.trainable = True

        if self.use_dataaug:
            data_augmentation = tf.keras.Sequential([
                    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
                ])
            x = tf.keras.Input(shape=(None, None, 3))
            x = data_augmentation(x)
            x = base_model(x, training=False)
        else:
            x = base_model.output

        if (self.ssdliteclassifier):
            #conv6
            x = tf.keras.layers.SeparableConv2D(
                filters=1024,
                kernel_size=(3, 3),
                padding='same',
                activation=None)(x)
            x = tf.keras.activations.relu(x, max_value=6.0)
            #conv7
            x = tf.keras.layers.SeparableConv2D(
                filters=1024,
                kernel_size=(3, 3),
                padding='same',
                activation=None)(x)
            x = tf.keras.activations.relu(x, max_value=6.0)
            #conv8
            x = tf.keras.layers.SeparableConv2D(
                filters=512,
                kernel_size=(3, 3),
                padding='same',
                activation=None)(x)
            x = tf.keras.activations.relu(x, max_value=6.0)
            #conv9
            x = tf.keras.layers.SeparableConv2D(
                filters=256,
                kernel_size=(3, 3),
                padding='same',
                activation=None)(x)
            x = tf.keras.activations.relu(x, max_value=6.0)
            #conv10
            x = tf.keras.layers.SeparableConv2D(
                filters=256,
                kernel_size=(3, 3),
                padding='same',
                activation=None)(x)
            x = tf.keras.activations.relu(x, max_value=6.0)
            #conv11
            x = tf.keras.layers.SeparableConv2D(
                filters=128,
                kernel_size=(3, 3),
                padding='same',
                activation=None)(x)
            x = tf.keras.activations.relu(x, max_value=6.0)
            
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if (self.custom_classifier is not None):
            for num in self.custom_classifier:
                x = tf.keras.layers.Dense(num, activation='relu')(x)
                #x = tf.keras.layers.Dense(num, activation='sigmoid')(x)
                

        x = tf.keras.layers.Dense(len(self.class_names),
                                                bias_initializer=output_bias,
                                                activation='sigmoid')(x)

        dl_model = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=x, #outputs=prediction_layer.output,
            name="tw_model")

        dl_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.base_learning_rate),
                         loss='binary_crossentropy',
                         metrics=[tf.keras.metrics.BinaryAccuracy()])
        return dl_model

    def _setup_checkpoint_callback(self):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.save_checkpoint_filepath or self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_binary_accuracy',
            mode='max',
            verbose=1,
            save_best_only=True)
        return model_checkpoint_callback

    def init_network(self):
        self.dl_model = self._create_network()

        self.model_ready = True
        self.model_checkpoint_callback = self._setup_checkpoint_callback()

    
    def balanced_shuffled_batches(
        self,
        tf_dataset,
        shuffle_buffer,
        batch_size):
        # Ref: Balancing https://www.tensorflow.org/guide/data
        def class_func(features, label):
            return label

        def remove_extra_label(extra_label, features_and_label):
            return features_and_label
            
        resampler = tf.data.experimental.rejection_resample(
            class_func, target_dist=[0.5, 0.5])

        resample_ds = tf_dataset.apply(resampler).shuffle(
            buffer_size=shuffle_buffer,
            reshuffle_each_iteration=True)
        balanced_ds = resample_ds.map(remove_extra_label)
        #balanced_ds = balanced_ds.shuffle(shuffle_buffer).batch(batch_size)
        balanced_ds = balanced_ds.batch(batch_size)
        return balanced_ds

    def create_tf_datasets(self):
        """
        Create two tensorflow datasets out of the labeled samples.
        One for training and one for validation.
        :return: (tf.data.Dataset, tf.data.Dataset)
        """
        images = []
        labels = []

        images = self.dataframe_labeled_samples.index.values

        labels.append(
                tuple(self.dataframe_labeled_samples['Intersection'].values.astype('uint8')))

        images = [
            os.path.join(
                os.path.dirname(
                    self.summary_manager.current_labelgui_summary_filepath),
                    img_name) for img_name in images]
        labels = list(chain.from_iterable(labels))


        if self.validation_split == 0:
            images = np.array([
                self.image_preprocessor(
                    imageio.imread(f)) for f in tqdm(images)])
            images = tf.data.Dataset.from_tensor_slices(images)
            labels = tf.data.Dataset.from_tensor_slices(labels)
            dataset = tf.data.Dataset.zip((images, labels))
            return dataset, None

        images, images_val, labels, labels_val = train_test_split(
            images, labels, test_size=self.validation_split, random_state=0)

        train_split_filename = ((
            f'{self.save_checkpoint_filepath or self.checkpoint_filepath}'
            f'_train_split.txt'
        ))
        print(f"Saving train split files to: {train_split_filename}")
        with open(train_split_filename, 'w+')\
            as train_split_file:
            for img in images:
                train_split_file.write(img + '\n')
        
        val_split_filename = ((
            f'{self.save_checkpoint_filepath or self.checkpoint_filepath}'
            f'_val_split.txt'
        ))
        print(f"Saving train split files to: {val_split_filename}")
        with open(val_split_filename, 'w+')\
            as val_split_file:
            for img in images_val:
                val_split_file.write(img + '\n')

        print(f"Loading validation image paths ({len(images)}) with preprocessor")
        images = np.array([
            self.image_preprocessor(
                imageio.imread(f)) for f in tqdm(images)])
        images = tf.data.Dataset.from_tensor_slices(images)

        print(f"Loading labels into tf tensor")
        labels = tf.data.Dataset.from_tensor_slices(labels)
        print(f"Creating zipped dataset with images and labels")
        dataset = tf.data.Dataset.zip((images, labels))

        print(f"Loading validation image paths ({len(images_val)}) with preprocessor")
        images_val = np.array([
            self.image_preprocessor(
                imageio.imread(f)) for f in tqdm(images_val)])
        images_val = tf.data.Dataset.from_tensor_slices(images_val)

        print(f"Loading validation labels into tf tensor")
        labels_val = tf.data.Dataset.from_tensor_slices(labels_val)

        print(f"Creating validation zipped dataset with images and labels")
        dataset_val = tf.data.Dataset.zip((images_val, labels_val))

        return dataset, dataset_val

    def train_network(self, num_epochs=10, shuffle_buffer=100, batch_size=100):
        if not self.model_ready:
            raise Exception("Model not ready")
        if os.path.exists(self.checkpoint_filepath + '.index'):
            print(f"Loading weights from: {self.checkpoint_filepath}")
            self.dl_model.load_weights(self.checkpoint_filepath)

        print(f"Loading dataset")
        tf_dataset, tf_dataset_val  = self.create_tf_datasets()

        print(f"Creating balanced and shuffled batches with size {batch_size}")
        balanced_ds = self.balanced_shuffled_batches(
            tf_dataset,
            shuffle_buffer=shuffle_buffer,
            batch_size=batch_size)

        fitParams = {
            'epochs': num_epochs,
            'callbacks':[self.model_checkpoint_callback]
            }
        
        if self.validation_split >= 0.0:
            print(f"Now validation dataset")
            balanced_ds_val = self.balanced_shuffled_batches(
                tf_dataset_val,
                shuffle_buffer=shuffle_buffer,
                batch_size=batch_size)
            fitParams['validation_data'] = balanced_ds_val
        
        print(f"Starting fit")
        history = self.dl_model.fit(
            balanced_ds,
            **fitParams)
        return history

    def check_checkpoint_variable(self, alternative_checkpoint_filepath=None):
        if alternative_checkpoint_filepath is None:
            if self.checkpoint_filepath is not None:
                alternative_checkpoint_filepath = self.checkpoint_filepath
        return alternative_checkpoint_filepath

    def evaluate_saved_network(
        self,
        batch_size,
        outputfile,
        alternative_checkpoint_filepath=None,
        repetitions=10):
        if not self.model_ready:
            raise Exception("Model not ready")
        alternative_checkpoint_filepath = self.check_checkpoint_variable(alternative_checkpoint_filepath)
        print(f"Cleaning evaluation file {outputfile}")
        open(outputfile, 'w').close()

        for _ in range(2):
            dataset, _ = self.create_tf_datasets()
            balanced_ds = self.balanced_shuffled_batches(dataset,
                shuffle_buffer=self.dataset_size,
                batch_size=batch_size)
            # Evaluate the model
            _, acc1 = self.dl_model.evaluate(balanced_ds, verbose=2)
            # Loads the weights
            with open(outputfile, 'a+') as f:
                f.write(f"{time.ctime()} - " +\
                    "Untrained model, accuracy val: {:5.2f}%".format(100 * acc1) + "\n")
        self.dl_model.load_weights(alternative_checkpoint_filepath)
        for _ in range(repetitions):
            dataset, _ = self.create_tf_datasets()
            balanced_ds = self.balanced_shuffled_batches(dataset,
                shuffle_buffer=self.dataset_size,
                batch_size=batch_size)
            # Re-evaluate the model
            _, acc2 = self.dl_model.evaluate(balanced_ds, verbose=2)
            with open(outputfile, 'a+') as f:
                f.write(f"{time.ctime()} - " + "Restored model, accuracy val: {:5.2f}%".format(100 * acc2) + "\n")

    def generate_predictions(self, batch_size, alternative_checkpoint_filepath=None):
        if not self.model_ready:
            raise Exception("Model not ready")
        alternative_checkpoint_filepath = self.check_checkpoint_variable(alternative_checkpoint_filepath)
        if alternative_checkpoint_filepath is not None:
            if os.path.exists(alternative_checkpoint_filepath + '.index'):
                print(f"Checkpoint '{alternative_checkpoint_filepath}' found! Loading weights!")
                self.dl_model.load_weights(alternative_checkpoint_filepath)
            else:
                print(f"Checkpoint '{alternative_checkpoint_filepath}' not found, it will be created")
        batches = self.get_image_paths_batch_linear(batch_size)
        predictions_list = []
        image_paths = []

        with tf.device('/gpu:0'):
            for batch in tqdm(batches):
                samples = []
                for image_name in batch[0]:
                    image_path = os.path.join(os.path.dirname(
                        self.summary_manager.current_labelgui_summary_filepath),
                        image_name)
                    samples.append(imageio.imread(image_path))
                samples = self.image_preprocessor(np.asarray(samples))
                predictions = self.dl_model.predict_on_batch(samples)
                predictions[predictions > 0.5] = 1.
                predictions[predictions <= 0.5] = 0.
                predictions_list.append(predictions.astype('int'))
                image_paths.append(batch[0])
        return image_paths, predictions_list[:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            f'This script can be used to train, evaluate or predict \n'
            f'data from the treewires_labeler project.\n'
            f'It is assumed that it is in the same folder as the model.py\n'
            f'script.\n'
            f'\n'
        ),
        usage=(
            f'It has a single parameter indicating the \'mode\' of use:\n\n'
            f'train: Trains the default network as defined \'hardcodedly\'\n'
            f'evaluate: Evaluates the default network\n'
            f'predict: Creates an output file with predictions for each image\n'
        )
    )
    parser.add_argument('mode',
                        type=str,
                        choices=['train', 't', 'evaluate', 'e', 'predict', 'p'],
                        help=
                        (
                            f'Defines the mode os use as \'train\' or \'t\'for training, '
                            f'\'evaluate\' or \'e\' for evaluating and '
                            f'\'predict\' or \'p\' for predicting.'
                        ))
    parser.add_argument('checkpoint',
                        type=str,
                        help=
                        ((
                            f'Sets a checkpoint file (e.g. mobilenetv2_testX.index...) '
                            f'to be used to load the weights. If the file doesn\'t exists it '
                            f'and the flag --savecheckpoint is not set then '
                            f'then this a checkpoint with this name will be created. '
                        )))
    parser.add_argument('--savecheckpoint',
                        type=str,
                        help=
                        ((
                            f'Sets a saving checkpoint file (e.g. mobilenetv2_testX.index...) '
                            f'to be used to save the weights. It must be used with the checkpoint '
                            f'paramether as well, the "checkpoint" parameter will load a checkpoint '
                            f'and this one will save a checkpoint with another name.'
                        )))
    parser.add_argument('--numepochs',
                        type=int,    
                        help=
                        (
                            f'If in train mode, it defines the number of epochs '
                            f'of training.'
                            f'The dafault is 1000 (one thousand).'
                        ),
                        default=1000)
    parser.add_argument('--test',
                        help=
                        (
                            f'Uses the defined test folder'
                        ),
                        action='store_true')
    parser.add_argument('--val_split',
                        type=float,
                        help=
                        ((
                            f'Sets a train/validation split for the dataset if used '
                            f'with the train mode. Has no effect in the other modes. '
                            f'Higher values imply lower training sets and bigger '
                            f'validation sets.'
                        )))
    parser.add_argument('--customclassifier',
                        type=str,
                        help=
                        (
                            f'Uses a custom classification head. '
                            f'This parameter defines a sequence '
                            f'of fully connected layers (dense layers) '
                            f'each one with a given number of neurons '
                            f'specified as a list of comma separated '
                            f'numbers. '
                            f'e.g. with the following argument: '
                            f' --customclassifier 1080,540 '
                            f' After the GAP layer two dense layers with '
                            f' 1080 and 540 neurons respectively will be created. '
                            f' After them there will always be a final dense layer '
                            f' with as many neurons as classes in the dataset. '
                        ))
    parser.add_argument('--ssdliteclassifier',
                        help=
                        (
                            f'Uses a classification head '
                            f'based on the SSDLite as presented at '
                            f'the paper \'SSD: Single Shot MultiBox Detector\'. '
                            f'The difference is that the Conv layers are '
                            f'replaced by SeparableConv ones as per the '
                            f'paper '
                            f'\'MobileNetV2: Inverted Residuals and Linear Bottlenecks\'.'
                            f'If used with the customclassifier flag then the '
                            f'layers from latter will be ahead (the final) ones.'
                        ),
                        action='store_true')
    parser.add_argument('--dataaug',
                        help=
                        (
                            f'Uses the hard-coded preprocessing '
                            f'data augmentation algorithm.'
                        ),
                        action='store_true')
    parser.add_argument('--finetune',
                        type=int,
                        help=
                        (
                            f'Unfreezes from the last layer up to this number '
                            f'of previous layers. '
                            f'The default is zero, meaning no layers from the base '
                            f'model will be unfreezed.'
                        ),
                        default=0)
    
    args = parser.parse_args()
    mode = args.mode
    test_mode = args.test
    arg_val_split = args.val_split
    arg_checkpoint_filepath = args.checkpoint
    arg_save_checkpoint_filepath = args.savecheckpoint
    customclassifier = args.customclassifier
    use_dataaug = args.dataaug
    fine_tune = args.finetune
    ssdliteclassifier = args.ssdliteclassifier
    numepochs = int(args.numepochs)

    print(f'Validation split argument: {arg_val_split}')
    print(f'checkpoint argument: {arg_checkpoint_filepath}')
    if arg_save_checkpoint_filepath is not None:
        print(f'save checkpoint argument: {arg_save_checkpoint_filepath}')
    if use_dataaug:
        print(f'Using data augmentation')
    if customclassifier is not None:
        print(f'Using custom classification head: {customclassifier}')
        customclassifier = [int(_) for _ in customclassifier.split(',')]


    # Used to automatically set the configuration
    import socket
    print(socket.gethostname())

    base_network = 'mobilenetv2'
    checkpoint_filepath = arg_checkpoint_filepath
    save_checkpoint_filepath = arg_save_checkpoint_filepath
    batch_size_hc = 8

    num_epochs_hc = numepochs
    eval_repetitions = 5

    print("Executing main")
    print("Loading images folder")
    train_folder = './Training_Data'
    test_folder = './Test_Data'

    predictions_report_file = f'predictions_{arg_checkpoint_filepath}.txt'
    evaluation_report_file = f'evaluation_{arg_checkpoint_filepath}.txt'

    if test_mode:
        pictures_folder = test_folder
        predictions_report_file = f'predictions_test_{arg_checkpoint_filepath}.txt'
        evaluation_report_file = f'evaluation_test_{arg_checkpoint_filepath}.txt'
    else:
        pictures_folder = train_folder

    summary_manager.load_images_folder(pictures_folder)
    print("Images folder loaded")
    print("Instantiating classifier")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # gpus = True
    if gpus:
        try:
            gpus_idx = list(range(len(gpus)))
            selected_gpus = list(itemgetter(*gpus_idx)(gpus))
            tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')
            for gpu in selected_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    C = Classifier(summary_manager,
        training_mode = ((mode == "train") or (mode == "t")),
        checkpoint_filepath=checkpoint_filepath,
        save_checkpoint_filepath=save_checkpoint_filepath,
        base_network=base_network,
        custom_classifier=customclassifier,
        ssdliteclassifier=ssdliteclassifier,
        fine_tune=fine_tune)
    print("Labels present:")
    for class_name in C.class_names:
        print(class_name)
        
    print("Initializing classifier")
    C.init_network()
    if (mode == "predict") or (mode == "p"):
        print(f"Initializing prediction with batch size of {batch_size_hc}")
        prediction_batches = C.generate_predictions(batch_size_hc, checkpoint_filepath)
        with open(predictions_report_file, 'w+') as f:
            for batch in zip(prediction_batches[0], prediction_batches[1]):
                for sample in zip(batch[0], batch[1]):
                    aux = sample[1].astype('str')
                    f.write(f"{sample[0]}, {','.join(aux)}\n")
    elif (mode == "evaluate") or (mode == "e"):
        C.validation_split = 0.0
        C.evaluate_saved_network(batch_size_hc,
                                outputfile=evaluation_report_file,
                                alternative_checkpoint_filepath=checkpoint_filepath,
                                repetitions=eval_repetitions)
    elif (mode == "train") or (mode == "t"):
        C.validation_split = arg_val_split
        C.use_dataaug = use_dataaug
        print(f"Initializing training with {num_epochs_hc} epochs and batch size of {batch_size_hc}")
        C.train_network(
            num_epochs=num_epochs_hc,
            shuffle_buffer=100,
            batch_size=batch_size_hc)


