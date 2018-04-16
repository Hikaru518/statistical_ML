# Import necessary modules and set global constants and variables. 

import tensorflow as tf            
import pandas as pd                 
import numpy as np                                       
import sklearn.model_selection     # For using KFold
import keras.preprocessing.image   # For using image generation
import datetime                    # To measure running time 
import skimage.transform           # For resizing images
import skimage.morphology          # For using image labeling
import cv2                         # To read and manipulate images
import os                          # For filepath, directory handling
import sys                         # System-specific parameters and functions
import tqdm                        # Use smart progress meter
import seaborn as sns              # For pairplots
import matplotlib.pyplot as plt    # Python 2D plotting library
import matplotlib.cm as cm         # Color map

for ii in range(3,61):



	# Global constants.
	IMG_WIDTH = 384       # Default image width
	IMG_HEIGHT = 384      # Default image height
	IMG_CHANNELS = 3      # Default number of channels
	CW_DIR = os.getcwd()  
	TRAIN_DIR = os.path.join(os.path.dirname(CW_DIR),'stage2', 'stage1_train')
	TEST_DIR = os.path.join(os.path.dirname(CW_DIR), 'stage2','stage2_test_divide\\'+str(ii) )
	IMG_TYPE = '.png'         # Image type
	IMG_DIR_NAME = 'images'   # Folder name including the image
	MASK_DIR_NAME = 'masks'   # Folder name including the masks
	LOGS_DIR_NAME = 'logs'    # Folder name for TensorBoard summaries 
	SAVES_DIR_NAME = 'saves'  # Folder name for storing network parameters
	SEED = 123                # Random seed for splitting train/validation sets
	    
	# Global variables.
	min_object_size = 1       # Minimal nucleous size in pixels
	x_train = []
	y_train = []
	x_test = []
	y_test_pred_proba = {}
	y_test_pred = {}

	# Display working/train/test directories.
	#print('CW_DIR = {}'.format(CW_DIR))
	#print('TRAIN_DIR = {}'.format(TRAIN_DIR))
	print('TEST_DIR = {}'.format(TEST_DIR))

	# Collection of methods for data operations. Implemented are functions to read  
	# images/masks from files and to read basic properties of the train/test data sets.

	def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None):
	    """Read an image from a file and resize it."""
	    img = cv2.imread(filepath, color_mode)
	    if target_size: 
	        img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)
	    return img

	def read_mask(directory, target_size=None):
	    """Read and resize masks contained in a given directory."""
	    for i,filename in enumerate(next(os.walk(directory))[2]):
	        mask_path = os.path.join(directory, filename)
	        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, target_size)
	        if not i: mask = mask_tmp
	        else: mask = np.maximum(mask, mask_tmp)
	    return mask 

	def read_train_data_properties(train_dir, img_dir_name, mask_dir_name):
	    """Read basic properties of training images and masks"""
	    tmp = []
	    for i,dir_name in enumerate(next(os.walk(train_dir))[1]):

	        img_dir = os.path.join(train_dir, dir_name, img_dir_name)
	        mask_dir = os.path.join(train_dir, dir_name, mask_dir_name) 
	        num_masks = len(next(os.walk(mask_dir))[2])
	        img_name = next(os.walk(img_dir))[2][0]
	        img_name_id = os.path.splitext(img_name)[0]
	        img_path = os.path.join(img_dir, img_name)
	        img_shape = read_image(img_path).shape
	        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
	                    img_shape[0]/img_shape[1], img_shape[2], num_masks,
	                    img_path, mask_dir])

	    train_df = pd.DataFrame(tmp, columns = ['img_id', 'img_height', 'img_width',
	                                            'img_ratio', 'num_channels', 
	                                            'num_masks', 'image_path', 'mask_dir'])
	    return train_df

	def read_test_data_properties(test_dir, img_dir_name):
	    """Read basic properties of test images."""
	    tmp = []
	    for i,dir_name in enumerate(next(os.walk(test_dir))[1]):

	        img_dir = os.path.join(test_dir, dir_name, img_dir_name)
	        img_name = next(os.walk(img_dir))[2][0]
	        img_name_id = os.path.splitext(img_name)[0]
	        img_path = os.path.join(img_dir, img_name)
	        img_shape = read_image(img_path).shape
	        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
	                    img_shape[0]/img_shape[1], img_shape[2], img_path])

	    test_df = pd.DataFrame(tmp, columns = ['img_id', 'img_height', 'img_width',
	                                           'img_ratio', 'num_channels', 'image_path'])
	    return test_df

	def imshow_args(x):
	    """Matplotlib imshow arguments for plotting."""
	    if len(x.shape)==2: return x, cm.gray
	    if x.shape[2]==1: return x[:,:,0], cm.gray
	    return x, None

	def load_raw_data(image_size=(IMG_HEIGHT, IMG_WIDTH)):
	    """Load raw data."""
	    # Python lists to store the training images/masks and test images.
	    x_train, y_train, x_test = [],[],[]

	    # Read and resize train images/masks. 
	    print('Loading and resizing train images and masks ...')
	    sys.stdout.flush()
	    for i, filename in tqdm.tqdm(enumerate(train_df['image_path']), total=len(train_df)):
	        img = read_image(train_df['image_path'].loc[i], target_size=image_size)
	        mask = read_mask(train_df['mask_dir'].loc[i], target_size=image_size)
	        x_train.append(img)
	        y_train.append(mask)

	    # Read and resize test images. 
	    print('Loading and resizing test images ...')
	    sys.stdout.flush()
	    for i, filename in tqdm.tqdm(enumerate(test_df['image_path']), total=len(test_df)):
	        img = read_image(test_df['image_path'].loc[i], target_size=image_size)
	        x_test.append(img)

	    # Transform lists into 4-dim numpy arrays.
	    x_train = np.array(x_train)
	    y_train = np.expand_dims(np.array(y_train), axis=4)
	    x_test = np.array(x_test)

	    print('x_train.shape: {} of dtype {}'.format(x_train.shape, x_train.dtype))
	    print('y_train.shape: {} of dtype {}'.format(y_train.shape, x_train.dtype))
	    print('x_test.shape: {} of dtype {}'.format(x_test.shape, x_test.dtype))
	    
	    return x_train, y_train, x_test


	train_df = read_train_data_properties(TRAIN_DIR, IMG_DIR_NAME, MASK_DIR_NAME)
	test_df = read_test_data_properties(TEST_DIR, IMG_DIR_NAME)

	df = pd.DataFrame([[x] for x in zip(train_df['img_height'], train_df['img_width'])])

	df = pd.DataFrame([[x] for x in zip(test_df['img_height'], test_df['img_width'])])

	x_train, y_train, x_test = load_raw_data()

	# Collection of methods for basic data manipulation like normalizing, inverting, 
	# color transformation and generating new images/masks

	def normalize_imgs(data):
	    """Normalize images."""
	    return normalize(data, type_=1)

	def normalize_masks(data):
	    """Normalize masks."""
	    return normalize(data, type_=1)
	    
	def normalize(data, type_=1): 
	    """Normalize data."""
	    if type_==0:
	        # Convert pixel values from [0:255] to [0:1] by global factor
	        data = data.astype(np.float32) / data.max()
	    if type_==1:
	        # Convert pixel values from [0:255] to [0:1] by local factor
	        div = data.max(axis=tuple(np.arange(1,len(data.shape))), keepdims=True) 
	        div[div < 0.01*data.mean()] = 1. # protect against too small pixel intensities
	        data = data.astype(np.float32)/div
	    if type_==2:
	        # Standardisation of each image 
	        data = data.astype(np.float32) / data.max() 
	        mean = data.mean(axis=tuple(np.arange(1,len(data.shape))), keepdims=True) 
	        std = data.std(axis=tuple(np.arange(1,len(data.shape))), keepdims=True) 
	        data = (data-mean)/std

	    return data

	def trsf_proba_to_binary(y_data):
	    """Transform propabilities into binary values 0 or 1."""  
	    return np.greater(y_data,.5).astype(np.uint8)

	def invert_imgs(imgs, cutoff=.5):
	    '''Invert image if mean value is greater than cutoff.'''
	    imgs = np.array(list(map(lambda x: 1.-x if np.mean(x)>cutoff else x, imgs)))
	    return normalize_imgs(imgs)

	def imgs_to_grayscale(imgs):
	    '''Transform RGB images into grayscale spectrum.''' 
	    if imgs.shape[3]==3:
	        imgs = normalize_imgs(np.expand_dims(np.mean(imgs, axis=3), axis=3))
	    return imgs

	def generate_images(imgs, seed=None):
	    """Generate new images."""
	    # Transformations.
	    image_generator = keras.preprocessing.image.ImageDataGenerator(
	        rotation_range = 90., width_shift_range = 0.02 , height_shift_range = 0.02,
	        zoom_range = 0.10, horizontal_flip=True, vertical_flip=True)
	    
	    # Generate new set of images
	    imgs = image_generator.flow(imgs, np.zeros(len(imgs)), batch_size=len(imgs),
	                                shuffle = False, seed=seed).next()    
	    return imgs[0]

	def generate_images_and_masks(imgs, masks):
	    """Generate new images and masks."""
	    seed = np.random.randint(10000) 
	    imgs = generate_images(imgs, seed=seed)
	    masks = trsf_proba_to_binary(generate_images(masks, seed=seed))
	    return imgs, masks

	def preprocess_raw_data(x_train, y_train, x_test, grayscale=False, invert=False):
	    """Preprocessing of images and masks."""
	    # Normalize images and masks
	    x_train = normalize_imgs(x_train)
	    y_train = trsf_proba_to_binary(normalize_masks(y_train))
	    x_test = normalize_imgs(x_test)
	    #print('Images normalized.')
	 
	    if grayscale:
	        # Remove color and transform images into grayscale spectrum.
	        x_train = imgs_to_grayscale(x_train)
	        x_test = imgs_to_grayscale(x_test)
	        #print('Images transformed into grayscale spectrum.')

	    if invert:
	        # Invert images, such that each image has a dark background.
	        x_train = invert_imgs(x_train)
	        x_test = invert_imgs(x_test)
	        #print('Images inverted to remove light backgrounds.')

	    return x_train, y_train, x_test

	x_train, y_train, x_test = preprocess_raw_data(x_train, y_train, x_test, invert=True)

	""" Collection of methods to compute the score.

	1. We start with a true and predicted mask, corresponding to one train image.

	2. The true mask is segmented into different objects. Here lies a main source 
	of error. Overlapping or touching nuclei are not separated but are labeled as 
	one object. This means that the target mask can contain less objects than 
	those that have been originally identified by humans.

	3. In the same manner the predicted mask is segmented into different objects.

	4. We compute all intersections between the objects of the true and predicted 
	masks. Starting with the largest intersection area we assign true objects to 
	predicted ones, until there are no true/pred objects left that overlap. 
	We then compute for each true/pred object pair their corresponding intersection 
	over union (iou) ratio. 

	5. Given some threshold t we count the object pairs that have an iou > t, which
	yields the number of true positives: tp(t). True objects that have no partner are 
	counted as false positives: fp(t). Likewise, predicted objects without a counterpart
	a counted as false negatives: fn(t).

	6. Now, we compute the precision tp(t)/(tp(t)+fp(t)+fn(t)) for t=0.5,0.55,0.60,...,0.95
	and take the mean value as the final precision (score).
	"""

	def get_labeled_mask(mask, cutoff=.5):
	    """Object segmentation by labeling the mask."""
	    mask = mask.reshape(mask.shape[0], mask.shape[1])
	    lab_mask = skimage.morphology.label(mask > cutoff) 
	    
	    # Keep only objects that are large enough.
	    (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
	    if (mask_sizes < min_object_size).any():
	        mask_labels = mask_labels[mask_sizes < min_object_size]
	        for n in mask_labels:
	            lab_mask[lab_mask == n] = 0
	        lab_mask = skimage.morphology.label(lab_mask > cutoff) 
	    
	    return lab_mask  

	def get_iou(y_true_labeled, y_pred_labeled):
	    """Compute non-zero intersections over unions."""
	    # Array of different objects and occupied area.
	    (true_labels, true_areas) = np.unique(y_true_labeled, return_counts=True)
	    (pred_labels, pred_areas) = np.unique(y_pred_labeled, return_counts=True)

	    # Number of different labels.
	    n_true_labels = len(true_labels)
	    n_pred_labels = len(pred_labels)

	    # Each mask has at least one identified object.
	    if (n_true_labels > 1) and (n_pred_labels > 1):
	        
	        # Compute all intersections between the objects.
	        all_intersections = np.zeros((n_true_labels, n_pred_labels))
	        for i in range(y_true_labeled.shape[0]):
	            for j in range(y_true_labeled.shape[1]):
	                m = y_true_labeled[i,j]
	                n = y_pred_labeled[i,j]
	                all_intersections[m,n] += 1 

	        # Assign predicted to true background.
	        assigned = [[0,0]]
	        tmp = all_intersections.copy()
	        tmp[0,:] = -1
	        tmp[:,0] = -1

	        # Assign predicted to true objects if they have any overlap.
	        for i in range(1, np.min([n_true_labels, n_pred_labels])):
	            mn = list(np.unravel_index(np.argmax(tmp), (n_true_labels, n_pred_labels)))
	            if all_intersections[mn[0], mn[1]] > 0:
	                assigned.append(mn)
	            tmp[mn[0],:] = -1
	            tmp[:,mn[1]] = -1
	        assigned = np.array(assigned)

	        # Intersections over unions.
	        intersection = np.array([all_intersections[m,n] for m,n in assigned])
	        union = np.array([(true_areas[m] + pred_areas[n] - all_intersections[m,n]) 
	                           for m,n in assigned])
	        iou = intersection / union

	        # Remove background.
	        iou = iou[1:]
	        assigned = assigned[1:]
	        true_labels = true_labels[1:]
	        pred_labels = pred_labels[1:]

	        # Labels that are not assigned.
	        true_not_assigned = np.setdiff1d(true_labels, assigned[:,0])
	        pred_not_assigned = np.setdiff1d(pred_labels, assigned[:,1])
	        
	    else:
	        # in case that no object is identified in one of the masks
	        iou = np.array([])
	        assigned = np.array([])
	        true_labels = true_labels[1:]
	        pred_labels = pred_labels[1:]
	        true_not_assigned = true_labels
	        pred_not_assigned = pred_labels
	        
	    # Returning parameters.
	    params = {'iou': iou, 'assigned': assigned, 'true_not_assigned': true_not_assigned,
	             'pred_not_assigned': pred_not_assigned, 'true_labels': true_labels,
	             'pred_labels': pred_labels}
	    return params

	def get_score_summary(y_true, y_pred):
	    """Compute the score for a single sample including a detailed summary."""
	    
	    y_true_labeled = get_labeled_mask(y_true)  
	    y_pred_labeled = get_labeled_mask(y_pred)  
	    
	    params = get_iou(y_true_labeled, y_pred_labeled)
	    iou = params['iou']
	    assigned = params['assigned']
	    true_not_assigned = params['true_not_assigned']
	    pred_not_assigned = params['pred_not_assigned']
	    true_labels = params['true_labels']
	    pred_labels = params['pred_labels']
	    n_true_labels = len(true_labels)
	    n_pred_labels = len(pred_labels)

	    summary = []
	    for i,threshold in enumerate(np.arange(0.5, 1.0, 0.05)):
	        tp = np.sum(iou > threshold)
	        fn = n_true_labels - tp
	        fp = n_pred_labels - tp
	        if (tp+fp+fn)>0: 
	            prec = tp/(tp+fp+fn)
	        else: 
	            prec = 0
	        summary.append([threshold, prec, tp, fp, fn])

	    summary = np.array(summary)
	    score = np.mean(summary[:,1]) # Final score.
	    params_dict = {'summary': summary, 'iou': iou, 'assigned': assigned, 
	                   'true_not_assigned': true_not_assigned, 
	                   'pred_not_assigned': pred_not_assigned, 'true_labels': true_labels,
	                   'pred_labels': pred_labels, 'y_true_labeled': y_true_labeled,
	                   'y_pred_labeled': y_pred_labeled}
	    
	    return score, params_dict

	def get_score(y_true, y_pred):
	    """Compute the score for a batch of samples."""
	    scores = []
	    for i in range(len(y_true)):
	        score,_ = get_score_summary(y_true[i], y_pred[i])
	        scores.append(score)
	    return np.array(scores)

	# Study how many objects in the masks can be identified. This is a limiting factor
	# for the overall performance.
	min_pixels_per_object = 1
	summary = []
	for n in range(len(y_train)):
	    img = y_train[n,:,:,0]
	    lab_img=get_labeled_mask(img)
	    img_labels, img_area = np.unique(lab_img, return_counts=True)
	    img_labels = img_labels[img_area>=min_pixels_per_object]
	    img_area = img_area[img_area>=min_pixels_per_object]
	    n_true_labels = train_df['num_masks'][n]
	    n_ident_labels = len(img_labels)
	    diff = np.abs(n_ident_labels-n_true_labels)
	    summary.append([n_true_labels, n_ident_labels, diff])

	sum_df = pd.DataFrame(summary, columns=(['true_objects', 'identified_objects', 'subtraction']))
	#sum_df.describe()

	class NeuralNetwork():
	    """ Implements a neural network.
	        
	        TensorFlow is used to implement the U-Net, which consists of convolutional
	        and max pooling layers. Input and output shapes coincide. Methods are
	        implemented to train the model, to save/load the complete session and to 
	        attach summaries for visualization with TensorBoard. 
	    """

	    def __init__(self, nn_name='tmp', nn_type='UNet', log_step=0.2, keep_prob=0.33, 
	                 mb_size=16, input_shape=[IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS], 
	                 output_shape=[IMG_HEIGHT,IMG_WIDTH,1]):
	        """Instance constructor."""
	        
	        # Tunable hyperparameters for training.
	        self.mb_size = mb_size       # Mini batch size
	        self.keep_prob = keep_prob   # Keeping probability with dropout regularization 
	        self.learn_rate_step = 3     # Step size in terms of epochs
	        self.learn_rate_alpha = 0.25 # Reduction of learn rate for each step 
	        self.learn_rate_0 = 0.001    # Starting learning rate 
	        self.dropout_proba = 0.15     # == 1-keep_probability
	        
	        # Set helper variables.
	        self.input_shape = input_shape
	        self.output_shape = output_shape
	        self.nn_type = nn_type                # Type of neural network
	        self.nn_name = nn_name                # Name of neural network
	        self.params = {}                      # For storing parameters
	        self.learn_rate_pos = 0                
	        self.learn_rate = self.learn_rate_0
	        self.index_in_epoch = 0 
	        self.epoch = 0. 
	        self.log_step = log_step              # Log results in terms of epochs
	        self.n_log_step = 0                   # Count number of mini batches  
	        self.train_on_augmented_data = False  # True = use augmented data 
	        self.use_tb_summary = False           # True = use TensorBoard summaries
	        self.use_tf_saver = False             # True = save the session
	        
	        # Parameters that should be stored.
	        self.params['train_loss']=[]
	        self.params['valid_loss']=[]
	        self.params['train_score']=[]
	        self.params['valid_score']=[]
	        
	    def get_learn_rate(self):
	        """Compute the current learning rate."""
	        if False:
	            # Fixed learnrate
	            learn_rate = self.learn_rate_0
	        else:
	            # Decreasing learnrate each step by factor 1-alpha
	            learn_rate = self.learn_rate_0*(1.-self.learn_rate_alpha)**self.learn_rate_pos
	        return learn_rate

	    def next_mini_batch(self):
	        """Get the next mini batch."""
	        start = self.index_in_epoch
	        self.index_in_epoch += self.mb_size           
	        self.epoch += self.mb_size/len(self.x_train)
	        
	        # At the start of the epoch.
	        if start == 0:
	            np.random.shuffle(self.perm_array) # Shuffle permutation array.
	   
	        # In case the current index is larger than one epoch.
	        if self.index_in_epoch > len(self.x_train):
	            self.index_in_epoch = 0
	            self.epoch -= self.mb_size/len(self.x_train) 
	            return self.next_mini_batch() # Recursive use of function.
	        
	        end = self.index_in_epoch
	        
	        # Original data.
	        x_tr = self.x_train[self.perm_array[start:end]]
	        y_tr = self.y_train[self.perm_array[start:end]]
	        
	        # Use augmented data.
	        if self.train_on_augmented_data:
	            x_tr, y_tr = generate_images_and_masks(x_tr, y_tr)
	            y_tr = trsf_proba_to_binary(y_tr)
	        
	        return x_tr, y_tr
	 
	    def weight_variable(self, shape, name=None):
	        """ Weight initialization """
	        #initializer = tf.truncated_normal(shape, stddev=0.1)
	        initializer = tf.contrib.layers.xavier_initializer()
	        #initializer = tf.contrib.layers.variance_scaling_initializer()
	        return tf.get_variable(name, shape=shape, initializer=initializer)

	    def bias_variable(self, shape, name=None):
	        """Bias initialization."""
	        #initializer = tf.constant(0.1, shape=shape)  
	        initializer = tf.contrib.layers.xavier_initializer()
	        #initializer = tf.contrib.layers.variance_scaling_initializer()
	        return tf.get_variable(name, shape=shape, initializer=initializer)
	     
	    def conv2d(self, x, W, name=None):
	        """ 2D convolution. """
	        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

	    def max_pool_2x2(self, x, name=None):
	        """ Max Pooling 2x2. """
	        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',
	                              name=name)
	    
	    def conv2d_transpose(self, x, filters, name=None):
	        """ Transposed 2d convolution. """
	        return tf.layers.conv2d_transpose(x, filters=filters, kernel_size=2, 
	                                          strides=2, padding='SAME') 
	    
	    def leaky_relu(self, z, name=None):
	        """Leaky ReLU."""
	        return tf.maximum(0.01 * z, z, name=name)
	    
	    def activation(self, x, name=None):
	        """ Activation function. """
	        a = tf.nn.elu(x, name=name)
	        #a = self.leaky_relu(x, name=name)
	        #a = tf.nn.relu(x, name=name)
	        return a 
	    
	    def loss_tensor(self):
	        """Loss tensor."""
	        if True:
	            # Dice loss based on Jaccard dice score coefficent.
	            axis=np.arange(1,len(self.output_shape)+1)
	            offset = 1e-5
	            corr = tf.reduce_sum(self.y_data_tf * self.y_pred_tf, axis=axis)
	            l2_pred = tf.reduce_sum(tf.square(self.y_pred_tf), axis=axis)
	            l2_true = tf.reduce_sum(tf.square(self.y_data_tf), axis=axis)
	            dice_coeff = (2. * corr + 1e-5) / (l2_true + l2_pred + 1e-5)
	            # Second version: 2-class variant of dice loss
	            #corr_inv = tf.reduce_sum((1.-self.y_data_tf) * (1.-self.y_pred_tf), axis=axis)
	            #l2_pred_inv = tf.reduce_sum(tf.square(1.-self.y_pred_tf), axis=axis)
	            #l2_true_inv = tf.reduce_sum(tf.square(1.-self.y_data_tf), axis=axis)
	            #dice_coeff = ((corr + offset) / (l2_true + l2_pred + offset) +
	            #             (corr_inv + offset) / (l2_pred_inv + l2_true_inv + offset))
	            loss = tf.subtract(1., tf.reduce_mean(dice_coeff))
	        if False:
	            # Sigmoid cross entropy. 
	            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
	                    labels=self.y_data_tf, logits=self.z_pred_tf))
	        return loss 
	    
	    def optimizer_tensor(self):
	        """Optimization tensor."""
	        # Adam Optimizer (adaptive moment estimation). 
	        optimizer = tf.train.AdamOptimizer(self.learn_rate_tf).minimize(
	                    self.loss_tf, name='train_step_tf')
	        return optimizer
	   
	    def batch_norm_layer(self, x, name=None):
	        """Batch normalization layer."""
	        if False:
	            layer = tf.layers.batch_normalization(x, training=self.training_tf, 
	                                                  momentum=0.9, name=name)
	        else: 
	            layer = x
	        return layer
	    
	    def dropout_layer(self, x, name=None):
	        """Dropout layer."""
	        if False:
	            layer = tf.layers.dropout(x, self.dropout_proba, training=self.training_tf,
	                                     name=name)
	        else:
	            layer = x
	        return layer

	    def num_of_weights(self,tensors):
	        """Compute the number of weights."""
	        sum_=0
	        for i in range(len(tensors)):
	            m = 1
	            for j in range(len(tensors[i].shape)):
	              m *= int(tensors[i].shape[j])
	            sum_+=m
	        return sum_

	            
	    def summary_variable(self, var, var_name):
	        """ Attach summaries to a tensor for TensorBoard visualization. """
	        with tf.name_scope(var_name):
	            mean = tf.reduce_mean(var)
	            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	            tf.summary.scalar('mean', mean)
	            tf.summary.scalar('stddev', stddev)
	            tf.summary.scalar('max', tf.reduce_max(var))
	            tf.summary.scalar('min', tf.reduce_min(var))
	            tf.summary.histogram('histogram', var)

	    def attach_summary(self, sess):
	        """ Attach TensorBoard summaries to certain tensors. """
	        self.use_tb_summary = True
	        
	        # Create summary tensors for TensorBoard.
	        tf.summary.scalar('loss_tf', self.loss_tf)

	        # Merge all summaries.
	        self.merged = tf.summary.merge_all()

	        # Initialize summary writer.
	        timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
	        filepath = os.path.join(os.getcwd(), LOGS_DIR_NAME, (self.nn_name+'_'+timestamp))
	        self.train_writer = tf.summary.FileWriter(os.path.join(filepath,'train'), sess.graph)
	        self.valid_writer = tf.summary.FileWriter(os.path.join(filepath,'valid'), sess.graph)

	    def attach_saver(self):
	        """ Initialize TensorFlow saver. """
	        with self.graph.as_default():
	            self.use_tf_saver = True
	            self.saver_tf = tf.train.Saver()

	    def save_model(self, sess):
	        """ Save parameters, tensors and summaries. """
	        if not os.path.isdir(os.path.join(CW_DIR, SAVES_DIR_NAME)):
	            os.mkdir(SAVES_DIR_NAME)
	        filepath = os.path.join(os.getcwd(), SAVES_DIR_NAME , self.nn_name+'_params.npy')
	        np.save(filepath, self.params) # save parameters of the network

	        # TensorFlow saver
	        if self.use_tf_saver:
	            filepath = os.path.join(os.getcwd(),  self.nn_name)
	            self.saver_tf.save(sess, filepath)

	        # TensorBoard summaries
	        if self.use_tb_summary:
	            self.train_writer.close()
	            self.valid_writer.close()
	        
	    def load_session_from_file(self, filename):
	        """ Load session from a file, restore the graph, and load the tensors. """
	        tf.reset_default_graph()
	        filepath = os.path.join(os.getcwd(), filename + '.meta')
	        saver = tf.train.import_meta_graph(filepath)
	        sess = tf.Session() # default session
	        saver.restore(sess, filename) # restore session
	        self.graph = tf.get_default_graph() # save default graph
	        self.load_parameters(filename) # load parameters
	        self.load_tensors(self.graph) # define relevant tensors as variables 
	        return sess
	    
	    def load_parameters(self, filename):
	        '''Load helper and tunable parameters.'''
	        filepath = os.path.join(os.getcwd(), SAVES_DIR_NAME, filename+'_params.npy')
	        self.params = np.load(filepath).item() # load parameters of network
	        
	        self.nn_name = filename
	        self.learn_rate = self.params['learn_rate']
	        self.learn_rate_0 = self.params['learn_rate_0']
	        self.learn_rate_step = self.params['learn_rate_step']
	        self.learn_rate_alpha = self.params['learn_rate_alpha']
	        self.learn_rate_pos = self.params['learn_rate_pos']
	        self.keep_prob = self.params['keep_prob']
	        self.epoch = self.params['epoch'] 
	        self.n_log_step = self.params['n_log_step']
	        self.log_step = self.params['log_step']
	        self.input_shape = self.params['input_shape']
	        self.output_shape = self.params['output_shape'] 
	        self.mb_size = self.params['mb_size']   
	        self.dropout_proba = self.params['dropout_proba']
	        
	        print('Parameters of the loaded neural network')
	        print('\tnn_name = {}, epoch = {:.2f}, mb_size = {}'.format(
	            self.nn_name, self.epoch, self.mb_size))
	        print('\tinput_shape = {}, output_shape = {}'.format(
	            self.input_shape, self.output_shape))
	        print('\tlearn_rate = {:.10f}, learn_rate_0 = {:.10f}, dropout_proba = {}'.format(
	            self.learn_rate, self.learn_rate_0, self.dropout_proba))
	        print('\tlearn_rate_step = {}, learn_rate_pos = {}, learn_rate_alpha = {}'.format(
	            self.learn_rate_step, self.learn_rate_pos, self.learn_rate_alpha))

	    def load_tensors(self, graph):
	        """ Load tensors from a graph. """
	        # Input tensors
	        self.x_data_tf = graph.get_tensor_by_name("x_data_tf:0")
	        self.y_data_tf = graph.get_tensor_by_name("y_data_tf:0")

	        # Tensors for training and prediction.
	        self.learn_rate_tf = graph.get_tensor_by_name("learn_rate_tf:0")
	        self.keep_prob_tf = graph.get_tensor_by_name("keep_prob_tf:0")
	        self.loss_tf = graph.get_tensor_by_name('loss_tf:0')
	        self.train_step_tf = graph.get_operation_by_name('train_step_tf')
	        self.z_pred_tf = graph.get_tensor_by_name('z_pred_tf:0')
	        self.y_pred_tf = graph.get_tensor_by_name("y_pred_tf:0")
	        self.training_tf = graph.get_tensor_by_name("training_tf:0")
	        self.extra_update_ops_tf = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	    def get_prediction(self, sess, x_data, keep_prob=1.0):
	        """ Prediction of the neural network graph. """
	        return sess.run(self.y_pred_tf, feed_dict={self.x_data_tf: x_data,
	                                                     self.keep_prob_tf: keep_prob})
	       
	    def get_loss(self, sess, x_data, y_data, keep_prob=1.0):
	        """ Compute the loss. """
	        return sess.run(self.loss_tf, feed_dict={self.x_data_tf: x_data, 
	                                                 self.y_data_tf: y_data,
	                                                 self.keep_prob_tf: keep_prob})

	mn = 'nn6_256_256_3' 
	u_net = NeuralNetwork()
	sess = u_net.load_session_from_file(mn)
	sess.close()
	train_loss = u_net.params['train_loss']
	valid_loss = u_net.params['valid_loss']
	train_score = u_net.params['train_score']
	valid_score = u_net.params['valid_score']

	# Collection of methods for run length encoding. 
	# For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included 
	# in the mask. The pixels are one-indexed and numbered from top to bottom, 
	# then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

	def rle_of_binary(x):
	    """ Run length encoding of a binary 2D array. """
	    dots = np.where(x.T.flatten() == 1)[0] # indices from top to down
	    run_lengths = []
	    prev = -2
	    for b in dots:
	        if (b>prev+1): run_lengths.extend((b + 1, 0))
	        run_lengths[-1] += 1
	        prev = b
	    return run_lengths

	def mask_to_rle(mask, cutoff=.5, min_object_size=1.):
	    """ Return run length encoding of mask. """
	    # segment image and label different objects
	    lab_mask = skimage.morphology.label(mask > cutoff)
	    
	    # Keep only objects that are large enough.
	    (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
	    if (mask_sizes < min_object_size).any():
	        mask_labels = mask_labels[mask_sizes < min_object_size]
	        for n in mask_labels:
	            lab_mask[lab_mask == n] = 0
	        lab_mask = skimage.morphology.label(lab_mask > cutoff) 
	        
	    # Loop over each object excluding the background labeled by 0.
	    for i in range(1, lab_mask.max() + 1):
	        yield rle_of_binary(lab_mask == i)
	        
	def rle_to_mask(rle, img_shape):
	    ''' Return mask from run length encoding.'''
	    mask_rec = np.zeros(img_shape).flatten()
	    for n in range(len(rle)):
	        for i in range(0,len(rle[n]),2):
	            for j in range(rle[n][i+1]): 
	                mask_rec[rle[n][i]-1+j] = 1
	    return mask_rec.reshape(img_shape[1], img_shape[0]).T

	# Load neural network, make prediction for test masks, resize predicted
	# masks to original image size and apply run length encoding for the
	# submission file. 

	# Load neural network and make prediction for masks.
	#nn_name = ['nn0_512_512_3']
	nn_name = ['nn6_256_256_3']

	# Soft voting majority.
	for i,mn in enumerate(nn_name):
	    u_net = NeuralNetwork()
	    sess = u_net.load_session_from_file(mn)
	    if i==0: 
	        y_test_pred_proba = u_net.get_prediction(sess, x_test)/len(nn_name)
	    else:
	        y_test_pred_proba += u_net.get_prediction(sess, x_test)/len(nn_name)
	    sess.close()

	y_test_pred = trsf_proba_to_binary(y_test_pred_proba)
	#print('y_test_pred.shape = {}'.format(y_test_pred.shape))

	# Resize predicted masks to original image size.
	y_test_pred_original_size = []
	for i in range(len(y_test_pred)):
	    res_mask = trsf_proba_to_binary(skimage.transform.resize(np.squeeze(y_test_pred[i]),
	        (test_df.loc[i,'img_height'], test_df.loc[i,'img_width']), 
	        mode='constant', preserve_range=True))
	    y_test_pred_original_size.append(res_mask)
	y_test_pred_original_size = np.array(y_test_pred_original_size)

	#print('y_test_pred_original_size.shape = {}'.format(y_test_pred_original_size.shape))
	   
	# Run length encoding of predicted test masks.
	test_pred_rle = []
	test_pred_ids = []
	for n, id_ in enumerate(test_df['img_id']):
	    min_object_size = 20*test_df.loc[n,'img_height']*test_df.loc[n,'img_width']/(256*256)
	    rle = list(mask_to_rle(y_test_pred_original_size[n], min_object_size=min_object_size))
	    test_pred_rle.extend(rle)
	    test_pred_ids.extend([id_]*len(rle))

	# Create submission file
	sub = pd.DataFrame()
	sub['ImageId'] = test_pred_ids
	sub['EncodedPixels'] = pd.Series(test_pred_rle).apply(lambda x: ' '.join(str(y) for y in x))
	sub.to_csv(str(ii)+'.csv', index=False)

