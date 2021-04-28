import tensorflow as tf
from keras.losses import binary_crossentropy
import keras

# BCE loss for the discriminator
def d_bce_loss(mask):
    def loss(y_true, y_pred):
        d_bce_loss = binary_crossentropy(y_true, y_pred)
        return d_bce_loss

    return loss

# trajLoss for the generator
def trajLoss(real_traj, gen_traj):
    def loss(y_true, y_pred):
        traj_length = keras.backend.sum(real_traj[4],axis=1)
        
        bce_loss = binary_crossentropy(y_true, y_pred)
        
        masked_latlon_full = keras.backend.sum(keras.backend.sum(tf.multiply(tf.multiply((gen_traj[0]-real_traj[0]),(gen_traj[0]-real_traj[0])),tf.concat([real_traj[4] for x in range(2)],axis=2)),axis=1),axis=1,keepdims=True)
        masked_latlon_mse = keras.backend.sum(tf.math.divide(masked_latlon_full,traj_length))
        
        ce_category = tf.nn.softmax_cross_entropy_with_logits_v2(gen_traj[1],real_traj[1])
        ce_day = tf.nn.softmax_cross_entropy_with_logits_v2(gen_traj[2],real_traj[2])
        ce_hour = tf.nn.softmax_cross_entropy_with_logits_v2(gen_traj[3],real_traj[3])
        
        ce_category_masked = tf.multiply(ce_category,keras.backend.sum(real_traj[4],axis=2))
        ce_day_masked = tf.multiply(ce_day,keras.backend.sum(real_traj[4],axis=2))
        ce_hour_masked = tf.multiply(ce_hour,keras.backend.sum(real_traj[4],axis=2))
        
        ce_category_mean = keras.backend.sum(tf.math.divide(ce_category_masked,traj_length))
        ce_day_mean = keras.backend.sum(tf.math.divide(ce_day_masked,traj_length))
        ce_hour_mean = keras.backend.sum(tf.math.divide(ce_hour_masked,traj_length))
        
        p_bce = 1
        p_latlon = 10
        p_cat = 1
        p_day = 1
        p_hour = 1
        
        return bce_loss*p_bce + masked_latlon_mse*p_latlon + ce_category_mean*p_cat + ce_day_mean*p_day + ce_hour_mean*p_hour

    return loss