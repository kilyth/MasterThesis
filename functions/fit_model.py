import tensorflow as tf
from tensorflow import keras
import numpy as np
from functions import oversample_indices

loss_object = tf.keras.losses.CategoricalCrossentropy()

def loss(model, x, y, training):
    y_pred = model(x, training = training)
    return loss_object(y_true = y, y_pred = y_pred)

def apply_gradient(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training = True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
    
    
def fit_model(model, x_train, y_train, output_dir, x_test = None, y_test = None, batch_size = 1, epochs = 1, 
              optimizer = tf.keras.optimizers.Adam(), augmentation = True, augment_batch = None, balance = False):
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    
    n = y_train.shape[0]
    n_test = y_test.shape[0]
    if(balance):
        batch_idx = oversample_indices(y_train)
    else:
        batch_idx = np.arange(n)
    batch_idx_test = np.arange(n_test)
    
    for epoch in range(epochs):
        np.random.shuffle(batch_idx)
        batch_tmp = 0
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy() # ylabels not in one hot encoding format
        epoch_loss_avg_test = tf.keras.metrics.Mean()
        epoch_accuracy_test = tf.keras.metrics.SparseCategoricalAccuracy() # ylabels not in one hot encoding format
        
        for i in range(int(n/batch_size)):
            x_batch = x_train[batch_idx[batch_tmp:(batch_tmp + batch_size)]]
            y_batch = y_train[batch_idx[batch_tmp:(batch_tmp + batch_size)]]
            
            if(augmentation):
                x_batch = augment_batch(x_batch)
            loss_value, grads = apply_gradient(model, x_batch, y_batch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Sanity check: print(np.histogram(np.argmax(y_batch, axis = 1), bins = y_batch.shape[1]))
            # batches are balanced!
        
            # Track progress
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(np.argmax(y_batch, axis = 1), model(x_batch, training = True))
            
            batch_tmp += batch_size
    
        # End epoch
        train_loss_results.append(epoch_loss_avg.result().numpy())
        train_accuracy_results.append(epoch_accuracy.result().numpy())
    
        # Calculate test accuracy:
        if(y_test is not None):
            np.random.shuffle(batch_idx_test)
            batch_tmp = 0
            
            for i in range(int(n_test/batch_size)):
                x_batch_test = x_test[batch_idx_test[batch_tmp:(batch_tmp + batch_size)]]
                y_batch_test = y_test[batch_idx_test[batch_tmp:(batch_tmp + batch_size)]]
                loss_value_test = loss(model, x_batch_test, y_batch_test, training = False)
                
                # Track progress
                epoch_loss_avg_test.update_state(loss_value_test)
                epoch_accuracy_test.update_state(np.argmax(y_batch_test, axis = 1), model(x_batch_test, training = False))
                
                batch_tmp += batch_size
            
            # End test epoch
            test_loss_results.append(epoch_loss_avg_test.result().numpy())
            test_accuracy_results.append(epoch_accuracy_test.result().numpy())
            
            if(epoch == 0):
                prev_test_loss = 1000
            else:
                prev_test_loss = np.min(test_loss_results[:-1]) 
            if(test_loss_results[epoch] <= prev_test_loss):
#                 model.model.save_weights(output_dir + 'model-' + str(epoch) + '.hdf5')
                model.model.save_weights('{}model-{:03d}.hdf5'.format(output_dir, epoch))
            
            # Print output
            print("Epoch {:03d}: Train loss: {:.4f}, Train accuracy: {:.2%}, Test loss: {:.4f}, Test accuracy: {:.2%}".format(epoch,
                                                                                                                              epoch_loss_avg.result(), 
                                                                                                                              epoch_accuracy.result(),
                                                                                                                              epoch_loss_avg_test.result(),
                                                                                                                              epoch_accuracy_test.result()))
    else:
        print("Epoch {:03d}: Train loss: {:.4f}, Train accuracy: {:.2%}".format(epoch,
                                                                                epoch_loss_avg.result(), 
                                                                                epoch_accuracy.result()))
    return {"train_loss": train_loss_results, 'train_acc': train_accuracy_results, 'test_loss': test_loss_results, 'test_acc': test_accuracy_results} 