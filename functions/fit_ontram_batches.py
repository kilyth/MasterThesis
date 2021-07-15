import tensorflow as tf
import numpy as np
from functions.apply_gradient import apply_gradient
from functions.methods import predict
from functions.compute_logLik import compute_logLik
from functions.oversample_indices import oversample_indices


def fit_ontram_batches(model, y_train, x_train = None, x_train_im = None, x_test = None, x_test_im = None, y_test = None,
               epochs = 10, batch_size = 1, batch_size2 = 10, optimizer = tf.keras.optimizers.Adam(), output_dir = None, 
               balance_batches = False, augment_batch = None):
    
    # define the input to theta
    if(model.response_varying):
        x_train_bl = x_train_im
        if(y_test is not None):
            x_test_bl = x_test_im
    else:
        x_train_bl = np.ones((y_train.shape[0], 1), dtype = 'float32')
        if(y_test is not None):
            x_test_bl = np.ones((y_test.shape[0], 1), dtype = 'float32')
        

    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    
    n = y_train.shape[0]
    if(balance_batches):
        idx_reps = oversample_indices(y_train)
        batch_idx = idx_reps
    else:
        batch_idx = np.arange(n)
       
    # for batch training
    n_img = x_train_im.shape[3]
    n = batch_size2
    
    for epoch in range(epochs):
        np.random.shuffle(batch_idx)
        batch_tmp = 0
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Accuracy()
        epoch_loss_avg_test = tf.keras.metrics.Mean()
        epoch_accuracy_test = tf.keras.metrics.Accuracy()
        
        for i in range(int(n/batch_size)):
            x_batch = None
            x_batch_im = None
            for j in range(n_img-n+1):
                x_batch_bl = x_train_bl[batch_idx[batch_tmp:(batch_tmp + batch_size)],:,:,j:(j+n),:]
                if(model.response_varying and (augment_batch is not None)):
                    x_batch_bl = augment_batch(x_batch_bl)
                y_batch = y_train[batch_idx[batch_tmp:(batch_tmp + batch_size)]]
                if(model.nn_x != None):
                    x_batch = x_train[batch_idx[batch_tmp:(batch_tmp + batch_size)]]
                if(model.nn_im != None):
                    x_batch_im = x_train_im[batch_idx[batch_tmp:(batch_tmp + batch_size)],:,:,j:(j+n),:]
                    if(augment_batch is not None):
                        x_batch_im = augment_batch(x_batch_im)
                loss_value, grads = apply_gradient(model, bl = x_batch_bl, x = x_batch, x_im = x_batch_im, y = y_batch)
                optimizer.apply_gradients(zip(grads, model.model.trainable_variables))
                
                # Track progress
                epoch_loss_avg.update_state(loss_value)
                out_ = predict(model, y = y_batch, bl = x_batch_bl, x = x_batch, x_im = x_batch_im)
                epoch_accuracy.update_state(out_['response'], np.argmax(y_batch, axis=1))
                
            batch_tmp += batch_size
        
        # End epoch
        train_loss_results.append(epoch_loss_avg.result().numpy())
        train_accuracy_results.append(epoch_accuracy.result().numpy())
        
        # Calculate test accuracy:
        # Calculate test accuracy:
        if(y_test is not None):
            n_test = y_test.shape[0]
            batch_idx_test = np.arange(n_test)
            np.random.shuffle(batch_idx_test)
            batch_tmp = 0
            
            for i in range(int(n_test/batch_size)):
                x_batch_test = None
                x_batch_im_test = None
                for j in range(n_img-n+1):
                    x_batch_bl_test = x_test_bl[batch_idx_test[batch_tmp:(batch_tmp + batch_size)],:,:,j:(j+n),:]
                    y_batch_test = y_test[batch_idx_test[batch_tmp:(batch_tmp + batch_size)]]
                    if(model.nn_x != None):
                        x_batch_test = x_test[batch_idx_test[batch_tmp:(batch_tmp + batch_size)]]
                    if(model.nn_im != None):
                        x_batch_im_test = x_test_im[batch_idx_test[batch_tmp:(batch_tmp + batch_size)],:,:,j:(j+n),:]
                    test_out_ = predict(model, y = y_batch_test, bl = x_batch_bl_test, x = x_batch_test, 
                                        x_im = x_batch_im_test)
                    loss_value_test = test_out_['negLogLik']
                    
                    # Track progress
                    epoch_loss_avg_test.update_state(loss_value_test)
                    epoch_accuracy_test.update_state(test_out_['response'], np.argmax(y_batch_test, axis = 1))
                
                batch_tmp += batch_size
            
            # End test epoch
            test_loss_results.append(epoch_loss_avg_test.result().numpy())
            test_accuracy_results.append(epoch_accuracy_test.result().numpy())
            
            if(epoch == 0):
                prev_test_loss = 1000
            else:
                prev_test_loss = np.min(test_loss_results[:-1]) 
            if((test_loss_results[epoch] <= prev_test_loss) and (output_dir is not None)):
#                 model.model.save_weights(output_dir + 'model-' + str(epoch) + '.hdf5')
                model.model.save_weights('{}model-{:03d}.hdf5'.format(output_dir, epoch))
            
            # Print output
            print('Epoch {:03d}: Train loss: {:.4f}, Train accuracy: {:.2%}, Test loss: {:.4f}, Test accuracy: {:.2%}'.format(epoch,    epoch_loss_avg.result(),                                                                                                                  epoch_accuracy.result(),                                                                                                  epoch_loss_avg_test.result(),                                                                                                                epoch_accuracy_test.result()))
            
        else:
            print('Epoch {:03d}: Train loss: {:.4f}, Train accuracy: {:.2%}'.format(epoch,
                                                                                    epoch_loss_avg.result(), 
                                                                                    epoch_accuracy.result()))
    return {'train_loss': train_loss_results, 'train_acc': train_accuracy_results, 'test_loss': test_loss_results, 'test_acc': test_accuracy_results} 