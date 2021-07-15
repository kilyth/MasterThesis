import tensorflow as tf
import numpy as np
from functions.apply_gradient import apply_gradient
from functions.methods import predict
from functions.compute_logLik import compute_logLik
from functions.oversample_indices import oversample_indices


def fit_ontram(model, y_train, x_train = None, x_train_im = None, x_test = None, x_test_im = None, y_test = None,
               epochs = 10, batch_size = 1, optimizer = tf.keras.optimizers.Adam(), output_dir = None, balance_batches = False,
               augment_batch = None, multilabel = False, combined_loss = False, model_selection = None):
    
    # define the input to theta
    if(model.response_varying):
        x_train_bl = x_train_im
        if(y_test is not None):
            x_test_bl = x_test_im
    else:
        if(combined_loss):
            x_train_bl = np.ones((y_train[0].shape[0], 1), dtype = 'float32')
        else:
            x_train_bl = np.ones((y_train.shape[0], 1), dtype = 'float32')
        if(y_test is not None):
            if(combined_loss):
                x_test_bl = np.ones((y_test[0].shape[0], 1), dtype = 'float32')
            else:
                x_test_bl = np.ones((y_test.shape[0], 1), dtype = 'float32')
        

    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    if(combined_loss):
        train_loss_img_results = []
        train_loss_pat_results = []
        train_accuracy_img_results = []
        train_accuracy_pat_results = []
        test_loss_img_results = []
        test_loss_pat_results = []
        test_accuracy_img_results = []
        test_accuracy_pat_results = []
    
    if(combined_loss):
        n = y_train[0].shape[0]
    else:
        n = y_train.shape[0]
        
    if(balance_batches):
        if(combined_loss):
            idx_reps = oversample_indices(y_train[1])
        else:
            idx_reps = oversample_indices(y_train)
        batch_idx = idx_reps
    else:
        batch_idx = np.arange(n)
       
    
    for epoch in range(epochs):
        np.random.shuffle(batch_idx)
        batch_tmp = 0
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_loss_avg_test = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Accuracy()
        epoch_accuracy_test = tf.keras.metrics.Accuracy()
        if(combined_loss):
            epoch_loss_img_avg = tf.keras.metrics.Mean()
            epoch_loss_img_avg_test = tf.keras.metrics.Mean()
            epoch_accuracy_img = tf.keras.metrics.Accuracy()  
            epoch_accuracy_test_img = tf.keras.metrics.Accuracy()  
            epoch_loss_pat_avg = tf.keras.metrics.Mean()
            epoch_loss_pat_avg_test = tf.keras.metrics.Mean()
            epoch_accuracy_pat = tf.keras.metrics.Accuracy()
            epoch_accuracy_test_pat = tf.keras.metrics.Accuracy()  
        
        for i in range(int(n/batch_size)):
            x_batch = None
            x_batch_im = None
            x_batch_bl = x_train_bl[batch_idx[batch_tmp:(batch_tmp + batch_size)]]
            if(combined_loss):
                y_batch_img = y_train[0][batch_idx[batch_tmp:(batch_tmp + batch_size)]]
                y_batch_pat = y_train[1][batch_idx[batch_tmp:(batch_tmp + batch_size)]]
                y_batch = [y_batch_img, y_batch_pat]
            else:
                y_batch = y_train[batch_idx[batch_tmp:(batch_tmp + batch_size)]]
            if(model.response_varying and (augment_batch is not None)):
                x_batch_bl = augment_batch(x_batch_bl)
            if(model.nn_x != None):
                x_batch = x_train[batch_idx[batch_tmp:(batch_tmp + batch_size)]]
            if(model.nn_im != None):
                x_batch_im = x_train_im[batch_idx[batch_tmp:(batch_tmp + batch_size)]]
                if(augment_batch is not None):
                    x_batch_im = augment_batch(x_batch_im)
            if(combined_loss):
                loss_value, loss_value_img, loss_value_pat, grads = apply_gradient(model, bl = x_batch_bl, x = x_batch, 
                                                                                   x_im = x_batch_im, y = y_batch, 
                                                                                   multilabel = multilabel, 
                                                                                   combined_loss = combined_loss)
            else:
                loss_value, grads = apply_gradient(model, bl = x_batch_bl, x = x_batch, x_im = x_batch_im, y = y_batch, 
                                                   multilabel = multilabel, combined_loss = combined_loss)
            optimizer.apply_gradients(zip(grads, model.model.trainable_variables))
            
            # Track progress: 
            # loss
            epoch_loss_avg.update_state(loss_value)
            if(combined_loss):
                epoch_loss_img_avg.update_state(loss_value_img)
                epoch_loss_pat_avg.update_state(loss_value_pat)
            out_ = predict(model, y = y_batch, bl = x_batch_bl, x = x_batch, x_im = x_batch_im, 
                           multilabel = multilabel, combined_loss = combined_loss)
            # accuracy
            if(combined_loss):
                y_batch_pred_img = tf.reshape(y_batch[0], shape = tf.math.multiply(y_batch[0].shape[0], 
                                                                                   y_batch[0].shape[1]))
                y_batch_pred_pat = y_batch[1]
                epoch_accuracy_img.update_state(out_['response_img'], y_batch_pred_img)
                epoch_accuracy_pat.update_state(out_['response_pat'], np.argmax(y_batch_pred_pat, axis=1))
            else:
                if(multilabel):
                    y_batch_pred = tf.reshape(y_batch, shape = tf.math.multiply(y_batch.shape[0], y_batch.shape[1]))
                    epoch_accuracy.update_state(out_['response'], y_batch_pred)
                else:
                    epoch_accuracy.update_state(out_['response'], np.argmax(y_batch, axis=1))
            
            batch_tmp += batch_size
        
        # End training epoch: summarize results
        if(combined_loss):
            train_loss_results.append(epoch_loss_avg.result().numpy())
            train_loss_img_results.append(epoch_loss_img_avg.result().numpy())
            train_loss_pat_results.append(epoch_loss_pat_avg.result().numpy())
            train_accuracy_img_results.append(epoch_accuracy_img.result().numpy())
            train_accuracy_pat_results.append(epoch_accuracy_pat.result().numpy())
        else:
            train_loss_results.append(epoch_loss_avg.result().numpy())
            train_accuracy_results.append(epoch_accuracy.result().numpy())
            

        # Track progress on validation data:
        if(y_test is not None):
            if(combined_loss):
                n_test = y_test[0].shape[0]
            else:
                n_test = y_test.shape[0]
            batch_idx_test = np.arange(n_test)
            np.random.shuffle(batch_idx_test)
            batch_tmp = 0
            
            for i in range(int(n_test/batch_size)):
                x_batch_test = None
                x_batch_im_test = None
                x_batch_bl_test = x_test_bl[batch_idx_test[batch_tmp:(batch_tmp + batch_size)]]
                if(combined_loss):
                    y_batch_test_img = y_test[0][batch_idx_test[batch_tmp:(batch_tmp + batch_size)]]
                    y_batch_test_pat = y_test[1][batch_idx_test[batch_tmp:(batch_tmp + batch_size)]]
                    y_batch_test = [y_batch_test_img, y_batch_test_pat]
                else:
                    y_batch_test = y_test[batch_idx_test[batch_tmp:(batch_tmp + batch_size)]]
                if(model.nn_x != None):
                    x_batch_test = x_test[batch_idx_test[batch_tmp:(batch_tmp + batch_size)]]
                if(model.nn_im != None):
                    x_batch_im_test = x_test_im[batch_idx_test[batch_tmp:(batch_tmp + batch_size)]]
                test_out_ = predict(model, y = y_batch_test, bl = x_batch_bl_test, x = x_batch_test, x_im = x_batch_im_test, 
                                    multilabel = multilabel, combined_loss = combined_loss)
                loss_value_test = test_out_['negLogLik']
                if(combined_loss):
                    loss_value_img_test = test_out_['negLogLik_img']
                    loss_value_pat_test = test_out_['negLogLik_pat']
                
                # Track progress:
                # loss
                epoch_loss_avg_test.update_state(loss_value_test)
                if(combined_loss):
                    epoch_loss_img_avg_test.update_state(loss_value_img_test)
                    epoch_loss_pat_avg_test.update_state(loss_value_pat_test)
                # accuracy
                if(combined_loss):
                    y_batch_test_pred_img = tf.reshape(y_batch_test[0], shape = tf.math.multiply(y_batch_test[0].shape[0],
                                                                                                 y_batch_test[0].shape[1]))
                    y_batch_test_pred_pat = y_batch_test[1]
                    epoch_accuracy_test_img.update_state(test_out_['response_img'], y_batch_test_pred_img)
                    epoch_accuracy_test_pat.update_state(test_out_['response_pat'], np.argmax(y_batch_test_pred_pat, axis=1))
                else:
                    if(multilabel):
                        y_batch_test_pred = tf.reshape(y_batch_test, shape = tf.math.multiply(y_batch_test.shape[0], 
                                                                                              y_batch_test.shape[1]))
                        epoch_accuracy_test.update_state(test_out_['response'], y_batch_test_pred)
                    else:
                        epoch_accuracy_test.update_state(test_out_['response'], np.argmax(y_batch_test, axis = 1))
                
                batch_tmp += batch_size
           
            
            # End test epoch: summarize results
            if(combined_loss):
                test_loss_results.append(epoch_loss_avg_test.result().numpy())
                test_loss_img_results.append(epoch_loss_img_avg_test.result().numpy())
                test_loss_pat_results.append(epoch_loss_pat_avg_test.result().numpy())
                test_accuracy_img_results.append(epoch_accuracy_test_img.result().numpy())
                test_accuracy_pat_results.append(epoch_accuracy_test_pat.result().numpy())
            else:
                test_loss_results.append(epoch_loss_avg_test.result().numpy())
                test_accuracy_results.append(epoch_accuracy_test.result().numpy())
            
            if(epoch < 20): # save always the first 20 epochs, smaller epochs are not expected to have learned sth.
                prev_test_loss = 1000
            else:
                prev_test_loss = np.min(test_loss_results[:-1]) 
            
            if(epoch > 1):
                prev_train_loss = np.min(train_loss_results[:-1])
            else:
                prev_train_loss = 1000
            
            if((model_selection == "test") or (model_selection is None)):
                if((test_loss_results[epoch] <= prev_test_loss) and (output_dir is not None)):
                    model.model.save_weights('{}model-{:03d}.hdf5'.format(output_dir, epoch))
            
            if(model_selection == "train"):
                if((train_loss_results[epoch] <= prev_train_loss) and (output_dir is not None)):# save best train model
                    model.model.save_weights('{}model-{:03d}.hdf5'.format(output_dir, epoch))
            
            if(model_selection == "last"):
                if((epoch == range(epochs)[-1]) and (output_dir is not None)): # always save last model:
                    model.model.save_weights('{}model-{:03d}.hdf5'.format(output_dir, epoch))
            
            # Print output
            if(combined_loss):
                print('Epoch {:03d}: Train loss: {:.4f}, Train loss (img): {:.4f}, Train loss (pat): {:.4f}, Train accuracy (img): {:.2%}, Train accuracy (pat): {:.2%}, Test loss: {:.4f}, Test loss (img): {:.4f}, Test loss (pat): {:.4f}, Test accuracy (img): {:.2%}, Test accuracy (pat): {:.2%}'.format(epoch, epoch_loss_avg.result(), epoch_loss_img_avg.result(), epoch_loss_pat_avg.result(), epoch_accuracy_img.result(), epoch_accuracy_pat.result(), epoch_loss_avg_test.result(), epoch_loss_img_avg_test.result(), epoch_loss_pat_avg_test.result(), epoch_accuracy_test_img.result(), epoch_accuracy_test_pat.result()))
            else:
                print('Epoch {:03d}: Train loss: {:.4f}, Train accuracy: {:.2%}, Test loss: {:.4f}, Test accuracy: {:.2%}'.format(epoch, epoch_loss_avg.result(), epoch_accuracy.result(), epoch_loss_avg_test.result(), epoch_accuracy_test.result()))
            
        else: # if there is no test data provided
            if(combined_loss):
                print('Epoch {:03d}: Train loss: {:.4f}, Train loss (img): {:.4f}, Train loss (pat): {:.4f}, Train accuracy (img): {:.2%}, Train accuracy (pat): {:.2%}'.format(epoch, epoch_loss_avg.result(), epoch_loss_img_avg.result(), epoch_loss_pat_avg.result(), epoch_accuracy_img.result(), epoch_accuracy_pat.result()))
            else:
                print('Epoch {:03d}: Train loss: {:.4f}, Train accuracy: {:.2%}'.format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
    if(combined_loss):
        return {'train_loss': train_loss_results, 'train_loss_img': train_loss_img_results, 
                'train_loss_pat': train_loss_pat_results,
                'train_acc_img': train_accuracy_img_results, 'train_acc_pat': train_accuracy_pat_results, 
                'test_loss': test_loss_results, 'test_loss_img': test_loss_img_results, 
                'test_loss_pat': test_loss_pat_results,
                'test_acc_img': test_accuracy_img_results, 'test_acc_pat': test_accuracy_pat_results} 
    else:
        return {'train_loss': train_loss_results, 'train_acc': train_accuracy_results, 
                'test_loss': test_loss_results, 'test_acc': test_accuracy_results} 