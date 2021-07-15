import tensorflow as tf
from functions.compute_logLik import compute_logLik

def apply_gradient(model, bl, x, x_im, y, multilabel, combined_loss):
    with tf.GradientTape() as tape:
        if((model.nn_x == None) & (model.nn_im == None)):
            gamma = model.model({"bl_in": bl})
            beta = 0
            eta = 0
        if((model.nn_x != None) & (model.nn_im == None)):
            gamma, beta = model.model({"bl_in": bl, "x_in": x})
            eta = 0
        if((model.nn_x == None) & (model.nn_im != None)):
            gamma, eta = model.model({"bl_in": bl, "im_in": x_im})
            beta = 0
        if((model.nn_x != None) & (model.nn_im != None)):
            gamma, beta, eta = model.model({"bl_in": bl, "x_in": x, "im_in": x_im})
        if(combined_loss):
            gamma_img = gamma[0]
            gamma_pat = gamma[1]
            y_img = y[0]
            y_pat = y[1]
            loss_value_img = compute_logLik(model, gamma_img, beta = 0, eta = 0, y = y_img, multilabel = True)
            loss_value_pat = compute_logLik(model, gamma_pat, beta, eta, y_pat, multilabel = False)
            loss_value = loss_value_img + loss_value_pat
            return loss_value, loss_value_img, loss_value_pat, tape.gradient(loss_value, model.model.trainable_variables)
        else:
            loss_value = compute_logLik(model, gamma, beta, eta, y, multilabel)
            return loss_value, tape.gradient(loss_value, model.model.trainable_variables)
 