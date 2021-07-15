import tensorflow as tf
import numpy as np
from functions.utils import gamma_to_theta, tf_diff_axis1
from functions.compute_logLik import compute_logLik

def predict(model, y, bl = None, x = None, x_im = None, multilabel = False, combined_loss = False):
    if(combined_loss):
        y_img = y[0]
        y_pat = y[1]
    if(bl is None):
        if(combined_loss):
            bl = np.ones((y_img.shape[0], 1), dtype = "float32")
        else:
            bl = np.ones((y.shape[0], 1), dtype = "float32")
    if((model.nn_x == None) & (model.nn_im == None)):
        gamma = model.model.predict({"bl_in": bl})
        beta = 0
        eta = 0
        beta_w = 0
    if((model.nn_x != None) & (model.nn_im == None)):
        gamma, beta = model.model.predict({"bl_in": bl, "x_in": x})
        eta = 0
        beta_w = model.model.get_layer('x_out').get_weights()
    if((model.nn_x == None) & (model.nn_im != None)):
        gamma, eta = model.model.predict({"bl_in": bl, "im_in": x_im})
        beta = 0
        beta_w = 0
    if((model.nn_x != None) & (model.nn_im != None)):
        gamma, beta, eta = model.model.predict({"bl_in": bl, "x_in": x, "nn_im": x_im})
        beta_w = model.model.get_layer('x_out').get_weights()
    
    if(combined_loss):
        # image: do not consider the patient tabular output
        beta_img = 0
        eta_img = 0
        gamma_img = gamma[0]
        gamma_img = tf.reshape(gamma_img, shape = (tf.math.multiply(gamma_img.shape[0], gamma_img.shape[1]), 1))
        theta_img = gamma_to_theta(gamma_img)
        probs_img = model.distr.cdf(theta_img - beta_img - eta_img)
        dens_img = tf_diff_axis1(probs_img)
        cls_img = tf.math.argmax(dens_img, axis = 1)
        nll_img = compute_logLik(model, gamma_img, beta = beta_img, eta = eta_img, y = y_img, multilabel = True)
        # patient
        gamma_pat = gamma[1]
        theta_pat = gamma_to_theta(gamma_pat)
        probs_pat = model.distr.cdf(theta_pat - beta - eta)
        dens_pat = tf_diff_axis1(probs_pat)
        cls_pat = tf.math.argmax(dens_pat, axis = 1)
        nll_pat = compute_logLik(model, gamma_pat, beta, eta, y_pat, multilabel = False)
        nll = nll_img + nll_pat
        return {"cdf_img": probs_img.numpy(), "pdf_img": dens_img.numpy(), "response_img": cls_img.numpy(), 
                "negLogLik_img": nll_img.numpy(), "theta_img": np.delete(theta_img.numpy(), [0, theta_img.shape[1]-1], 1), 
                "cdf_pat": probs_pat.numpy(), "pdf_pat": dens_pat.numpy(), "response_pat": cls_pat.numpy(), 
                "negLogLik_pat": nll_pat.numpy(), "theta_pat": np.delete(theta_pat.numpy(), [0, theta_pat.shape[1]-1], 1),
                "negLogLik": nll.numpy(), "beta": beta, "eta": eta, "beta_w": beta_w} 
    else:
        if(multilabel):
            gamma = tf.reshape(gamma, shape = (tf.math.multiply(gamma.shape[0], gamma.shape[1]), 1))
        theta = gamma_to_theta(gamma)
        probs = model.distr.cdf(theta - beta - eta)
        dens = tf_diff_axis1(probs)
        cls = tf.math.argmax(dens, axis = 1) # predicted class
        nll = compute_logLik(model, gamma, beta, eta, y, multilabel)
        return {"cdf": probs.numpy(), "pdf": dens.numpy(), "response": cls.numpy(), "negLogLik": nll.numpy(), 
                "theta": np.delete(theta.numpy(), [0, theta.shape[1]-1], 1), "beta": beta, "eta": eta, "beta_w": beta_w}