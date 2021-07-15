import matplotlib.pyplot as plt
import numpy as np

def plot_results(dat, train_loss, test_loss, train_acc = None, test_acc = None):
    if(train_acc is not None):
        plt.plot(dat.loc[:,train_acc], 'blue')
        plt.plot(dat.loc[:,test_acc], 'magenta')
        plt.ylim(0, 1)
        plt.title('Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'valid'], loc = 'lower right')
        plt.show()
    plt.plot(dat.loc[:,train_loss], 'blue')
    plt.plot(dat.loc[:,test_loss], 'magenta')
    plt.ylim(np.min([np.min(dat.loc[:,test_loss]), np.min(dat.loc[:,train_loss])])-0.1, 
             np.max([np.max(dat.loc[:,test_loss]), np.max(dat.loc[:,train_loss])])+0.1)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc = 'lower right')
    plt.show()
    if(train_acc is not None):
        print("Max. validation accuracy: ", np.max(dat.loc[:,test_acc]))
        print('In epoch: ', np.where(dat.loc[:,test_acc] == np.max(dat.loc[:,test_acc])))
    print("Min. validation loss: ", np.min(dat.loc[:,test_loss]))
    print('In epoch: ', np.where(dat.loc[:,test_loss] == np.min(dat.loc[:,test_loss])))