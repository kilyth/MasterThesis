from matplotlib import pyplot as plt

def plot_slices(X, pat, plane = ["axial", "coronal", "sagittal"], modality = "perfusion"):
    # total figure size (including all subplots)
    nslices = X.shape[2]
    ncols = 6
    nrows = int(nslices / ncols)
    base_size = 2
    aspect_ratio = 0.5
    # ax_aspect = pat.pixel_spacing_y.values[0]/pat.pixel_spacing_x.values[0]
    # cor_aspect = pat.pixel_spacing_z.values[0]/pat.pixel_spacing_x.values[0]
    # sag_aspect = pat.pixel_spacing_y.values[0]/pat.pixel_spacing_z.values[0]

    figsize = (ncols*3, nrows*3)
    fig = plt.figure(figsize = figsize)
    
    if plane == "axial":
        fig_all = []
        for i in range(1, ncols*nrows):
            if modality != "perfusion":
                img = X[:,:,i,0]
                fig_all.append(fig.add_subplot(nrows, ncols, i))
                plt.imshow(img, cmap='gray')
            else:
                img = X[:,:,i,:]
                fig_all.append(fig.add_subplot(nrows, ncols, i))
                plt.imshow(img)
        plt.show()
    if plane == "coronal":
        # which images do we want to consider
        idx = int(X.shape[1]/(ncols*nrows))
        idx = list(range(0, X.shape[1], idx))
        fig_all = []
        for i in range(1, ncols*nrows):
            if modality != "perfusion":
                img = X[idx[i],:,:,0]
                fig_all.append(fig.add_subplot(nrows, ncols, i))
                plt.imshow(img, aspect = "auto", cmap='gray')
            else:
                img = X[idx[i],:,:,:]
                fig_all.append(fig.add_subplot(nrows, ncols, i))
                plt.imshow(img, aspect = "auto")
        plt.show()
    if plane == "sagittal":
        # which images do we want to consider
        idx = int(X.shape[0]/(ncols*nrows))
        idx = list(range(0, X.shape[0], idx))
        fig_all = []
        for i in range(1, ncols*nrows):
            if modality != "perfusion":
                img = X[:,idx[i],:,0]
                fig_all.append(fig.add_subplot(nrows, ncols, i))
                plt.imshow(img, aspect = "auto", cmap='gray')
            else:
                img = X[:,idx[i],:,:]
                fig_all.append(fig.add_subplot(nrows, ncols, i))
                plt.imshow(img, aspect = "auto")
        plt.show()