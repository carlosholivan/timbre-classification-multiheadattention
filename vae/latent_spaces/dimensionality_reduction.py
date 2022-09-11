from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import umap.plot


def pca_reduction(df, model_name, encoder='encoder1', projection='2d'):
    
    df_pca = df.copy(deep=True)
        
    n_components = int(projection.split('d')[0])

    if model_name == 'TimeConv2D_cut':
        if encoder == 'encoder1':
            means = df_pca["means_1"].tolist()
        elif encoder == 'encoder2':
            means = df_pca["means_2"].tolist()
    else:
        means = df_pca["means"].tolist()
    pca = PCA(n_components=n_components)
    pca_embedding = pca.fit_transform(means)

    #add tsne coordinates columns in pandas dataframe
    for i in range(0, n_components):
        df_pca.insert(i+2, str(i), pca_embedding[:, i].tolist(), True)

    return df_pca


def tsne(df, dims, reduction='all', projection='2d', perplexity=30):
    
    df_tsne = df[['dynamic', 'epochs', 'family', 'file', 'instrument', 'latent_dims', 'note', 'technique']].copy()
        
    n_components = int(projection.split('d')[0])

    if reduction == 'all':
        new_list = df['means'].tolist()
    elif reduction == 'pca':
        list_cols = []
        for i in df.columns:
                try:
                    j = int(i)
                    list_cols.append(i)
                except:
                    continue
        # if all is True we do not specify dimension, but we reduce all the dimensions
        list = []
        new_list = []
        prev = 0
        for index, row in df.iterrows():
            for j in list_cols:
                list.append(df[j][index])
            new_list.append(list)
            list = []
    elif reduction == 'dims':
        list = []
        new_list = []
        for index, row in df.iterrows():
            list.append(df[str(dims[0])][index])
            list.append(df[str(dims[1])][index])
            if n_components == 3:
                list.append(df[str(dims[2])][index])
            new_list.append(list)
            list = []
    tsne_embedded = TSNE(n_components=n_components, perplexity=perplexity).fit_transform(new_list) #perform t-sne

    #add tsne coordinates columns in pandas dataframe
    for i in range(0, n_components):
        df_tsne.insert(i+2, str(i), tsne_embedded[:, i].tolist(), True)
    return df_tsne


def u_map(df, projection='2d', n_neighbors=15, min_dist=0.1):
    df_umap = df.copy(deep=True)
    
    if projection == '2d':
        n_components = 2
    if projection == '3d':
        n_components = 3
        
    fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    means = df["means"] #select means columns of the dataframe
    umap_embedded = fit.fit_transform(means.tolist())
    mapper = umap.UMAP().fit(means.tolist())
    
    #add umaps coordinates columns in pandas dataframe
    if 'x' and 'y' not in df_umap:
        df_umap.insert(3, "x", umap_embedded[:, 0].tolist(), True)
        df_umap.insert(4, "y", umap_embedded[:, 1].tolist(), True)
        if n_components == 3:
            df_umap.insert(5, "z", umap_embedded[:, 2].tolist(), True)
    return df_umap, mapper