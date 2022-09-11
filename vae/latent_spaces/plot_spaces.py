
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import os

# Our modules
from vae import configs
from vae.latent_spaces import dimensionality_reduction

def plot_reduced_latent_space(df, 
                             projection, 
                             projection_plot,
                             dims,
                             method, 
                             perplexity=30,
                             n_neighbors=15, 
                             min_dist=0.1,
                             plt_plot=False,
                             plot_html=True,
                             save_html=False,
                             save_png=False,
                             plot_connectivity=False,
                             filename='',
                             label='instrument'):

    if method == 'tsne':
        df_embeed = dimensionality_reduction.tsne(df, projection)

    elif method == 'umap':
        df_embeed, mapper = dimensionality_reduction.u_map(df, projection, n_neighbors, min_dist)
        
    elif method == 'pca':
        df_embeed = dimensionality_reduction.pca_reduction(df, projection)
        
    else:
        raise ValueError('Bad method argument.')

    if plt_plot:
        plot_latent_space_matplotlib(df_embeed, projection_plot, label)
    if plot_html:
        plot_latent_space_html(df_embeed, projection_plot, dims, label, save_html, save_png, plot_connectivity, filename)

    return


def plot_latent_space_html(df, projection_plot, dims, label, save_html, save_png, plot_connectivity, filename):

    title = 'Latent space of VAE trained {} epochs'.format(int(df['epochs'][0]))
    
    color_discrete_map, hover_data = get_hover_data(label)
    
    if projection_plot == '2d':
        if plot_connectivity:
            fig1 = px.line(df, x=str(dims[0]), y=str(dims[1]))
            fig1.update_traces(line=dict(color = 'rgba(56,45,45,0.2)'), hoverinfo='skip')
            
            fig2 = px.scatter(df, x=str(dims[0]), y=str(dims[1]), color=label, hover_name=label, hover_data=hover_data,
                              color_discrete_map=color_discrete_map)
            fig = go.Figure(layout={'title': title,
                                    'template': 'plotly_dark',
                                    'width': 700,
                                    'height': 600},
                            data=fig1.data + fig2.data)
        else:
            fig = px.scatter(df, x=str(dims[0]), y=str(dims[1]), color=label, title=title, hover_name=label, hover_data=hover_data,
                         width=700, height=600, color_discrete_map=color_discrete_map)
        fig.update_traces(marker=dict(opacity=0.8))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_showgrid=False, yaxis_showgrid=False,
                          xaxis_zeroline=False, yaxis_zeroline=False)
                              

    elif projection_plot == '3d':
        if plot_connectivity:
            fig = px.scatter_3d(df, x=str(dims[0]), y=str(dims[1]), z=str(dims[2]), color=label, title=title,
                            width=600, height=600, color_discrete_map=color_discrete_map,)
        else:
            fig = px.scatter_3d(df, x=str(dims[0]), y=str(dims[1]), z=str(dims[2]), color=label, title=title,
                            width=600, height=600, color_discrete_map=color_discrete_map)
        fig.update_traces(marker=dict(size=3, opacity=0.5))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                          scene = dict(
                              xaxis_showspikes=False,
                              yaxis_showspikes=False,
                              zaxis_showspikes=False,
                              xaxis = dict(
                                  showgrid=False,
                                  showline=True,),
                              yaxis = dict(
                                  showgrid=False,
                                  showline=True,),
                              zaxis = dict(
                                  showgrid=False,
                                  showline=True,),))
        
    if save_html or save_png:
        if not os.path.exists(configs.PlotsConfig.PLOTS_PATH):
            os.mkdir(configs.PlotsConfig.PLOTS_PATH) 

        if save_html:
            fig.write_html(configs.PlotsConfig.PLOTS_PATH + '/' + filename + str(int(df['epochs'][0])) + 'epochs.html')
        if save_png:
            fig.write_image(configs.PlotsConfig.PLOTS_PATH + '/' + filename + str(int(df['epochs'][0])) + 'epochs.png')
        print(filename + str(int(df['epochs'][0])) + 'epochs.png', 'saved in', configs.PlotsConfig.PLOTS_PATH)
    fig.show()

    return


def plot_latent_space_matplotlib(df, projection='2d', label='instrument', save_png=False, name_fig='plot'):

    fig = plt.figure(figsize=(8, 8))
    title = 'Latent space of VAE trained {} epochs'.format(int(df['epochs'][0]))
    
    s = df.shape[0] // 40 # size of points in plot
    
    #groups = df.groupby(label)
    #for name, group in groups:
    if projection == '2d':
        #ax = fig.add_subplot(111)
        #ax.scatter(group["x"], group["y"], marker="o", s=s, label=name, alpha=0.5)
        
        g = sns.jointplot(data=df, x="x", y="y", hue=label)
        #g.plot_joint(sns.kdeplot, cmap="Blues", shade=True, shade_lowest=False, alpha=0.5)
        #g.set_axis_labels()
        #ax = plt.gca()
        #ax.legend()
        #ax.plot_marginals(sns.kdeplot, color='b', shade=True, alpha=0.2, legend=False)
    
    elif projection == '3d':
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(group["x"], group["y"], group["z"], marker="o", s=s, label=name, alpha=0.5)
    #ax.set_title(title)
    #ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1))

    if save_png:
        if not os.path.exists(configs.PlotsConfig.PLOTS_PATH):
            os.mkdir(configs.PlotsConfig.PLOTS_PATH)
        fig.write_image(configs.PlotsConfig.PLOTS_PATH + '/' + name_fig + str(trained_epochs) + 'epochs.png')
        print(name_fig + str(trained_epochs) + 'epochs.png', 'Plot saved in', configs.PlotsConfig.PLOTS_PATH)
    return


def explore_latent_vars(df, dim_1, dim_2, label="note"):
    
    df_new = df.copy(deep=True)

    #add umaps coordinates columns in pandas dataframe
    x, y = [], []
    for sample in range(len(df)):
        x.append(df["means"][sample][dim_1])
        y.append(df["means"][sample][dim_2])
        
    if 'x' and 'y' not in df_new:
        df_new.insert(3, "x", x, True)
        df_new.insert(4, "y", y, True)
        
    color_discrete_map, hover_data = get_hover_data(label)

    fig = px.scatter(df_new, x="x", y="y", color=label, hover_name=label, hover_data=hover_data, 
                    color_discrete_map=color_discrete_map,
                    width=700, height=600)
    fig.update_traces(marker=dict(opacity=0.5))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_showgrid=False, yaxis_showgrid=False,
                          xaxis_zeroline=False, yaxis_zeroline=False)
    fig.show()
    return

def get_hover_data(label):

    if label == 'instrument':
        color_discrete_map = configs.PlotsConfig.COLORS_INSTRUMENTS
        hover_data = ['note', 'dynamic', 'technique', 'family']
    elif label == 'note':
        color_discrete_map = configs.PlotsConfig.COLORS_NOTE
        hover_data = ['instrument', 'dynamic', 'technique', 'family']
    elif label == 'dynamic':
        color_discrete_map = configs.PlotsConfig.COLORS_DYNAMICS
        hover_data = ['instrument', 'note', 'technique', 'family']
    elif label == 'technique':
        color_discrete_map = configs.PlotsConfig.COLORS_TECHNIQUE
        hover_data = ['instrument', 'note', 'dynamic', 'family']
    elif label == 'family':
        color_discrete_map = configs.PlotsConfig.COLORS_TECHNIQUE
        hover_data = ['instrument', 'note', 'dynamic', 'technique']

    return color_discrete_map, hover_data