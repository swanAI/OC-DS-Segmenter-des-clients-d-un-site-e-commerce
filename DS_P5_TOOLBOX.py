import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
from termcolor import colored
from IPython.display import display_html
from tabulate import tabulate
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import normaltest



#Fonction pour réduire la mémoire du df     
def reduce_memory_usage(df, verbose=True):
    """
    Réduire mémoire du dataframe  

    Parameters:
    -----------------
    df (dataframe) : DataFrame analysis
        
    Returns:
    -----------------
    Dataframe réduction memory
    """
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df



def display_dfs(dfs, gap=50, justify='center'):

    """
    Afficher plusieurs dataframe 

    Parameters:
    -----------------
    dfs (dictionnaire) : dictionnaire comprenant keys: nom du dataframe et values: Dataframe 
        
    Returns:
    -----------------
    plusieurs dataframe afficher à l'horizontal
    """

    html = ""
    for title, df in dfs.items():  
        df_html = df._repr_html_()
        cur_html = f'<div> <h3>{title}</h3> {df_html}</div>'
        html +=  cur_html
    html= f"""
    <div style="display:flex; gap:{gap}px; justify-content:{justify};">
        {html}
    </div>
    """
    display_html(html, raw=True)



    # describe pour  plusieurs dataframes dans dictionnaire
def describe_all_multi_dfs(dfs):

    """
    Afficher plusieurs tabulate avec les informations describe

    Parameters:
    -----------------
    dfs(dictionnaire) : dictionnaire comprenant keys: nom du dataframe et values: Dataframe 
        
    Returns:
    -----------------
    plusieurs tabulate
    """
    from tabulate import tabulate
    for keys, df in dfs.items():
          print(f"Le dataset : {colored(keys.upper(),'red')}")
          print(tabulate(df.describe(include='all'),headers='keys',tablefmt='pretty'))
          print('')
    
    
    # fonction detecte les doublons pour plusieurs dataframe dans un dictionnaire 
def doublons_check_multi_dfs(dfs):
     """
    Afficher les doublons de plusiseurs dataframes 

    Parameters:
    -----------------
    dfs (dictionnaire) : dictionnaire comprenant keys: nom du dataframe et values: Dataframe 
        
    Returns:
    -----------------
    series doublons 
    """
     for keys, df in dfs.items():
        print(f"Les doublons du df {colored(keys.upper(),'red')}: " ,len(df[df.duplicated()]))
        
    
    
    #Fontion pour detecter les NaN dans les colonnes pour plusieurs dataframes dans dictionnaire 
def NaN_columns_check_multi_dfs(dfs):

    """
    Afficher les informartions concernant les valeurs manquantes de plusiseurs dataframes 

    Parameters:
    -----------------
    dfs (dictionnaire) : dictionnaire comprenant keys: nom du dataframe et values: Dataframe 
        
    Returns:
    -----------------
    tabulates des informartions concernant les valeurs manquantes de plusiseurs dataframes 
    """
    for keys, df in dfs.items():
        print(f"Le dataset : {colored(keys.upper(),'red')}")
        print(f'Les dimensions du {keys} : {df.shape}')
        
        dtypes = df.dtypes
        missing_count = df.isnull().sum()
        value_counts = df.isnull().count()
        missing_pct = (missing_count/value_counts)*100
        missing_total = df.isna().sum().sum()
        df_missing = pd.DataFrame({'Count_NaN':missing_count, 'Pct_NaN':missing_pct, 'Types':dtypes, 'Total_NaN_in_dataset': missing_total})
        df_missing = df_missing.sort_values(by='Pct_NaN', ascending=False)
        
        print('Les valeurs manquantes pour chaques colonnes :')
        print('')
        print(tabulate(df_missing,headers='keys',tablefmt='pretty'))
        
        plt.style.use('ggplot')
        plt.figure(figsize=(14,10))
        plt.title(f'Le pourcentage de valeurs manquantes pour la colonne {keys}', size=20)
        plt.plot( df.isna().sum()/df.shape[0])
        plt.xticks(rotation = 90) 
        plt.show()
        print('')
        print('----------------------------------------------------------------------------------')
        print('')
    
    
def NaN_ROWS_df(df):

    """
    Afficher les informartions concernant les valeurs manquantes des rows

    Parameters:
    -----------------
    df (DataFrame) : Dataframe analysis
        
    Returns:
    -----------------
    tabulates les valeurs manquantes des rows
    """

    print(f"Le dataset : {colored(keys.upper(),'red')}")
      
    nb_rows = pd.DataFrame(df.T.isnull().count()).rename(columns={0:'Nb_rows'})
    missing_count_rows = pd.DataFrame(df.T.isnull().sum()).rename(columns={0:'Count_NaN'})
    pct_rows_NaN = pd.DataFrame((df.T.isnull().sum()/EdStatsData.T.isnull().count()*100)).rename(columns={0:'Pct_NaN_rows'})
      
    df_rows_NaN = pd.concat([missing_count_rows,nb_rows, pct_rows_NaN], axis=1)
    print(tabulate(df_rows_NaN,headers='keys',tablefmt='pretty'))
         
            
            
    # Graphique pour voir le pourcentage de valeurs manquantes pour les colonnes 
def plot_pourcentage_NaN_features(df):

     """
    Afficher les informartions concernant les valeurs manquantes des rows

    Parameters:
    -----------------
    df (DataFrame) : Dataframe analysis
        
    Returns:
    -----------------
    tabulates les valeurs manquantes des rows
    """
    
     plt.figure(figsize=(20,18))
     plt.title('Le pourcentage de valeurs manquantes pour les features', size=20)
     plt.plot((df.isna().sum()/df.shape[0]*100).sort_values(ascending=True))
     plt.xlabel('Features dataset', fontsize=18)
     plt.ylabel('Pourcentage NaN dans features', fontsize=18)
     plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
     plt.show()

     pct_dataset = pd.DataFrame((df.isna().sum()/df.shape[0]*100).sort_values(ascending=False))
     pct_dataset = pct_dataset.rename(columns={0:'Pct_NaN_colonne'})
     pct_dataset =pct_dataset.style.background_gradient(cmap='YlOrRd')
     return pct_dataset


import scipy.stats as stats
def diagnostic_plots(df, variable):

     """
    Afficher diagnostic_plots afin de vérifier diagnostic des distributions 
    voir les outliers et normalités

    Parameters:
    -----------------
    df (DataFrame) : Dataframe analysis
    variable (series) : Colonne du dataframe
        
    Returns:
    -----------------
    plot diagnostic et test normalité 
    """
    
     plt.figure(figsize=(16, 4))

     # histogram
     plt.subplot(1, 3, 1)
     sns.histplot(df[variable], bins=30)
     plt.title('Histogram')
 
     # Q-Q plot
     plt.subplot(1, 3, 2)
     stats.probplot(df[variable], dist="norm", plot=plt)
     plt.ylabel('Variable quantiles')
 
     # boxplot
     plt.subplot(1, 3, 3)
     sns.boxplot(y=df[variable])
     plt.title('Boxplot')
 
     print('Test Shapiro')
     data = df[variable].values
     stat, p = shapiro(data)
     print('stat=%.3f, p=%.3f' % (stat, p))
     if p > 0.05:
          print('Probablement Gaussien ')
     else:
          print('Probablement pas  Gaussien ')
             
     print('Test normaltest')        
     data = df[variable].values
     stat, p = normaltest(data)
     print('stat=%.3f, p=%.3f' % (stat, p))
     if p > 0.05:
         print('Probablement Gaussien ')
     else:
         print('Probablement pas  Gaussien ')
 
 
     plt.show()
 


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(Z, names):
    plt.figure(figsize=(25,10))
    plt.title('Hierarchical Clustering Dendrogram', size=50)
    plt.ylabel('Distance euclidienne ', size=20)
    plt.xlabel('Pays', size=20)
    dendrogram(
        Z,
        labels = names,
        #orientation = "left"
        leaf_rotation=90)
    plt.show()


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)

  # For reproducibility



range_n_clusters = [2,3,4,5,6,7,8,9,10]


def silhouette_score_plot(range_n_clusters, data_scaled):

  """
    Plot silhouette score visualisation 

    Parameters:
    -----------------
    range_n_clusters (liste) : liste éléments du nombre de clusters à tester
    data_scaled (numpy.ndarray): numpy.ndarray (les données scaled et compression)
        
    Returns:
    -----------------
    Plot silhouette score visualisation 
    """
  
  for n_clusters in range_n_clusters:
    
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(df_segmentation) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=39)
    cluster_labels = clusterer.fit_predict(data_scaled)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data_scaled, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(pca2d[:,0], pca2d[:,1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()


# Fonction silouette pour clusters
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.spatial.distance import pdist
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_silhouettes(X_scaled, clusters):
    # calcul du coefficient de silhouette total
    # et des coefficients de silhouette associés à chaque point de chaque cluster
    silhouette_avg = silhouette_score(X_scaled, clusters)
    sample_silhouette_values = silhouette_samples(X_scaled, clusters)
    n_cluster = max(clusters+1)

    # instanciation de la figure pour tracer les silhouette
    fig, ax = plt.subplots()

    ax.set_title(f"coefficients de silhouette pour {n_cluster-1} clusters")
    ax.set_xlabel("coefficient de silhouette")
    ax.set_ylabel("cluster")
    ax.set_ylim([0, len(X_scaled) + 10*n_cluster])
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", label="coefficient de silhouette moyen")

    y_lower = 10

    for i in range(1, n_cluster):
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # on color les coefficients de silhouette
        color = cm.nipy_spectral(float(i+1) / n_cluster)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, ith_cluster_silhouette_values,
            facecolor=color, edgecolor=color, alpha=0.7,
        )

        # on annote le diagramme avec le numéro et la population du cluster
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        ax.text(0.1, y_lower + 0.5 * size_cluster_i, str(size_cluster_i))

        y_lower = y_upper + 10

    ax.set_yticks([])
    ax.legend()
    
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.close()
    
    return fig


# Fonction pour afficher les graphique (cercle corrélation , eboulie, )

def plot_sortie_acf( y_acf, y_len, pacf=False):
    "représentation de la sortie ACF"
    if pacf:
        y_acf = y_acf[1:]
    plt.figure(figsize=(14,6))
    plt.bar(range(len(y_acf)), y_acf, width = 0.1)
    plt.xlabel('lag')
    plt.ylabel('ACF')
    plt.axhline(y=0, color='black')
    plt.axhline(y=-1.96/np.sqrt(y_len), color='b', linestyle='--', linewidth=0.8)
    plt.axhline(y=1.96/np.sqrt(y_len), color='b', linestyle='--', linewidth=0.8)
    plt.ylim(-1, 1)
    plt.show()
    return



import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=True):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(10, 10))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1),size=20)
            plt.show(block=False)

def display_scree_plot(pca):
    """
    Afficher Eboulis des valeurs propres pour voir le pourcentage de variance expliquée par chaque composante

    Parameters:
    -----------------
    pca (sklearn.decomposition._pca.PCA) :  Calcul des composantes principales
        
    Returns:
    -----------------
    plot Eboulis des valeurs propres
    """
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)







def pca_factorial_plane(pca, component_plane, groups, columns, labels, centroids, savefig):
    
    import matplotlib.colors as colors
    
    # component_plane transformation
    component_plane = np.array(component_plane)
    component_plane -= 1
    
    # matrix preparation
    matrix = pca.fit_transform(columns)
    
    # figure preparation
    fig = plt.figure(figsize=(7,7))
    max_dimension = 3
    plt.xlim([-max_dimension, max_dimension])
    plt.ylim([-max_dimension, max_dimension])
    
    # components axes
    plt.plot([-100, 100], [0, 0], color='grey', ls='--', alpha=0.3)
    plt.plot([0, 0], [-100, 100], color='grey', ls='--', alpha=0.3)
    
    cmap = colors.LinearSegmentedColormap.from_list('', ['lightcoral','mediumseagreen'])
    plt.scatter(matrix[:, component_plane[0]], matrix[:, component_plane[1]], c=groups, cmap=cmap)  
    
    # centroids visual
    if centroids:
        
        # centroids computation
        centroids = np.concatenate([matrix,np.array(groups, dtype=int).reshape(-1, 1)], axis=1)
        centroids = pd.DataFrame(centroids).groupby(3).median()
        
        # red dots
        plt.scatter(centroids.iloc[:, component_plane[0]], centroids.iloc[:, component_plane[1]], 
                    c='black', alpha=0.7)
        
        # text
        [plt.text(x=centroids.iloc[i, component_plane[0]], 
                  y=centroids.iloc[i, component_plane[1]], 
                  s=int(centroids.index[i]), 
                  c='black',
                  fontsize=20) for i in np.arange(0, centroids.shape[0])];
                    
    # naming dots
    if labels:
        [plt.text(x=matrix[i, component_plane[0]], 
                  y=matrix[i, component_plane[1]], 
                  s=columns.index[i], 
                  c='r',
                  fontsize=8, alpha=0.7) for i in np.arange(0, columns.shape[0])];

    # nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(component_plane[0]+1, int(round(100*pca.explained_variance_ratio_[component_plane[0]],0))))
    plt.ylabel('F{} ({}%)'.format(component_plane[1]+1, int(round(100*pca.explained_variance_ratio_[component_plane[1]],0))))
    
    # dumping graph
    if savefig:
        plt.tight_layout()
        plt.savefig('P6_05_factorial_plane_'+ str(component_plane[0] + 1) + '_' 
                    + str(component_plane[1] + 1) + '.png')
    plt.show()



# Library of Functions for the OpenClassrooms Multivariate Exploratory Data Analysis Course

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
import seaborn as sns


palette = sns.color_palette("bright", 10)

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """Display correlation circles, one for each factorial plane"""

    # For each factorial plane
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # Initialise the matplotlib figure
            fig, ax = plt.subplots(figsize=(10,10))

            # Determine the limits of the chart
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 20 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # Add arrows
            # If there are more than 30 arrows, we do not display the triangle at the end
            if pcs.shape[1] < 20 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (see the doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # Display variable names
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='10', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.4)
            
            # Display circle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # Define the limits of the chart
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # Display grid lines
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Correlation Circle (PC{} and PC{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    '''Display a scatter plot on a factorial plane, one for each factorial plane'''

    # For each factorial plane
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # Initialise the matplotlib figure      
            fig = plt.figure(figsize=(7,6))
        
            # Display the points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # Display grid lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection of points (on PC{} and PC{})".format(d1+1, d2+1))
            #plt.show(block=False)
   
def display_scree_plot(pca):
    '''Display a scree plot for the pca'''

    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Number of principal components")
    plt.ylabel("Percentage explained variance")
    plt.title("Scree plot")
    plt.show(block=False)

def append_class(df, class_name, feature, thresholds, names):
    '''Append a new class feature named 'class_name' based on a threshold split of 'feature'.  Threshold values are in 'thresholds' and class names are in 'names'.'''
    
    n = pd.cut(df[feature], bins = thresholds, labels=names)
    df[class_name] = n

def plot_dendrogram(Z, names, figsize=(10,25)):
    '''Plot a dendrogram to illustrate hierarchical clustering'''

    plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    #plt.show()

def addAlpha(colour, alpha):
    '''Add an alpha to the RGB colour'''
    
    return (colour[0],colour[1],colour[2],alpha)

def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df
    
    Parameters:
    -----------------
    df (dataframe) : DataFrame centroïdes des groupes et leurs coordonnées dans chacune des dimensions
    num_clusters (int) : Le nombre de clusters 
        
    Returns:
    -----------------
    plot display_parallel_coordinates
    
    '''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster==i])
    
    # Create the plot
    fig = plt.figure(figsize=(12, 15))
    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):    
        plt.subplot(num_clusters, 1, i+1)
        for j,c in enumerate(cluster_points): 
            if i!= j:
                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])
        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.5)])

        # Stagger the axes
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)        


def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)   






























# il faut mettre les observations ou clusters en set_index()