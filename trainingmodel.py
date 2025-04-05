
# import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import warnings,os,json
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize']=[15,5]
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import AgglomerativeClustering, MeanShift, OPTICS, SpectralClustering, DBSCAN, KMeans, kmeans_plusplus
# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, davies_bouldin_score, adjusted_rand_score, silhouette_score,calinski_harabasz_score, mean_squared_error
# from scipy.spatial import distance
# from scipy.stats.mstats import winsorize

#loading the data
df = pd.read_csv('static/data/Data_final.csv')
codebook_path = "static/data/code_book.csv"
df_codebook = pd.read_csv(codebook_path)
scoring_path = "static/data/scoring.csv"
df_scoring = pd.read_csv(scoring_path)
# create a df_reversed from df_scoring where df_scoring['direction']=='reversed'
df_reversed = df_scoring[df_scoring['direction'] == 'reversed']

def rename_cols(df):
  df.columns = df.columns.str.replace(' ', '_')
  df.columns = df.columns.str.replace('(', '_')
  df.columns = df.columns.str.replace(')', '')
  df.columns = df.columns.str.replace('-', '_')
  df.columns = df.columns.str.replace('/', '_')
  df.columns = df.columns.str.replace('?', '')
  df.columns = df.columns.str.replace('!', '')
  return df

def clean_data(df):
  if df.isnull().sum().any():
    df.dropna(inplace=True)
  if df.duplicated().sum().any():
    df.drop_duplicates(inplace=True)
  return df

# Change float64 to int
def convert_int(df): # Change float64 to int but not its caption
    for col in df.select_dtypes('float64'):
        if col!='Career':
            df[col] = df[col].astype('int64')
    # for col in df.select_dtypes(include='float64'):
    #     # print(col)
    #     if df[col].dtypes=='float64':
    #         df[col]=df[col].astype(int)
    return df

def assign_datasets():
  x_train=df[['O_score','C_score','E_score','A_score','N_score']]
  y_train=df[['Career']]
  return x_train,y_train

# def visualize_data(df):
#     y = df['Career']
#     x = df.drop('Career', axis=1)
#     x.boxplot(rot=40)
#     # Explanation of  the graph info for each column
#     info = '''The boxplot shows the distribution of each numerical feature in the dataset.
#     Each box represents the interquartile range (IQR), which contains the middle 50% of the data.
#     The line inside the box represents the median value.
#     The whiskers extend to 1.5 times the IQR from the box.
#     Any points outside the whiskers are considered outliers.
#     '''
#     print(info)

def handle_outliers(df):
  for col in df.select_dtypes(include='number'):
    q1,q3=df[col].quantile([0.25,0.75])
    iqr=q3-q1
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    df[col]=np.where(df[col]<lower_bound,lower_bound,df[col])
    df[col]=np.where(df[col]>upper_bound,upper_bound,df[col])
    # df[col]=df[col].clip(lower=df[col].quantile(0.01),upper=df[col].quantile(0.99))
    df.boxplot(rot=40)
  return df

def handle_outliers1(x_train):
  # Outlier Detection using Isolation Forest
  iso = IsolationForest(contamination=0.1)
  yhat = iso.fit_predict(x_train)
  mask = yhat != -1
  cleaned_data = x_train[mask]
  cleaned_data.info()
  return cleaned_data


from sklearn.decomposition import PCA
# import hdbscan
def cluster_analysis(cleaned_data):
  n_clusters=12

  # Apply PCA for visualization
  pca = PCA(n_components=2)
  components = pca.fit_transform(cleaned_data)

  # Function to evaluate and plot clustering results
  def evaluate_clustering(model, data, model_name):
      clusters = model.fit_predict(data)
      print(clusters)
      silhouette = silhouette_score(data, clusters)
      db_index = davies_bouldin_score(data, clusters)
      print(f'{model_name} - Silhouette Score: {silhouette}, Davies-Bouldin Index: {db_index}, ')
      print(f'Number of clusters: {len(np.unique(clusters))}')

      # Plotting the results
      # plt.figure()
      # plt.scatter(components[:, 0], components[:, 1], c=clusters)
      # plt.title(f'PCA Visualization of {model_name}')
      # plt.show()

  # DBSCAN Clustering
  dbscan = DBSCAN(eps=0.2, min_samples=5)
  evaluate_clustering(dbscan, cleaned_data, 'DBSCAN')

  # Mean Shift Clustering
  ms = MeanShift()
  evaluate_clustering(ms, cleaned_data, 'Mean Shift')

  # # HDBSCAN Clustering
  # hdbscan = HDBSCAN(min_cluster_size=5)  # Adjust parameters as needed
  # evaluate_clustering(hdbscan, cleaned_data, 'HDBSCAN')

  # OPTICS Clustering
  optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)  # Adjust parameters as needed
  evaluate_clustering(optics, cleaned_data, 'OPTICS')

  # Spectral Clustering
  spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')  # Adjust parameters as needed
  evaluate_clustering(spectral, cleaned_data, 'Spectral Clustering')

  # Gaussian Mixture Models (GMM)
  gmm = GaussianMixture(n_components=5)  # Adjust number of components as needed
  evaluate_clustering(gmm, cleaned_data, 'Gaussian Mixture Models')

  # Hierarchical Clustering
  hc = AgglomerativeClustering(n_clusters=n_clusters)  # Adjust number of clusters as needed
  evaluate_clustering(hc, cleaned_data, 'Hierarchical Clustering')

  # Gaussian Mixture Models (GMM)
  gmm = GaussianMixture(n_components=5)  # Adjust number of components as needed
  evaluate_clustering(gmm, cleaned_data, 'Gaussian Mixture Models')

  #Kmeans Clustering
  kmeans = KMeans(n_clusters=n_clusters)  # Adjust number of clusters as needed
  evaluate_clustering(kmeans, cleaned_data, 'Kmeans')

  kmeans = KMeans(n_clusters=n_clusters).fit(cleaned_data)
  initial_centroids = kmeans.cluster_centers_
  kmeans_plusplus = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1)
  evaluate_clustering(kmeans_plusplus, cleaned_data, 'Kmeans-plusplus')

def kmeans_plusplus(data):
  # Create a KMeans model with 12 clusters and KMeans++ initialization
  kmeans = KMeans(n_clusters=12, init='k-means++', random_state=42)
  # Fit the model to the cleaned_data
  kmeans.fit(data)
  # Get the cluster labels for each data point
  labels = kmeans.labels_
  # Get the cluster centroids
  initial_centroids = kmeans.cluster_centers_
  # Print the inertia (sum of squared distances of samples to their closest cluster center)
  print("Inertia:", kmeans.inertia_)
  kmeans_plus1 = KMeans(n_clusters=12, init=initial_centroids, n_init=1)
  kmeans_plus1.fit_predict(data)
  centroids1=kmeans_plus1.cluster_centers_
  print("Inertia:", kmeans_plus1.inertia_)
  print("Cluster_centers:", kmeans_plus1.cluster_centers_)
  print("Labels:", kmeans_plus1.labels_)

  # # Calculate distances from each data point to its closest cluster centroid
  # distances = distance.cdist(data, centroids1, 'euclidean')
  # closest_centroid_distances = distances.min(axis=1)

  # # Identify outliers (e.g., data points with distances > 2 standard deviations)
  # outlier_threshold = 2 * np.std(closest_centroid_distances)
  # outliers = data[closest_centroid_distances > outlier_threshold]

  # Remove outliers from the dataset
  # data[closest_centroid_distances > outlier_threshold] = outlier_threshold
  dbscan = DBSCAN(eps=0.2, min_samples=4)
  dbscan.fit(data)
  kmeans_plus2 = KMeans(n_clusters=12, init=centroids1, n_init=1)
  kmeans_plus2.fit_predict(data)
  centroids2=kmeans_plus2.cluster_centers_
  print("Inertia:", kmeans_plus2.inertia_)
  print("Cluster_centers:", kmeans_plus2.cluster_centers_)
  print("Labels:", kmeans_plus2.labels_)
  return kmeans_plus2

def clustering_info(trained_model):
    cluster=trained_model[['Career','Cluster']]
    #     cluster_info=cluster.value_counts().values
    # print(cluster_info)
    cluster_0 = cluster[cluster['Cluster'] == 0]
    career_0 = set(cluster_0['Career'].unique())
    cluster_1 = cluster[cluster['Cluster'] == 1]
    career_1 = set(cluster_1['Career'].unique())
    cluster_2 = cluster[cluster['Cluster'] == 2]
    career_2 = set(cluster_2['Career'].unique())
    cluster_3 = cluster[cluster['Cluster'] == 3]
    career_3 = set(cluster_3['Career'].unique())
    cluster_4 = cluster[cluster['Cluster'] == 4]
    career_4 = set(cluster_4['Career'].unique())

    # print(df.groupby('Cluster').mean())
    return career_0,career_1,career_2,career_3,career_4

def printing_cluster(cluster):
    for i in range(5):
        print(f'Cluster {i}')
        print(cluster[cluster['Cluster']==i]['Career'].value_counts())
        # pd.crosstab(cluster['Career'], cluster['clusters'])

# Group Names and Columns
def group_cols(data):
  EXT = [column for column in data.columns if column.startswith('EXT')]
  NES = [column for column in data.columns if column.startswith('NES')]
  CON = [column for column in data.columns if column.startswith('CON')]
  AGR = [column for column in data.columns if column.startswith('AGR')]
  OPN = [column for column in data.columns if column.startswith('OPN')]
  return EXT,NES,CON,AGR,OPN

def calculate_reverse(user_data):
    df_data_transposed = user_data.transpose()
    df_data_transposed = df_data_transposed.reset_index()
    df_data_transposed = df_data_transposed.rename(columns={'index': 'item'})
    #inner join of user_data and codebook on item
    user_data = pd.merge(df_data_transposed, df_codebook, on='item', how='inner')
    # update the reversed values
    rows = user_data.shape[0]
    for row in range(rows):
        if user_data.iloc[row, 4] == 'reversed':
            user_data.iloc[row, 1] = 6 - int(user_data.iloc[row, 1])
    user_data.drop(['direction', 'trait', 'facet'], axis=1, inplace=True)
    # take transpose of user_data with values of item as column names
    user_data = user_data.set_index('item').transpose()
    return user_data

def calculate_scores(user_data,OPN,CON,EXT,AGR,NES):
    df_input = pd.DataFrame()
    df_temp=pd.DataFrame()
    df_input['O_score'] = O_score = round(user_data[OPN].mean(axis=1), 2)
    df_input['C_score'] = C_score = round(user_data[CON].mean(axis=1), 2)
    df_input['E_score'] = E_score = round(user_data[EXT].mean(axis=1), 2)
    df_input['A_score'] = A_score = round(user_data[AGR].mean(axis=1), 2)
    df_input['N_score'] = N_score = round(user_data[NES].mean(axis=1), 2)
    df_input['Numerical_Aptitude'] = C_score * 0.5 + O_score * 0.2 #Conscientiousness (for diligence) and Openness (for analytical thinking).
    df_input['Spatial_Aptitude'] = O_score * 0.5 + E_score * 0.2 #Openness (for creativity) and Conscientiousness (for attention to detail).
    df_input['Perceptual_Aptitude'] = E_score * 0.4 + N_score * 0.3 #Openness (for sensory experiences) and Neuroticism (for attention to detail).
    df_input['Abstract_Reasoning'] = O_score * 0.5 + C_score * 0.3 #Openness (for innovative thinking) and Conscientiousness (for systematic approach)
    df_input['Verbal_Reasoning'] = N_score * 0.5 + A_score * 0.2   #Openness (for language learning) and Extraversion (for communication skills).
    # Save scores to a JSON file
    file_path = os.path.join('static','scores.json')
    df_temp=df_input.copy()*10
    scores = df_temp.to_dict(orient='records')[0]

    print(scores)
    with open(file_path, 'w') as json_file:
        json.dump(scores, json_file)
    print("scored updated at trainingmodel")
    return df_input

# def predict_cluster(df_input,model):
#     # Predict the cluster for the user input (assuming 'model' is the trained KMeans model)
#     predicted_cluster = model.predict(df_input)[0]
#     print(f"Predicted Cluster: {predicted_cluster}")
#     # Suggest careers based on the predicted cluster
#     return predicted_cluster

def suggestCareers(suggested_careers):
    print("Suggested Careers being called here")
    # Convert the set to a list
    careers_list = list(suggested_careers)
    # Convert the list to a dictionary
    # careers_dict = {'careers': careers_list}
    file_path = os.path.join('static','suggested_careers.json')
    print(file_path)
    with open(file_path, 'w') as json_file:
        json.dump(careers_list, json_file)
    # print("Suggested careers updated at training model.py")
    # # Top 5 recommendations
    # print("Suggested Careers (Top 5):")
    # for i, career in enumerate(careers_list):
    #     if i < 5:
    #         print(career)

def train_model():
    rename_cols(df)
    clean_data(df)
    # convert_int(df)
    # visualize_data(df)
    handle_outliers(df)
    handle_outliers(df)
    x_train, y_train = assign_datasets()
    cleaned_data = handle_outliers1(x_train)
    # cluster_analysis(cleaned_data)

    model = kmeans_plusplus(cleaned_data)
    kmeans_labels = model.labels_
    silhouette_avg = silhouette_score(cleaned_data, kmeans_labels)
    print(f"The average silhouette score is: {silhouette_avg}")
    trained_model = cleaned_data.copy()
    trained_model['Cluster'] = model.labels_
    # trained_data['Cluster2'] =kmeans_plusplus.labels_
    trained_model['Career'] = y_train['Career']
    trained_model.head()
    career_0,career_1,career_2,career_3,career_4=clustering_info(trained_model)

    #Predicting the cluster for the user input
    user_data = pd.read_excel('static/data/test_data.xlsx')
    user_data.shape
    EXT, NES, CON, AGR, OPN=group_cols(user_data)
    calculate_reverse(user_data)
    x_test=calculate_scores(user_data,OPN,CON,EXT,AGR,NES)
    x_test=x_test[['O_score','C_score','E_score','A_score','N_score']]
    predicted_cluster = model.predict(x_test)
    print(predicted_cluster)
    # predicted_cluster= predict_cluster(df_input,kmeans)
    if predicted_cluster == 0:
        suggested_careers = career_0
    elif predicted_cluster == 1:
        suggested_careers = career_1
    elif predicted_cluster == 2:
        suggested_careers = career_2
    elif predicted_cluster == 3:
        suggested_careers = career_3
    else:
        suggested_careers = career_4
    # Suggest careers based on the predicted cluster
    suggestCareers(suggested_careers)
    print(suggested_careers)
    print("End of line in training model")




